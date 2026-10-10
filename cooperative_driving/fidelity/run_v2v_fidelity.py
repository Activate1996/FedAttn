# -*- coding: utf-8 -*-
"""Measure prefill KV fidelity for one cooperative-driving task and group."""

# Run one task/group, then export the fidelity and plotting data.
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import gzip
import json
import os
from pathlib import Path
import socket
import signal
import sys
import time
from fidelity_plan import SCHEMA, METRICS, build_configs, select_samples, digest, file_hash, read_json, write_json
from v2v_runtime import resolve_task_paths

HERE=Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from check_assets import check_required_features
STOP_REQUESTED=False

# One task and one group are run each time.
ACTIVE_QUESTION = 'q1'
EXPERIMENT_GROUP = 'A'  # A, B, C, or D
GPU_ID = '0'
PROJECT_ROOT = HERE.parent  # sibling LLaVA/ and DMSTrack/
SOURCE_SCRIPT = Path('LLaVA/llava/eval/model_vqa_loader_Fed_v2v_llm.py')
WRAPPER_SCRIPT = Path('LLaVA/llava/eval/Wrapper.py')
OUTPUT_ROOT = HERE / 'results/v2v_fidelity'

A_SETTING = {
    'H_values': [2, 3, 4, 8],
    'seed': 0,
    'input_keep_ratios': [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005],
}

B_SETTING = {
    'H_values': [2, 3, 4, 8],
    'seed': 0,
    'comm_keep_ratios': [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005],
}

C_SETTING = {
    'publisher_intervals': 2,
    'seed': 0,
    'collaborator_intervals': [2, 4, 6, 8, 10, 12, 14, 16],
}

D_SETTING = {'H_values': [2, 3, 4, 8], 'gamma': 1.4}

# Apply task-specific quality sampling before fidelity subsampling.
QUALITY_SAMPLE_STRIDES = {'q1': 20, 'q4': 1, 'q5': 1}
SUBSAMPLE_EVERY = 1000 # Set to 10 for the paper sampling protocol.
NUM_SHARDS = 1
SHARD_INDEX = 0
RESUME = True


def request_stop(signum,frame):
    global STOP_REQUESTED
    STOP_REQUESTED=True
    print('Stop requested; finishing the current record before exit. Resume with the same command.',file=sys.stderr,flush=True)

def check_stop():
    if STOP_REQUESTED:
        raise KeyboardInterrupt('Stopped at a record boundary; completed records are preserved')

def absolute(value,base=HERE):
    p=Path(value).expanduser(); return (p if p.is_absolute() else base/p).resolve()

def locate_project(explicit=None):
    if explicit: return absolute(explicit)
    for start in (Path.cwd(),HERE):
        for p in (start,*start.parents):
            if (p/'LLaVA'/'llava').is_dir() and (p/'DMSTrack').is_dir(): return p
    raise FileNotFoundError('Cannot locate V2V project. Put this toolkit inside the project with sibling LLaVA/DMSTrack, or set PROJECT_ROOT to the directory containing LLaVA/ and DMSTrack/')

from fidelity_locks import result_lock as output_lock, model_load_lock

def immutable_json(path,value):
    if path.exists():
        if read_json(path)!=value: raise ValueError(f'{path}: experiment settings/data/code changed. Use a new --output-root; existing records are preserved.')
    else: write_json(path,value)

def prepare(settings,root,project):
    source=absolute(settings['source_script']); wrapper=absolute(settings['wrapper_script'])
    for p in (source,wrapper):
        if not p.is_file(): raise FileNotFoundError(p)
    tasks=settings['tasks']
    if not tasks or len(set(tasks))!=len(tasks) or set(tasks)-{'q1','q4','q5'}:
        raise ValueError('tasks must be a unique subset of q1 q4 q5')
    task_plans={}; selected_rows={}
    for task in tasks:
        paths=resolve_task_paths(task,settings,source,project)
        rows=read_json(paths['question_file'])
        sampling=select_samples(rows,task,settings)
        schedules=build_configs(task,settings,source)
        task_plans[task]={**schedules,'paths':paths,'sample_plan':sampling,
                          'question_file_sha256':file_hash(paths['question_file'])}
        selected_rows[task]=[rows[s['sample_index']] for s in sampling['samples']]
        check_required_features(selected_rows[task], project, paths['num_input_frames'],
                                feature_source=paths['feature_source'])
    code_names=['fidelity_plan.py','v2v_runtime.py','fidelity_observer.py','run_v2v_fidelity.py','fidelity_locks.py']
    code_hashes={name:file_hash(HERE/name) for name in code_names}
    code_hashes.update(source_loader=file_hash(source),source_wrapper=file_hash(wrapper),
                       asset_check=file_hash(HERE.parent/'check_assets.py'))
    plan={'schema':SCHEMA,'tasks':tasks,'task_plans':task_plans,'measurement_definitions':METRICS,
          'settings':{k:v for k,v in settings.items() if k not in ('output_root','progress_every')},
          'project_root':str(project),'code_hashes':code_hashes,
          'source_script':str(source),'wrapper_script':str(wrapper)}
    plan['plan_id']=digest(plan)
    immutable_json(root/'experiment_plan.json',plan)
    for task in tasks:
        t=task_plans[task]; samples=t['sample_plan']['samples']; expected=len(samples)*len(t['configs'])
        print(f"{task}: {len(samples)} fixed samples × {len(t['configs'])} unique configurations = {expected} prefill records",flush=True)
        def preview(values):
            return values if len(values)<=20 else values[:10]+['...']+values[-3:]
        sp=t['sample_plan']
        print(f"  Global target={sp['n_global_expected_samples']}; shard {sp['shard_index']+1}/{sp['num_shards']}; method={sp['method']}",flush=True)
        print('  Selected original row indices:',preview([s['sample_index'] for s in samples]),flush=True)
        print('  Selected original sample IDs:',preview([s['sample_id'] for s in samples]),flush=True)
        print('  Complete row/ID lists saved in experiment_plan.json',flush=True)
        print(f"  Frames={t['paths']['num_input_frames']}; checkpoint={t['paths']['model_path']}",flush=True)
    return plan,selected_rows

def process(settings,root,plan,selected_rows,resume=False,no_export=False,load_lock_path=None):
    from v2v_runtime import Runtime
    from fidelity_observer import build_availability, validate_availability_against_masks, PrefillObserver
    from export_v2v_fidelity import read_records, validate_measurement, validate_reference
    for task in plan['tasks']:
        check_stop()
        tp=plan['task_plans'][task]
        compressed=settings.get('raw_compression','gzip')=='gzip'
        path=root/(f'records_{task}.jsonl.gz' if compressed else f'records_{task}.jsonl')
        if path.exists() and path.stat().st_size and not resume:
            raise FileExistsError(f'{path} already exists. Set RESUME=True for this identical experiment.')
        saved=read_records(root,plan,task,check_cancel=check_stop)
        expected=len(tp['sample_plan']['samples'])*len(tp['configs'])
        if len(saved)==expected:
            print(f'{task}: already complete ({expected}/{expected}); no model loaded.',flush=True)
            saved.close()
            continue
        runtime=None
        try:
            with model_load_lock(load_lock_path,check_cancel=check_stop):
                runtime=Runtime(task,settings,plan['source_script'],plan['wrapper_script'],plan['project_root'],selected_rows[task])
            check_stop()
            if runtime.n_layers!=32: raise ValueError('Expected the original 32-layer Vicuna backbone')
            print(f'{task}: model loaded; memory MiB {runtime.memory_stats()}',flush=True)
            metadata=runtime.metadata()
            provenance={'schema':SCHEMA,'plan_id':plan['plan_id'],'task':task,'runtime':metadata}
            provenance['run_id']=digest(provenance)
            immutable_json(root/f'runtime_{task}.json',provenance)
            run_id=provenance['run_id']; written=0
            for spec,row in zip(tp['sample_plan']['samples'],selected_rows[task]):
                check_stop()
                index=spec['sample_index']
                if all((index,c['config_id']) in saved for c in tp['configs']): continue
                sample_started=time.monotonic(); runtime.reset_memory_peak()
                sample=runtime.prepare_sample(row); reference=None; reference_input=None
                for config in tp['configs']:
                    check_stop()
                    cid=config['config_id']; is_cen=cid==tp['baselines']['CenAttn']; key=(index,cid)
                    if key in saved and not is_cen: continue
                    if not is_cen and reference is None: raise RuntimeError('Cen must run first')
                    prefill,decode=runtime.prepare_masks(sample,config)
                    availability=build_availability(config,sample['token_pos'],sample['token_chunks'],sample['sample_key'],sample['asker_cav_id'],runtime.n_layers)
                    validate_availability_against_masks(availability,prefill)
                    with PrefillObserver(runtime.model,sample['token_chunks'],sample['asker_cav_id'],availability,reference=reference) as observer:
                        runtime.forward(sample,prefill,decode)
                        measurement=observer.result()
                    input_fingerprint=runtime.last_prefill_input_sha256
                    validate_measurement(measurement,reference=is_cen)
                    if is_cen:
                        reference=observer.reference; reference_input=input_fingerprint
                        if key in saved:
                            validate_reference({'input_fingerprint':input_fingerprint,
                                                'measurement':measurement},saved[key])
                            del observer,measurement,prefill,decode,availability
                            continue
                    elif input_fingerprint!=reference_input:
                        raise ValueError('Expanded multimodal input changed across configurations')
                    record={'schema':SCHEMA,'plan_id':plan['plan_id'],'run_id':run_id,'task':task,
                            **spec,'config_id':cid,'config':config,'input_fingerprint':input_fingerprint,
                            'measurement':measurement,'generated_answer':False}
                    # Append and flush each completed sample/config independently.
                    opener=gzip.open if compressed else open
                    compression_kwargs={'compresslevel':1} if compressed else {}
                    with opener(path,'at',encoding='utf-8',**compression_kwargs) as f:
                        f.write(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n'); f.flush()
                    saved[key]=record; written+=1
                    if written==1 or written%settings.get('progress_every',20)==0 or len(saved)==expected:
                        print(f"{task} id={spec['sample_id']} config={config['name']} hidden={measurement['hidden_mean']:.6g} KV={measurement['kv_mean']:.6g} saved={len(saved)}/{expected}",flush=True)
                    del observer,measurement,prefill,decode,availability
                del sample,reference
                gc.collect()
                print(f"{task} sample={spec['sample_id']} completed in {time.monotonic()-sample_started:.1f}s; memory MiB {runtime.memory_stats()}",flush=True)
                write_json(root/f'status_{task}.json',{'schema':SCHEMA,'task':task,'expected':expected,'saved':len(saved),'complete':len(saved)==expected})
        finally:
            saved.close()
            if runtime is not None: runtime.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['q1', 'q4', 'q5'], default=ACTIVE_QUESTION)
    parser.add_argument('--group', choices=list('ABCD'), default=EXPERIMENT_GROUP)
    parser.add_argument('--gpu-id', default=GPU_ID)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--source-script', type=Path, default=SOURCE_SCRIPT)
    parser.add_argument('--wrapper-script', type=Path, default=WRAPPER_SCRIPT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--subsample-every', type=int, default=SUBSAMPLE_EVERY)
    parser.add_argument('--num-shards', type=int, default=NUM_SHARDS)
    parser.add_argument('--shard-index', type=int, default=SHARD_INDEX)
    parser.add_argument('--model-base', default='liuhaotian/llava-v1.5-7b')
    parser.add_argument('--dtype', choices=['float16', 'bfloat16', 'float32'], default='float16')
    parser.add_argument('--progress-every', type=int, default=20)
    args = parser.parse_args()
    if args.subsample_every < 1:
        parser.error('--subsample-every must be positive')
    if args.progress_every < 1:
        parser.error('--progress-every must be positive')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    project = locate_project(args.project_root)
    source = absolute(args.source_script, project)
    wrapper = absolute(args.wrapper_script, project)
    experiment_settings = {
        'A': A_SETTING, 'B': B_SETTING,
        'C': C_SETTING, 'D': D_SETTING,
    }
    settings = {
        'source_script': str(source),
        'wrapper_script': str(wrapper),
        'tasks': [args.task],
        'experiment_group': args.group,
        'experiment_settings': experiment_settings,
        'panels': [args.group],
        'sampling_mode': 'stride',
        'sample_stride': QUALITY_SAMPLE_STRIDES[args.task] * args.subsample_every,
        'sample_indices_by_task': {},
        'sample_ids_by_task': {},
        'num_shards': args.num_shards,
        'shard_index': args.shard_index,
        'seed': experiment_settings[args.group].get('seed', 0),
        'dtype': args.dtype,
        'device': 'cuda:0',
        'model_base': args.model_base,
        'feature_source': 'no_fusion_keep_all',
        'task_overrides': {},
        'progress_every': args.progress_every,
        'raw_compression': 'gzip',
    }
    output_base = absolute(args.output_root)
    root = output_base / args.task / args.group
    if args.num_shards > 1:
        root = root / f'shard{args.shard_index}'
    load_lock = output_base / '.model_load_locks' / f'gpu_{args.gpu_id.replace(",", "_")}.lock'

    from export_v2v_fidelity import export_results
    from prepare_plot_data import prepare as prepare_plot_data

    with output_lock(root):
        plan, rows = prepare(settings, root, project)
        try:
            process(settings, root, plan, rows, resume=RESUME,
                    load_lock_path=load_lock)
        except BaseException:
            try:
                summary = export_results(root, make_bundle=False)
                prepare_plot_data(summary, root / 'plot_inputs')
            except Exception as exc:
                print(f'Partial export unavailable: {exc}', file=sys.stderr, flush=True)
            raise
        summary = export_results(root, make_bundle=False)
        prepare_plot_data(summary, root / 'plot_inputs')


if __name__ == '__main__':
    main()
