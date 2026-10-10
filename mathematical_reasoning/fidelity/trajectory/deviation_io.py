"""Shared sample selection and provenance validation using the standard library."""
import hashlib
import json
import random
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def load_docs(path):
    docs=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not docs:
        raise ValueError('Dataset is empty')
    for i,d in enumerate(docs):
        if not isinstance(d.get('question'),str) or not isinstance(d.get('answer'),str):
            raise ValueError(f'Dataset row {i} must have string question and answer')
    return docs


def require_int(name,value,minimum=0):
    if type(value) is not int or value<minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def dataset_signature(docs):
    # Completions and accuracy may differ across runs; input identities must match.
    return digest([[d['question'],d['answer']] for d in docs])


def select_indices(total,settings):
    if 'sample_indices' in settings:
        ids=settings['sample_indices']
        if not isinstance(ids,list):
            raise ValueError('sample_indices must be a list of original zero-based row indices')
    else:
        mode=settings.get('sample_selection','stride')
        count=settings.get('sample_count')
        if count is not None:
            require_int('sample_count',count,1)
        if mode=='stride':
            start=require_int('sample_start',settings.get('sample_start',0))
            step=require_int('sample_step',settings.get('sample_step',20),1)
            stop=settings.get('sample_stop')
            stop=total if stop is None else require_int('sample_stop',stop)
            if stop>total or start>=stop:
                raise ValueError(f'Invalid sampling range [{start},{stop}) for {total} rows')
            ids=list(range(start,stop,step))
            if count is not None and count>len(ids):
                raise ValueError(f'Requested {count} samples, only {len(ids)} available with this start/step/stop')
            if count is not None:
                ids=ids[:count]
        elif mode=='random':
            count=64 if count is None else count
            if count>total:
                raise ValueError('sample_count exceeds dataset size')
            ids=sorted(random.Random(settings.get('subset_seed',20260917)).sample(range(total),total)[:count])
        elif mode=='even':
            count=64 if count is None else count
            if count>total:
                raise ValueError('sample_count exceeds dataset size')
            ids=[i*total//count for i in range(count)]
        else:
            raise ValueError('sample_selection must be stride, random or even')
    if not ids or any(type(i) is not int or not 0<=i<total for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Sample indices must be nonempty, unique, zero-based original row indices within the dataset')
    return list(ids)


def make_sample_plan(docs,settings):
    ids=select_indices(len(docs),settings)
    rows=[{'sample_index':i,'question_sha256':digest(docs[i]['question']),
           'answer_sha256':digest(docs[i]['answer'])} for i in ids]
    return {'schema_version':2,'dataset_size':len(docs),'dataset_questions_sha256':dataset_signature(docs),
            'sample_indices':ids,'sample_questions':rows,'sample_set_sha256':digest(rows),
            'n_expected_samples':len(ids)}


def resolve_sample_plan(docs,manifest):
    if manifest.get('sample_plan_json'):
        path=Path(manifest['sample_plan_json'])
        stored=json.loads(path.read_text(encoding='utf-8'))
        actual=make_sample_plan(docs,{'sample_indices':stored['sample_indices']})
        for key in ('dataset_size','dataset_questions_sha256','sample_set_sha256','sample_questions'):
            if stored.get(key)!=actual[key]:
                raise ValueError(f'Shared sample plan mismatch in {key}: {path}; dataset order/content must match')
        if 'sample_indices' in manifest and manifest['sample_indices']!=actual['sample_indices']:
            raise ValueError('Manifest sample_indices disagree with shared sample plan')
        return actual
    return make_sample_plan(docs,manifest)


def point_metadata(manifest,config,hyperparams):
    figure=manifest.get('figure')
    if figure not in (2,3):
        return {}
    model=manifest.get('model',Path(manifest['model_path']).name)
    split=manifest['split_way']
    axis='num_local_forwards' if figure==2 else 'num_clients'
    value=1 if config['name']=='CenAttn' else hyperparams['num_local_forward' if figure==2 else 'num_clients']
    letter='H' if figure==2 else 'N'
    return {'figure':figure,'model':model,'split_way':split,'x_axis':axis,'x_value':value,
            'point_key':f'fig{figure}|{model}|{split}|{letter}={value}'}


def planned_points(manifest,configs):
    points=[]
    for c in configs:
        hp=dict(manifest['hyperparams']);hp.update(c)
        meta=point_metadata(manifest,c,hp)
        points.append({'config':c['name'],**meta})
    keys=[p.get('point_key',p['config']) for p in points]
    if len(keys)!=len(set(keys)):
        raise ValueError('Duplicate planned figure/x point; omit H1/N1 because CenAttn supplies x=1')
    return points


def read_completed(path,provenance,resume=False):
    path=Path(path)
    sidecar=Path(str(path)+'.manifest.json')
    if not path.exists():
        if sidecar.exists():
            old=json.loads(sidecar.read_text(encoding='utf-8'))['provenance']
            if old.get('run_id')!=provenance['run_id']:
                raise ValueError('Existing provenance belongs to a different run')
        return {}
    if not resume:
        raise FileExistsError(f'{path} already exists; use --resume to continue the identical run')
    if not sidecar.exists():
        raise ValueError('Cannot resume without existing provenance sidecar')
    old=json.loads(sidecar.read_text(encoding='utf-8'))['provenance']
    if old.get('run_id')!=provenance['run_id']:
        raise ValueError('Resume rejected: dataset, sample plan, config, source or runtime identity changed')
    expected={(i,p['config']) for i in provenance['sample_indices'] for p in provenance['planned_points']}
    questions={r['sample_index']:r['question_sha256'] for r in provenance['sample_questions']}
    complete={}
    for number,line in enumerate(path.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():
            continue
        try:
            r=json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Invalid saved JSONL line {number}; inspect interrupted final write before resume') from exc
        key=(r['sample_index'],r['config'])
        if key not in expected or r.get('run_id')!=provenance['run_id'] or r.get('question_sha256')!=questions[key[0]]:
            raise ValueError(f'Saved row {number} does not match this plan')
        if key in complete and r!=complete[key]:
            raise ValueError('Conflicting duplicate saved sample/config')
        complete[key]=r
    return complete
