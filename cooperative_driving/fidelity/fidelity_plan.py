"""CPU-only fixed sampling and exact experiment schedules for V2V fidelity."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

SCHEMA = 'v2v-prefill-fidelity-v1'
N_LAYERS = 32
METRICS = {
    'hidden': {'id': 'prefill_mean_block_global_hidden_relative_frobenius_v1',
               'formula': 'mean_sample mean_block ||X_Fed-X_Cen||_F / max(||X_Cen||_F,1e-12)',
               'tokens': 'All effective prefill input positions, including expanded sensor tokens and prompt text; no generated answer tokens.',
               'stage': 'Output of the complete decoder block, before final model norm; no participant weighting.'},
    'kv': {'id': 'prefill_available_global_kv_mean_block_participant_v1',
           'formula': 'mean_sample mean_block mean_participant sqrt(||K_available-K_Cen||_F^2+||V_available-V_Cen||_F^2)/max(sqrt(||K_Cen||_F^2+||V_Cen||_F^2),1e-12)',
           'tokens': 'Global aligned token axis; unavailable current-layer KV rows contribute zero on the participant side; full Cen reference retained.',
           'stage': 'Actual cached post-RoPE K and projected V with native KV heads; availability before causal query masking.'},
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def source_config_functions(path):
    """Load the inference configuration builder and settings without GPU execution."""
    names={'build_experiment_configs','stride_progressive_dec_back'}
    setting_names={f'{group}_SETTING' for group in 'HABCD'}
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'),str(path))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=names:
        raise ValueError('Source loader is missing the current experiment configuration functions')
    env={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in setting_names:
                    env[target.id]=ast.literal_eval(node.value)
    if not setting_names.issubset(env):
        raise ValueError('Source loader is missing one of H/A/B/C/D_SETTING')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),env)
    return env

def build_configs(task,settings,source_script):
    functions=source_config_functions(source_script)
    experiment_group=settings['experiment_group']
    if experiment_group not in ('A','B','C','D'):
        raise ValueError('experiment_group must be A, B, C or D')
    for group,value in settings.get('experiment_settings',{}).items():
        if group not in ('A','B','C','D') or not isinstance(value,dict):
            raise ValueError('experiment_settings must map A/B/C/D to setting dictionaries')
        functions[f'{group}_SETTING']=dict(value)
    functions['EXPERIMENT_GROUP']=experiment_group
    selected=functions[f'{experiment_group}_SETTING']
    groups=functions['build_experiment_configs'](N_LAYERS,task)
    seed=selected.get('seed',0)
    configs=[]; points=[]; seen={}
    def add(name,global_layers,setting,H=None):
        gl=sorted(global_layers)
        if len(set(gl))!=len(gl) or any(type(i)!=int or not 0<=i<32 for i in gl):
            raise ValueError('Invalid zero-based global layer list')
        value={'name':name,'global_layers':gl,'local_layers':sorted(set(range(32))-set(gl)),
               'setting':setting,'H':H}
        physical={'global_layers':gl,'setting':{k:v for k,v in setting.items()
                  if k not in ('name','panel','shared_baseline','c_sweep_role')}}
        cid=digest(physical)[:20]
        if cid not in seen:
            value['config_id']=cid; configs.append(value); seen[cid]=value
        return cid
    full={'name':f'full_s{seed}','panel':'reference','ratios':{},'input_keep_ratio':1.0,
          'seed':seed,'collaborator_global_layers':None}
    cen=add('CenAttn',range(32),dict(full))
    def point(panel,H,x,cid,role=None,strategy=None,G=None):
        p={'task':task,'panel':panel,'H':H,'x':x,'role':role,'strategy':strategy,'G':G,
           'config_id':cid,'metric':'kv'}
        if p not in points: points.append(p)
    for group in groups:
        for layer_name,gl,ll in group['layer_configs']:
            if sorted(ll)!=sorted(set(range(N_LAYERS))-set(gl)):
                raise ValueError('Source global/local layers do not partition all layers')
            h=group['H']
            strategy=None
            if experiment_group=='D':
                strategy,separator,budget=layer_name.rpartition('_N')
                if (not separator or strategy not in
                        ('cumul_shallow','cumul_deep','cumul_sparse','cumul_dense')
                        or int(budget)!=len(gl)):
                    raise ValueError(f'Unexpected D configuration name: {layer_name}')
                candidates=[value for value in selected['H_values']
                            if len(range(0,N_LAYERS,value))==len(gl)]
                if len(candidates)!=1:
                    raise ValueError('D H_values must have distinct synchronization budgets')
                h=candidates[0]
            for original_setting in group['settings']:
                setting=dict(original_setting)
                if experiment_group=='D':
                    setting.setdefault('input_keep_ratio',1.0)
                    setting.setdefault('seed',0)
                    setting.setdefault('collaborator_global_layers',None)
                cid=add(layer_name+'__'+setting['name'],gl,setting,h)
                if experiment_group=='A':
                    point('A',h,setting['input_keep_ratio'],cid)
                elif experiment_group=='B':
                    point('B',h,setting['ratios'].get('ego',1.0),cid)
                elif experiment_group=='C':
                    point('C',h,setting['collaborator_interval'],cid,role='collaborator')
                else:
                    point('D',h,h,cid,strategy=strategy,G=len(gl))
    required={p['config_id'] for p in points}|{cen}
    configs=[c for c in configs if c['config_id'] in required]
    return {'configs':configs,'points':points,'baselines':{'CenAttn':cen}}

def select_samples(rows,task,settings):
    """Select dataset rows, then partition their selected positions among shards.

    For stride 20, using row_index % 2 would assign all selected rows to
    shard 0. Each shard records the complete target set and its own subset.
    """
    if not isinstance(rows,list) or not rows:
        raise ValueError(f'{task}: QA JSON must contain a nonempty list')
    mode=settings.get('sampling_mode','count')
    if task in settings.get('sample_indices_by_task',{}):
        indices=settings['sample_indices_by_task'][task]; method='explicit_row_indices'
    elif task in settings.get('sample_ids_by_task',{}):
        indices=[]; method='explicit_sample_ids'
        for sample_id in settings['sample_ids_by_task'][task]:
            found=[i for i,r in enumerate(rows) if str(r.get('id'))==str(sample_id)]
            if len(found)!=1: raise ValueError(f'{task}: ID {sample_id} has {len(found)} matches; use explicit row indices')
            indices.extend(found)
    elif mode=='all':
        indices=list(range(len(rows))); method='all_full_QA_rows'
    elif mode=='stride':
        stride=settings.get('sample_stride')
        if type(stride)!=int or stride<1:
            raise ValueError('sample_stride must be a positive integer')
        indices=list(range(0,len(rows),stride)); method='fixed_stride_full_QA_rows'
    elif mode=='count':
        n=settings.get('sample_count',8)
        if type(n)!=int or not 1<=n<=len(rows): raise ValueError(f'{task}: invalid sample_count={n}')
        indices=[0] if n==1 else [i*(len(rows)-1)//(n-1) for i in range(n)]
        method='evenly_spaced_full_QA_rows'
    else:
        raise ValueError(f'Unknown sampling_mode {mode!r}')
    if not isinstance(indices,list) or not indices or len(set(indices))!=len(indices) or any(type(i)!=int or not 0<=i<len(rows) for i in indices):
        raise ValueError(f'{task}: sample indices must be unique valid zero-based row indices')
    num_shards=settings.get('num_shards',1); shard_index=settings.get('shard_index',0)
    if type(num_shards)!=int or num_shards<1 or num_shards>len(indices):
        raise ValueError(f'{task}: num_shards must be between 1 and selected sample count')
    if type(shard_index)!=int or not 0<=shard_index<num_shards:
        raise ValueError('shard_index must be in [0, num_shards)')
    cap=settings.get('max_samples_per_shard')
    if cap is not None and (type(cap)!=int or cap<1):
        raise ValueError('max_samples_per_shard must be positive')
    partitions=[indices[i::num_shards] for i in range(num_shards)]
    if cap is not None: partitions=[part[:cap] for part in partitions]
    retained={i for part in partitions for i in part}
    global_indices=[i for i in indices if i in retained]
    selected=[]
    for i in global_indices:
        row=rows[i]
        if not isinstance(row,dict) or not isinstance(row.get('conversations'),list) or not row['conversations']:
            raise ValueError(f'{task}[{i}]: missing conversations')
        if row.get('id') is None: raise ValueError(f'{task}[{i}]: missing ID required by original sparse-mask RNG')
        selected.append({'sample_index':i,'sample_id':row['id'],'row_sha256':digest(row),
                         'asker_cav_id':row.get('asker_cav_id','ego')})
    local_set=set(partitions[shard_index]); local=[s for s in selected if s['sample_index'] in local_set]
    return {'sampling_schema_version':2,'task':task,'dataset_size':len(rows),
            'method':method,'sample_stride':settings.get('sample_stride') if mode=='stride' else None,
            'num_shards':num_shards,'shard_index':shard_index,
            'max_samples_per_shard':cap,'smoke_limited':cap is not None,
            'n_selected_before_limit':len(indices),
            'n_global_expected_samples':len(selected),'global_samples':selected,
            'global_sample_set_sha256':digest(selected),
            'n_expected_samples':len(local),'samples':local,'sample_set_sha256':digest(local)}
