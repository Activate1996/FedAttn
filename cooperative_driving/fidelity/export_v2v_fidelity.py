"""CPU-only validation/export of saved sampled prefill measurements."""
from __future__ import annotations
import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import shutil
import statistics
import zipfile
from fidelity_plan import SCHEMA, METRICS, digest, read_json, write_json

def require(ok,message):
    if not ok: raise ValueError(message)

def finite(value,label):
    require(type(value) in (int,float) and math.isfinite(value) and value>=0,f'{label}: invalid nonnegative number')
    return value

def close(a,b,label):
    require(math.isclose(a,b,rel_tol=2e-5,abs_tol=1e-7),f'{label}: inconsistent calculation ({a}, {b})')

def validate_measurement(m,reference=False):
    require(m.get('hidden_metric_id')==METRICS['hidden']['id'] and m.get('kv_metric_id')==METRICS['kv']['id'], 'Wrong measurement definition')
    require(m.get('layer_count')==32 and type(m.get('n_tokens'))==int and m['n_tokens']>0,'Expected 32 complete blocks and valid token count')
    n=m['n_tokens']; hidden=m.get('hidden_layers',[]); kv=m.get('kv_layers',[])
    require(len(hidden)==len(kv)==32,'Missing layer measurements')
    require([x.get('block_index') for x in hidden]==list(range(1,33)) and [x.get('block_index') for x in kv]==list(range(1,33)),'Layers must appear once in order')
    publisher=None
    for h,k in zip(hidden,kv):
        require(h.get('n_tokens')==n,'Hidden token count differs across blocks')
        for key in ('relative_frobenius','absolute_frobenius','reference_frobenius'): finite(h.get(key),'hidden '+key)
        close(h['relative_frobenius'],h['absolute_frobenius']/max(h['reference_frobenius'],1e-12),'hidden norm')
        ref=k['reference_kv']
        require(ref.get('n_tokens')==n and all(type(ref.get(x))==int and ref[x]>0 for x in ('n_kv_heads','head_dim')),'Invalid KV shape')
        denominator=finite(ref['reference_frobenius'],'KV reference norm')
        ps=k.get('participant_kv',[])
        require(len(ps)==2 and [p.get('participant') for p in ps]==['ego','car1'],'Expected both participant views')
        pub=[p['participant'] for p in ps if p.get('is_publisher') is True]
        require(len(pub)==1 and (publisher is None or pub[0]==publisher),'Publisher changed between blocks')
        publisher=pub[0]
        for i,p in enumerate(ps):
            require(p.get('participant_index')==i and type(p.get('is_publisher'))==bool,'Invalid participant identity')
            require(p.get('n_tokens')==n and type(p.get('n_available_tokens'))==int and 0<=p['n_available_tokens']<=n,'Invalid availability count')
            for field in ('relative_frobenius','absolute_frobenius','reference_frobenius','coverage','available_error_squared','missing_reference_squared','total_reference_squared'):
                finite(p.get(field),'KV '+field)
            close(p['reference_frobenius'],denominator,'global Cen denominator')
            close(p['total_reference_squared'],denominator**2,'Cen squared norm')
            close(p['coverage'],p['n_available_tokens']/n,'KV coverage')
            require(p['missing_reference_squared']<=p['total_reference_squared']+max(1e-7,p['total_reference_squared']*2e-5),'Missing energy exceeds full reference')
            close(p['absolute_frobenius'],math.sqrt(p['available_error_squared']+p['missing_reference_squared']),'error decomposition')
            close(p['relative_frobenius'],p['absolute_frobenius']/max(denominator,1e-12),'KV relative norm')
            if p['n_available_tokens']==n: close(p['missing_reference_squared'],0,'full-view missing energy')
            if p['n_available_tokens']==0: close(p['available_error_squared'],0,'empty-view available error')
            if reference: require(p['relative_frobenius']==0 and p['n_available_tokens']==n,'CenAttn must have zero error and full coverage')
        close(finite(k['participant_mean_relative_kv_frobenius'],'layer KV mean'),statistics.fmean(p['relative_frobenius'] for p in ps),'participant average')
        if reference: require(h['relative_frobenius']==0,'Cen hidden self-error must be zero')
    close(finite(m['hidden_mean'],'hidden mean'),statistics.fmean(h['relative_frobenius'] for h in hidden),'block hidden average')
    close(finite(m['kv_mean'],'KV mean'),statistics.fmean(k['participant_mean_relative_kv_frobenius'] for k in kv),'block KV average')
    ps=m['participant_means']; require(len(ps)==2,'Missing participant mean')
    for i,p in enumerate(ps):
        require(p.get('participant_index')==i and p.get('participant')==['ego','car1'][i] and p.get('is_publisher')==(p['participant']==publisher),'Invalid averaged participant identity')
        close(finite(p['relative_kv_frobenius'],'participant mean'),statistics.fmean(k['participant_kv'][i]['relative_frobenius'] for k in kv),'participant block average')


# Raw records contain 32 blocks and two participant views. Keeping all of them
# in RAM becomes prohibitively expensive for full-QA runs. The temporary SQLite
# index below holds keys plus only one small Cen reference per sample on disk.
import contextlib
import sqlite3
import tempfile


def compact_record(row):
    m = row['measurement']
    return {'input_fingerprint': row['input_fingerprint'], 'measurement': {
        'n_tokens': m['n_tokens'],
        'hidden_layers': [{'reference_frobenius': h['reference_frobenius']} for h in m['hidden_layers']],
        'kv_layers': [{'reference_kv': k['reference_kv']} for k in m['kv_layers']],
    }}


class ResumeIndex:
    """Disk-backed mapping of completed (sample_index, config_id) keys.

    Only Cen payloads are retained. Non-Cen values are empty dictionaries;
    callers should use membership tests for them. Assignment accepts a complete
    record and compacts it immediately. close() removes the temporary database.
    """
    def __init__(self, cen_id):
        self.cen_id = cen_id
        self._temp = tempfile.TemporaryDirectory(prefix='v2v-fidelity-index-')
        self._db = sqlite3.connect(str(Path(self._temp.name) / 'index.sqlite'))
        self._db.execute('PRAGMA journal_mode=OFF')
        self._db.execute('PRAGMA synchronous=OFF')
        self._db.execute('PRAGMA cache_size=-4096')
        self._db.execute('CREATE TABLE saved (sample INTEGER, config TEXT, value TEXT, PRIMARY KEY(sample, config)) WITHOUT ROWID')
        self._count = 0
        self._cached_cen = None

    def __len__(self):
        return self._count

    def __contains__(self, key):
        return self._db.execute('SELECT 1 FROM saved WHERE sample=? AND config=?', key).fetchone() is not None

    def __getitem__(self, key):
        if self._cached_cen is not None and self._cached_cen[0] == key:
            return self._cached_cen[1]
        found = self._db.execute('SELECT value FROM saved WHERE sample=? AND config=?', key).fetchone()
        if found is None:
            raise KeyError(key)
        value = json.loads(found[0]) if found[0] is not None else {}
        if key[1] == self.cen_id:
            self._cached_cen = (key, value)
        return value

    def __setitem__(self, key, row):
        value = compact_record(row) if key[1] == self.cen_id else None
        try:
            self._db.execute('INSERT INTO saved VALUES (?,?,?)', (*key, json.dumps(value) if value is not None else None))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f'Duplicate saved sample/config {key}; preserve and inspect the original JSONL') from exc
        self._count += 1
        if value is not None:
            self._cached_cen = (key, value)
        if self._count % 4096 == 0:
            self._db.commit()

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None
            self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if getattr(self, '_db', None) is not None:
            self.close()


def validate_reference(row, cen):
    require(row['input_fingerprint'] == cen['input_fingerprint'], 'Inputs differ from saved Cen reference')
    require(row['measurement']['n_tokens'] == cen['measurement']['n_tokens'], 'Token count differs from Cen reference')
    for h, ch, k, ck in zip(row['measurement']['hidden_layers'], cen['measurement']['hidden_layers'],
                           row['measurement']['kv_layers'], cen['measurement']['kv_layers']):
        close(h['reference_frobenius'], ch['reference_frobenius'], 'reference hidden norm changed')
        for field in ('n_tokens', 'n_kv_heads', 'head_dim'):
            require(k['reference_kv'][field] == ck['reference_kv'][field], 'reference KV dimensions changed')
        close(k['reference_kv']['reference_frobenius'], ck['reference_kv']['reference_frobenius'], 'reference KV norm changed')


def _record_stream(root, plan, task, index, check_cancel=None):
    tp = plan['task_plans'][task]
    candidates = [Path(root) / f'records_{task}.jsonl', Path(root) / f'records_{task}.jsonl.gz']
    existing = [p for p in candidates if p.exists()]
    require(len(existing) <= 1, f'{task}: both compressed and uncompressed records exist; do not mix formats')
    path = existing[0] if existing else candidates[0]
    expected = {s['sample_index']: s for s in tp['sample_plan']['samples']}
    configs = {c['config_id']: c for c in tp['configs']}
    if not path.exists() or not path.stat().st_size:
        return
    runtime_path = Path(root) / f'runtime_{task}.json'
    require(runtime_path.exists(), f'{task}: runtime metadata missing')
    provenance = read_json(runtime_path)
    require(provenance.get('schema') == SCHEMA and provenance.get('plan_id') == plan['plan_id'], 'Runtime metadata does not match plan')
    require(provenance.get('run_id') == digest({k: v for k, v in provenance.items() if k != 'run_id'}), 'Runtime identity mismatch')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if check_cancel is not None: check_cancel()
            if not line.strip():
                continue
            tag = f'{path}:{line_number}'
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(tag + ': interrupted/invalid JSONL; preserve the file and inspect the final line before resume') from exc
            sample_index = row.get('sample_index'); cid = row.get('config_id')
            require(row.get('schema') == SCHEMA and row.get('task') == task and row.get('plan_id') == plan['plan_id'], tag + ': wrong experiment')
            require(type(sample_index) is int and sample_index in expected and cid in configs, tag + ': unexpected sample/config')
            require(row.get('run_id') == provenance['run_id'], tag + ': run identity changed')
            require(row.get('sample_id') == expected[sample_index]['sample_id'] and row.get('row_sha256') == expected[sample_index]['row_sha256'], tag + ': sample changed')
            require(row.get('config') == configs[cid], tag + ': config changed')
            require(isinstance(row.get('input_fingerprint'), str) and row['input_fingerprint'], tag + ': no input fingerprint')
            require(row.get('generated_answer') is False, tag + ': expected prefill-only measurement')
            is_cen = cid == tp['baselines']['CenAttn']
            validate_measurement(row['measurement'], is_cen)
            if not is_cen:
                try:
                    cen = index[(sample_index, tp['baselines']['CenAttn'])]
                except KeyError as exc:
                    raise ValueError(tag + ': sample lacks an earlier saved Cen reference') from exc
                validate_reference(row, cen)
            index[(sample_index, cid)] = row
            yield row


def iter_validated_records(root, plan, task):
    """Yield validated full rows once, using disk-backed keys and Cen references."""
    with ResumeIndex(plan['task_plans'][task]['baselines']['CenAttn']) as index:
        yield from _record_stream(root, plan, task, index)


def read_records(root, plan, task, check_cancel=None):
    """Build the compact disk-backed resume index; never retain full measurements."""
    index = ResumeIndex(plan['task_plans'][task]['baselines']['CenAttn'])
    try:
        for _ in _record_stream(root, plan, task, index, check_cancel=check_cancel):
            pass
    except BaseException:
        index.close()
        raise
    return index


class RunningStats:
    def __init__(self):
        self.n = 0; self.mean = 0.0; self.m2 = 0.0

    def add(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)


class MeasurementAggregate:
    def __init__(self):
        self.metrics = {m: RunningStats() for m in METRICS}
        self.layers = {m: [RunningStats() for _ in range(32)] for m in METRICS}
        self.roles = {r: RunningStats() for r in ('publisher', 'collaborator')}

    def add(self, measurement):
        for metric in METRICS:
            self.metrics[metric].add(measurement[metric + '_mean'])
            field = 'relative_frobenius' if metric == 'hidden' else 'participant_mean_relative_kv_frobenius'
            for state, layer in zip(self.layers[metric], measurement[metric + '_layers']):
                state.add(layer[field])
        for participant in measurement['participant_means']:
            self.roles['publisher' if participant['is_publisher'] else 'collaborator'].add(participant['relative_kv_frobenius'])

    def result(self, expected, metric):
        state = self.metrics[metric]; n = state.n
        require(n <= expected, 'More records than planned samples')
        complete = n == expected and expected > 0
        status = 'complete' if complete else ('incomplete' if n else 'missing')
        return {'n': n, 'n_expected': expected, 'status': status,
                'mean': state.mean if complete else None,
                'std': math.sqrt(max(0.0, state.m2 / n)) if complete else None,
                'sem': math.sqrt(max(0.0, state.m2 / (n * (n - 1)))) if complete and n > 1 else None,
                'layer_means': [{'block_index': i + 1, 'mean': s.mean if complete else None} for i, s in enumerate(self.layers[metric])],
                'participant_means': [{'role': r, 'mean': s.mean if complete else None} for r, s in self.roles.items()] if metric == 'kv' else []}


def aggregate(rows, expected, metric):
    state = MeasurementAggregate()
    for row in rows:
        state.add(row['measurement'])
    return state.result(expected, metric)


@contextlib.contextmanager
def csv_writer(path, fields):
    path = Path(path); temp = path.with_name(path.name + '.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            yield writer
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


def write_csv(path, rows):
    rows = list(rows)  # Only small point tables use this compatibility helper.
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with csv_writer(path, fields) as writer:
        writer.writerows(rows)


def load_plan(root):
    plan = read_json(Path(root) / 'experiment_plan.json')
    require(plan.get('schema') == SCHEMA, 'Wrong experiment plan')
    require(plan.get('plan_id') == digest({k: v for k, v in plan.items() if k != 'plan_id'}), 'Plan identity mismatch')
    require(plan.get('measurement_definitions') == METRICS, 'Plan metric definitions changed')
    return plan


def _global_sampling(sample_plan):
    samples = sample_plan.get('global_samples', sample_plan['samples'])
    expected = sample_plan.get('n_global_expected_samples', len(samples))
    require(expected == len(samples), 'Global sample count differs from global sample list')
    require(sample_plan.get('global_sample_set_sha256', digest(samples)) == digest(samples), 'Global sample set identity mismatch')
    require(sample_plan['n_expected_samples'] == len(sample_plan['samples']), 'Local sample count differs from sample list')
    require(sample_plan['sample_set_sha256'] == digest(sample_plan['samples']), 'Local sample set identity mismatch')
    return samples


def _merge_sources(roots):
    """Validate disjoint, complete shard assignment before opening raw data."""
    sources = []; task_sources = {}; reference_code = None; reference_metrics = None
    for root in roots:
        root = Path(root).expanduser().resolve()
        require(root not in [r for r, _ in sources], f'Duplicate input directory: {root}')
        plan = load_plan(root)
        if reference_code is None:
            reference_code = plan['code_hashes']; reference_metrics = plan['measurement_definitions']
        require(plan['code_hashes'] == reference_code and plan['measurement_definitions'] == reference_metrics, 'Cannot merge different measurement code or metric definitions')
        sources.append((root, plan))
        for task in plan['tasks']:
            task_sources.setdefault(task, []).append((root, plan))
    merged_samples = {}; merged_task_plans = {}
    for task, entries in task_sources.items():
        first = entries[0][1]['task_plans'][task]
        global_samples = _global_sampling(first['sample_plan'])
        global_by_index = {s['sample_index']: s for s in global_samples}
        require(len(global_by_index) == len(global_samples), f'{task}: duplicate planned global sample')
        seen = set(); shard_ids = set(); runtime_reference = None
        n_shards = first['sample_plan'].get('num_shards', 1)
        for root, plan in entries:
            tp = plan['task_plans'][task]; sp = tp['sample_plan']
            for key in ('configs', 'points', 'baselines', 'paths', 'question_file_sha256'):
                require(tp[key] == first[key], f'{task}: incompatible {key} across input roots')
            require(_global_sampling(sp) == global_samples, f'{task}: different global sample selection across shards')
            require(sp.get('num_shards', 1) == n_shards, f'{task}: shard counts differ')
            shard_id = sp.get('shard_index', 0)
            require(type(shard_id) is int and 0 <= shard_id < n_shards and shard_id not in shard_ids, f'{task}: repeated/invalid shard index')
            shard_ids.add(shard_id)
            require(sp['samples'] == global_samples[shard_id::n_shards], f'{task}: shard sample assignment differs from its declared index')
            for sample in sp['samples']:
                idx = sample['sample_index']
                require(idx not in seen and sample == global_by_index.get(idx), f'{task}: overlapping or changed sample {idx}')
                seen.add(idx)
            # Physical GPU number can differ; the model, dtype, software and
            # checkpoint identities must agree for a given task.
            runtime_path = root / f'runtime_{task}.json'
            if runtime_path.exists():
                metadata = read_json(runtime_path)['runtime']
                comparable = {k: v for k, v in metadata.items() if k != 'device'}
                if runtime_reference is None:
                    runtime_reference = comparable
                require(comparable == runtime_reference, f'{task}: model/runtime identities differ across shards')
        require(seen == set(global_by_index) and shard_ids == set(range(n_shards)),
                f'{task}: input directories do not cover all {n_shards} shards; missing shard indices {sorted(set(range(n_shards)) - shard_ids)}')
        sp = dict(first['sample_plan'])
        sp.update(samples=global_samples, n_expected_samples=len(global_samples), sample_set_sha256=digest(global_samples))
        sp.pop('shard_index', None); sp['num_shards_merged'] = n_shards
        merged_samples[task] = sp
        merged_task_plans[task] = first
    return sources, task_sources, merged_task_plans, merged_samples


def _make_plot_bundle(root, summary):
    from prepare_plot_data import prepare

    prepared = prepare(summary, root / 'plot_inputs')
    entries = [prepared, prepared.parent / 'v2v_fidelity_plot_points.csv']
    bundle = root / 'v2v_fidelity_plot_inputs.zip'
    temp = bundle.with_name(bundle.name + '.tmp')
    with zipfile.ZipFile(temp, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in entries:
            archive.write(path, path.name)
    temp.replace(bundle)
    print('Plot-data bundle:', bundle, flush=True)


def _export_streaming(root, tasks, task_sources, task_plans, sample_plans, identity, make_bundle, export_layers):
    root.mkdir(parents=True, exist_ok=True)
    out = {'schema': SCHEMA, 'tasks': tasks, **identity, 'metric_definitions': METRICS,
           'sample_plans': sample_plans, 'points': [], 'baselines': [],
           'point_counts': {'complete': 0, 'incomplete': 0, 'missing': 0},
           'note': 'Fidelity uses fixed matched samples across configurations within each task. Each sample receives equal weight, including when shard sizes differ.'}
    base_fields = ['task', 'sample_index', 'sample_id', 'config_id', 'config_name']
    hidden_fields = ['block_index', 'n_tokens', 'relative_frobenius', 'absolute_frobenius', 'reference_frobenius']
    kv_fields = ['block_index', 'participant', 'participant_index', 'is_publisher', 'n_tokens', 'n_available_tokens', 'relative_frobenius', 'absolute_frobenius', 'reference_frobenius', 'coverage', 'available_error_squared', 'missing_reference_squared', 'total_reference_squared']
    with contextlib.ExitStack() as stack:
        sample_csv = stack.enter_context(csv_writer(root / 'v2v_fidelity_samples.csv', base_fields + ['hidden_mean', 'kv_mean', 'n_tokens']))
        hidden_csv = stack.enter_context(csv_writer(root / 'v2v_fidelity_hidden_layers.csv', base_fields + hidden_fields)) if export_layers else None
        kv_csv = stack.enter_context(csv_writer(root / 'v2v_fidelity_layer_participants.csv', base_fields + kv_fields)) if export_layers else None
        for task in tasks:
            tp = task_plans[task]
            expected = sample_plans[task]['n_expected_samples']
            states = {c['config_id']: MeasurementAggregate() for c in tp['configs']}
            count = 0
            for source_root, plan in task_sources[task]:
                for row in iter_validated_records(source_root, plan, task):
                    cid = row['config_id']; m = row['measurement']
                    states[cid].add(m)
                    base = {'task': task, 'sample_index': row['sample_index'], 'sample_id': row['sample_id'], 'config_id': cid, 'config_name': row['config']['name']}
                    sample_csv.writerow({**base, 'hidden_mean': m['hidden_mean'], 'kv_mean': m['kv_mean'], 'n_tokens': m['n_tokens']})
                    if export_layers:
                        for h in m['hidden_layers']:
                            hidden_csv.writerow({**base, **{k: h[k] for k in hidden_fields}})
                        for layer in m['kv_layers']:
                            for participant in layer['participant_kv']:
                                kv_csv.writerow({**base, 'block_index': layer['block_index'], **{k: participant[k] for k in kv_fields if k != 'block_index'}})
                    count += 1
                    if count % 10000 == 0:
                        print(f'Export {task}: {count} records validated', flush=True)
            for point in tp['points']:
                result = {**point, **states[point['config_id']].result(expected, point['metric'])}
                out['points'].append(result); out['point_counts'][result['status']] += 1
            for name, cid in tp['baselines'].items():
                for metric in METRICS:
                    out['baselines'].append({'task': task, 'name': name, 'metric': metric, **states[cid].result(expected, metric)})
    summary = root / 'v2v_fidelity_summary.json'
    write_json(summary, out)
    write_csv(root / 'v2v_fidelity_points.csv', [{k: v for k, v in p.items() if k not in ('layer_means', 'participant_means')} for p in out['points']])
    print('Fidelity points:', out['point_counts'], flush=True)
    print('Summary:', summary, flush=True)
    if make_bundle:
        _make_plot_bundle(root, summary)
    return summary


def export_results(root, make_bundle=True, export_layers=False):
    root = Path(root).expanduser().resolve(); plan = load_plan(root)
    tasks = plan['tasks']
    return _export_streaming(root, tasks, {t: [(root, plan)] for t in tasks}, plan['task_plans'],
                             {t: plan['task_plans'][t]['sample_plan'] for t in tasks},
                             {'plan_id': plan['plan_id']}, make_bundle, export_layers)


def merge_results(roots, output_root, make_bundle=True, export_layers=False):
    from fidelity_locks import result_lock
    require(bool(roots), 'At least one input directory is required')
    resolved=[Path(r).expanduser().resolve() for r in roots]
    root=Path(output_root).expanduser().resolve()
    require(root not in resolved, 'Merged output must be a separate directory')
    with contextlib.ExitStack() as locks:
        for source_root in sorted(set(resolved)):
            locks.enter_context(result_lock(source_root,reader=True))
        locks.enter_context(result_lock(root))
        sources, task_sources, task_plans, sample_plans = _merge_sources(resolved)
        tasks = [t for t in ('q1', 'q4', 'q5') if t in task_sources]
        identity = {'source_plans': [{'root': str(r), 'plan_id': p['plan_id']} for r, p in sources]}
        identity['plan_id'] = digest(identity)
        return _export_streaming(root, tasks, task_sources, task_plans, sample_plans, identity, make_bundle, export_layers)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True)
    p.add_argument('--export-layers', action='store_true', help='Also stream the very large per-block/participant CSVs; default exports summary, samples and points only')
    p.add_argument('--no-bundle', action='store_true')
    args = p.parse_args()
    from fidelity_locks import result_lock
    with result_lock(args.output_root):
        export_results(args.output_root, make_bundle=not args.no_bundle, export_layers=args.export_layers)


if __name__ == '__main__':
    main()
