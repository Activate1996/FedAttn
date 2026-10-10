#!/usr/bin/env python3
"""Prepare validated ABCD KV fidelity tables for visualization."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_ROOT = HERE / 'results/v2v_fidelity'
SUMMARY_PATHS = [
    RESULTS_ROOT / 'q1/A/v2v_fidelity_summary.json',
    RESULTS_ROOT / 'q1/B/v2v_fidelity_summary.json',
    RESULTS_ROOT / 'q1/C/v2v_fidelity_summary.json',
    RESULTS_ROOT / 'q1/D/v2v_fidelity_summary.json',
    RESULTS_ROOT / 'q4/A/v2v_fidelity_summary.json',
    RESULTS_ROOT / 'q4/B/v2v_fidelity_summary.json',
    RESULTS_ROOT / 'q4/C/v2v_fidelity_summary.json',
    RESULTS_ROOT / 'q4/D/v2v_fidelity_summary.json',
    RESULTS_ROOT / 'q5/A/v2v_fidelity_summary.json',
    RESULTS_ROOT / 'q5/B/v2v_fidelity_summary.json',
    RESULTS_ROOT / 'q5/C/v2v_fidelity_summary.json',
    RESULTS_ROOT / 'q5/D/v2v_fidelity_summary.json',
]
OUTPUT_DIR = RESULTS_ROOT / 'plot_inputs'


TASKS = ('q1', 'q4', 'q5')
EXPECTED_METRIC_IDS = {
    'hidden': 'prefill_mean_block_global_hidden_relative_frobenius_v1',
    'kv': 'prefill_available_global_kv_mean_block_participant_v1',
}
HS = (2, 3, 4, 8)
H_TO_G = {2: 16, 3: 11, 4: 8, 8: 4}
C_ROLES = ('collaborator', 'publisher')
SCHEMES = ('cumul_shallow', 'cumul_deep', 'cumul_sparse', 'cumul_dense')


def read_summary(source):
    if isinstance(source, (str, Path)):
        source = json.loads(Path(source).read_text(encoding='utf-8-sig'))
    if source.get('schema') != 'v2v-prefill-fidelity-v1':
        raise ValueError('Expected v2v-prefill-fidelity-v1 summary; do not mix GSM8K or old metrics.')
    for name, expected_id in EXPECTED_METRIC_IDS.items():
        actual_id = source.get('metric_definitions', {}).get(name, {}).get('id')
        if actual_id != expected_id:
            raise ValueError(f'Incompatible {name} metric: expected {expected_id}, got {actual_id!r}')
    return source


def metric_points(summary, task, panel, metric):
    """Retain all planned x positions; incomplete means stay None and break lines."""
    source = read_summary(summary)
    points = []
    for p in source.get('points', []):
        if p['task'] != task or p['panel'] != panel:
            continue
        if p.get('metric') != metric:
            continue
        mean, n = p.get('mean'), p.get('n', 0)
        if p.get('status') in ('incomplete', 'missing'):
            if mean is not None:
                raise ValueError('Incomplete/missing point must have mean=null; export complete sample sets first.')
            points.append(dict(p))
            continue
        if p.get('status') != 'complete' or n <= 0 or n != p.get('n_expected'):
            raise ValueError('Only complete point means can be plotted; rerun export after finishing samples.')
        if not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean < 0:
            raise ValueError(f'Invalid {task}/{panel}/{metric} mean: {mean!r}')
        points.append(dict(p))
    return points


def display_record(point):
    # Express mean relative deviations as percentages.
    mean = point.get('mean')
    return {**point, 'mean_percent': 100 * mean if mean is not None else None}


def write_csv(path, records, fields):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def ordered_points(points, panel):
    """Order ABCD configuration points and reject duplicate curve positions."""
    if any(point.get('H') not in HS for point in points):
        raise ValueError('Plot styles are defined for H=2,3,4,8 only')
    if panel in ('A', 'B'):
        curves = [[point for point in points if int(point['H']) == h]
                  for h in HS]
    elif panel == 'C':
        if any(point.get('role') not in C_ROLES or point['H'] != 2
               for point in points):
            raise ValueError('C plot expects sender roles at fixed base H=2')
        curves = [[point for point in points if point.get('role') == role]
                  for role in C_ROLES]
    else:
        strategies = SCHEMES
        if any(point.get('strategy') not in strategies or
               point.get('G') != H_TO_G[point['H']] for point in points):
            raise ValueError('D schedule or matched H-to-G budget is inconsistent')
        ordered = []
        for strategy in strategies:
            current = [point for point in points if point.get('strategy') == strategy]
            by_h = {int(point['H']): point for point in current}
            if len(by_h) != len(current):
                raise ValueError('Duplicate D schedule/budget points')
            ordered.extend(by_h[h] for h in HS if h in by_h)
        return ordered

    ordered = []
    for curve in curves:
        curve = sorted(curve, key=lambda point: point['x'])
        xs = [point['x'] for point in curve]
        if len(xs) != len(set(xs)):
            raise ValueError('Duplicate x values within a fidelity curve')
        ordered.extend(curve)
    return ordered


def prepare(summary_path, output_dir):
    """Export the ABCD KV summary and a point table with percentage means."""
    from fidelity_plan import write_json

    summary_path = Path(summary_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    summary = read_summary(summary_path)
    tasks = tuple(summary.get('tasks', TASKS))
    points = []
    for task in tasks:
        for panel in 'ABCD':
            selected = metric_points(summary, task, panel, 'kv')
            points.extend(ordered_points(selected, panel))
    baselines = [point for point in summary.get('baselines', [])
                 if point.get('name') == 'CenAttn' and point.get('metric') == 'kv']
    prepared = {
        **summary, 'points': points, 'baselines': baselines,
        'point_counts': {status: sum(point['status'] == status for point in points)
                         for status in ('complete', 'incomplete', 'missing')},
        'data_preparation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'display_conversion': 'CSV mean_percent = 100 * mean; missing means remain null.',
        'uncertainty_units': 'std and sem retain the original ratio units.',
    }
    if 'source_summaries' not in prepared:
        prepared['source_summaries'] = [{'path': str(summary_path),
            'sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            'plan_id': summary.get('plan_id')}]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_copy = output_dir / 'v2v_fidelity_summary.json'
    write_json(summary_copy, prepared)
    point_fields = ['task', 'panel', 'metric', 'config_id', 'H', 'G', 'role',
                    'strategy', 'x', 'n', 'n_expected', 'status', 'mean',
                    'mean_percent', 'std', 'sem']
    points_csv = output_dir / 'v2v_fidelity_plot_points.csv'
    write_csv(points_csv, [display_record(point) for point in points], point_fields)
    print('ABCD KV plot summary:', summary_copy)
    print('Plot-data CSV:', points_csv)
    return summary_copy


def source_identity(summary_path, summary, task, samples):
    """Validate experiment and runtime identities across merged sample shards."""
    from fidelity_plan import digest, read_json

    sources = summary.get('source_plans') or [
        {'root': str(summary_path.parent), 'plan_id': summary['plan_id']}]
    identity = None
    planned_points = None
    shard_ids = set()
    n_shards = None
    for source in sources:
        root = Path(source['root']).expanduser()
        if not root.is_absolute():
            root = summary_path.parent / root
        plan = read_json(root / 'experiment_plan.json')
        if (plan['plan_id'] != source['plan_id'] or plan['plan_id'] != digest(
                {k: v for k, v in plan.items() if k != 'plan_id'})):
            raise ValueError(f'{root}: experiment plan identity changed')
        if task not in plan['tasks']:
            continue
        if (plan['schema'] != summary['schema'] or
                plan['measurement_definitions'] != summary['metric_definitions']):
            raise ValueError(f'{root}: measurement definitions differ')
        tp = plan['task_plans'][task]
        sp = tp['sample_plan']
        if sp.get('global_samples', sp['samples']) != samples:
            raise ValueError(f'{root}: planned global samples differ')
        count = sp.get('num_shards', 1)
        shard = sp.get('shard_index', 0)
        if n_shards is None:
            n_shards = count
        if count != n_shards or not 0 <= shard < count or shard in shard_ids:
            raise ValueError(f'{root}: conflicting shard assignment')
        shard_ids.add(shard)
        if sp['samples'] != samples[shard::count]:
            raise ValueError(f'{root}: shard samples differ from declared assignment')
        if planned_points is None:
            planned_points = tp['points']
        if tp['points'] != planned_points:
            raise ValueError(f'{root}: point plans differ between shards')
        runtime = read_json(root / f'runtime_{task}.json')
        if (runtime['schema'] != summary['schema'] or runtime['task'] != task or
                runtime['plan_id'] != plan['plan_id'] or runtime['run_id'] != digest(
                    {k: v for k, v in runtime.items() if k != 'run_id'})):
            raise ValueError(f'{root}: runtime identity changed')
        current = {
            'code_hashes': {key: plan['code_hashes'][key] for key in (
                'fidelity_observer.py', 'v2v_runtime.py', 'source_loader', 'source_wrapper')},
            'paths': tp['paths'], 'question_file_sha256': tp['question_file_sha256'],
            'dtype': plan['settings']['dtype'],
            'runtime': {key: value for key, value in runtime['runtime'].items()
                        if key != 'device'},
        }
        if identity is not None and identity != current:
            raise ValueError(f'{root}: model/runtime identity differs between shards')
        identity = current
    if identity is None or shard_ids != set(range(n_shards)):
        raise ValueError(f'{summary_path}: missing task or sample shards')
    actual = [point for point in summary['points'] if point['task'] == task]
    if len(actual) != len(planned_points) or any(
            not any(all(point.get(k) == v for k, v in planned.items()) for point in actual)
            for planned in planned_points):
        raise ValueError(f'{summary_path}: exported points differ from the original plan')
    return identity


def combine_plot_inputs(summary_paths, output_dir):
    """Concatenate complete task/group points; never average group summaries."""
    from fidelity_plan import digest, read_json, write_json

    paths = [Path(path).expanduser().resolve() for path in summary_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Provide distinct nonempty summary paths')
    output_dir = Path(output_dir).expanduser().resolve()
    destination = output_dir / 'v2v_fidelity_summary.json'
    if destination in paths:
        raise ValueError('Combined output must not overwrite an input summary')
    result = {'schema': 'v2v-prefill-fidelity-v1', 'tasks': [], 'sample_plans': {},
              'points': [], 'baselines': [], 'source_summaries': [],
              'note': 'Complete configuration points concatenated without changing their statistics.'}
    identities, sample_sets, seen_groups, baselines = {}, {}, set(), {}
    for path in paths:
        summary = read_summary(read_json(path))
        if 'metric_definitions' not in result:
            result['metric_definitions'] = summary['metric_definitions']
        if summary['metric_definitions'] != result['metric_definitions']:
            raise ValueError(f'{path}: incompatible metric definitions')
        tasks = summary['tasks']
        if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
            raise ValueError(f'{path}: invalid task list')
        if any(row['task'] not in tasks for row in summary['points'] + summary['baselines']):
            raise ValueError(f'{path}: unlisted task in results')
        for task in tasks:
            sp = summary['sample_plans'][task]
            samples = sp['samples']
            expected = sp['n_expected_samples']
            if (expected <= 0 or len(samples) != expected or sp['sample_set_sha256'] != digest(samples)
                    or sp.get('global_samples', samples) != samples
                    or sp.get('n_global_expected_samples', expected) != expected
                    or sp.get('global_sample_set_sha256', digest(samples)) != digest(samples)
                    or (sp.get('num_shards', 1) > 1 and
                        sp.get('num_shards_merged') != sp['num_shards'])):
                raise ValueError(f'{path}: incomplete sample set; merge all shards first')
            identity = source_identity(path, summary, task, samples)
            if task in identities and (identities[task] != identity or sample_sets[task] != samples):
                raise ValueError(f'{path}: {task} samples, checkpoint, dtype or measurement code differ')
            identities[task], sample_sets[task] = identity, samples
            result['sample_plans'].setdefault(task, sp)
            rows = [row for row in summary['points'] + summary['baselines'] if row['task'] == task]
            if not rows or any(row.get('status') != 'complete' or row.get('n') != expected or
                               row.get('n_expected') != expected or row.get('mean') is None for row in rows):
                raise ValueError(f'{path}: finish all samples before combining plotting data')
            task_points = [row for row in summary['points'] if row['task'] == task]
            if not task_points or any(row['panel'] not in 'ABCD' or row['metric'] != 'kv' for row in task_points):
                raise ValueError(f'{path}: missing or invalid task/group points')
            for panel in 'ABCD':
                selected = metric_points(summary, task, panel, 'kv')
                if not selected:
                    continue
                if (task, panel) in seen_groups:
                    raise ValueError(f'{path}: duplicate task/group {task}/{panel}')
                seen_groups.add((task, panel))
                result['points'].extend(ordered_points(selected, panel))
            task_baselines = [row for row in summary['baselines'] if row['task'] == task]
            expected_baselines = {('CenAttn', metric) for metric in ('hidden', 'kv')}
            if len(task_baselines) != 2 or {(row['name'], row['metric']) for row in task_baselines} != expected_baselines:
                raise ValueError(f'{path}: missing or duplicate reference baselines')
            for row in task_baselines:
                key = (task, row['name'], row['metric'])
                if key in baselines and baselines[key] != row:
                    raise ValueError(f'{path}: conflicting baseline {key}; no baseline selected or averaged')
                baselines[key] = row
        result['source_summaries'].append({'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'plan_id': summary.get('plan_id')})
    result['tasks'] = [task for task in TASKS if task in sample_sets]
    result['baselines'] = [row for row in baselines.values() if row['metric'] == 'kv']
    result['point_counts'] = {'complete': len(result['points']), 'incomplete': 0, 'missing': 0}
    result['combination_id'] = digest(result['source_summaries'])
    write_json(destination, result)
    return prepare(destination, output_dir)


def main():
    combine_plot_inputs(SUMMARY_PATHS, OUTPUT_DIR)


if __name__ == '__main__':
    main()
