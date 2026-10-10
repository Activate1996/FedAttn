"""Check input feature availability before loading the model."""
from pathlib import Path


def check_required_features(rows, project_root, num_input_frames, *,
                            feature_source='no_fusion_keep_all', ego_only=False,
                            num_latency_frames=0, scene_feature_mode='shallow',
                            object_feature_mode='shallow', scene_level_only=False,
                            object_level_only=False):
    root = Path(project_root) / 'DMSTrack/V2V4Real/official_models' / feature_source / 'npy'
    required = set()
    for row in rows:
        if row.get('asker_cav_id', 'ego') not in ('ego', '1'):
            raise ValueError(f"Unsupported asker_cav_id for sample {row.get('id')}")
        cav_ids = ('ego',) if ego_only else ('ego', '1')
        for past in range(num_input_frames):
            # Follow the Dataset scene-boundary and latency conventions.
            frame = row['global_timestamp_index']
            if row['local_timestamp_index'] >= past:
                frame -= past
            if num_latency_frames != 0:
                frame = max(0, frame - num_latency_frames)
            for cav in cav_ids:
                # Detection boxes are loaded for every feature mode.
                required.add(root / 'co_llm' / cav / f'{frame:04d}_detection_box_score.npy')
                if not object_level_only:
                    names = (('regression_map', 'classification_map')
                             if scene_feature_mode == 'shallow' else ('spatial_features_2d',))
                    required.update(root / 'co_llm' / cav / f'{frame:04d}_{name}.npy' for name in names)
                if not scene_level_only and object_feature_mode == 'deep':
                    required.add(root / cav / f'{frame:04d}_feature.npy')
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        preview = '\n'.join(missing[:20])
        raise FileNotFoundError(
            f'{len(missing)} required feature files are missing. First paths:\n{preview}\n'
            'Extract the processed features under DMSTrack/V2V4Real/official_models/.')
    print(f'Input assets: {len(required)} required feature files present.', flush=True)
    return sorted(required)
