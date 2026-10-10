from __future__ import annotations

import ast
import builtins
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import sys
import symtable
import types


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _checkpoint_identity(path):
    """Hash checkpoint metadata and record sizes and timestamps for large weights."""
    root = Path(path).expanduser()
    if not root.is_dir():
        return {'requested': str(path), 'local_directory': False}
    root = root.resolve()
    entries = {}
    for file in sorted(root.rglob('*')):
        if not file.is_file() or file.suffix.lower() not in {'.json', '.bin', '.safetensors', '.model', '.txt', '.py'}:
            continue
        stat = file.stat()
        item = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        if stat.st_size <= 16 * 1024 * 1024:
            item['sha256'] = file_sha256(file)
        entries[str(file.relative_to(root))] = item
    return {'requested': str(root), 'local_directory': True, 'files': entries,
            'large_weight_identity': 'size_and_mtime_ns; files_at_most_16MiB_are_sha256_hashed'}


def _llava_source_identity(llava_root):
    root = Path(llava_root).resolve()
    paths = list((root / 'llava/model').rglob('*.py'))
    paths += [root / 'llava' / name for name in ('constants.py', 'conversation.py', 'mm_utils.py', 'utils.py')]
    return {str(path.relative_to(root)): file_sha256(path) for path in sorted(set(paths)) if path.is_file()}


def task_specs(source_script):
    """Read QUESTION_CONFIGS without executing the inference entry point."""
    source_script = Path(source_script)
    tree = ast.parse(source_script.read_text(encoding='utf-8-sig'), str(source_script))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'QUESTION_CONFIGS'
            for target in node.targets
        ):
            found.append(ast.literal_eval(node.value))
    if len(found) != 1 or not isinstance(found[0], dict):
        raise ValueError(f'{source_script}: expected exactly one QUESTION_CONFIGS dictionary')
    return found[0]


def resolve_task_paths(task, settings, source_script, project_root):
    """Resolve task-specific checkpoint and QA paths from the inference settings."""
    specs = task_specs(source_script)
    if task not in specs:
        raise ValueError(f'Unknown task {task!r}; available tasks: {list(specs)}')
    spec = dict(specs[task])
    overrides = dict(settings.get('task_overrides', {}).get(task, {}))
    project_root = Path(project_root).expanduser().resolve()
    llava_root = Path(settings.get('llava_root') or project_root / 'LLaVA').expanduser().resolve()
    feature_source = settings.get('feature_source', 'no_fusion_keep_all')
    # Match the checkpoint and QA directory conventions of the inference entry point.
    checkpoint_dir = llava_root / 'checkpoints' / 'llava-v1.5-7b-task-lora'
    checkpoint_dir /= f"llava-v1.5-7b-task-lora_v2v4real_3d_grounding_{spec['model_suffix']}"
    checkpoint_dir /= f"checkpoint-{spec['checkpoint']}"
    question_file = project_root / 'DMSTrack/V2V4Real/official_models' / feature_source
    question_file /= 'npy/co_llm'
    question_file /= spec['question_file']
    return {
        **spec,
        'task': task,
        'project_root': str(project_root),
        'llava_root': str(llava_root),
        'model_path': str(Path(overrides.get('model_path') or checkpoint_dir).expanduser().resolve()),
        'question_file': str(Path(overrides.get('question_file') or question_file).expanduser().resolve()),
        'model_base': overrides.get('model_base', settings.get('model_base', 'liuhaotian/llava-v1.5-7b')),
        'conv_mode': overrides.get('conv_mode', settings.get('conv_mode', 'vicuna_v1')),
        'num_input_frames': int(overrides.get('num_input_frames', spec['num_input_frames'])),
        'feature_source': feature_source,
    }


def _load_dataset_namespace(source_script, project_root):
    """Load the inference Dataset class and collate function without executing the entry point."""
    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    tree = ast.parse(Path(source_script).read_text(encoding='utf-8-sig'), str(source_script))
    wanted = {'V2V4RealCustomDataset', 'v2v4real_collate_fn'}
    body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    if {node.name for node in body} != wanted:
        raise ValueError(f'{source_script}: original V2V Dataset/collate definitions were not found')
    module = types.ModuleType('_v2v_fidelity_original_dataset')
    module.__dict__.update({
        '__file__': str(source_script), 'os': os, 'random': random, 'np': np,
        'torch': torch, 'Dataset': Dataset, 'PROJECT_ROOT': str(project_root),
        'DEFAULT_IMAGE_TOKEN': DEFAULT_IMAGE_TOKEN, 'IMAGE_TOKEN_INDEX': IMAGE_TOKEN_INDEX,
        'conv_templates': conv_templates, 'tokenizer_image_token': tokenizer_image_token,
    })
    selected = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    # Reject Dataset definitions with unresolved global dependencies.
    symbols = symtable.symtable(ast.unparse(selected), str(source_script), 'exec')
    required = set()
    def collect(table):
        required.update(item.get_name() for item in table.get_symbols()
                        if item.is_global() and item.is_referenced())
        for child in table.get_children():
            collect(child)
    collect(symbols)
    missing = required - set(module.__dict__) - set(vars(builtins)) - wanted
    if missing:
        raise ValueError(f'{source_script}: Dataset needs unsupported global names: {sorted(missing)}')
    exec(compile(selected, str(source_script), 'exec'), module.__dict__)
    return module


def _load_wrapper(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location('_v2v_fidelity_uploaded_wrapper', path)
    if spec is None or spec.loader is None:
        raise ValueError(f'Cannot import wrapper: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Runtime:
    """Task checkpoint and single-sample cooperative-driving preprocessing."""
    def __init__(self, task, settings, source_script, wrapper_script, project_root, rows):
        self.settings = dict(settings)
        self.source_script = str(Path(source_script).resolve())
        self.wrapper_script = str(Path(wrapper_script).resolve())
        self.paths = resolve_task_paths(task, settings, source_script, project_root)
        self.task = task
        self.rows = list(rows)
        self._row_indices = {id(row): index for index, row in enumerate(self.rows)}
        self.wrapper = None
        self.model = None
        self.last_prefill_input_sha256 = None
        if not self.rows:
            raise ValueError(f'{task}: no samples selected')
        for key in ('TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE', 'HF_HUB_OFFLINE'):
            os.environ[key] = '1'
        llava_root = Path(self.paths['llava_root'])
        if not (llava_root / 'llava').is_dir():
            raise FileNotFoundError(f'Cannot find LLaVA sources: {llava_root / "llava"}; set project_root or llava_root')
        sys.path.insert(0, str(llava_root))
        import numpy as np
        import torch
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.utils import disable_torch_init
        import transformers

        self.torch = torch
        self.device = torch.device(settings.get('device', 'cuda:0'))
        if self.device.type != 'cuda':
            raise ValueError('Runtime is intended for CUDA; planning/export-only run without CUDA')
        if self.device.index is None:
            self.device = torch.device('cuda:0')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable in this Python environment')
        dtype_name = settings.get('dtype', 'float16')
        if dtype_name not in ('float16', 'bfloat16', 'float32'):
            raise ValueError(f'Unsupported dtype: {dtype_name}')
        self.dtype = getattr(torch, dtype_name)
        self.transformers_version = transformers.__version__
        seed = int(settings.get('seed', 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.my_model_config = {
            'mm_scene_projector_input_size': 3072,
            'scene_level_only': False,
            'object_level_only': False,
            'scene_feature_mode': 'shallow',
            'object_feature_mode': 'shallow',
            'num_input_frames': self.paths['num_input_frames'],
            'ego_only': False,
            'feature_source': self.paths['feature_source'],
            'dataset_source': 'v2v4real',
            'num_latency_frames': 0,
            'positional_error_std': 0.0,
        }
        model_path = self.paths['model_path']
        if not Path(model_path).is_dir():
            raise FileNotFoundError(f'{task}: task checkpoint not found: {model_path}')
        disable_torch_init()
        loader_kwargs = {'my_model_config': self.my_model_config}
        loader_parameters = inspect.signature(load_pretrained_model).parameters
        if 'device_map' in loader_parameters:
            loader_kwargs['device_map'] = str(self.device)
        if 'device' in loader_parameters:
            loader_kwargs['device'] = str(self.device)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, self.paths['model_base'], get_model_name_from_path(model_path),
            **loader_kwargs,
        )
        self.model.eval()
        parameter_devices = {str(parameter.device) for parameter in self.model.parameters()}
        if parameter_devices != {str(self.device)}:
            raise RuntimeError(
                f'Expected checkpoint entirely on {self.device}; found parameter devices {sorted(parameter_devices)}. '
                'Use CUDA_VISIBLE_DEVICES with one physical GPU and a builder supporting device_map. '
                'CPU/disk offload and multiple-device dispatch are unsupported for this fidelity run.'
            )
        if self.dtype != torch.float16:
            self.model.to(dtype=self.dtype)
        self.effective_parameter_dtypes = sorted({
            str(parameter.dtype).replace('torch.', '')
            for parameter in self.model.parameters() if parameter.is_floating_point()
        })
        self.wrapper_module = _load_wrapper(self.wrapper_script)
        self.wrapper = self.wrapper_module.V2VFedAttnWrapper(self.model, device=str(self.device))
        self.n_layers = self.wrapper.n_layers
        self.dataset_module = _load_dataset_namespace(self.source_script, self.paths['project_root'])
        conv_mode = self.paths['conv_mode']
        model_name = get_model_name_from_path(model_path)
        if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in conv_mode:
            conv_mode += '_mmtag'
        self.paths['conv_mode'] = conv_mode
        self.dataset = self.dataset_module.V2V4RealCustomDataset(
            self.rows, '', self.tokenizer, self.image_processor, self.model.config,
            0, self.paths['num_input_frames'], 0, 0.0,
            self.paths['feature_source'], False, conv_mode,
        )

    def metadata(self):
        return {
            **self.paths,
            'dtype': str(self.dtype).replace('torch.', ''),
            'device': str(self.device),
            'num_layers': self.n_layers,
            'model_config': dict(self.my_model_config),
            'source_script_sha256': file_sha256(self.source_script),
            'wrapper_script_sha256': file_sha256(self.wrapper_script),
            'forward_method': 'original_multimodal_generate_max_new_tokens_1_prefill_only',
            'torch_version': self.torch.__version__,
            'transformers_version': self.transformers_version,
            'effective_parameter_dtypes': self.effective_parameter_dtypes,
            'task_checkpoint_identity': _checkpoint_identity(self.paths['model_path']),
            'base_model_identity': _checkpoint_identity(self.paths['model_base']),
            'base_model_commit_hash': getattr(self.model.config, '_commit_hash', None),
            'llava_source_sha256': _llava_source_identity(self.paths['llava_root']),
            'attention_implementation_source': {
                'path': inspect.getfile(type(self.wrapper.transformer_layers[0].self_attn)),
                'sha256': file_sha256(inspect.getfile(type(self.wrapper.transformer_layers[0].self_attn))),
            },
        }

    def prepare_sample(self, row):
        index = self._row_indices.get(id(row))
        if index is None:
            candidates = [i for i, item in enumerate(self.rows) if item.get('id') == row.get('id')]
            if len(candidates) != 1:
                raise ValueError('prepare_sample row must match exactly one selected QA record')
            index = candidates[0]
        data = self.dataset_module.v2v4real_collate_fn([self.dataset[index]])
        (input_ids, image_tensor, image_sizes, scene_point_feature_map, regression_map,
         classification_map, detection_box_score, object_features, active_agent_mask) = data
        torch = self.torch
        input_ids = input_ids.to(device=self.device)
        asker_cav_id = row.get('asker_cav_id', 'ego')
        if asker_cav_id not in ('ego', '1'):
            raise ValueError(f'Unexpected asker_cav_id: {asker_cav_id!r}')
        token_pos, token_chunks = self.wrapper_module.compute_token_positions(
            input_ids=input_ids, my_model_config=self.my_model_config,
            asker_cav_id=asker_cav_id, IMAGE_TOKEN_INDEX=-200,
        )
        all_tokens = [i for ids in token_chunks.values() for i in ids]
        if len(all_tokens) != len(set(all_tokens)) or sorted(all_tokens) != list(range(token_pos['seq_len'])):
            raise ValueError('Participant token chunks must partition the entire expanded input')
        def move(value):
            return value.to(device=self.device, dtype=self.dtype, non_blocking=True)
        sample = {
            'input_ids': input_ids,
            'token_pos': token_pos, 'token_chunks': token_chunks,
            'asker_cav_id': asker_cav_id, 'sample_key': row.get('id'),
            'seq_len': int(token_pos['seq_len']),
            'image_sizes': image_sizes,
            'images': move(image_tensor),
            'scene_point_feature_map': move(scene_point_feature_map),
            'regression_map': move(regression_map),
            'classification_map': move(classification_map),
            'detection_box_score': move(detection_box_score),
            'object_features': move(object_features),
            'active_agent_mask': active_agent_mask.to(device=self.device, dtype=torch.bool),
        }
        if sample['sample_key'] is None:
            raise ValueError('QA id is missing: cannot preserve original token-subset sampling identity')
        return sample

    def prepare_masks(self, sample, config):
        torch = self.torch
        setting = dict(config.get('setting', {}))
        setting_name = setting.get('name', config['name'])
        setting['name'] = setting_name
        global_layers = list(config['global_layers'])
        local_layers = list(config['local_layers'])
        if set(global_layers) & set(local_layers) or sorted(global_layers + local_layers) != list(range(self.n_layers)):
            raise ValueError(f'{config["name"]}: global/local layers do not partition all layers')
        dense_global, dense_local = self.wrapper_module.create_fedattn_mask_from_chunks(
            token_chunks=sample['token_chunks'], seq_len=sample['seq_len'], batch_size=1,
            com_ratios={}, asker_cav_id=sample['asker_cav_id'], dtype=self.dtype, device=str(self.device),
        )
        return self.wrapper_module.build_layer_masks(
            n_layers=self.n_layers,
            batch_fedattn_configs=[(config['name'], global_layers, local_layers, setting_name)],
            global_mask_template={setting_name: dense_global},
            local_mask_template={setting_name: dense_local},
            token_pos=sample['token_pos'], token_chunks=sample['token_chunks'],
            com_ratio_settings={setting_name: setting}, sample_key=sample['sample_key'],
            collaborator_global_layers=None, asker_cav_id=sample['asker_cav_id'],
        )

    def forward(self, sample, prefill_masks, decode_masks):
        """Run one multimodal prefill and discard the generated token."""
        torch = self.torch
        calls = []
        self.last_prefill_input_sha256 = None
        def check_expanded_input(module, args):
            hidden = args[0]
            calls.append(int(hidden.shape[1]))
            if len(calls) != 1 or hidden.shape[0] != 1 or hidden.shape[1] != sample['seq_len']:
                raise RuntimeError(
                    f'Expected one prefill of length {sample["seq_len"]}; got layer-0 calls {calls}. '
                    'Check feature expansion/token_positions; decode measurements are not permitted.'
                )
            digest = hashlib.sha256()
            digest.update(json.dumps(list(hidden.shape), separators=(',', ':')).encode('ascii'))
            digest.update(hidden.detach().float().cpu().contiguous().numpy().tobytes(order='C'))
            self.last_prefill_input_sha256 = digest.hexdigest()
        handle = self.wrapper.transformer_layers[0].register_forward_pre_hook(check_expanded_input)
        self.wrapper.set_layer_masks(prefill_masks, decode_masks)
        kwargs = {name: sample[name] for name in (
            'images', 'image_sizes', 'scene_point_feature_map', 'regression_map',
            'classification_map', 'detection_box_score', 'object_features', 'active_agent_mask',
        )}
        try:
            with torch.inference_mode():
                output = self.model.generate(
                    sample['input_ids'], attention_mask=torch.ones_like(sample['input_ids']),
                    do_sample=False, num_beams=1, max_new_tokens=1, use_cache=True, **kwargs,
                )
                del output
            if calls != [sample['seq_len']]:
                raise RuntimeError(f'Expected one complete prefill, got layer-0 calls {calls}')
        finally:
            handle.remove()
            self.wrapper.reset()

    def reset_memory_peak(self):
        """Begin a new allocator peak window, retaining the loaded model.

        Call before a complete sample to measure all of its configurations.
        This changes reporting only; it does not clear or limit the allocator.
        """
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def memory_stats(self):
        """Return MiB counters without retaining tensors or changing inference.

        Allocated/reserved/peak counters cover this process's PyTorch allocator.
        Device free/total counters cover the entire physical device, including
        other workers and non-PyTorch allocations. Consequently two processes'
        allocator peaks alone are not a guarantee that concurrent loading fits.
        """
        cuda = self.torch.cuda
        free_bytes, total_bytes = cuda.mem_get_info(self.device)
        mib = 1024 * 1024
        return {
            'device': str(self.device),
            'device_name': cuda.get_device_name(self.device),
            'allocated_mib': round(cuda.memory_allocated(self.device) / mib, 1),
            'reserved_mib': round(cuda.memory_reserved(self.device) / mib, 1),
            'peak_allocated_mib': round(cuda.max_memory_allocated(self.device) / mib, 1),
            'peak_reserved_mib': round(cuda.max_memory_reserved(self.device) / mib, 1),
            'device_free_mib': round(free_bytes / mib, 1),
            'device_total_mib': round(total_bytes / mib, 1),
        }

    def close(self):
        if self.wrapper is not None:
            self.wrapper.reset()
            self.wrapper.restore()
        self.wrapper = None
        self.model = None
        self.dataset = None
        self.tokenizer = None
        self.image_processor = None
        self._row_indices = {}
        gc.collect()
        if hasattr(self, 'torch') and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
