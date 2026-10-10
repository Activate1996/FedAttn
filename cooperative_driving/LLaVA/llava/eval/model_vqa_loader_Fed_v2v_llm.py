# -*- coding: utf-8 -*-
"""FedAttn inference on V2V4Real with an ego vehicle and one collaborator."""

import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

GPU_id = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id

os.chdir(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../..'))  

""" 0) Experiment settings """
EXPERIMENT_GROUP = 'H'  # H, A, B, C, or D
# Run two QA examples in demo mode; otherwise use the task-specific sample stride.
DEMO_MODE = True
DEMO_SAMPLE_INDICES = [0, 1]
QUALITY_SAMPLE_STRIDES = {'q1': 20, 'q4': 1, 'q5': 1}
H_SETTING = {'H_values': [2, 3, 4, 8]}
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


# Nonuniform synchronization schedules.
def stride_progressive_dec_back(start: int, end: int, k: int, gamma: float = 1.4):
    """Construct integer synchronization positions with decreasing step weights.

    The first and last positions span [start, end] when k > 1. A larger
    gamma places more weight on the early steps; k = 1 returns [end]."""
    if k <= 0:
        return []
    if k == 1:
        return [end]  # A single synchronization point is aligned with the final block.
    if end - start + 1 < k:
        raise ValueError(f"Range too small: [{start},{end}] cannot fit k={k} unique ints.")

    span = end - start
    gaps = k - 1

    # 1) Form decreasing continuous step weights.
    weights = list(range(gaps, 0, -1))          # gaps, gaps-1, ..., 1
    raw = [w ** gamma for w in weights]
    scale = span / sum(raw)
    steps = [max(1, int(round(scale * r))) for r in raw]

    # 2) Enforce nonincreasing integer steps of at least one.
    for i in range(1, gaps):
        steps[i] = min(steps[i], steps[i-1])
        steps[i] = max(1, steps[i])

    # 3) Adjust the total step length to span the requested range.
    total = sum(steps)

    # Reduce the earliest eligible step while preserving its order with the next step.
    while total > span:
        changed = False
        for i in range(gaps):
            if steps[i] > 1:
                newv = steps[i] - 1
                # Keep this step at least as large as the next step.
                if i == gaps - 1 or newv >= steps[i + 1]:
                    steps[i] = newv
                    total -= 1
                    changed = True
                    break
        if not changed:
            break

    # Increase eligible steps from the end of the schedule.
    while total < span:
        changed = False
        for i in range(gaps - 1, -1, -1):
            newv = steps[i] + 1
            # Apply the boundary condition or compare with the preceding step.
            if i == gaps - 1 or steps[i - 1] >= newv:
                steps[i] = newv
                total += 1
                changed = True
                break
        if not changed:
            # If no step is eligible, increase the first step.
            steps[0] += 1
            total += 1

    # 4) Accumulate the step lengths into block positions.
    pos = [start]
    cur = start
    for s in steps:
        cur += s
        pos.append(cur)

    # Align the final synchronization point with the end of the range.
    pos[-1] = end

    return pos

def build_experiment_configs(n_layers, task_name):

    group = EXPERIMENT_GROUP

    settings_by_group = {
        'H': H_SETTING,
        'A': A_SETTING,
        'B': B_SETTING,
        'C': C_SETTING,
        'D': D_SETTING,
    }

    selected = settings_by_group[group]

    # Group C fixes the publisher interval; the other groups use H_values.
    h_values = (
        [selected['publisher_intervals']]
        if group == 'C'
        else selected['H_values']
    )

    seed = selected.get('seed', 0)
    runs = []
    layer_configs = []

    if group == 'H':
        layer_configs.append(
            ('CenAttn', list(range(n_layers)), [])
        )

    for h in h_values:
        global_layers = list(range(0, n_layers, h))
        local_layers = sorted(
            set(range(n_layers)) - set(global_layers)
        )
        settings = []

        if group == 'H':
            layer_configs.append((
                f'strided_s{h}_G{len(global_layers)}',
                global_layers,
                local_layers,
            ))
            continue

        elif group == 'A':
            for r in selected['input_keep_ratios']:
                settings.append({
                    'name': f'A_input{r:g}_s{seed}',
                    'panel': 'A',
                    'ratios': {},
                    'input_keep_ratio': r,
                    'seed': seed,
                    'collaborator_global_layers': None,
                })

        elif group == 'B':
            for r in selected['comm_keep_ratios']:

                settings.append({
                    'name': f'B_comm{r:g}_s{seed}',
                    'panel': 'B',
                    'ratios': {'ego': r, 'car1': r},
                    'input_keep_ratio': 1.0,
                    'seed': seed,
                    'collaborator_global_layers': None,
                })

        elif group == 'C':
            for collaborator_h in selected['collaborator_intervals']:
                settings.append({
                    'name': (
                        f'full_s{seed}'
                        if collaborator_h == h
                        else f'C_colH{collaborator_h}_s{seed}'
                    ),
                    'panel': 'C',
                    'ratios': {},
                    'input_keep_ratio': 1.0,
                    'seed': seed,
                    'collaborator_interval': collaborator_h,
                    'collaborator_global_layers': list(
                        range(0, n_layers, collaborator_h)
                    ),
                    'shared_baseline': collaborator_h == h,
                })

        elif group == 'D':
            g = len(global_layers)
            dense = stride_progressive_dec_back(
                0, n_layers - 1, g,
                gamma=selected['gamma'],
            )

            schedules = {
                'cumul_shallow': list(range(g)),
                'cumul_deep': list(range(n_layers - g, n_layers)),
                'cumul_sparse': sorted(
                    n_layers - 1 - i for i in dense
                ),
                'cumul_dense': dense,
            }

            for name, layers in schedules.items():
                layer_configs.append((
                    f'{name}_N{g}',
                    layers,
                    sorted(set(range(n_layers)) - set(layers)),
                ))
            continue

        if group in ('A', 'B'):
            settings.append({
                'name': f'full_s{seed}',
                'panel': 'C',
                'ratios': {},
                'input_keep_ratio': 1.0,
                'seed': seed,
                'collaborator_interval': h,
                'collaborator_global_layers': list(global_layers),
                'shared_baseline': True,
            })

        names = [s['name'] for s in settings]

        runs.append({
            'H': h,
            'layer_configs': [
                (f'abc_{task_name}_H{h}', global_layers, local_layers)
            ],
            'settings': settings,
        })

    if group in ('H', 'D'):
        runs.append({
            'H': None,
            'layer_configs': layer_configs,
            'settings': [
                {'name': 'comm_100_100', 'ratios': {}}
            ],
        })

    return runs

""" 1) Standard library """
import argparse                                                                        
import json                                      
import math                                     
import time           
import random
import glob
import sys
sys.path.insert(0, PROJECT_ROOT)
from check_assets import check_required_features

""" 2) Third-party libraries """
import torch
import gc
import numpy as np
from tqdm import tqdm                        
import shortuuid         
from PIL import Image    
from torch.utils.data import Dataset, DataLoader

""" 3) LLaVA """
from llava.constants import (
    IMAGE_TOKEN_INDEX,                                
    DEFAULT_IMAGE_TOKEN,                 
    DEFAULT_IM_START_TOKEN,                     
    DEFAULT_IM_END_TOKEN     
)

from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (                       
    tokenizer_image_token,                                                 
    process_images,                               
    get_model_name_from_path  
)

""" 4) Wrapper """
from Wrapper import (
    V2VFedAttnWrapper,
    compute_token_positions,
    create_fedattn_mask_from_chunks,
    build_layer_masks,
)

"""5) Data-loading utilities"""
def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Line {i} error:", e)
    return data

""" V2V4RealCustomDataset """
class V2V4RealCustomDataset(Dataset):

    def __init__(self, list_data_dict, image_folder, tokenizer, image_processor, 
                 model_config, simplified_object_feature, num_input_frames, 
                 num_latency_frames, positional_error_std, 
                 feature_source, ego_only, conv_mode):
        
        self.list_data_dict = list_data_dict      
        self.image_folder = image_folder          
        self.tokenizer = tokenizer                
        self.image_processor = image_processor    
        self.model_config = model_config         
        self.simplified_object_feature = simplified_object_feature 
        self.num_input_frames = num_input_frames  
        self.num_latency_frames = num_latency_frames  
        self.positional_error_std = positional_error_std  
        
        self.last_global_timestamp_index = 1992
        
        self.feature_source = feature_source 
        self.ego_only = ego_only
        self.conv_mode = conv_mode
        
        self.llm_data_path = os.path.join(
            PROJECT_ROOT,
            'DMSTrack/V2V4Real/official_models/', 
            self.feature_source,  
            'npy/co_llm'         
        ).replace("\\", "/")
        
        print('self.llm_data_path: ', self.llm_data_path)
        
        # Check that the first and last selected samples can be loaded.
        self.__getitem__(0)                 
        self.__getitem__(self.__len__() - 1)  

    def zero_padded_roll(self, array, shift, axis):
        result = np.zeros_like(array)

        if shift == 0:
            return array.copy()

        if shift > 0:
            slicer_src = [slice(None)] * array.ndim
            slicer_dst = [slice(None)] * array.ndim
            slicer_src[axis] = slice(0, -shift)
            slicer_dst[axis] = slice(shift, None)
        else:  # negative shift
            slicer_src = [slice(None)] * array.ndim
            slicer_dst = [slice(None)] * array.ndim
            slicer_src[axis] = slice(-shift, None)
            slicer_dst[axis] = slice(0, shift)

        result[tuple(slicer_dst)] = array[tuple(slicer_src)]
        return result

    def load_single_frame_feature_map(self, cav_ids, feature_map_name, single_frame_global_timestamp_index):

        non_ego_single_frame_global_timestamp_index = single_frame_global_timestamp_index - self.num_latency_frames
        
        if non_ego_single_frame_global_timestamp_index < 0:
            non_ego_single_frame_global_timestamp_index = 0

        if self.positional_error_std > 0:
            non_ego_positional_shift_meters = np.random.normal(
                loc=0.0, 
                scale=self.positional_error_std
            )
            feature_map_resolution = 0.4 * 4 
            non_ego_positional_shift_num_pixels = int(
                non_ego_positional_shift_meters / feature_map_resolution
            )
            non_ego_positional_shift_spatial_dim_idx = random.randint(2, 3)
        else:
            non_ego_positional_shift_num_pixels = 0
            non_ego_positional_shift_meters = 0
            non_ego_positional_shift_spatial_dim_idx = None
        
        scene_spatial_features_2d_all = []  
        default_shapes = {
        'spatial_features_2d': [1, 256, 50, 88],
        'regression_map': [1, 14, 50, 88],
        'classification_map': [1, 2, 50, 88]
        }
        feature_shape = default_shapes.get(feature_map_name)
        
        for cav_id in cav_ids:
            if self.num_latency_frames == 0:
                scene_spatial_features_2d_file = os.path.join(
                    self.llm_data_path, 
                    cav_id,  
                    '%04d_%s.npy' % (single_frame_global_timestamp_index, feature_map_name)
                    ).replace("\\", "/") 
            else:
                scene_spatial_features_2d_file = os.path.join(
                    self.llm_data_path, 
                    cav_id, 
                    '%04d_%s.npy' % (non_ego_single_frame_global_timestamp_index, feature_map_name)
                    ).replace("\\", "/")
            try:
                scene_spatial_features_2d = np.load(scene_spatial_features_2d_file)
                feature_shape = scene_spatial_features_2d.shape   
            except FileNotFoundError:
                scene_spatial_features_2d = np.zeros(feature_shape)

            if non_ego_positional_shift_num_pixels != 0:
                scene_spatial_features_2d = self.zero_padded_roll(
                    scene_spatial_features_2d,                   
                    non_ego_positional_shift_num_pixels,         
                    non_ego_positional_shift_spatial_dim_idx     
                )

            scene_spatial_features_2d = torch.from_numpy(scene_spatial_features_2d[0])
            scene_spatial_features_2d_all.append(scene_spatial_features_2d)
        
        scene_spatial_features_2d = torch.stack(scene_spatial_features_2d_all, dim=0)
        
        return scene_spatial_features_2d

    def load_single_frame_object_features(self, cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav):
        
        non_ego_single_frame_global_timestamp_index = single_frame_global_timestamp_index - self.num_latency_frames
        if non_ego_single_frame_global_timestamp_index < 0:
            non_ego_single_frame_global_timestamp_index = 0

        object_features_all = []  
        feature_shape = [max_num_boxes_per_cav, 256]  
        
        npy_base_path = os.path.dirname(self.llm_data_path)  
        for cav_id in cav_ids:
            if self.num_latency_frames == 0:
                object_features_file = os.path.join(
                    npy_base_path,
                    cav_id,           
                    '%04d_feature.npy' % single_frame_global_timestamp_index
                ).replace("\\", "/")
            else:
                object_features_file = os.path.join(
                    npy_base_path, 
                    cav_id, 
                    '%04d_feature.npy' % non_ego_single_frame_global_timestamp_index
                ).replace("\\", "/")

            try:
                object_features = np.load(object_features_file)
                object_features = object_features[:, :, 2, 2]
                feature_shape = [max_num_boxes_per_cav, object_features.shape[1]]  
            except FileNotFoundError:
                object_features = np.zeros(feature_shape)

            if object_features.shape[0] > max_num_boxes_per_cav:
                object_features = object_features[:max_num_boxes_per_cav]
                
            if object_features.shape[0] < max_num_boxes_per_cav:
                object_features = np.pad(
                    object_features,
                    ((0, max_num_boxes_per_cav - object_features.shape[0]), (0, 0)),
                    'constant', 
                    constant_values=(0, 0)
                )
            
            object_features = torch.from_numpy(object_features)
            object_features_all.append(object_features)

        object_features = torch.stack(object_features_all, dim=0)
        
        return object_features

    def load_single_frame_detection_box_score(self, cav_ids, single_frame_global_timestamp_index, max_num_boxes_per_cav):
        
        non_ego_single_frame_global_timestamp_index = single_frame_global_timestamp_index - self.num_latency_frames
        if non_ego_single_frame_global_timestamp_index < 0:
            non_ego_single_frame_global_timestamp_index = 0

        detection_box_score_all = []  
        feature_shape = None          
        
        for cav_id in cav_ids:
            if self.num_latency_frames == 0:
                detection_box_score_file = os.path.join(
                    self.llm_data_path,  
                    cav_id,             
                    '%04d_detection_box_score.npy' % single_frame_global_timestamp_index
                ).replace("\\", "/")
            else:
                detection_box_score_file = os.path.join(
                    self.llm_data_path, 
                    cav_id, 
                    '%04d_detection_box_score.npy' % non_ego_single_frame_global_timestamp_index
                ).replace("\\", "/")

            try:
                detection_box_score = np.load(detection_box_score_file)
                feature_shape = detection_box_score.shape
            except FileNotFoundError:
                detection_box_score = np.zeros([max_num_boxes_per_cav, feature_shape[1]])

            if self.positional_error_std > 0:
                non_ego_positional_shift_meters = np.random.normal(
                    loc=0.0, 
                    scale=self.positional_error_std, 
                    size=detection_box_score.shape  
                )
                detection_box_score[:, :6] += non_ego_positional_shift_meters[:, :6]
            
            if self.simplified_object_feature == 8:
                detection_box_score = np.concatenate([
                    detection_box_score[:, 0:4],   
                    detection_box_score[:, 5:6],   
                    detection_box_score[:, 4:5],   
                    detection_box_score[:, 6:8]   
                ], axis=1)
                
            elif self.simplified_object_feature == 3:
                detection_box_score = np.concatenate([
                    detection_box_score[:, 3:4],  
                    detection_box_score[:, 5:6],  
                    detection_box_score[:, 7:8]    
                ], axis=1)
                
            elif self.simplified_object_feature == 2:
                detection_box_score = np.concatenate([
                    detection_box_score[:, 3:4],   
                    detection_box_score[:, 5:6]    
                ], axis=1)

            if detection_box_score.shape[0] > max_num_boxes_per_cav:
                detection_box_score = detection_box_score[:max_num_boxes_per_cav]
                
            if detection_box_score.shape[0] < max_num_boxes_per_cav:
                detection_box_score = np.pad(
                    detection_box_score,
                    ((0, max_num_boxes_per_cav - detection_box_score.shape[0]), (0, 0)),
                    'constant', 
                    constant_values=(0, 0)
                )
            
            detection_box_score = torch.from_numpy(detection_box_score)
            detection_box_score_all.append(detection_box_score)

        detection_box_score = torch.stack(detection_box_score_all, dim=0)
        
        return detection_box_score

    def __getitem__(self, index):
        
        data_dict = self.list_data_dict[index]
        
        global_timestamp_index = data_dict['global_timestamp_index']  
        local_timestamp_index = data_dict['local_timestamp_index']  
     
        if self.ego_only:
            cav_ids = ['ego']
        else:
            if 'asker_cav_id' not in data_dict:
                cav_ids = ['ego', '1']
            elif data_dict["asker_cav_id"] == "ego":
                cav_ids = ['ego', '1']
            elif data_dict["asker_cav_id"] == '1':
                cav_ids = ['1', 'ego']     
                
        num_input_frames = self.num_input_frames
        
        load_data_dict = {}  
        
        for feature_map_name in ['spatial_features_2d', 'regression_map', 'classification_map']:
            
            feature_map_all_frames = []  
            
            for past_idx in range(num_input_frames):  
                
                if local_timestamp_index < past_idx:
                    single_frame_global_timestamp_index = global_timestamp_index
                else:
                    single_frame_global_timestamp_index = global_timestamp_index - past_idx

                single_frame_feature_map = self.load_single_frame_feature_map(
                    cav_ids,                         
                    feature_map_name,                    
                    single_frame_global_timestamp_index 
                )
                
                feature_map_all_frames.append(single_frame_feature_map)

            feature_map_all_frames = torch.stack(feature_map_all_frames, dim=0)

            if feature_map_name == 'spatial_features_2d':
                load_data_dict['scene_point_feature_map'] = feature_map_all_frames 
            else:
                load_data_dict[feature_map_name] = feature_map_all_frames

        scene_point_feature_map = load_data_dict['scene_point_feature_map']      
        regression_map = load_data_dict['regression_map']      
        classification_map = load_data_dict['classification_map']
        
        max_num_boxes_per_cav = 50  

        object_features_all_frames = []
        
        for past_idx in range(num_input_frames):

            if local_timestamp_index < past_idx:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx

            single_frame_object_features = self.load_single_frame_object_features(
                cav_ids,                         
                single_frame_global_timestamp_index,  
                max_num_boxes_per_cav             
            )
            
            object_features_all_frames.append(single_frame_object_features)

        object_features = torch.stack(object_features_all_frames, dim=0)

        detection_box_score_all_frames = []
        
        for past_idx in range(num_input_frames):
            if local_timestamp_index < past_idx:
                single_frame_global_timestamp_index = global_timestamp_index
            else:
                single_frame_global_timestamp_index = global_timestamp_index - past_idx

            single_frame_detection_box_score = self.load_single_frame_detection_box_score(
                cav_ids,                             
                single_frame_global_timestamp_index,  
                max_num_boxes_per_cav           
            )
            
            detection_box_score_all_frames.append(single_frame_detection_box_score)

        detection_box_score = torch.stack(detection_box_score_all_frames, dim=0)

        active_agent_mask = torch.from_numpy(np.ones([num_input_frames, len(cav_ids)]))

        question = data_dict['conversations'][0]['value']
        qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        conv = conv_templates[self.conv_mode].copy()  
        conv.append_message(conv.roles[0], qs)        
        conv.append_message(conv.roles[1], None)   
        prompt = conv.get_prompt()

        image_tensor = torch.zeros(3, 336, 336)
        image_sizes = [3, 336, 336]

        input_ids = tokenizer_image_token(
            prompt,          
            self.tokenizer,         
            IMAGE_TOKEN_INDEX,       
            return_tensors='pt'      
        )
        
        return (
            input_ids,                 
            image_tensor,            
            image_sizes,               
            scene_point_feature_map,    
            regression_map,            
            classification_map,     
            detection_box_score,      
            object_features,           
            active_agent_mask           
        )

    def __len__(self):
        return len(self.list_data_dict)

def v2v4real_collate_fn(batch):
    """
    V2V4Real collate function
    """

    (input_ids, image_tensors, image_sizes, scene_point_feature_map,
     regression_map, classification_map, detection_box_score,
     object_features, active_agent_mask) = zip(*batch)

    input_ids = torch.stack(input_ids, dim=0)  # [batch_size, seq_len]
    image_tensors = torch.stack(image_tensors, dim=0)  # [batch_size, 3, 336, 336]
    scene_point_feature_map = torch.stack(scene_point_feature_map, dim=0)  # [batch_size, frames, agents, C, H, W]
    regression_map = torch.stack(regression_map, dim=0)  # [batch_size, frames, agents, 14, H, W]
    classification_map = torch.stack(classification_map, dim=0)  # [batch_size, frames, agents, 2, H, W]  
    detection_box_score = torch.stack(detection_box_score, dim=0)  # [batch_size, frames, agents, max_boxes, box_dim]  
    object_features = torch.stack(object_features, dim=0)  # [batch_size, frames, agents, max_boxes, feat_dim]
    active_agent_mask = torch.stack(active_agent_mask, dim=0)  # [batch_size, frames, agents]

    return (input_ids, image_tensors, image_sizes, scene_point_feature_map,
            regression_map, classification_map, detection_box_score,
            object_features, active_agent_mask)

def create_data_loader_v2v4real(list_data_dict, image_folder, tokenizer, image_processor,
                                model_config, simplified_object_feature, num_input_frames,
                                num_latency_frames, positional_error_std, 
                                feature_source, ego_only, conv_mode,
                                batch_size=1, num_workers=0):

    assert batch_size == 1, "batch_size must be 1"
    
    dataset = V2V4RealCustomDataset(
        list_data_dict, image_folder, tokenizer, image_processor,
        model_config, simplified_object_feature, num_input_frames,
        num_latency_frames, positional_error_std, 
        feature_source, ego_only, conv_mode,
    )
    
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        shuffle=False, 
        collate_fn=v2v4real_collate_fn
    )
    
    return data_loader

def load_completed_results(folder_path, activate_question, pattern="*.jsonl"):

    completed = set()

    if not folder_path or not os.path.isdir(folder_path):
        print(f"Warning: folder not found: {folder_path}")
        return completed

    completed_file = os.path.join(folder_path, f"completed_v2v_llm_{activate_question}.jsonl").replace("\\", "/")
    if os.path.exists(completed_file):
        try:
            with open(completed_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        qid = obj.get("id")
                        cfg = obj.get("config_name")
                        com = obj.get("com_ratio_name", "comm_100_100")
                        if qid is not None and cfg:
                            completed.add((qid, cfg, com))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: failed to read completed_file: {e}")

    jsonl_files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if not jsonl_files:
        print(f"Warning: no jsonl files matched in folder: {folder_path} (pattern={pattern})")
        return completed

    loaded_lines = 0
    skipped_lines = 0

    for fp in jsonl_files:
        if os.path.abspath(fp) == os.path.abspath(completed_file):
            continue
        try:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    loaded_lines += 1
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        skipped_lines += 1
                        continue

                    question_id = data.get("id")
                    fedattn_setting = data.get("fedattn_setting", {}) or {}
                    config_name = fedattn_setting.get("config_name")
                    com_ratio_name = fedattn_setting.get("com_ratio_name", "comm_100_100")

                    if question_id is not None and config_name:
                        completed.add((question_id, config_name, com_ratio_name))
                    else:
                        skipped_lines += 1

        except Exception as e:
            print(f"Warning: failed to read {fp}: {e}")

    print(
        f"Loaded {len(completed)} completed tuples "
        f"from {len(jsonl_files)} files ({loaded_lines} lines, skipped {skipped_lines}) "
        f"in {folder_path}"
    )

    try:
        os.makedirs(folder_path, exist_ok=True)
        with open(completed_file, "w", encoding="utf-8") as wf:
            for (qid, cfg, com) in sorted(completed):
                wf.write(json.dumps(
                    {"id": qid, "config_name": cfg, "com_ratio_name": com},
                    ensure_ascii=False
                ) + "\n")
    except Exception as e:
        print(f"Warning: failed to write completed_file: {e}")

    print(f"Completed tuples: {len(completed)}")
    return completed

""" inference main function """
def inference_v2v4real_3d_grounding(args):
    """
    V2V-LLM main inference function for 3D grounding task on V2V4Real dataset.
    """

    """
    V2V-LLM Multimodal Configuration
    """
    my_model_config = {
        'mm_scene_projector_input_size': args.mm_scene_projector_input_size,
        'scene_level_only': args.scene_level_only,  
        'object_level_only': args.object_level_only,
        'scene_feature_mode': args.scene_feature_mode,
        'object_feature_mode': args.object_feature_mode,
        'num_input_frames': args.num_input_frames, 
        'ego_only': args.ego_only,
        'feature_source': args.feature_source,
        'dataset_source': args.dataset_source,
        'num_latency_frames': args.num_latency_frames,
        'positional_error_std': args.positional_error_10_std * 0.1,
    }

    if args.feature_source in ['cobevt', 'early', 'v2xvit', 'attfuse', 'no_fusion']:
        assert args.ego_only, \
            f"feature_source={args.feature_source} must set ego_only=True"
    if my_model_config['scene_level_only']:
        assert not my_model_config['object_level_only'], \
            "scene_level_only and object_level_only cannot be True simultaneously"
    if my_model_config['object_level_only']:
        assert not my_model_config['scene_level_only'], \
            "scene_level_only and object_level_only cannot be True simultaneously"


    # Select and check input files before allocating the model on the GPU.
    with open(args.question_file, "r", encoding="utf-8") as stream:
        list_data_dict = json.load(stream)
    if DEMO_MODE:
        list_data_dict = [list_data_dict[i] for i in DEMO_SAMPLE_INDICES]
    else:
        list_data_dict = list_data_dict[::QUALITY_SAMPLE_STRIDES[args.activate_question]]
    if not list_data_dict:
        raise ValueError("No QA samples selected")
    list_data_dict = get_chunk(list_data_dict, args.num_chunks, args.chunk_idx)
    if not list_data_dict:
        raise ValueError("The selected QA chunk is empty")
    check_required_features(
        list_data_dict, PROJECT_ROOT, args.num_input_frames,
        feature_source=args.feature_source, ego_only=args.ego_only,
        num_latency_frames=args.num_latency_frames,
        scene_feature_mode=args.scene_feature_mode,
        object_feature_mode=args.object_feature_mode,
        scene_level_only=args.scene_level_only,
        object_level_only=args.object_level_only,
    )

    """
    load model and tokenizer
    """
    disable_torch_init() 
    model_path = os.path.expanduser(args.model_path) 
    model_name = get_model_name_from_path(model_path) 
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,                     
        args.model_base,                 
        model_name,                      
        my_model_config=my_model_config  
    )

    """
    hook & attention mask
    """  
    wrapper = V2VFedAttnWrapper(model, device='cuda')


    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)  

    # Conversation template.
    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'Auto switching to {args.conv_mode}.')

    # Data loader.
    if args.dataset_source == 'v2v4real':
        data_loader = create_data_loader_v2v4real(
            list_data_dict,                 
            args.image_folder,               
            tokenizer,                     
            image_processor,                 
            model.config,                    
            args.simplified_object_feature, 
            args.num_input_frames,          
            args.num_latency_frames,         
            args.positional_error_10_std * 0.1,  
            args.feature_source,
            args.ego_only,
            args.conv_mode
        )

    
    """
    Configurations
    """
    
    n_layers = wrapper.n_layers
    experiment_configs = build_experiment_configs(n_layers, args.activate_question)
    print(f"Experiment group: {EXPERIMENT_GROUP}")
    print(f"Configurations per QA: {sum(len(group['layer_configs']) * len(group['settings']) for group in experiment_configs)}")

    answers_file = os.path.expanduser(args.answers_file)
    answers_dir = os.path.dirname(answers_file)          
    completed_results = load_completed_results(answers_dir, args.activate_question)
    print(f"Loaded {len(completed_results)} completed results")

    
    """
    run inference loop
    """ 
    for (input_ids, image_tensor, image_sizes, scene_point_feature_map,
            regression_map, classification_map, detection_box_score,
            object_features, active_agent_mask), data_dict in tqdm( 
                zip(data_loader, list_data_dict), total=len(list_data_dict), ncols=100): 
        
        question_id = data_dict.get('id')
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        token_pos, token_chunks = compute_token_positions(
            input_ids=input_ids,
            my_model_config=my_model_config,
            asker_cav_id = data_dict.get('asker_cav_id', 'ego'),
            IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX
            ) 
        
        global_mask_template, local_mask_template = {}, {}
        if EXPERIMENT_GROUP in ('A', 'B', 'C'):
            dense_global, dense_local = create_fedattn_mask_from_chunks(
                token_chunks=token_chunks,
                seq_len=token_pos['seq_len'],
                batch_size=1,
                com_ratios={},
                asker_cav_id=data_dict.get('asker_cav_id', 'ego'),
                dtype=torch.float16,
                device='cuda',
            )
            for group in experiment_configs:
                for setting in group['settings']:
                    global_mask_template[setting['name']] = dense_global
                    local_mask_template[setting['name']] = dense_local
        else:
            for com_ratio_setting in experiment_configs[0]['settings']:
                com_ratio_name = com_ratio_setting['name']
                com_ratios = com_ratio_setting['ratios']
                global_mask_template[com_ratio_name], local_mask_template[com_ratio_name] = create_fedattn_mask_from_chunks(
                    token_chunks=token_chunks,
                    seq_len=token_pos['seq_len'],
                    batch_size=1,
                    com_ratios=com_ratios,
                    dtype=torch.float16,
                    device='cuda')

        pending_batches = []
        for group in experiment_configs:
            configs_to_run = []
            for setting in group['settings']:
                com_ratio_name = setting['name']
                for config in group['layer_configs']:
                    if (question_id, config[0], com_ratio_name) not in completed_results:
                        configs_to_run.append(config + (com_ratio_name,))
            for offset in range(0, len(configs_to_run), args.batch_size):
                pending_batches.append((group, configs_to_run[offset:offset + args.batch_size]))

        if not pending_batches:
            continue

        for group, batch_fedattn_configs in pending_batches:
            current_h = group['H']

            ratio_settings_by_name = {
                setting['name']: setting
                for setting in group['settings']
            }

            batch_size = len(batch_fedattn_configs)

            batch_scene_point_feature_map = scene_point_feature_map.repeat(batch_size, 1, 1, 1, 1, 1).to(
                dtype=torch.float16,       
                device='cuda',             
                non_blocking=True          
                )
            batch_regression_map = regression_map.repeat(batch_size, 1, 1, 1, 1, 1).to(dtype=torch.float16, device='cuda', non_blocking=True)
            batch_classification_map = classification_map.repeat(batch_size, 1, 1, 1, 1, 1).to(dtype=torch.float16, device='cuda', non_blocking=True)
            batch_detection_box_score = detection_box_score.repeat(batch_size, 1, 1, 1, 1).to(dtype=torch.float16, device='cuda', non_blocking=True)
            batch_object_features = object_features.repeat(batch_size, 1, 1, 1, 1).to(dtype=torch.float16, device='cuda', non_blocking=True)
            batch_active_agent_mask = active_agent_mask.repeat(batch_size, 1, 1).to(dtype=torch.bool, device='cuda', non_blocking=True)
            batch_input_ids = input_ids.repeat(batch_size, 1).to(device='cuda', non_blocking=True)            

            if EXPERIMENT_GROUP in ('A', 'B', 'C'):
                prefill_masks, decode_masks = build_layer_masks(
                    n_layers=wrapper.n_layers,
                    batch_fedattn_configs=batch_fedattn_configs,
                    global_mask_template=global_mask_template,
                    local_mask_template=local_mask_template,
                    token_pos=token_pos,
                    token_chunks=token_chunks,
                    com_ratio_settings=ratio_settings_by_name,
                    sample_key=question_id,
                    collaborator_global_layers=None,
                    asker_cav_id=data_dict.get('asker_cav_id', 'ego'),
                )
            else:
                prefill_masks, decode_masks = build_layer_masks(
                    n_layers=wrapper.n_layers,
                    batch_fedattn_configs=batch_fedattn_configs,
                    global_mask_template=global_mask_template,
                    local_mask_template=local_mask_template,
                )

            wrapper.set_layer_masks(prefill_masks, decode_masks)    
    
            with torch.inference_mode(): 
                output_ids = model.generate(
                    batch_input_ids,             
                    attention_mask=torch.ones_like(batch_input_ids),
                    
                    images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True), # [1, 3, 336, 336] 
                    image_sizes=image_sizes,
                    
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,            
                    num_beams=args.num_beams,    
                    max_new_tokens=args.max_new_tokens, 
                    use_cache=True,              
                    
                    scene_point_feature_map=batch_scene_point_feature_map,  
                    regression_map=batch_regression_map,                    
                    classification_map=batch_classification_map,             
                    detection_box_score=batch_detection_box_score,          
                    object_features=batch_object_features,                   
                    active_agent_mask=batch_active_agent_mask                
                )
    
            batch_outputs = tokenizer.batch_decode(     
                output_ids,                    
                skip_special_tokens=True,           
                )
            batch_outputs = [s.strip() for s in batch_outputs]  

            wrapper.reset()
            
            del prefill_masks, decode_masks
            del output_ids
            
            results_to_write = [] 
            for idx, outputs in enumerate(batch_outputs): 
                result_dict = data_dict.copy()                    
                config_name, global_layers, local_layers, com_ratio_name = batch_fedattn_configs[idx]
                
                result_dict['fedattn_setting'] = {
                    'config_name': config_name,
                    'global_layers': global_layers,
                    'local_layers': local_layers,
                    'com_ratio_name': com_ratio_name,
                }          
                
                if EXPERIMENT_GROUP in ('A', 'B', 'C'):
                    setting = ratio_settings_by_name[com_ratio_name]
                    result_dict['fedattn_setting'].update({
                        'com_ratios': setting['ratios'],
                        'input_keep_ratio': setting.get('input_keep_ratio', 1.0),
                        'seed': setting.get('seed', 0),
                        'panel': setting['panel'],
                        'base_interval': current_h,
                        'collaborator_interval': setting.get('collaborator_interval'),
                        'collaborator_global_layers': setting['collaborator_global_layers'],
                        'shared_baseline': setting.get('shared_baseline', False),
                    })
                    if 'publisher_global_layers' in setting:
                        result_dict['fedattn_setting'].update({
                            'c_sweep_role': setting['c_sweep_role'],
                            'publisher_interval': setting['publisher_interval'],
                            'publisher_global_layers': setting['publisher_global_layers'],
                        })

                result_dict['outputs'] = outputs
                results_to_write.append(result_dict)
    
                completed_results.add((question_id, config_name, com_ratio_name))
                
            with open(answers_file, "a", encoding="utf-8") as f:
                for result in results_to_write:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")                             

            del batch_input_ids, batch_scene_point_feature_map
            del batch_regression_map, batch_classification_map
            del batch_detection_box_score, batch_object_features
            del batch_active_agent_mask
            
            torch.cuda.empty_cache()
            gc.collect()                
            
if __name__ == "__main__":

    QUESTION_CONFIGS = {

        'q1': {
            'model_suffix': 'v2vllmq1_10ep_both_shallow_f1',
            'checkpoint': 1387,
            'question_file': 'v2v4real_3d_grounding_qa_dataset_v2vllmq1.json',
            'num_input_frames': 1,
            'batch_size': 1,
        },

        'q4': {
            'model_suffix': 'v2vllmq4_10ep_both_shallow_f1',
            'checkpoint': 490,
            'question_file': 'v2v4real_3d_grounding_qa_dataset_v2vllmq4.json',
            'num_input_frames': 1,
            'batch_size': 9,
        },
        
        'q5': {
            'model_suffix': 'v2vllmq5_10ep_both_shallow_f2',
            'checkpoint': 490,
            'question_file': 'v2v4real_3d_grounding_qa_dataset_v2vllmq5.json',
            'num_input_frames': 2,  
            'batch_size': 9,
        },
    }
    
    ACTIVE_QUESTION = 'q1'  
    
    config = QUESTION_CONFIGS[ACTIVE_QUESTION]
    print(f"\n{'='*60}")
    print(f"   Running Question Set: {ACTIVE_QUESTION}")
    print(f"   Model: {config['model_suffix']}")
    print(f"   Checkpoint: {config['checkpoint']}")
    print(f"   Question File: {config['question_file']}")
    print(f"   Input Frames: {config['num_input_frames']}")
    print(f"{'='*60}\n")

    
    parser = argparse.ArgumentParser(description="V2V-LLM Fedarated")
  
    parser.add_argument("--batch_size", type=int, default=config['batch_size'],
        help="batch_size(Fed mask settings)")  

    parser.add_argument("--activate_question", type=str, default=ACTIVE_QUESTION)    
    parser.add_argument("--model-path", type=str, 
        default=os.path.join(
            os.path.abspath(os.path.join(SCRIPT_DIR, '../..')), 
            "checkpoints/llava-v1.5-7b-task-lora",
            f"llava-v1.5-7b-task-lora_v2v4real_3d_grounding_{config['model_suffix']}",
            f"checkpoint-{config['checkpoint']}"
            ).replace("\\", "/")
        )
    parser.add_argument("--model-base", type=str, default="liuhaotian/llava-v1.5-7b")
        
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, 
        default=os.path.join(
            os.path.abspath(os.path.join(SCRIPT_DIR, '../../..')),
            "DMSTrack/V2V4Real/official_models/no_fusion_keep_all/npy/co_llm",
            config["question_file"],
            ).replace("\\", "/")
        )
    parser.add_argument("--answers-file", type=str, 
        default=os.path.join(
            os.path.abspath(os.path.join(SCRIPT_DIR, '../..')),
            "playground/data/eval",
            f"v2v4real_3d_grounding_{config['model_suffix']}_{config['checkpoint']}",
            "answers", "val", "llava-v1.5-7b", EXPERIMENT_GROUP,
            f"v2v_llm_{ACTIVE_QUESTION}.jsonl",  
            ).replace("\\", "/")
        )

    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--simplified-object-feature", type=int, default=0)
    parser.add_argument("--mm_scene_projector_input_size", type=int, default=3072)
    parser.add_argument("--scene_level_only", type=bool, default=False)
    parser.add_argument("--object_level_only", type=bool, default=False)
    parser.add_argument("--scene_feature_mode", type=str, default='shallow')
    parser.add_argument("--object_feature_mode", type=str, default='shallow')  
    parser.add_argument("--num_input_frames", type=int, default=config["num_input_frames"])
    parser.add_argument("--ego_only", type=bool, default=False)
    parser.add_argument("--feature_source", type=str, default='no_fusion_keep_all')
    parser.add_argument("--dataset_source", type=str, default='v2v4real')
    
    parser.add_argument("--num_latency_frames", type=int, default=0)
    parser.add_argument("--positional_error_10_std", type=int, default=0)  
    
    args = parser.parse_args()
    if args.activate_question != ACTIVE_QUESTION:
        parser.error("Edit ACTIVE_QUESTION in this file to select a task together with its checkpoint, QA file and frame count.")

    inference_v2v4real_3d_grounding(args)
    
    
    