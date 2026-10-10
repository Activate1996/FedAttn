# -*- coding: utf-8 -*-
"""Layer-wise FedAttn attention masks for cooperative-driving inference."""

import torch
from typing import List, Optional, Tuple
import random

class V2VFedAttnWrapper:
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.transformer_layers = model.model.layers
        else:
            raise ValueError("can not find transformer layers")
        
        self.n_layers = len(self.transformer_layers)
        print(f"[V2VFedAttnWrapper] find {self.n_layers} layers of Transformer")
        

        self.prefill_layer_masks = None  # List[Tensor], [batch, 1, seq_len, seq_len]
        self.decode_layer_masks = None   # List[Tensor], [batch, 1, 1, seq_len]

        self.capture_attention = False
        self.captured_attentions = {}  # {layer_idx: tensor}
        
        self._orig_forwards = {}
        self._apply_patch()
    
    def _apply_patch(self):
        wrapper = self
        
        for layer_idx, layer in enumerate(self.transformer_layers):
            attn = layer.self_attn
            self._orig_forwards[layer_idx] = attn.forward
            orig_fwd = attn.forward
            
            def make_patched_forward(orig, idx):
                def patched_forward(
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_value = None,
                    output_attentions: bool = False,
                    use_cache: bool = False,
                    **kwargs
                ):
                    q_len = hidden_states.shape[1]

                    if wrapper.capture_attention and q_len > 1:
                        output_attentions = True
                    
                    if q_len > 1:
                        # Prefill 
                        custom_mask = wrapper.prefill_layer_masks[idx] if wrapper.prefill_layer_masks else None
                    else:
                        # Decode 
                        custom_mask = wrapper.decode_layer_masks[idx] if wrapper.decode_layer_masks else None
                    
                    if custom_mask is not None:
                        custom_mask = custom_mask.to(hidden_states.device, hidden_states.dtype)
                        
                        if q_len == 1 and past_key_value is not None:
                            if hasattr(past_key_value, 'get_seq_length'):
                                kv_len = past_key_value.get_seq_length(idx) + 1
                            elif hasattr(past_key_value, '__getitem__'):
                                kv_len = past_key_value[0].shape[2] + 1  
                            if custom_mask.shape[-1] < kv_len:
                                pad_len = kv_len - custom_mask.shape[-1]
                                pad = torch.zeros(
                                    (*custom_mask.shape[:-1], pad_len),
                                    device=custom_mask.device,
                                    dtype=custom_mask.dtype
                                )
                                custom_mask = torch.cat([custom_mask, pad], dim=-1)
                        
                        if attention_mask is not None:
                            attention_mask = torch.minimum(attention_mask, custom_mask)
                        else:
                            attention_mask = custom_mask
                
                    result = orig(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        **kwargs
                    )
                    
                    if wrapper.capture_attention and q_len > 1 and output_attentions:
                        if isinstance(result, tuple) and len(result) >= 2:
                            attn_weights = result[1]
                            if attn_weights is not None:
                                wrapper.captured_attentions[idx] = attn_weights.detach()
                    
                    return result                
                
                return patched_forward
            
            attn.forward = make_patched_forward(orig_fwd, layer_idx)
        
        print(f"[V2VFedAttnWrapper] Already patch {self.n_layers} layers")

    def enable_attention_capture(self):
        self.capture_attention = True
        self.captured_attentions = {}
    
    def disable_attention_capture(self):
        self.capture_attention = False
    
    def get_captured_attentions(self):
        return self.captured_attentions
    
    def clear_captured_attentions(self):
        self.captured_attentions = {}
    
    def set_layer_masks(
        self, 
        prefill_masks: Optional[List[torch.Tensor]] = None,
        decode_masks: Optional[List[torch.Tensor]] = None
    ):

        self.prefill_layer_masks = prefill_masks
        self.decode_layer_masks = decode_masks
    
    def reset(self):
        self.prefill_layer_masks = None
        self.decode_layer_masks = None
    
    def restore(self):
        for idx, layer in enumerate(self.transformer_layers):
            if idx in self._orig_forwards:
                layer.self_attn.forward = self._orig_forwards[idx]
        self._orig_forwards.clear()
    
    def __del__(self):
        try:
            self.restore()
        except:
            pass

def create_causal_mask(batch_size, seq_len, dtype=torch.float16, device='cuda'):

    neg_inf = torch.finfo(dtype).min if dtype.is_floating_point else float("-inf")
    
    mask = torch.triu(
        torch.full((seq_len, seq_len), neg_inf, dtype=dtype, device=device),
        diagonal=1
    )
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    return mask.clone()

def create_fedattn_mask_from_chunks(
    token_chunks: dict,   
    seq_len: int,
    batch_size: int = 1,
    com_ratios: dict = {},  
    asker_cav_id = 'ego',
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    ):
    
    neg_inf = torch.finfo(dtype).min if dtype.is_floating_point else float("-inf")
    causal_mask = create_causal_mask(batch_size, seq_len, dtype, device)
    
    # ---------- Global ----------
    if com_ratios:
        global_mask = torch.full(
            (batch_size, 1, seq_len, seq_len),
            neg_inf,
            device=device,
            dtype=dtype
            )  
        
        all_global_indices = [] 
        for group_name, indices in token_chunks.items():
            if len(indices) == 0:
                continue
            
            idx = torch.tensor(indices, device=device, dtype=torch.long)
            global_mask[:, 0, idx[:, None], idx[None, :]] = 0.0

            if (asker_cav_id == 'ego' and group_name == 'ego') or (asker_cav_id in ['1','car1'] and group_name == 'car1'):
                num_global = max(1, int(len(indices)))
            else:
                num_global = max(1, int(len(indices) * com_ratios[group_name]))             
            global_indices = random.sample(indices, num_global)
            all_global_indices.extend(global_indices)
        
        if len(all_global_indices) > 0:
            global_idx = torch.tensor(all_global_indices, device=device, dtype=torch.long)
            global_mask[:, 0, :, global_idx] = 0.0
        
        global_mask = torch.minimum(global_mask, causal_mask)

    else:
        global_mask = causal_mask

        
    # ---------- Local ----------    
    local_mask = torch.full(
        (batch_size, 1, seq_len, seq_len),
        neg_inf,
        device=device,
        dtype=dtype)

    for _, indices in token_chunks.items():
        if len(indices) == 0:
            continue

        idx = torch.tensor(indices, device=device, dtype=torch.long)
        local_mask[:, 0, idx[:, None], idx[None, :]] = 0.0

    local_mask = torch.minimum(local_mask, causal_mask)

    return global_mask, local_mask

def build_layer_masks( 
    n_layers,
    batch_fedattn_configs,
    global_mask_template,  # [1, 1, seq_len, seq_len]
    local_mask_template,   # [1, 1, seq_len, seq_len]
    token_pos=None,               
    token_chunks=None,               
    com_ratio_settings=None,         
    sample_key=None,                 
    collaborator_global_layers=None, 
    asker_cav_id='ego',              
    ):

    global_mask = next(iter(global_mask_template.values())).squeeze(0)  # [1, seq_len, seq_len]
    batch_size = len(batch_fedattn_configs)
    seq_len = global_mask.shape[-1]
    collaborator_group = 'car1' if asker_cav_id == 'ego' else 'ego'
    publisher_group = 'ego' if asker_cav_id == 'ego' else 'car1'

    settings = com_ratio_settings or {}
    input_drops = {}
    for _, _, _, name in batch_fedattn_configs:
        if name in input_drops:
            continue
        setting = settings.get(name, {})
        keep = float(setting.get('input_keep_ratio', 1.0))
        if not 0.0 <= keep <= 1.0:
            raise ValueError(f"Invalid input_keep_ratio: {keep}")
        dropped = []
        if keep < 1.0:
            for group in token_chunks:
                sensors = list(token_pos[group]) 
                rng = random.Random(f"{sample_key}|{setting.get('seed', 0)}|A|{group}")
                order = rng.sample(sensors, len(sensors))
                dropped.extend(order[int(len(sensors) * keep):])
        input_drops[name] = dropped

    prefill_masks = []
    
    for layer_idx in range(n_layers): 
        
        layer_mask = torch.empty( 
            batch_size, 1, seq_len, seq_len, 
            dtype=global_mask.dtype, 
            device=global_mask.device 
            )      
        
        for batch_idx, (config_name, global_layers, local_layers, comm_ratio) in enumerate(batch_fedattn_configs):           
            if layer_idx in local_layers:
                layer_mask[batch_idx] = local_mask_template[comm_ratio].squeeze(0)
            elif layer_idx in global_layers:
                layer_mask[batch_idx] = global_mask_template[comm_ratio].squeeze(0)
            else:
                raise ValueError(
                    f"[FedAttnMaskError] layer_idx={layer_idx} not in local_layers or global_layers "
                    f"for config={config_name}. "
                    f"len(global_layers)={len(global_layers)}, len(local_layers)={len(local_layers)}"
                )      

            if layer_idx in global_layers and com_ratio_settings is not None:
                setting = settings[comm_ratio]
                allowed = setting.get(
                    'collaborator_global_layers',
                    (collaborator_global_layers or {}).get(config_name),
                )
                collaborator_off = (
                    allowed is not None
                    and layer_idx not in allowed
                )
                publisher_allowed = setting.get(
                    'publisher_global_layers'
                )
                publisher_off = (
                    publisher_allowed is not None
                    and layer_idx not in publisher_allowed
                )
                mask = layer_mask[batch_idx]
                neg_inf = torch.finfo(mask.dtype).min
                for group, source_tokens in token_chunks.items():
                    keep = float(setting.get('ratios', {}).get(group, 1.0))
                    if not 0.0 <= keep <= 1.0:
                        raise ValueError(f"Invalid communication retention: {keep}")
                    dropped = []
                    if group == collaborator_group and collaborator_off:
                        dropped = list(source_tokens)

                    elif group == publisher_group and publisher_off:
                        dropped = list(source_tokens)

                    elif keep < 1.0:
                        sensors = list(token_pos[group])
                        rng = random.Random(
                            f"{sample_key}|{setting.get('seed', 0)}|B|{layer_idx}|{group}")
                        order = rng.sample(sensors, len(sensors))
                        dropped = order[int(len(sensors) * keep):]
                    if dropped:
                        receiver_tokens = [i for g, ids in token_chunks.items()
                                           if g != group for i in ids]
                        rows = torch.tensor(receiver_tokens, device=mask.device, dtype=torch.long)
                        cols = torch.tensor(dropped, device=mask.device, dtype=torch.long)
                        mask[:, rows[:, None], cols[None, :]] = neg_inf

            dropped = input_drops[comm_ratio]
            if dropped:
                mask = layer_mask[batch_idx]
                idx = torch.tensor(dropped, device=mask.device, dtype=torch.long)
                neg_inf = torch.finfo(mask.dtype).min
                mask[:, :, idx] = neg_inf 
                mask[:, idx, :] = neg_inf 
                mask[:, idx, idx] = 0.0  

        prefill_masks.append(layer_mask)
    
    decode_masks = [m[:, :, -1:, :].clone() for m in prefill_masks]
    
    return prefill_masks, decode_masks

def compute_token_positions(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    my_model_config: Optional[dict] = None,
    asker_cav_id: str = 'ego',
    IMAGE_TOKEN_INDEX: int = -200
    ) -> dict:
    
    if my_model_config is None:
        my_model_config = {}
    
    num_frames = my_model_config.get('num_input_frames', 2)
    ego_only = my_model_config.get('ego_only', False)
    
    dataset_source = my_model_config.get('dataset_source', 'v2v4real')
    
    scene_feature_mode = my_model_config.get('scene_feature_mode', 'shallow')
    object_feature_mode = my_model_config.get('object_feature_mode', 'shallow')
    scene_level_only = my_model_config.get('scene_level_only', False)
    object_level_only = my_model_config.get('object_level_only', False)
    
    
    if dataset_source == 'v2v4real':
        if scene_feature_mode == 'shallow':
            num_scene_tokens = 220 
        else:  # deep
            num_scene_tokens = 374 
    elif dataset_source == 'v2xreal':
        if scene_feature_mode == 'shallow':
            num_scene_tokens = 264 
        else:  # deep
            num_scene_tokens = 374  
    elif dataset_source == 'cobevt':
        if scene_feature_mode == 'shallow':
            num_scene_tokens = 288  
        else:  # deep
            num_scene_tokens = 512
    
    num_scene_tokens = my_model_config.get('num_scene_tokens', num_scene_tokens) 
    
    if dataset_source == 'v2v4real':
        num_object_tokens = 50
    elif dataset_source == 'v2xreal':
        num_object_tokens = 100
    
    num_object_tokens = my_model_config.get('num_object_queries', num_object_tokens) 
    
    if scene_level_only:
        tokens_per_agent = num_scene_tokens
    elif object_level_only:
        tokens_per_agent = num_object_tokens
    else:
        tokens_per_agent = num_scene_tokens + num_object_tokens  
    
    if 'num_cavs' in my_model_config:
        num_agents = my_model_config['num_cavs']
    elif ego_only:
        num_agents = 1
    elif dataset_source == 'v2v4real': 
        num_agents = 2
    elif dataset_source == 'v2xreal':
        num_agents = 4  
    elif dataset_source == 'cobevt':
        num_agents = 1 
        
    tokens_per_frame = tokens_per_agent * num_agents 
    total_v2v_tokens = tokens_per_frame * num_frames

    batch_idx = 0
    cur_input_ids = input_ids[batch_idx]
    
    if attention_mask is not None:  
        cur_attention_mask = attention_mask[batch_idx].bool()
        cur_input_ids = cur_input_ids[cur_attention_mask]
    
    img_positions = (cur_input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
    
    img_pos = img_positions[0].item()
    num_text_before = img_pos
    num_text_after = cur_input_ids.shape[0] - img_pos - 1
    new_seq_len = num_text_before + total_v2v_tokens + num_text_after
    
    positions = {}
    positions['text_before'] = list(range(num_text_before))
    
    v2v_start = num_text_before
    
    if num_agents == 1:
        agent_names = ['ego']
    elif num_agents == 2:
        if asker_cav_id == "ego":
            agent_names = ['ego', 'car1']
        else:
            agent_names = ['car1','ego']
    
    agent_all = {name: [] for name in agent_names}
    
    for f in range(num_frames):
        frame_start = v2v_start + f * tokens_per_frame
        
        for agent_idx, agent_name in enumerate(agent_names):
            agent_offset = agent_idx * tokens_per_agent
            agent_start = frame_start + agent_offset
            
            if scene_level_only:
                scene_start = agent_start
                scene_end = scene_start + num_scene_tokens
                positions[f'{agent_name}_scene_f{f}'] = list(range(scene_start, scene_end))
                agent_all[agent_name].extend(positions[f'{agent_name}_scene_f{f}'])
                
            elif object_level_only:
                object_start = agent_start
                object_end = object_start + num_object_tokens
                positions[f'{agent_name}_object_f{f}'] = list(range(object_start, object_end))
                agent_all[agent_name].extend(positions[f'{agent_name}_object_f{f}'])
                
            else:
                scene_start = agent_start
                scene_end = scene_start + num_scene_tokens
                positions[f'{agent_name}_scene_f{f}'] = list(range(scene_start, scene_end))
                
                object_start = scene_end
                object_end = object_start + num_object_tokens
                positions[f'{agent_name}_object_f{f}'] = list(range(object_start, object_end))
                
                agent_all[agent_name].extend(positions[f'{agent_name}_scene_f{f}'])
                agent_all[agent_name].extend(positions[f'{agent_name}_object_f{f}'])
    
    text_after_start = v2v_start + total_v2v_tokens
    positions['text_after'] = list(range(text_after_start, new_seq_len))
    
    positions['text'] = positions['text_before'] + positions['text_after']
    positions['ego'] = agent_all.get('ego', [])
    positions['car1'] = agent_all.get('car1', [])
    
    all_v2v = []
    for name in agent_names:
        all_v2v.extend(agent_all[name])
    positions['all_cars'] = all_v2v
    
    positions['seq_len'] = new_seq_len
    
    positions['_config'] = {
        'num_frames': num_frames,
        'num_agents': num_agents,
        'agent_names': agent_names,
        'ego_only': ego_only,
        'dataset_source': dataset_source,
        'scene_feature_mode': scene_feature_mode,
        'object_feature_mode': object_feature_mode,
        'scene_level_only': scene_level_only,
        'object_level_only': object_level_only,
        'num_scene_tokens': num_scene_tokens,
        'num_object_tokens': num_object_tokens,
        'tokens_per_agent': tokens_per_agent,
        'tokens_per_frame': tokens_per_frame,
        'total_v2v_tokens': total_v2v_tokens,
    }

    token_chunks = {}
    for agent_name in agent_names:
        token_chunks[agent_name] = positions[agent_name] 
    
    if asker_cav_id == 'ego':
        token_chunks['ego'] = token_chunks['ego'] + positions['text']
    elif asker_cav_id == '1':
        token_chunks['car1'] = token_chunks['car1'] + positions['text']      
    
    return positions, token_chunks