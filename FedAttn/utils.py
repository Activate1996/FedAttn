# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 12:11:47 2025

@author: DengXiumei
"""
import random
import torch
import copy

# 均匀: 半开区间 [start, end) 内等间距选 k 个整数（k <= end-start）
def k_uniform_ints(start: int, end: int, k: int):
    n = end - start
    if k < 0 or k > n:
        raise ValueError(f"k={k} 超过区间长度 {n}")
    if k == 0:
        return []
    return [start + (i * n) // k for i in range(k)]


# 非均匀: 间隔递增/递减
def weighted_spaced(start: int, end: int, k: int, mode: str):
    if k < 2:
        return [start] if k == 1 else []
    gaps = k - 1
    weights = list(range(1, gaps + 1))
    if mode == "dec":
        weights.reverse()
    total_w = sum(weights)
    ratios, acc = [0.0], 0
    for w in weights:
        acc += w
        ratios.append(acc / total_w)
    pos = [round(start + (end - start) * r) for r in ratios]
    pos[0], pos[-1] = start, end
    # 校正避免重复
    fixed = [pos[0]]
    for v in pos[1:]:
        fixed.append(max(fixed[-1] + 1, v))
    overshoot = max(0, fixed[-1] - end)
    for i in range(overshoot):
        fixed[-2 - i] -= 1
    return fixed


def get_points(N, M, strategy="uni_fr"):
    """
    - N: 总范围大小，相当于 range(N)
    - M: 整除间隔（用于 uni_div）
    - strategy:
        "uni_fr"  -> 前半区间等间距，数量与 uni_div 一样
        "uni_bk"  -> 直接用 N-1-uni_fr 的镜像（升序）
        "inc_gp"  -> 全区间，间隔递增
        "dec_gp"  -> 全区间，间隔递减
    """
    base = len([x for x in range(N) if x % M == 0])     

    if strategy == "uni_fr":
        return k_uniform_ints(0, N // 2, base)
    elif strategy == "uni_bk":
        fr = k_uniform_ints(0, N // 2, base)
        bk = [N - 1 - x for x in fr]
        return sorted(bk)
    elif strategy == "inc_gp":
        return weighted_spaced(0, N - 1, base, "inc")
    elif strategy == "dec_gp":
        return weighted_spaced(0, N - 1, base, "dec")


# 辅助函数
def token_chunk(seq_lens, num_clients, way, prompt_token_nums):
    """
    分割token序列为多个客户端的部分
    
    Args:
        seq_lens: 序列长度列表，每个元素表示一个样本的总token长度
        num_clients: 客户端数量
        way: 分割方式 - "even"/"even_question_last"/"smart"/"smart_question_last"
        prompt_token_nums: 每个prompt段的token数量列表
        
    Returns:
        splits: 分割结果
    """

    if way == "even":
        # 均匀分割策略
        chunks = []
        for seq_len in seq_lens:
            
            chunk = []
            chunk_size = seq_len // num_clients
            
            for i in range(num_clients):
                start = i * chunk_size 
                end = start + chunk_size if i < num_clients - 1 else seq_len
                chunk.append(list(range(start, end)))  
                    
            chunks.append(chunk)
       
        return chunks

    elif way == "even_question_last":

        """
        新的分割策略：最后一个客户端处理问题，其他客户端均匀分配prompt部分
        question长度 = seq_len - sum(prompt_token_nums)
        """
        all_chunks = []
        prompt_total_tokens = sum(prompt_token_nums)  # 所有prompt的总token数
        prompt_clients = num_clients - 1              # 处理prompt的客户端数量
        
        for seq_len in seq_lens:
            client_token_chunks = [[] for _ in range(num_clients)]
            
            # 为前面的客户端均匀分配prompt tokens
            prompt_chunk_size = prompt_total_tokens // prompt_clients
            
            for i in range(prompt_clients):
                start = i * prompt_chunk_size
                end = start + prompt_chunk_size if i < prompt_clients - 1 else prompt_total_tokens
                client_token_chunks[i] = list(range(start, end))
            
            # 最后一个客户端分配问题部分的tokens
            question_start = prompt_total_tokens
            question_end = seq_len
            client_token_chunks[-1] = list(range(question_start, question_end))

            all_chunks.append(client_token_chunks)
        
        return all_chunks


    elif way == "smart":
        # 智能分割策略
        num_prompts = len(prompt_token_nums)
    
        # 预先决定每个client分到哪些prompt段
        client_prompt_indices = []
        prompt_size = num_prompts // num_clients
        
        for i in range(num_clients):
            start = i * prompt_size
            end = start + prompt_size if i < num_clients - 1 else num_prompts
            client_prompt_indices.append(list(range(start, end)))          
    
        all_chunks = []
        for seq_len in seq_lens:
            client_token_chunks = [[] for _ in range(num_clients)]
            token_cursor = 0
    
            for seg_idx, seg_len in enumerate(prompt_token_nums):
                token_range = list(range(token_cursor, token_cursor + seg_len))
                token_cursor += seg_len
                for client_id, indices in enumerate(client_prompt_indices):
                    if seg_idx in indices:
                        client_token_chunks[client_id].extend(token_range)
                        break
    
            # 剩余 token 是问题部分，交给最后一个 client
            question_tokens = list(range(token_cursor, seq_len))
            client_token_chunks[-1].extend(question_tokens)
    
            all_chunks.append(client_token_chunks)
    
        return all_chunks   

    elif way == "smart_question_last":
        """
        新的分割策略：最后一个客户端只处理问题，其他客户端平均分配所有prompt
        """
            
        num_prompts = len(prompt_token_nums)
        prompt_clients = num_clients - 1  # 处理prompt的客户端数量
        
        # 预先决定每个处理prompt的客户端分到哪些prompt段
        client_prompt_indices = []
        prompt_size = num_prompts // prompt_clients
        
        for i in range(prompt_clients):
            start = i * prompt_size
            end = start + prompt_size if i < prompt_clients - 1 else num_prompts
            client_prompt_indices.append(list(range(start, end)))
        
        # 最后一个客户端不处理任何prompt段
        client_prompt_indices.append([])  # 空列表表示不处理prompt
        
        all_chunks = []
        for seq_len in seq_lens:
            client_token_chunks = [[] for _ in range(num_clients)]
            token_cursor = 0
            
            # 分配prompt段给对应的客户端
            for seg_idx, seg_len in enumerate(prompt_token_nums):
                token_range = list(range(token_cursor, token_cursor + seg_len))
                token_cursor += seg_len
                for client_id, indices in enumerate(client_prompt_indices):
                    if seg_idx in indices:
                        client_token_chunks[client_id].extend(token_range)
                        break
            
            # 最后一个客户端只处理问题部分
            question_tokens = list(range(token_cursor, seq_len))
            client_token_chunks[-1].extend(question_tokens)
            
            all_chunks.append(client_token_chunks)        
        
        return all_chunks
    else:
        raise ValueError(f"Unknown split_way: {way}") # 不支持的分割方式

def create_causal_mask(batch_size, seq_len, dtype, device):
    """Create causal mask once and reuse"""
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), fill_value=torch.finfo(dtype).min, device=device), 
        diagonal=1
        )
    return causal_mask.expand(batch_size, 1, seq_len, seq_len)

def create_base_mask(batch_size, seq_len, dtype, device):    
    """Create base mask once and reuse"""
    prefill_mask_infs = torch.full((batch_size, 1, seq_len, seq_len), fill_value = torch.finfo(dtype).min, device = device)  # 创建全为-inf的mask矩阵（表示所有位置都不可见）
    decode_mask_infs = torch.full((batch_size, 1, 1, seq_len), fill_value = torch.finfo(dtype).min, device = device)  # 创建全为-inf的mask矩阵（表示所有位置都不可见）
    # mask_zeros = torch.zeros((batch_size, 1, seq_len, seq_len), dtype = dtype, device = device) 
    
    return prefill_mask_infs, decode_mask_infs

def apply_offset_to_chunks(tok_chunks, seq_offsets, device):
    """
    给tok_chunks中的每个token索引加上对应的偏移量
    Args:
        tok_chunks: [batch_size][num_chunks][chunk_tokens] 每个token都是int
        seq_offsets: [batch_size] 每个序列的偏移量
        seq_lens: [batch_size] 每个序列的长度，用于边界检查
    Returns:
        offset_chunks: 加上偏移后的chunks
    """
    offset_chunks = []

    for batch_idx, (chunks, offset) in enumerate(zip(tok_chunks, seq_offsets)):
        offset_chunk = [ 
            torch.tensor(chunk, device=device) + offset 
            for chunk in chunks
            ]
        offset_chunks.append(offset_chunk)       
        
    return offset_chunks

def build_prefill_and_decode_mask_main(padding_side, dtype, device, tok_chunks, seq_lens, max_len, 
                                  batch_size, num_clients, causal_mask, prefill_mask_infs, decode_mask_infs): 
    """
    构建 local attention mask：每个客户端只能看到自己的 token
    tok_chunks: List[List[List[int]]], 
    seq_lens: List[int], 
    padding_masks: torch.Tensor, 
    dtype: torch.dtype, 
    device: torch.device,
    ...
    -> Tuple[torch.Tensor, torch.Tensor]    
    """
  
    seq_offsets = max_len - seq_lens if padding_side == "left" else seq_lens - seq_lens
    offset_chunks = apply_offset_to_chunks(tok_chunks, seq_offsets, device) 
    
    if padding_side == "left":
        for batch_idx, (chunks, offset) in enumerate(zip(offset_chunks, seq_offsets)):
            if offset > 0:
                prefill_mask_infs[batch_idx, :, :offset, :offset] = 0 
            
    elif padding_side == "right":
        for batch_idx, (chunks, offset) in enumerate(zip(offset_chunks, seq_offsets)):
            prefill_mask_infs[batch_idx, :, seq_lens[batch_idx]:, seq_lens[batch_idx]:] = 0     
    
    local_prefill_masks = copy.deepcopy(prefill_mask_infs)
    global_prefill_masks = copy.deepcopy(prefill_mask_infs) 
    
    local_decode_masks = [copy.deepcopy(decode_mask_infs) for _ in range(num_clients)]
    global_decode_masks = copy.deepcopy(decode_mask_infs)    
    
    if padding_side == "left":
        for batch_idx, (chunks, offset) in enumerate(zip(offset_chunks, seq_offsets)):
            for client_idx, chunk in enumerate(chunks):
                local_prefill_masks[batch_idx, :, chunk.unsqueeze(1), chunk.unsqueeze(0)] = 0   
                local_decode_masks[client_idx][batch_idx, :, :, chunk] = 0                 
            global_prefill_masks[batch_idx, :, offset:, offset:] = 0
            global_decode_masks[batch_idx, :, :, offset:] = 0             
    elif padding_side == "right":
        for batch_idx, (chunks, offset) in enumerate(zip(offset_chunks, seq_offsets)):
            for client_idx, chunk in enumerate(chunks):
                local_prefill_masks[batch_idx, :, chunk.unsqueeze(1), chunk.unsqueeze(0)] = 0   
                local_decode_masks[client_idx][batch_idx, :, :, chunk] = 0   
            global_prefill_masks[batch_idx, :, :seq_lens[batch_idx], :seq_lens[batch_idx]] = 0  
            global_decode_masks[batch_idx, :, :, :seq_lens[batch_idx]] = 0 
    
    return torch.minimum(local_prefill_masks, causal_mask), torch.minimum(global_prefill_masks, causal_mask), local_decode_masks, global_decode_masks


def build_prefill_and_decode_mask_sample(padding_side, dtype, device, tok_chunks, seq_lens, max_len, 
                                         batch_size, num_clients, causal_mask, prefill_mask_infs, decode_mask_infs,
                                         num_comms, ratios_comp, ratios_comm):
    
    tok_chunks_comp = copy.deepcopy(tok_chunks)
    tok_chunks_not_comp = []
    for idx_chunks, chunks in enumerate(tok_chunks):
        tok_chunk_not_comp = []
        for idx_chunk, chunk in enumerate(chunks):
            tok_chunks_comp[idx_chunks][idx_chunk] = random.sample(chunk, max(int(ratios_comp[idx_chunk]*len(chunk)),1))
            tok_chunk_not_comp = tok_chunk_not_comp + [item for item in chunk if item not in set(tok_chunks_comp[idx_chunks][idx_chunk])]
            
        tok_chunks_not_comp.append([tok_chunk_not_comp])

    tok_chunks_comms = []
    for _ in range(num_comms):
        tok_chunks_comm = []
        for idx_chunks, chunks in enumerate(tok_chunks_comp):
            tok_chunk_comm_sum = []
            for idx_chunk, chunk in enumerate(chunks):
                tok_chunk_comm_sum = tok_chunk_comm_sum + random.sample(chunk, max(int(ratios_comm[idx_chunk]*len(chunk)),1))    
            tok_chunks_comm.append(tok_chunk_comm_sum)    
        tok_chunks_comms.append(tok_chunks_comm)

    tok_chunks_comms_clients = []
    for comm_idx in range(num_comms):
        tok_chunks_comm_clients = copy.deepcopy(tok_chunks_comp)
        for idx_chunks, chunks in enumerate(tok_chunks_comp):
            for idx_chunk, chunk in enumerate(chunks):
                tok_chunks_comm_clients[idx_chunks][idx_chunk] = list(set(tok_chunks_comp[idx_chunks][idx_chunk] + tok_chunks_comms[comm_idx][idx_chunks]))
        tok_chunks_comms_clients.append(tok_chunks_comm_clients)


    seq_offsets = max_len - seq_lens if padding_side == "left" else seq_lens - seq_lens
    offset_tok_chunks_comp = apply_offset_to_chunks(tok_chunks_comp, seq_offsets, device)
    offset_tok_chunks_comms = [apply_offset_to_chunks(tok_chunks_comm, seq_offsets, device) for tok_chunks_comm in tok_chunks_comms_clients] 
    
    offset_tok_chunks_not_comp = apply_offset_to_chunks(tok_chunks_not_comp, seq_offsets, device)
    
    global_prefill_mask_init = copy.deepcopy(prefill_mask_infs)
    for batch_idx, offset_tok_chunk_not_comp in enumerate(offset_tok_chunks_not_comp):
        if len(offset_tok_chunk_not_comp[0])>0:
            global_prefill_mask_init[batch_idx, :, offset_tok_chunk_not_comp[0], offset_tok_chunk_not_comp[0]] = 0

    if padding_side == "left":        
        for batch_idx, offset in enumerate(seq_offsets):
            if offset > 0:                
                global_prefill_mask_init[batch_idx, :, :offset, :offset] = 0     
    if padding_side == "right":             
        for batch_idx, offset in enumerate(seq_offsets):             
            global_prefill_mask_init[batch_idx, :, seq_lens[batch_idx]:, seq_lens[batch_idx]:] = 0    
 
    global_prefill_masks = [copy.deepcopy(global_prefill_mask_init) for _ in range(num_comms)] 
    global_decode_masks = [[copy.deepcopy(decode_mask_infs) for _ in range(num_clients)] for _ in range(num_comms)] 
    local_prefill_masks = copy.deepcopy(global_prefill_mask_init)
    local_decode_masks = [copy.deepcopy(decode_mask_infs) for _ in range(num_clients)]
     
    for comm_idx, offset_tok_chunks_comm in enumerate(offset_tok_chunks_comms):
        for batch_idx, offset_chunks in enumerate(offset_tok_chunks_comm): 
            for client_idx, offset_chunk in enumerate(offset_chunks):          
                global_prefill_masks[comm_idx][batch_idx, :, offset_tok_chunks_comp[batch_idx][client_idx].unsqueeze(1), offset_chunk.unsqueeze(0)] = 0
                global_decode_masks[comm_idx][client_idx][batch_idx, :, :, offset_chunk] = 0   
            
    for batch_idx, chunks in enumerate(offset_tok_chunks_comp):                
        for client_idx, chunk in enumerate(chunks):                
            local_prefill_masks[batch_idx, :, chunk.unsqueeze(1), chunk.unsqueeze(0)] = 0 
            local_decode_masks[client_idx][batch_idx, :, :, chunk] = 0 
    
    global_prefill_masks = [torch.minimum(global_prefill_mask, causal_mask) for global_prefill_mask in global_prefill_masks] 
    
    return torch.minimum(local_prefill_masks, causal_mask), global_prefill_masks, local_decode_masks, global_decode_masks



def build_prefill_and_decode_mask(comm_policy, padding_side, dtype, device, tok_chunks, seq_lens, max_len, 
                                  batch_size, num_clients, causal_mask, prefill_mask_infs, decode_mask_infs,
                                  num_comms = None, ratios_comp = None, ratios_comm = None):


    if comm_policy in ["main", "uni_extra", "uni_fr", "uni_bk", "inc_gp", "dec_gp"]:
        return build_prefill_and_decode_mask_main(padding_side = padding_side, dtype = dtype, device = device, tok_chunks = tok_chunks, seq_lens = seq_lens, max_len = max_len, 
                                             batch_size = batch_size, num_clients = num_clients, causal_mask = causal_mask, prefill_mask_infs = prefill_mask_infs, decode_mask_infs = decode_mask_infs)                 

    elif "sample" in comm_policy:
        return build_prefill_and_decode_mask_sample(padding_side = padding_side, dtype = dtype, device = device, tok_chunks = tok_chunks, seq_lens = seq_lens, max_len = max_len, 
                                                    batch_size = batch_size, num_clients = num_clients, causal_mask = causal_mask, prefill_mask_infs = prefill_mask_infs, decode_mask_infs = decode_mask_infs, 
                                                    num_comms = num_comms, ratios_comp = ratios_comp, ratios_comm = ratios_comm)     
        

def build_prefill_masks_main(num_layers, num_local_forward, global_prefill_mask, local_prefill_mask):
    
    prefill_masks = []
    for layer_idx in range(num_layers):
        if layer_idx % num_local_forward == 0:
            prefill_masks.append(global_prefill_mask)
        else:
            prefill_masks.append(local_prefill_mask)

    return prefill_masks

def build_decode_masks_main(num_layers, num_local_forward, global_decode_mask, local_decode_masks):
    
    decode_masks = []    
    for idx, mask in enumerate(local_decode_masks):
        decode_mask = []
        for layer_idx in range(num_layers):
            if layer_idx % num_local_forward == 0:
                decode_mask.append(global_decode_mask)
            else:
                decode_mask.append(mask)
        decode_masks.append(decode_mask)

    return decode_masks


def build_prefill_masks_sample(num_layers, num_local_forward, global_prefill_masks, local_prefill_mask):
    
    prefill_masks = []
    idx_com = 0
    for layer_idx in range(num_layers):
        if layer_idx % num_local_forward == 0:
            prefill_masks.append(global_prefill_masks[idx_com])
            idx_com += 1
        else:
            prefill_masks.append(local_prefill_mask)

    return prefill_masks

def build_decode_masks_sample(num_layers, num_local_forward, global_decode_masks, local_decode_masks):
    
    decode_masks = []
    for idx, mask in enumerate(local_decode_masks):
        decode_mask = []
        idx_com = 0
        for layer_idx in range(num_layers):
            if layer_idx % num_local_forward == 0:
                decode_mask.append(global_decode_masks[idx_com][idx])
                idx_com += 1
            else:
                decode_mask.append(mask)
        decode_masks.append(decode_mask)

    return decode_masks


def build_prefill_masks_customized(num_layers, num_local_forward, global_prefill_mask, local_prefill_mask, customized_prefill_mask, num_local_forward_last):
        
    prefill_masks = []
    for layer_idx in range(num_layers):
        if layer_idx % num_local_forward == 0:
            prefill_masks.append(global_prefill_mask)
        else:
            if layer_idx % num_local_forward_last == 0:
                prefill_masks.append(customized_prefill_mask)
            else:
                prefill_masks.append(local_prefill_mask)                

    return prefill_masks

def build_decode_masks_customized(num_layers, num_local_forward, global_decode_mask, local_decode_masks, customized_decode_masks, num_local_forward_last):
    
    decode_masks = []
    for idx, mask in enumerate(local_decode_masks):
        decode_mask = []
        for layer_idx in range(num_layers):
            if layer_idx % num_local_forward == 0:
                decode_mask.append(global_decode_mask)
            else:
                if layer_idx % num_local_forward_last == 0:
                    decode_mask.append(customized_decode_masks[idx])
                else:
                    decode_mask.append(mask)
        decode_masks.append(decode_mask)

    return decode_masks


def build_prefill_masks_unipolicy(num_layers, num_local_forward, global_prefill_mask, local_prefill_mask, comm_policy):
    
    comm_points = get_points(num_layers, num_local_forward, comm_policy)
    prefill_masks = []
    for layer_idx in range(num_layers):
        if layer_idx in comm_points:
            prefill_masks.append(global_prefill_mask)
        else:
            prefill_masks.append(local_prefill_mask)

    return prefill_masks

def build_decode_masks_unipolicy(num_layers, num_local_forward, global_decode_mask, local_decode_masks, comm_policy):
    
    comm_points = get_points(num_layers, num_local_forward, comm_policy)
    decode_masks = []
    for idx, mask in enumerate(local_decode_masks):
        decode_mask = []
        for layer_idx in range(num_layers):
            if layer_idx in comm_points:
                decode_mask.append(global_decode_mask)
            else:
                decode_mask.append(mask)
        decode_masks.append(decode_mask)

    return decode_masks
