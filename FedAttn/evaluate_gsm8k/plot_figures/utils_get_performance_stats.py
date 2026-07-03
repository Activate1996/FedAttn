# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:57:02 2025

@author: DengXiumei
"""

from transformers import AutoConfig
import torch

"""
calculate FLOPs and Memory
"""

# ================= Utils =================

def torch_dtype_from_cfg(x):
    """
    根据 config 里读到的 dtype 字段，返回对应的 torch.dtype。

    参数:
        x: 可能是 torch.dtype 本身，也可能是字符串（如 "float16"），
           或者 None/其他。

    返回:
        torch.dtype 类型，若无法解析则默认返回 torch.float16。
    """    
    if isinstance(x, torch.dtype):  # 已经是 torch.dtype，直接返回
        return x
    if isinstance(x, str):          # 如果是字符串，则 getattr(torch, x)，例如 "float32" -> torch.float32
        return getattr(torch, x)
    return torch.float16            # 其他情况，回退到 float16

def bytes_per_element(dtype):
    """
    返回给定 dtype 的单元素占用字节数。

    参数:
        dtype: torch.dtype

    返回:
        int，单元素字节数。
    """    
    if dtype in (torch.float16, torch.bfloat16): return 2  # 半精度和 bf16 占 2 bytes
    if dtype == torch.float32: return 4                    # 单精度 4 bytes
    if dtype == torch.float64: return 8                    # 双精度 8 bytes
    if dtype in (torch.int8, torch.uint8): return 1        # 8-bit 整数 1 byte
    return 2                                               # 其他情况默认 2 bytes

# ============== Read model shape ==============

def read_shape(repo_id):
      
    # 读取模型配置
    cfg = AutoConfig.from_pretrained(repo_id)

    # ========= 兼容不同模型字段名，提取隐藏维 d =========
    d  = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "model_dim", None)
    if d is None: 
        raise AttributeError("hidden size not found")      

    # ========= 层数 L =========
    L  = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None)
    if L is None:
        raise AttributeError("num_hidden_layers not found")

    # ========= 注意力总头数 H =========
    H  = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if H is None:
        raise AttributeError("num_attention_heads not found")
        
    # ========= KV 头数 H_kv（支持 GQA/MQA） =========
    H_kv = getattr(cfg, "num_key_value_heads", None) or H

    # ========= 单头维度 d_h =========
    d_h  = d // H

    # ========= FFN 中间维 d_ff =========
    d_ff = getattr(cfg, "intermediate_size", None) or (4*d)
    
    # ========= 激活函数种类 =========
    act  = (getattr(cfg, "hidden_act", "") or "").lower()
    if any(k in act for k in ["silu","swish","swiglu","glu"]):
        ffn_weight, ffn_flop = 3, 6
    else:
        ffn_weight, ffn_flop = 2, 4
        
        
    # ========= dtype 与每元素字节数 =========
    torch_dtype = torch_dtype_from_cfg(getattr(cfg, "torch_dtype", torch.float16))
    bytes_per   = bytes_per_element(torch_dtype)
    
    # ========= 词表大小 =========。    
    vocab_size  = getattr(cfg, "vocab_size", None) or 0

    return {
        "d": d, "L": L, "H": H, "H_kv": H_kv, "d_h": d_h, "d_ff": d_ff,
        "dtype": torch_dtype, "bytes": bytes_per,
        "ffn_weight": ffn_weight, "ffn_flop": ffn_flop,
        "vocab_size": vocab_size
    }

C_ATTN = 1.0   
C_FFN  = 1.0   

def kv_bytes_per_layer(T_kv_i: int, shape: dict) -> int:   
    return int(T_kv_i * shape["H_kv"] * shape["d_h"] * 2 * shape["bytes"])

def ws_block_per_layer(T_q: int, T_kv_i: int, shape: dict) -> int:
    b   = shape["bytes"]     
    H   = shape["H"]        
    d   = shape["d"]        
    d_ff= shape["d_ff"]    
    r   = shape["H_kv"] / shape["H"]  
    
    memory_steps = {}
    
    step1_x = b * T_q * d
    memory_steps["step1_initial_activations"] = {
        "x": step1_x,
        "total": step1_x
    }
    
    step2_x = step1_x
    step2_norm1 = b * T_q * d
    step2_total = step2_x + step2_norm1
    memory_steps["step2_layernorm1"] = {
        "x": step2_x,
        "norm1_output": step2_norm1,
        "total": step2_total
    }
    
    step3_x = step1_x
    step3_norm1 = step2_norm1
    step3_q = b * T_q * d
    step3_k = b * T_kv_i * d * r
    step3_v = b * T_kv_i * d * r
    step3_total = step3_x + step3_norm1 + step3_q + step3_k + step3_v
    memory_steps["step3_qkv_projection"] = {
        "x": step3_x,
        "norm1_output": step3_norm1,
        "Q": step3_q,
        "K": step3_k,
        "V": step3_v,
        "total": step3_total
    }
    
    step4_x = step1_x
    step4_q = step3_q
    step4_k = step3_k
    step4_v = step3_v
    
    step4_attn_scores = b * H * T_q * T_kv_i
    
    step4_total = step4_x + step4_q + step4_k + step4_v + step4_attn_scores
    memory_steps["step4_attention_scores"] = {
        "x": step4_x,
        "Q": step4_q,
        "K": step4_k,
        "V": step4_v,
        "attention_scores": step4_attn_scores,
        "total": step4_total
    }
    
    step5_x = step1_x
    step5_q = step3_q
    step5_k = step3_k
    step5_v = step3_v 
    step5_attn_scores = step4_attn_scores
    step5_attn_output = b * T_q * d
    step5_total = step5_x + step5_attn_output  + step5_q + step5_k + step5_v + step5_attn_scores
    memory_steps["step5_attention_output"] = {
        "x": step5_x,
        "attention_output": step5_attn_output,
        "Q": step5_q,
        "K": step5_k,
        "V": step5_v,
        "attention_scores": step5_attn_scores,        
        "total": step5_total
    }
    
    step6_attn_residual = b * T_q * d
    memory_steps["step6_first_residual"] = {
        "attn_with_residual": step6_attn_residual,
        "total": step6_attn_residual
    }
    
    step7_attn_residual = step6_attn_residual
    step7_norm2 = b * T_q * d
    step7_total = step7_attn_residual + step7_norm2
    memory_steps["step7_layernorm2"] = {
        "attn_with_residual": step7_attn_residual,
        "norm2_output": step7_norm2,
        "total": step7_total
    }
    
    step8_attn_residual = step7_attn_residual
    step8_norm2 = step7_norm2
    step8_intermediate = b * T_q * d_ff
    step8_activated = b * T_q * d_ff
    step8_total = step8_attn_residual + step8_norm2 + step8_intermediate + step8_activated
    memory_steps["step8_ffn_intermediate"] = {
        "attn_with_residual": step8_attn_residual,
        "norm2_output": step8_norm2,
        "ffn_intermediate": step8_intermediate,
        "ffn_activated": step8_activated,
        "total": step8_total
    }
    
    step9_attn_residual = step7_attn_residual
    step9_activated = step8_activated
    step9_ffn_output = b * T_q * d
    step9_total = step9_attn_residual + step9_activated + step9_ffn_output
    memory_steps["step9_ffn_output"] = {
        "attn_with_residual": step9_attn_residual,
        "ffn_activated": step9_activated,
        "ffn_output": step9_ffn_output,
        "total": step9_total
    }
    
    step10_final = b * T_q * d
    memory_steps["step10_final_residual"] = {
        "final_output": step10_final,
        "total": step10_final
    }
    
    peak_memory = 0
    peak_step = ""
    for step_name, step_data in memory_steps.items():
        if step_data["total"] > peak_memory:
            peak_memory = step_data["total"]
            peak_step = step_name
    
    return {
        # "memory_steps": memory_steps,
        "memory_step_peak": memory_steps[peak_step],
        "peak_memory": peak_memory,
        "peak_step": peak_step
    }

def flops_block_per_layer(T_q: int, T_kv_i: int, shape: dict) -> int:  
    d  = shape["d"]
    r = shape["H_kv"] / shape["H"]
    
    flops_linear = 2 * T_q * d * d * (2 + 2*r)
    flops_attn  = 4 * T_q * T_kv_i * d
    flops_ffn   = shape["ffn_flop"] * T_q * d * shape["d_ff"]
    
    return {"flops_total": int(flops_linear + flops_attn + flops_ffn), 
            "flops_linear": flops_linear, 
            "flops_attn": flops_attn, 
            "flops_ffn": flops_ffn}

def ws_embedding(T_q: int, shape: dict) -> int:   
    return int(shape["bytes"] * T_q * shape["d"])

def flops_embedding(T_q: int, shape: dict) -> int:  
    return 0  # embedding lookup

def ws_lm_head(T_logits: int, shape: dict) -> int: 
    return int(shape["bytes"] * T_logits * shape["vocab_size"])

def flops_lm_head(T_logits: int, shape: dict) -> int:  
    return int(2 * T_logits * shape["d"] * shape["vocab_size"])

def get_model_weight_byte(shape: dict):   
    d   = shape["d"]
    L   = shape["L"]
    H   = shape["H"]
    H_kv= shape["H_kv"]
    d_ff= shape["d_ff"]
    r   = H_kv / H

    per_layer_params = (2+2*r)*d*d + shape["ffn_weight"]*d*d_ff
    return {"model": L * per_layer_params * shape["bytes"] + shape["vocab_size"] * d * shape["bytes"],
            "emb_layer": shape["vocab_size"] * d * shape["bytes"],
            "trans_layer": per_layer_params * shape["bytes"]
        }

def summarize_prefill(model_w_byte, kv_comm, shape, T_q, T_kv_list, T_logits = 1):
    L = shape["L"]
    if len(T_kv_list) != L:
        raise ValueError(f"len(T_kv_list)={len(T_kv_list)} != L={L}")
    
    kv_i   = [kv_bytes_per_layer(tk, shape) for tk in T_kv_list]
    flops_i= [flops_block_per_layer(T_q, tk, shape) for tk in T_kv_list]
    ws_i   = [ws_block_per_layer(T_q, tk, shape) for tk in T_kv_list]

    kv_cache = []
    acc = 0
    for x in kv_i:
        acc += x
        kv_cache.append(acc)

    emb_ws = ws_embedding(T_q, shape)
    emb_f  = flops_embedding(T_q, shape)
    peak_emb_load_per_layer = model_w_byte["emb_layer"] + max(emb_ws, 0) 
    peak_emb_load_all = model_w_byte["model"] + max(emb_ws, 0) 

    layers = []
    layers.append({
        "name":"embedding",
        "flops_total":emb_f,
        "ws_peak":emb_ws,
        "kv_cache":0,        
        "overall_mem_peak_load_per_layer":peak_emb_load_per_layer,
        "overall_mem_peak_load_all":peak_emb_load_all
        })

    for i in range(L):
        peak_i_load_per_layer = model_w_byte["trans_layer"] + kv_cache[i] + ws_i[i]["peak_memory"]
        peak_i_load_all = model_w_byte["model"] + kv_cache[i] + ws_i[i]["peak_memory"]
        
        layers.append({
            "name":f"block_{i}",
            "flops_total":flops_i[i]["flops_total"],
            "ws_peak":ws_i[i]["peak_memory"],
            "kv_cache":kv_cache[i],
            "overall_mem_peak_load_per_layer":peak_i_load_per_layer,
            "overall_mem_peak_load_all":peak_i_load_all            
            # "appendix_kv_i":kv_i[i],
            # "appendix_flops_i":flops_i[i],
            # "appendix_ws_i":ws_i[i]            
        })

    lm_f = flops_lm_head(T_logits, shape)
    lm_ws= ws_lm_head(T_logits, shape)
    peak_lm_load_per_layer = model_w_byte["emb_layer"] + kv_cache[-1] + lm_ws
    peak_lm_load_all = model_w_byte["model"] + kv_cache[-1] + lm_ws    
    layers.append({
        "name":"lm_head",
        "flops_total":lm_f,
        "ws_peak":lm_ws,
        "kv_cache":kv_cache[-1],  
        "overall_mem_peak_load_per_layer":peak_lm_load_per_layer,
        "overall_mem_peak_load_all":peak_lm_load_all                
        })

    flops_total = sum([l["flops_total"] for l in layers])
    peak_overall_load_per_layer = max(l["overall_mem_peak_load_per_layer"] for l in layers)
    peak_layer_idx_load_per_layer = int(max(range(len(layers)), key=lambda k: layers[k]["overall_mem_peak_load_per_layer"]))
    
    peak_overall_load_only_trans = max(l["overall_mem_peak_load_per_layer"] for l in layers[1:-1]) 
    peak_layer_idx_load_only_trans = int(max(range(1,len(layers)-1), key=lambda k: layers[k]["overall_mem_peak_load_per_layer"]))

    peak_overall_load_all = max(l["overall_mem_peak_load_all"] for l in layers)
    peak_layer_idx_load_all = int(max(range(len(layers)), key=lambda k: layers[k]["overall_mem_peak_load_all"]))
    

    return {
        "layers": {
            "mem_peak_layer_load_per_layer": layers[peak_layer_idx_load_per_layer],
            "mem_peak_layer_load_only_trans": layers[peak_layer_idx_load_only_trans],
            "mem_peak_layer_load_all": layers[peak_layer_idx_load_all]
            },
        "totals": {
            "kv_comm": kv_comm,
            "kv_cache": kv_cache[-1],
            "flops": flops_total,
            "mem_peak_load_per_layer": peak_overall_load_per_layer,
            "mem_peak_load_only_trans": peak_overall_load_only_trans,          
            "mem_peak_load_all": peak_overall_load_all         
        }
    }

def summarize_decode(model_w_byte, shape, T0_kv_list, G):

    L = shape["L"]
    if len(T0_kv_list) != L:
        raise ValueError(f"len(T0_kv_list)={len(T0_kv_list)} != L={L}")


    d  = shape["d"]
    r = shape["H_kv"] / shape["H"]
    const_per_step_per_layer = 2*d*d*(2+2*r) + shape["ffn_flop"]*d*shape["d_ff"]

    attn_sum = 0
    for T0 in T0_kv_list:
        attn_sum += G*T0 + (G*(G-1))//2

    flops_total = int(L*G*const_per_step_per_layer + 4*d*attn_sum + 2*G*d*shape["vocab_size"])

    kv_cache_final_step = 0
    per_layer_ws_final_step = []
    for T0 in T0_kv_list:
        Tkv_final_step = T0 + G  
        kv_cache_final_step += kv_bytes_per_layer(Tkv_final_step, shape)  
        per_layer_ws_final_step.append(ws_block_per_layer(1, Tkv_final_step, shape)["peak_memory"])  

    lm_ws = ws_lm_head(1, shape)  
    ws_max_final_step = max(max(per_layer_ws_final_step), lm_ws)

    peak_overall_load_per_layer = model_w_byte["trans_layer"] + ws_max_final_step + kv_cache_final_step
    peak_overall_only_trans = model_w_byte["trans_layer"] + max(per_layer_ws_final_step) + kv_cache_final_step
    peak_overall_load_all = model_w_byte["model"] + ws_max_final_step + kv_cache_final_step   
    

    return {
        "flops_total": flops_total,
        "kv_cache_final_step": kv_cache_final_step,
        "ws_max_final_step": ws_max_final_step,
        "final_step_mem_peak_load_per_layer":peak_overall_load_per_layer,
        "final_step_mem_peak_only_trans":peak_overall_only_trans,
        "final_step_mem_peak_load_all":peak_overall_load_all          
        
    }




