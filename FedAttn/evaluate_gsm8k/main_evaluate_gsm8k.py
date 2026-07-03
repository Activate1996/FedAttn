# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:28:30 2025

@author: DengXiumei
"""

import re
import jsonlines
import json
import sys
import os
from datetime import datetime
import gc
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Wrapper import LayerMaskedTransformerWrapper
from utils import (
    token_chunk,
    create_causal_mask, create_base_mask,
    build_prefill_and_decode_mask, 
    build_prefill_masks_main, build_decode_masks_main,
    build_prefill_masks_customized, build_decode_masks_customized,
    build_prefill_masks_unipolicy, build_decode_masks_unipolicy,
    build_prefill_masks_sample, build_decode_masks_sample,
)

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def parse_number(answer):
    """
    尝试将字符串格式的答案转换为整数（GSM8K 都是整数）
    支持：整数、小数、分数、科学计数法
    """
    answer = answer.strip()

    # 整数
    if answer.isdigit() or (answer.startswith('-') and answer[1:].isdigit()):
        return int(answer)

    # 小数（如 42.0）
    if '.' in answer and answer.replace('.', '', 1).replace('-', '', 1).isdigit():
        return int(float(answer))

    # 分数（如 3/4）
    if '/' in answer:
        parts = answer.split('/')
        if len(parts) == 2 and parts[0].strip().lstrip('-').isdigit() and parts[1].strip().isdigit():
            num = int(parts[0])
            denom = int(parts[1])
            if denom != 0:
                return int(num / denom)

    # 科学计数法或其他合法格式
    try:
        return int(float(answer))
    except ValueError:
        return None

def extract_boxed_answer(text):
    """
    提取 \boxed{...} 中的答案，返回整数或 None。
    """
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    for ans in matches:
        parsed = parse_number(ans)
        if parsed is not None:
            return parsed
    return None

def extract_keyword_answer(text):
    """
    提取类似 "The answer is 42" 的句式中的数字，返回整数或 None。
    """
    patterns = [
        r'(?:answer is|answer:|final answer is?)\s*(-?\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?)',
        r'the answer is\s*[:\-]?\s*(-?\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?)',
        r'answer\s*[:\-]?\s*(-?\d+(?:\.\d+)?(?:/\d+)?(?:e[+-]?\d+)?)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parsed = parse_number(match.group(1))
            if parsed is not None:
                return parsed
    return None

def extract_fallback_answer(text, max_lines_back=5):
    """
    提取文本中最后一个数字作为兜底手段。
    仅考虑最后 max_lines_back 行，减少误提风险。
    """
    lines = text.strip().splitlines()[-max_lines_back:]
    fallback_text = "\n".join(lines)
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", fallback_text)
    for num in reversed(numbers):
        parsed = parse_number(num)
        if parsed is not None:
            return parsed
    return None

def extract_answer(text):
    """
    GSM8K 答案提取主函数。优先级：
    1. \boxed{}
    2. 关键词句式 ("The answer is 42")
    3. 最后几行中的数字（保守回退）
    返回整数或 None。
    """
    if not text or not isinstance(text, str):
        return None

    for extractor in [extract_boxed_answer, extract_keyword_answer, extract_fallback_answer]:
        result = extractor(text)
        if result is not None:
            return result
    return None

def extract_answer_hf(text):
    match = ANS_RE.search(text)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS

def is_correct(completion, answer):
    gold = extract_answer_hf(answer)
    assert gold != INVALID_ANS, "No ground truth answer found in the document."
    return extract_answer(completion) == gold

def doc_to_text(doc, fewshot_prompt):
    return (
            fewshot_prompt
            + "\nQuestion: "
            + doc["question"]
            + "\nLet's think step by step\n"
            )


def generate_responses(model, dtype, wrapper, device, tokenizer, tokenizer_padding_side, 
                       input_txt, prompt_token_nums, loaded_chunks_batch, 
                       hyperparams, split_way, comm_policy, ratios_comp, ratios_comm):
    """1. tokenizer"""
    inputs = tokenizer(input_txt, return_tensors="pt", padding=True).to(device)
    
    """2. 生成token_chunks"""
    input_ids = inputs["input_ids"]
    padding_mask = inputs["attention_mask"]
    
    batch_size, input_len = input_ids.shape
    seq_lens = padding_mask.sum(dim=1)

    token_chunks = loaded_chunks_batch if loaded_chunks_batch is not None else token_chunk(seq_lens, hyperparams['num_clients'], split_way, prompt_token_nums)

    """3. 生成casual_mask,mask_infs""" 
    causal_mask = create_causal_mask(batch_size = batch_size, seq_len = input_len, dtype = dtype, device = device)
    prefill_mask_infs, decode_mask_infs = create_base_mask(batch_size = batch_size, seq_len = input_len, dtype = dtype, device = device)
 
    """4. 生成local and global masks""" 
    num_comms = int((wrapper.n_layers + hyperparams['num_local_forward'] - 1) // hyperparams['num_local_forward'])
    local_prefill_masks, global_prefill_masks, local_decode_masks, global_decode_masks = build_prefill_and_decode_mask(comm_policy = comm_policy,
        padding_side = tokenizer_padding_side, dtype = dtype, device = device, tok_chunks = token_chunks, seq_lens = seq_lens, max_len = input_len, 
        batch_size = batch_size, num_clients = hyperparams['num_clients'], causal_mask = causal_mask, prefill_mask_infs = prefill_mask_infs, decode_mask_infs = decode_mask_infs,
        num_comms = num_comms, ratios_comp = ratios_comp, ratios_comm = ratios_comm)    
    
    if comm_policy == "main":  
        
        prefill_layer_masks = build_prefill_masks_main(
            num_layers = wrapper.n_layers, 
            num_local_forward = hyperparams['num_local_forward'], 
            global_prefill_mask = global_prefill_masks, 
            local_prefill_mask = local_prefill_masks
            ) 
        decode_layer_masks = build_decode_masks_main(
            num_layers = wrapper.n_layers, 
            num_local_forward = hyperparams['num_local_forward'], 
            global_decode_mask = global_decode_masks, 
            local_decode_masks = local_decode_masks
            )

    elif "sample" in comm_policy:   
    
        prefill_layer_masks = build_prefill_masks_sample(
            num_layers = wrapper.n_layers, 
            num_local_forward = hyperparams['num_local_forward'], 
            global_prefill_masks = global_prefill_masks, 
            local_prefill_mask = local_prefill_masks
            )
        decode_layer_masks = build_decode_masks_sample(
            num_layers = wrapper.n_layers, 
            num_local_forward = hyperparams['num_local_forward'], 
            global_decode_masks = global_decode_masks, 
            local_decode_masks = local_decode_masks
            )
        
    elif comm_policy == "uni_extra":                

        customized_prefill_masks, customized_decode_masks = copy.deepcopy(local_prefill_masks), copy.deepcopy(local_decode_masks) 
        pad_offsets = input_len - seq_lens if tokenizer_padding_side == "left" else seq_lens - seq_lens
        for idx, seq_len in enumerate(seq_lens):
            last_chunk = torch.tensor(token_chunks[idx][-1], device=device) + pad_offsets[idx]
            customized_prefill_masks[idx][:, :, last_chunk] = 0
            for customized_decode_mask in customized_decode_masks:
                customized_decode_mask[idx][:, :, last_chunk] = 0
                
        prefill_layer_masks = build_prefill_masks_customized(
            num_layers = wrapper.n_layers, 
            num_local_forward = hyperparams['num_local_forward'], 
            global_prefill_mask = global_prefill_masks, 
            local_prefill_mask = local_prefill_masks, 
            customized_prefill_mask = torch.minimum(customized_prefill_masks, causal_mask), 
            num_local_forward_last = hyperparams["last_num_local_forward"]
            )
        decode_layer_masks = build_decode_masks_customized(
            num_layers = wrapper.n_layers,
            num_local_forward = hyperparams['num_local_forward'], 
            global_decode_mask = global_decode_masks, 
            local_decode_masks = local_decode_masks,
            customized_decode_masks = customized_decode_masks,
            num_local_forward_last = hyperparams["last_num_local_forward"]
            )
        
    elif comm_policy in ["uni_fr", "uni_bk", "inc_gp", "dec_gp"]: 
        
        prefill_layer_masks = build_prefill_masks_unipolicy(
            num_layers = wrapper.n_layers, 
            num_local_forward = hyperparams['num_local_forward'], 
            global_prefill_mask = global_prefill_masks, 
            local_prefill_mask = local_prefill_masks,
            comm_policy = comm_policy
            )
        decode_layer_masks = build_decode_masks_unipolicy(
            num_layers = wrapper.n_layers, 
            num_local_forward = hyperparams['num_local_forward'], 
            global_decode_mask = global_decode_masks, 
            local_decode_masks = local_decode_masks,
            comm_policy = comm_policy
            )            
        
    else:
        raise ValueError(f"不支持的通信策略: {comm_policy}")            

    responses = []
    for decode_layer_mask in decode_layer_masks:

        out = wrapper.generate(
            input_ids,
            padding_side=tokenizer_padding_side,
            padding_mask=padding_mask,
            prefill_layer_masks=prefill_layer_masks,
            decode_layer_masks=decode_layer_mask,
            max_new_tokens=hyperparams['max_new_tokens'],
            do_sample=hyperparams['do_sample'],
            temperature=hyperparams.get("temperature", 0.2),
            top_k=hyperparams.get("top_k", 20),
            top_p=hyperparams.get("top_p", 0.9)
        )
        wrapper.reset()
        responses.append(tokenizer.batch_decode(
            out[:, input_len:], 
            skip_special_tokens=True
        ))    
    return responses


def load_prompts(id_list, tokenizer, is_loaded_chunks, prompt_prefix="gsm8k_prompt_", prompt_suffix=".txt"):
    """
    加载多个 prompt 文件，拼接为一个完整 prompt，并统计每段（含 \n\n）在 inference 配置下的 token 数。
    
    Args:
        id_list: List[int], prompt 文件编号，如 [1, 2, 3]
        tokenizer: HF tokenizer（你后续做 inference 时用的）
        prompt_prefix: 文件名前缀
        prompt_suffix: 文件名后缀

    Returns:
        full_prompt: str, 拼接后的完整 prompt，段落间以 \n\n 分隔
        token_lengths: List[int], 每段（含空行）tokenizer 后的 token 数（去除 padding）
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(current_dir, "prompt")
    
    prompt_segments = []
    
    for pid in id_list:
        filename = os.path.join(prompt_dir, f"{prompt_prefix}{pid}{prompt_suffix}")
        content = open(filename, "r", encoding="utf-8").read()
        prompt_segments.append(content)
        
    full_prompt = "\n".join(prompt_segments)  # 拼接为整体 prompt（中间加换行）

    if is_loaded_chunks:
        segments_with_newlines = [seg + "\n" for seg in prompt_segments[:-1]] + [prompt_segments[-1]]
    
        # tokenize（使用 inference 的配置）
        encodings = tokenizer(
            segments_with_newlines,
            return_tensors="pt",
            padding=True
        )
        token_lengths = encodings["attention_mask"].sum(dim=1).tolist()  # 每段的有效 token 数（去除 padding）
    else:
        token_lengths = None

    return full_prompt, token_lengths


def create_exp_name(hyperparams, split_way, comm_policy):

    exp_name = f"shot{hyperparams['num_shots']}_c{hyperparams['num_clients']}_lf{hyperparams['num_local_forward']}_sample_{hyperparams['do_sample']}"
    if hyperparams['do_sample']:
        exp_name += f"_temp{hyperparams['temperature']}_k{hyperparams['top_k']}_p{hyperparams['top_p']}"   
    exp_name_appendix = {
        "uni_extra": f"_llf{hyperparams.get('last_num_local_forward')}",
        "sample_uni_comp": f"_rp{hyperparams.get('ratio_comp')}",
        "sample_uni_comm": f"_rm{hyperparams.get('ratio_comm')}",
        "sample_uni_comp_comm": f"_rp{hyperparams.get('ratio_comp')}_rm{hyperparams.get('ratio_comm')}",
        "sample_last_comp": f"_rp{hyperparams.get('ratio_comp')}_{hyperparams.get('ratio_last_comp')}",
        "sample_last_comm": f"_rm{hyperparams.get('ratio_comm')}_{hyperparams.get('ratio_last_comm')}",
        "sample_last_comp_comm": f"_rp{hyperparams.get('ratio_comp')}_{hyperparams.get('ratio_last_comp')}_rm{hyperparams.get('ratio_comm')}_{hyperparams.get('ratio_last_comm')}",
        }          
    exp_name += exp_name_appendix.get(comm_policy,"")  
    
    return f"{hyperparams['max_new_tokens']}_{exp_name}_{split_way}"


def main(hyperparams, test_data, device, split_way, batch_size, comm_policy, script_dir, results_dir):
    print(f"Running experiment with hyperparams: {hyperparams}, split way: {split_way}, comm_policy: {comm_policy}")
    
    """ 0.1. 将文件路径指向results目录 """    
    exp_name = create_exp_name(hyperparams, split_way, comm_policy)          
    model_name = os.path.basename(hyperparams['checkpoint_path'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{timestamp}_{model_name}_{exp_name}"            
    output_file_path = os.path.join(results_dir, output_file)
    output_files = { 
        idx: jsonlines.Writer(open(f"{output_file_path}_client{idx}.jsonl", "w", encoding="utf-8")) 
        for idx in range(hyperparams['num_clients']) 
        }
    acc = {idx: [] for idx in range(hyperparams['num_clients'])}
    
    """ 0.2. 加载token_chunks目录 """ 
    load_file_token_chunks = f"data_process_{model_name}_shot{hyperparams['num_shots']}_c{hyperparams['num_clients']}_{split_way}"
    file_dir_token_chunks = os.path.join(script_dir, "data_process_token_chunks")
    try:
        with open(f"{os.path.join(file_dir_token_chunks, load_file_token_chunks)}.json", "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
        loaded_chunks = None     
    
    """ 1. 加载模型和分词器 """
    tokenizer = AutoTokenizer.from_pretrained(hyperparams['checkpoint_path'])
    model = AutoModelForCausalLM.from_pretrained(hyperparams['checkpoint_path']).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    """ 2. 创建包装器 """
    wrapper = LayerMaskedTransformerWrapper(model, device)
    
    """ 3. 创建prompt """
    fewshot_prompt, prompt_token_nums = load_prompts(list(range(hyperparams['num_shots'])), tokenizer, loaded_chunks == None)  
    ratios_comp = [hyperparams.get('ratio_comp', 1.0) for _ in range(hyperparams['num_clients']-1)] + [hyperparams.get('ratio_last_comp', hyperparams.get('ratio_comp', 1.0))]
    ratios_comm = [hyperparams.get('ratio_comm', 1.0) for _ in range(hyperparams['num_clients']-1)] + [hyperparams.get('ratio_last_comm', hyperparams.get('ratio_comm', 1.0))] 
    
    """ 4. 运行测试 """
    for batch_idx in range(0, len(test_data), batch_size):
        if batch_idx % 100 == 0:
            print(f"Processing {batch_idx}/{len(test_data)}")     

        """ 4.1. load batch prompt """
        batch_docs = [dict(test_data[i]) for i in range(batch_idx, min(batch_idx + batch_size, len(test_data)))]
        batch_prompts = [doc_to_text(doc, fewshot_prompt) for doc in batch_docs]
        loaded_chunks_batch = [loaded_chunks[i] for i in range(batch_idx, min(batch_idx + batch_size, len(test_data)))] if loaded_chunks is not None else None            
        
        """ 4.2. inference batch """
        batch_completions = generate_responses( 
            model = model, dtype = model.dtype, wrapper = wrapper, device = device, 
            tokenizer = tokenizer, tokenizer_padding_side = tokenizer.padding_side, 
            input_txt = batch_prompts, prompt_token_nums = prompt_token_nums, loaded_chunks_batch = loaded_chunks_batch, 
            hyperparams = hyperparams, split_way = split_way, comm_policy = comm_policy, ratios_comp = ratios_comp, ratios_comm = ratios_comm, 
            )               
        
        """ 4.3. 写入每个 client 的输出结果和acc """
        for client_id, client_completions in enumerate(batch_completions):
            for doc, completion in zip(batch_docs, client_completions):
                record = { 
                    "question": doc["question"],
                    "answer": doc["answer"],
                    "completion": completion,
                    "acc": is_correct(completion, doc["answer"]) 
                    }
                output_files[client_id].write(record)
                acc[client_id].append(record["acc"])

    for writer in output_files.values():
        writer.close() # 关闭所有 writer
        
    """ 5. 计算 acc 平均值 并保存 """
    avg_acc = {k: sum(v) / len(v) if v else 0 for k, v in acc.items()}
    avg_values = list(avg_acc.values())
    avg_acc["avg"] = sum(avg_values) / len(avg_values)

    with open(os.path.join(results_dir, "accs", f"acc_{output_file}.json"), "w", encoding="utf-8") as f:
        json.dump(acc, f, indent=4, ensure_ascii=False)    
    with open(os.path.join(results_dir, "avg_acc", f"avg_acc_{output_file}.json"), "w", encoding="utf-8") as f:
        json.dump(avg_acc, f, indent=4, ensure_ascii=False)    
    with open(os.path.join(results_dir, "avg_acc", f"hyperparams_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False) 
    
    # 清理显存资源
    del model, tokenizer, wrapper
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.synchronize()

