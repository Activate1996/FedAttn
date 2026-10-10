# -*- coding: utf-8 -*-


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
    """Convert integer, decimal, fraction, or scientific-notation text to an integer.

    Non-integer numeric values are truncated toward zero. Return None when
    conversion fails.
    """
    answer = answer.strip()

    # Integers.
    if answer.isdigit() or (answer.startswith('-') and answer[1:].isdigit()):
        return int(answer)

    # Decimals, such as 42.0.
    if '.' in answer and answer.replace('.', '', 1).replace('-', '', 1).isdigit():
        return int(float(answer))

    # Fractions, such as 3/4.
    if '/' in answer:
        parts = answer.split('/')
        if len(parts) == 2 and parts[0].strip().lstrip('-').isdigit() and parts[1].strip().isdigit():
            num = int(parts[0])
            denom = int(parts[1])
            if denom != 0:
                return int(num / denom)

    # Scientific notation and other float-compatible formats.
    try:
        return int(float(answer))
    except ValueError:
        return None

def extract_boxed_answer(text):
    """Extract an integer from a LaTeX boxed answer, or return None."""
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    for ans in matches:
        parsed = parse_number(ans)
        if parsed is not None:
            return parsed
    return None

def extract_keyword_answer(text):
    """Extract an integer from an answer phrase, or return None."""
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
    """Return the last parseable number from the final max_lines_back lines."""
    lines = text.strip().splitlines()[-max_lines_back:]
    fallback_text = "\n".join(lines)
    numbers = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", fallback_text)
    for num in reversed(numbers):
        parsed = parse_number(num)
        if parsed is not None:
            return parsed
    return None

def extract_answer(text):
    """Extract an integer answer, or return None.

    Try boxed answers first, followed by answer phrases and numbers in the
    final lines of the response.
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
    """1. Tokenize the inputs."""
    inputs = tokenizer(input_txt, return_tensors="pt", padding=True).to(device)
    
    """2. Partition input tokens among participants."""
    input_ids = inputs["input_ids"]
    padding_mask = inputs["attention_mask"]
    
    batch_size, input_len = input_ids.shape
    seq_lens = padding_mask.sum(dim=1)

    token_chunks = loaded_chunks_batch if loaded_chunks_batch is not None else token_chunk(seq_lens, hyperparams['num_clients'], split_way, prompt_token_nums)

    """3. Create the causal mask and base masks.""" 
    causal_mask = create_causal_mask(batch_size = batch_size, seq_len = input_len, dtype = dtype, device = device)
    prefill_mask_infs, decode_mask_infs = create_base_mask(batch_size = batch_size, seq_len = input_len, dtype = dtype, device = device)
 
    """4. Build local and global attention masks.""" 
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
    """Load and join prompt segments with a single newline.

    Args:
        id_list: Prompt file indices.
        tokenizer: Tokenizer used for inference.
        is_loaded_chunks: Whether to compute segment token counts.
        prompt_prefix: Prompt filename prefix.
        prompt_suffix: Prompt filename suffix.

    Returns:
        The combined prompt and, when requested, token counts for each segment
        including its trailing separator and excluding padding. Otherwise,
        the token-count result is None.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(current_dir, "prompt")
    
    prompt_segments = []
    
    for pid in id_list:
        filename = os.path.join(prompt_dir, f"{prompt_prefix}{pid}{prompt_suffix}")
        content = open(filename, "r", encoding="utf-8").read()
        prompt_segments.append(content)
        
    full_prompt = "\n".join(prompt_segments)  # Join prompt segments with a newline.

    if is_loaded_chunks:
        segments_with_newlines = [seg + "\n" for seg in prompt_segments[:-1]] + [prompt_segments[-1]]
    
        # Tokenize with the inference settings.
        encodings = tokenizer(
            segments_with_newlines,
            return_tensors="pt",
            padding=True
        )
        token_lengths = encodings["attention_mask"].sum(dim=1).tolist()  # Count non-padding tokens in each segment.
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
    
    """0.1. Set the result paths."""    
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
    
    """0.2. Load cached token partitions.""" 
    load_file_token_chunks = f"data_process_{model_name}_shot{hyperparams['num_shots']}_c{hyperparams['num_clients']}_{split_way}"
    file_dir_token_chunks = os.path.join(script_dir, "data_process_token_chunks")
    try:
        with open(f"{os.path.join(file_dir_token_chunks, load_file_token_chunks)}.json", "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
        loaded_chunks = None     
    
    """1. Load the model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(hyperparams['checkpoint_path'])
    model = AutoModelForCausalLM.from_pretrained(hyperparams['checkpoint_path']).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    """2. Initialize the model wrapper."""
    wrapper = LayerMaskedTransformerWrapper(model, device)
    
    """3. Build the few-shot prompt."""
    fewshot_prompt, prompt_token_nums = load_prompts(list(range(hyperparams['num_shots'])), tokenizer, loaded_chunks == None)  
    ratios_comp = [hyperparams.get('ratio_comp', 1.0) for _ in range(hyperparams['num_clients']-1)] + [hyperparams.get('ratio_last_comp', hyperparams.get('ratio_comp', 1.0))]
    ratios_comm = [hyperparams.get('ratio_comm', 1.0) for _ in range(hyperparams['num_clients']-1)] + [hyperparams.get('ratio_last_comm', hyperparams.get('ratio_comm', 1.0))] 
    
    """4. Run evaluation."""
    for batch_idx in range(0, len(test_data), batch_size):
        if batch_idx % 100 == 0:
            print(f"Processing {batch_idx}/{len(test_data)}")     

        """4.1. Build the batch prompts."""
        batch_docs = [dict(test_data[i]) for i in range(batch_idx, min(batch_idx + batch_size, len(test_data)))]
        batch_prompts = [doc_to_text(doc, fewshot_prompt) for doc in batch_docs]
        loaded_chunks_batch = [loaded_chunks[i] for i in range(batch_idx, min(batch_idx + batch_size, len(test_data)))] if loaded_chunks is not None else None            
        
        """4.2. Run inference for the batch."""
        batch_completions = generate_responses( 
            model = model, dtype = model.dtype, wrapper = wrapper, device = device, 
            tokenizer = tokenizer, tokenizer_padding_side = tokenizer.padding_side, 
            input_txt = batch_prompts, prompt_token_nums = prompt_token_nums, loaded_chunks_batch = loaded_chunks_batch, 
            hyperparams = hyperparams, split_way = split_way, comm_policy = comm_policy, ratios_comp = ratios_comp, ratios_comm = ratios_comm, 
            )               
        
        """4.3. Save participant responses and accuracy values."""
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
        writer.close() # Close all output writers.
        
    """5. Compute and save mean accuracy."""
    avg_acc = {k: sum(v) / len(v) if v else 0 for k, v in acc.items()}
    avg_values = list(avg_acc.values())
    avg_acc["avg"] = sum(avg_values) / len(avg_values)

    with open(os.path.join(results_dir, "accs", f"acc_{output_file}.json"), "w", encoding="utf-8") as f:
        json.dump(acc, f, indent=4, ensure_ascii=False)    
    with open(os.path.join(results_dir, "avg_acc", f"avg_acc_{output_file}.json"), "w", encoding="utf-8") as f:
        json.dump(avg_acc, f, indent=4, ensure_ascii=False)    
    with open(os.path.join(results_dir, "avg_acc", f"hyperparams_{timestamp}.json"), "w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=4, ensure_ascii=False) 
    
    # Release model/tokenizer references and cached GPU memory.
    del model, tokenizer, wrapper
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    torch.cuda.synchronize()

