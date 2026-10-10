# -*- coding: utf-8 -*-

import re
import jsonlines

from transformers import AutoTokenizer
import torch
import json
import datasets
from datasets import load_dataset
import sys
import os
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import product

import gc
from utils import token_chunk
from para_dict import get_para_dict

def load_prompts(id_list, tokenizer, prompt_prefix="gsm8k_prompt_", prompt_suffix=".txt"):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(current_dir, "prompt")
    
    prompt_segments = []

    for pid in id_list:
        filename = os.path.join(prompt_dir, f"{prompt_prefix}{pid}{prompt_suffix}")
        content = open(filename, "r", encoding="utf-8").read()
        prompt_segments.append(content)

    full_prompt = "\n".join(prompt_segments)
    segments_with_newlines = [seg + "\n" for seg in prompt_segments[:-1]] + [prompt_segments[-1]]

    encodings = tokenizer(
        segments_with_newlines,
        return_tensors="pt",
        padding=True
    )
    attention_mask = encodings["attention_mask"]
    token_lengths = attention_mask.sum(dim=1).tolist()

    return full_prompt, token_lengths

def doc_to_text(doc, fewshot_prompt):
    return (
            fewshot_prompt
            + "\nQuestion: "
            + doc["question"]
            + "\nLet's think step by step\n"
            )

def main(hyperparams, test_data, device, batch_size):
    print(f"Running experiment with hyperparams: {hyperparams}")
    model_name = os.path.basename(hyperparams['checkpoint_path'])
    output_file = f"data_process_{model_name}_shot{hyperparams['num_shots']}_c{hyperparams['num_clients']}_{hyperparams['split_way']}"    

    # Create the output directories.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "data_process"), exist_ok=True)

    results_dir_token_chunks = os.path.join(script_dir, "data_process", "data_process_token_chunks") 
    os.makedirs(results_dir_token_chunks, exist_ok=True)
    
    results_dir_token_split = os.path.join(script_dir, "data_process", "data_process_token_split") 
    os.makedirs(results_dir_token_split, exist_ok=True)
    
    token_split_file_path = os.path.join(results_dir_token_split, output_file)    
    token_split_file = jsonlines.Writer(open(f"{token_split_file_path}.jsonl", "w", encoding="utf-8"))
    
    # if os.path.exists(f"{os.path.join(results_dir_token_chunks, output_file)}.json") and os.path.exists(f"{token_split_file_path}.jsonl"):
    #     return
    
    """1. Initialize the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(hyperparams['checkpoint_path'])
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    fewshot_prompt, prompt_token_num = load_prompts(list(range(hyperparams['num_shots'])), tokenizer)
    
    """2. Partition the test prompts."""
    
    for batch_idx in range(0, len(test_data), batch_size):
        batch_docs = [dict(test_data[i]) for i in range(batch_idx, min(batch_idx + batch_size, len(test_data)))]    
        batch_prompts = [doc_to_text(doc, fewshot_prompt) for doc in batch_docs]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)    
        padding_mask = inputs["attention_mask"]
        seq_lens = padding_mask.sum(dim=1)
    
        token_chunks = token_chunk(seq_lens, hyperparams['num_clients'], hyperparams['split_way'], prompt_token_num)

        with open(f"{os.path.join(results_dir_token_chunks, output_file)}.json", "w", encoding="utf-8") as f:
            json.dump(token_chunks, f, ensure_ascii=False, indent=2)

        for idx, doc in enumerate(batch_docs):
            record = { 
                "question": doc["question"],
                "answer": doc["answer"],
                "token_chunks": token_chunks[idx],
                "token_chunks_lens": [len(sublist) for sublist in token_chunks[idx]] 
                }
            token_split_file.write(record)
        
    token_split_file.close()

    # Release model/tokenizer references and cached GPU memory.
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def sort_dict(d):
    """Return a dictionary with sorted keys for comparing configurations."""
    return {k: d[k] for k in sorted(d)}

def task_generator(base_grid):

    base_keys = list(base_grid.keys())
    base_combos = list(product(*[base_grid[k] for k in base_keys]))

    all_combinations = []
    for base_vals in base_combos:
        base = dict(zip(base_keys, base_vals))

        all_combinations.append(base)  # Omit sampling parameters.
         
    all_combinations = [
        combo for combo in all_combinations
        if not (
                (combo["split_way"] == "smart" and combo["num_shots"] < combo["num_clients"]) or
                (combo["split_way"] == "smart_question_last" and combo["num_shots"] < combo["num_clients"] - 1) or
                (combo["num_shots"] == 0 and combo["split_way"] in {"smart", "even_question_last", "smart_question_last"})or
                (combo["num_clients"] == 1 and combo["split_way"] in {"smart", "even_question_last", "smart_question_last"})
                )
        ] 
    
    return all_combinations   
    
def generate_token_chunks_and_splits_file():
    
    combinations_file = "data_process_combinations.pkl" 
    status_file = combinations_file.replace("combinations","experiment_status").replace(".pkl",".json")
    
    """0.1. Load existing combinations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "experiment_records"), exist_ok=True)
    
    combinations_file_path = os.path.join(script_dir, "experiment_records", combinations_file)
    if os.path.exists(combinations_file_path):
        with open(combinations_file_path, 'rb') as f:
            old_combinations = pickle.load(f)
        print(f"Loaded {len(old_combinations)} old combinations from {combinations_file}")
    else:
        old_combinations = []
        print("No previous combinations found, starting fresh.")       

    """0.2. Define the hyperparameter grid."""      
    hyperparams_grid = {
        'checkpoint_path': 
            [
                'Qwen/Qwen2.5-0.5B',
                'Qwen/Qwen2.5-1.5B',
                'Qwen/Qwen2.5-3B',
                'Qwen/Qwen2.5-7B'
                ],
        'num_clients': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 24, 32],            # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 24, 32]                      
        'num_shots': [1, 2, 3, 4, 5, 6, 7, 8],                 # [0, 1, 2, 3, 4, 5, 6, 7, 8]
        'split_way': ["even", "smart", "even_question_last", "smart_question_last"]
        } 
         
    """0.3. Generate new combinations."""   
    new_combinations = task_generator(base_grid=hyperparams_grid)
    print(f"Generated {len(new_combinations)} new combinations.")
    
    """0.4. Append combinations absent from the existing list.""" 
    old_sorted = [sort_dict(c) for c in old_combinations]
    appended = 0
    for combo in new_combinations:
        if sort_dict(combo) not in old_sorted:
            old_combinations.append(combo)
            appended += 1
    print(f"Appended {appended} new combinations.")
    
    """0.5. Save the updated combinations."""
    with open(combinations_file_path, 'wb') as f:
        pickle.dump(old_combinations, f)
    print(f"Saved total {len(old_combinations)} combinations to {combinations_file}")    

    """0.6. Load the experiment status."""
    status_file_path = os.path.join(script_dir, "experiment_records", status_file)
    if os.path.exists(status_file_path):
        with open(status_file_path, 'r') as f:
            status = json.load(f)
        completed = set(status.get('completed', []))
    else:
        completed = set()
        status = {'completed': [], 'total': len(old_combinations)}

        # Create the initial status file.
        with open(status_file_path, 'w') as f:
            json.dump(status, f, indent=2)      
    
    """Identify the remaining configurations.""" 
    remaining_combinations = [i for i in range(len(old_combinations)) if i not in completed]
    print(f"Remaining: {len(remaining_combinations)} experiments")
    
    """1. Select the compute device.""" 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    """2. Load the dataset."""
    config = datasets.DownloadConfig(resume_download=True, max_retries=100) # Enable resumable downloads and retries.   
    dataset = load_dataset("gsm8k", "main", download_config=config) # Load the GSM8K dataset.    
    test_data = dataset["test"].select(range(len(dataset["test"]))) # Use the complete test split. 
    
    """3. Run each remaining configuration."""
    for i in remaining_combinations:
        hyperparams = old_combinations[i]
        
        print(f"Experiment {i+1}/{len(old_combinations)} Start")

        try:
            main(
                hyperparams=hyperparams,           # Experiment hyperparameters.
                test_data=test_data,               # Test examples.
                device=device,                     # Compute device.
                batch_size=len(dataset["test"])  # Process the complete selected split in one batch.
            )  
            print(f"Completed {i+1}/{len(old_combinations)}")
            completed.add(i)

            # Record successfully completed experiments.
            status = {
                'completed': sorted(list(completed)),
                'total': len(old_combinations)
            }
            with open(status_file_path, 'w') as f:
                json.dump(status, f, indent=2)   
                
        except Exception as e:
            # Report the exception.       
            print(f"Error: {e}")

    print("All token_chunks and token_split completed.")


def generate_res_lens_file(folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "main")):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    # Create the output directories.
    results_dir = os.path.join(folder_path, "data_process_responses")
    os.makedirs(results_dir, exist_ok=True)


    for model in ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/" + model)
        for shot_item in ["shot0", "shot1", "shot2", "shot3", "shot4", "shot5", "shot6", "shot7", "shot8"]:
                
            records = {}
            
            for file_name in os.listdir(folder_path):
                if ("client" in file_name) and (model in file_name) and (shot_item in file_name) and file_name.endswith(".jsonl"):
                    client_id = int(re.search(r"client(\d+)\.jsonl$", file_name).group(1))
                    output_file_path = os.path.join(results_dir, "data_process_" + file_name.replace("_client" + str(client_id) + ".jsonl", ''))            
        
                    if os.path.exists(output_file_path + '.json'):
                        continue
        
                    else:
                        records.setdefault(output_file_path, {}).setdefault(client_id, [])
            
                        with jsonlines.open(os.path.join(folder_path, file_name), "r") as reader:
                            for row in reader:
                                # Skip null records.
                                if row is None:
                                    continue
                                # Skip records without a completion.
                                completion = row.get("completion")
                                if completion is None:
                                    continue  
                                records[output_file_path][client_id].append(completion)
            
                        # Tokenize with the inference settings.
                        if records[output_file_path][client_id]:
                            encodings = tokenizer(
                                records[output_file_path][client_id],
                                return_tensors="pt",
                                padding=True
                            ).to(device)
                            records[output_file_path][client_id] = encodings["attention_mask"].sum(dim=1).tolist()
        
                        
            # Save the output files.
            for file_path, data in records.items():
                with open(file_path + '.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"保存文件: {file_path}")  

        # Release model/tokenizer references and cached GPU memory.
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()                         

if __name__ == "__main__":
    
    comm_policy = "main"
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    
    # generate_token_chunks_and_splits_file()
    generate_res_lens_file(os.path.join(folder_path,comm_policy))
    
    
    
    
