# -*- coding: utf-8 -*-


import sys
import os
gpu_id = 0
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_evaluate_gsm8k_multiprocess import create_paths, task_generator

import copy
import json
import gc
from itertools import product
import torch
import datasets
from datasets import load_dataset
from datasets import load_from_disk

import subprocess
import pickle

from main_evaluate_gsm8k import main
from para_dict import get_para_dict


"""Example configuration."""
num_local_forward_map = {
    "Qwen/Qwen2.5-0.5B": [2, 3, 4, 6, 8, 12, 18, 24],  # 24 Transformer blocks.
    "Qwen/Qwen2.5-1.5B": [2, 3, 4, 8, 10, 14, 21, 28],  # 28 Transformer blocks.
    "Qwen/Qwen2.5-3B": [2, 3, 4, 9, 12, 18, 27, 36],  # 36 Transformer blocks.
    "Qwen/Qwen2.5-7B": [2, 3, 4, 8, 10, 14, 21, 28]  # 28 Transformer blocks.
    }      
sample_hyperparams_grid = {
    'temperature': [0.2],
    'top_k': [20],
    'top_p': [0.9]
    }
hyperparams_grid = {
    'checkpoint_path': 
        [
            'Qwen/Qwen2.5-0.5B',
            # 'Qwen/Qwen2.5-1.5B',
            # 'Qwen/Qwen2.5-3B',
            # 'Qwen/Qwen2.5-7B'
            ],
    'num_clients': [4],            # [2, 4, 8, 16, 32]  [2, 4, 6, 8, 10, 16, 24, 32]             
    'num_local_forward': [5],
    'do_sample': [False],             # True, False
    'max_new_tokens': [2],                  # 2, 3, 4, 8
    'num_shots': [4],                 # 4, 8
    'split_way': ["even_question_last"] # ["even", "smart", "even_question_last", "smart_question_last"] 
    } 
      
other_hyperparams_grid = {
    "uni_extra": {'last_num_local_forward': [2]},
    "sample_uni_comp": {'ratio_comp': [0.9]}, # [0.9, 0.7, 0.5, 0.3, 0.1]
    "sample_uni_comm": {'ratio_comm': [0.9]}, 
    "sample_uni_comp_comm": {'ratio_comp': [0.9],'ratio_comm': [0.9]}, 
    "sample_last_comp": {'ratio_comp': [0.9], 'ratio_last_comp': [0.9]}, 
    "sample_last_comm": {'ratio_comm': [0.9,], 'ratio_last_comm': [0.9]}, 
    "sample_last_comp_comm": {'ratio_comp': [0.9],'ratio_comm': [0.9], 'ratio_last_comp': [0.9],'ratio_last_comm': [0.9]}, 
    }  
"""Example configuration."""

if __name__ == "__main__":
    
    """0. Configure the experiments."""
    for comm_policy in [
            "main", 
            # "uni_extra", 
            # "uni_fr", "uni_bk", "inc_gp", "dec_gp", 
            # "sample_uni_comp", "sample_last_comp",
            # "sample_uni_comm", 
            # "sample_last_comm",
            # "sample_uni_comp_comm", "sample_last_comp_comm",
            ]:

        
        """0.1. Set the experiment paths."""
        load_path, write_path = create_paths(comm_policy)

        """Demo configuration."""
        write_path = os.path.join(load_path, "demo_runs", "inference_evaluation")
        for subfolder in ("", "accs", "avg_acc"):
            os.makedirs(os.path.join(write_path, "results", comm_policy, subfolder), exist_ok=True) # Demo output directories.
        
        combinations_file = comm_policy + "_combinations_demo2.pkl"  
        # Build the hyperparameter grid.
        num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid = get_para_dict(comm_policy)

        """Demo configuration."""
        hyperparams_grid.update({
            "checkpoint_path": ["Qwen/Qwen2.5-0.5B"],
            "num_clients": [4],
            "num_local_forward": [2],
            "do_sample": [False],
            "max_new_tokens": [256],
            "num_shots": [4],
            "split_way": ["even"],
        }) # Demo: Qwen2.5-0.5B, four participants, H=2, greedy decoding, up to 256 new tokens, four shots, and even segmentation.
        
        combinations = task_generator(comm_policy, combinations_file, load_path, num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid)      
    
        """0.2. Load the experiment status."""
        status_file = combinations_file.replace("combinations","experiment_status").replace(".pkl",".json")
        status_file_path = os.path.join(load_path, "experiment_records", status_file)
        if os.path.exists(status_file_path):
            with open(status_file_path, 'r') as f:
                status = json.load(f)
            completed = set(status.get('completed', []))
        else:
            completed = set()
            status = {'completed': [], 'total': len(combinations)}

            with open(status_file_path, 'w') as f:
                json.dump(status, f, indent=2)       
        
        """0.3. Identify the remaining experiment configurations.""" 
        remaining_combinations = [i for i in range(len(combinations)) if i not in completed]
        print(f"Remaining: {len(remaining_combinations)} experiments")
        
        """1. Select the compute device.""" 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use CUDA when available.
        
        """2. Load the dataset."""
        config = datasets.DownloadConfig(resume_download=True, max_retries=100) # Enable resumable downloads and retries. 

        try:
            dataset = load_dataset("gsm8k", "main", download_config=config) # Load the GSM8K dataset.  
        except Exception:
            dataset_path_maps = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k")]
            
            for dataset_path_map in dataset_path_maps:
                try:
                    dataset = load_from_disk(dataset_path_map) # Load the GSM8K dataset.  
                    break
                except Exception:
                    continue                       
    
        # test_data = dataset["test"].select(range(len(dataset["test"]))) # Use the complete test split. 
        
        """Demo configuration."""
        test_data = dataset["test"].select(range(2)) ## Use the first two test examples for the demo.
        
        # Select batch sizes by model and number of shots.  
        model_batch_map = {
            "Qwen/Qwen2.5-0.5B": {
                0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
            "Qwen/Qwen2.5-1.5B": {
                0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
            "Qwen/Qwen2.5-3B": {
                0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
            "Qwen/Qwen2.5-7B": {
                0: 2, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2},
            }
        """Example configuration."""
        
        for i in remaining_combinations:
            hyperparams = combinations[i]
            
            # # Example selection criteria for the main experiments.
            # run_conditions = [
            #     hyperparams.get('checkpoint_path') in [
            #         # 'Qwen/Qwen2.5-0.5B',
            #         'Qwen/Qwen2.5-1.5B',
            #         # 'Qwen/Qwen2.5-3B',
            #         # 'Qwen/Qwen2.5-7B'
            #         ],
            #     (hyperparams.get("num_shots") in [4] and hyperparams.get("num_clients") in [4]) or (hyperparams.get("num_shots") in [8] and hyperparams.get("num_local_forward") in [4] and hyperparams.get("num_clients") <= hyperparams.get("num_shots")),            
            #     hyperparams.get("do_sample") is False,
            #     ]

            # # Selection criteria for the other communication policies.
            # run_conditions = [
            #     hyperparams.get('checkpoint_path') in [
            #         'Qwen/Qwen2.5-0.5B',
            #         'Qwen/Qwen2.5-1.5B',
            #         'Qwen/Qwen2.5-3B',
            #         # 'Qwen/Qwen2.5-7B'
            #         ],
            #     ]
            
            # if not all(run_conditions):  # Skip configurations that fail any selection criterion.
            #     print(f"Skipping experiment {i+1}/{len(combinations)}")
            #     continue 
            # print(f"Experiment {i+1}/{len(combinations)} Start")
            
            """3. Run one experiment."""  
            try:
                main(
                    hyperparams=hyperparams,           # Experiment hyperparameters.
                    test_data=test_data,               # Test examples.
                    device=device,                     # Compute device.
                    split_way=hyperparams["split_way"],                 # Input segmentation setting.
                    batch_size=min(model_batch_map[hyperparams['checkpoint_path']][hyperparams['num_shots']], len(test_data)),  # Select the batch size for the model.
                    comm_policy=comm_policy,
                    script_dir=load_path,
                    results_dir=os.path.join(write_path, "results", comm_policy) 
                )
                print(f"Completed {i+1}/{len(combinations)}")
                completed.add(i)
    
                # Record successfully completed experiments.
                status = {
                    'completed': sorted(list(completed)),
                    'total': len(combinations)
                }
                with open(status_file_path, 'w') as f:
                    json.dump(status, f, indent=2) 
                     
            except Exception as e:
                print(f"Error: {e}") # Report the exception. 
    
        print("All experiments completed.")
