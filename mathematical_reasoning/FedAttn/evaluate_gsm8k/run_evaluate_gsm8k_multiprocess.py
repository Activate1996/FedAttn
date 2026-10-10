# -*- coding: utf-8 -*-


import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gpu_id = 0
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def sort_dict(d):
    """Return a dictionary with sorted keys for comparing configurations."""
    return {k: d[k] for k in sorted(d)}

def create_paths(comm_policy):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir, "experiment_records"), exist_ok=True)
    
    results_dir = script_dir    
        
    os.makedirs(os.path.join(results_dir, "results"), exist_ok=True)
    
    os.makedirs(os.path.join(results_dir, "results", comm_policy), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "results", comm_policy, "accs"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "results", comm_policy, "avg_acc"), exist_ok=True)
    
    return script_dir, results_dir

def task_generator(comm_policy, combinations_file, script_dir, num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid):

    """Load and extend the saved hyperparameter combinations."""
    # Load the existing combinations, if available.
    if os.path.exists(os.path.join(script_dir, "experiment_records", combinations_file)):
        with open(os.path.join(script_dir, "experiment_records", combinations_file), 'rb') as f:
            old_combinations = pickle.load(f)
        print(f"Loaded {len(old_combinations)} old combinations from {combinations_file}")
    else:
        old_combinations = []
        print("No previous combinations found, starting fresh.")        
         
    # Generate the new combinations.    
    new_combinations = new_combos_generator(base_grid=hyperparams_grid, sample_grid=sample_hyperparams_grid, other_grid = other_hyperparams_grid.get(comm_policy, None), comm_policy = comm_policy, num_local_forward_map = num_local_forward_map)
    print(f"Generated {len(new_combinations)} new combinations.")
    
    # Append combinations that are absent from the existing list.
    old_sorted = [sort_dict(c) for c in old_combinations]
    appended = 0
    for combo in new_combinations:
        if sort_dict(combo) not in old_sorted:
            old_combinations.append(combo)
            appended += 1
    print(f"Appended {appended} new combinations.")
    
    # Save the updated list of combinations.
    with open(os.path.join(script_dir, "experiment_records", combinations_file), 'wb') as f:
        pickle.dump(old_combinations, f)
    print(f"Saved total {len(old_combinations)} combinations to {combinations_file}") 
    
    return old_combinations

def new_combos_generator(base_grid, sample_grid, other_grid, comm_policy, num_local_forward_map):
   
    """Expand the hyperparameter grids according to do_sample.

    For sampled decoding, combine each base configuration with every sampling
    configuration. For greedy decoding, use only the base parameters.
    """
    
    all_combinations_all_models = []
    base_grids = copy.deepcopy(base_grid)
    
    for model in base_grids['checkpoint_path']:
        
        base_grid['checkpoint_path'] = [model]
        if not base_grids['num_local_forward']:
            base_grid['num_local_forward'] = num_local_forward_map[model]          
        
        base_keys = list(base_grid.keys())
        base_vals = [base_grid[k] for k in base_keys]
    
        if other_grid is not None:
            other_keys = list(other_grid.keys())
            base_keys.extend(other_keys)
            base_vals.extend([other_grid[k] for k in other_keys])
    
        base_combos = list(product(*base_vals))
        
        sample_keys = list(sample_grid.keys())
        sample_combos = list(product(*[sample_grid[k] for k in sample_keys]))
    
        all_combinations = []
        for base_vals in base_combos:
            base = dict(zip(base_keys, base_vals))
    
            if base["do_sample"]:
                for sample_vals in sample_combos:
                    sample = dict(zip(sample_keys, sample_vals))
                    combined = {**base, **sample}
                    all_combinations.append(combined)
            else:
                all_combinations.append(base)  # Omit sampling parameters.
        
        all_combinations_all_models = all_combinations_all_models + all_combinations
        
    # For smart segmentation, require at least one shot per participant.
    # For smart_question_last, require at least num_clients - 1 shots.
    # Exclude zero-shot configurations for smart, even_question_last, and smart_question_last.
    # Exclude single-participant configurations for those three segmentation settings.
    # For a single participant, exclude synchronization intervals greater than one.
    # When specified, last_num_local_forward must be smaller than num_local_forward.
    # Apply the remaining policy-specific restrictions below.
    
    all_combinations_all_models = [
        combo for combo in all_combinations_all_models
        if not (
                (combo["split_way"] == "smart" and combo["num_shots"] < combo["num_clients"]) or
                (combo["split_way"] == "smart_question_last" and combo["num_shots"] < combo["num_clients"] - 1) or
                (combo["num_shots"] == 0 and combo["split_way"] in {"smart", "even_question_last", "smart_question_last"}) or
                (combo["num_clients"] == 1 and combo["split_way"] in {"smart", "even_question_last", "smart_question_last"}) or
                (combo["num_clients"] == 1 and combo["num_local_forward"] > 1) or 
                (combo["num_clients"] > 1 and combo["num_local_forward"] == 1 and "sample" not in comm_policy) or 
                (combo["num_clients"] == 1 and (combo.get('ratio_comm') is not None or combo.get('ratio_last_comp') is not None or combo.get('ratio_last_comm') is not None)) or
                (combo["num_clients"] == 1 and combo.get("last_num_local_forward") is not None) or
                (combo.get("last_num_local_forward") is not None and combo["last_num_local_forward"] >= combo["num_local_forward"]) or
                (combo["num_local_forward"] == 1 and comm_policy in ["uni_extra", "uni_fr", "uni_bk", "inc_gp", "dec_gp"])
                )
        ]

    return all_combinations_all_models



if __name__ == "__main__":
        
    # Run one experiment in the subprocess using command-line hyperparameters.
    if len(sys.argv) > 1 and sys.argv[1].startswith('{'):
        hyperparams = json.loads(sys.argv[1])  # Parse the JSON hyperparameters from the command line.
        comm_policy = sys.argv[2]
        write_path = sys.argv[3]
        load_path = sys.argv[4]
        
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

        test_data = dataset["test"].select(range(len(dataset["test"]))) # Use the complete test split. 
        
        # Select batch sizes by model and number of shots.  
        model_batch_map = {
            "Qwen/Qwen2.5-0.5B": {
                0: 64, 1: 64, 2: 48, 3: 32, 4: 32, 5: 24, 6: 20, 7: 16, 8: 16},
            "Qwen/Qwen2.5-1.5B": {
                0: 32, 1: 64, 2: 32, 3: 18, 4: 30, 5: 12, 6: 10, 7: 8, 8: 14},
            "Qwen/Qwen2.5-3B": {
                0: 24, 1: 24, 2: 12, 3: 8, 4: 6, 5: 4, 6: 4, 7: 4, 8: 8},
            "Qwen/Qwen2.5-7B": {
                0: 6, 1: 6, 2: 46, 3: 4, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1},
            }
    
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

        except Exception as e:
            print(f"Error: {e}") # Report the exception. 
            sys.exit(1)
            
        finally:
            torch.cuda.empty_cache() # Release GPU memory after the experiment.
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            torch.cuda.synchronize()
        
        sys.exit(0) # Exit after the subprocess finishes.


    """0. Configure the experiments."""
    for comm_policy in ["uni_fr", "uni_bk", "inc_gp", "dec_gp"]:
               
        # comm_policy = "uni_fr" 
        # [
        #  "main", 
        #  "uni_extra", 
        #  "uni_fr", "uni_bk", "inc_gp", "dec_gp", 
        #  "sample_uni_comp", "sample_last_comp",
        #  "sample_uni_comm", "sample_last_comm",
        #  #  "sample_uni_comp_comm", "sample_last_comp_comm"
        #  ]
        
        """0.1. Set the experiment paths.""" 
        load_path, write_path = create_paths(comm_policy)
        
        combinations_file = comm_policy + "_combinations.pkl" 
        # Build the hyperparameter grid.
        num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid = get_para_dict(comm_policy)
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
        
        for i in remaining_combinations:
            hyperparams = combinations[i]

            # Candidate selection criteria for the main experiments.
            run_conditions = [
                hyperparams.get('checkpoint_path') in [
                    'Qwen/Qwen2.5-0.5B',
                    'Qwen/Qwen2.5-1.5B',
                    'Qwen/Qwen2.5-3B',
                    # 'Qwen/Qwen2.5-7B'
                    ],
                (
                    (
                        hyperparams.get("num_shots") in [4] and hyperparams.get("num_clients") in [4]
                        ) or 
                    (
                        hyperparams.get('checkpoint_path') in ['Qwen/Qwen2.5-0.5B','Qwen/Qwen2.5-1.5B','Qwen/Qwen2.5-7B'] and 
                        hyperparams.get("num_shots") in [8] and 
                        hyperparams.get("num_local_forward") in [4] and 
                        hyperparams.get("num_clients") <= hyperparams.get("num_shots")
                        ) or     
                    (
                        hyperparams.get('checkpoint_path') in ['Qwen/Qwen2.5-3B'] and 
                        hyperparams.get("num_shots") in [8] and 
                        hyperparams.get("num_local_forward") in [9] and 
                        hyperparams.get("num_clients") <= hyperparams.get("num_shots")
                        ) or 
                    (
                        hyperparams.get("num_clients") in [1])
                    ),
                hyperparams.get("do_sample") is False,
                ]

            # Selection criteria for the other communication policies.
            run_conditions = [
                hyperparams.get('checkpoint_path') in [
                    'Qwen/Qwen2.5-0.5B',
                    'Qwen/Qwen2.5-1.5B',
                    'Qwen/Qwen2.5-3B',
                    'Qwen/Qwen2.5-7B'
                    ],
                ]
            
            if not all(run_conditions):  # Skip configurations that fail any selection criterion.
                print(f"Skipping experiment {i+1}/{len(combinations)}")
                continue 
            print(f"Experiment {i+1}/{len(combinations)} Start")
            
            cmd = [sys.executable, __file__, json.dumps(hyperparams), comm_policy, write_path, load_path]
            try:
                result = subprocess.run(cmd, check=True, env=env)
                print(f"Completed {i+1}/{len(combinations)}")
                completed.add(i)
                
                # Record successfully completed experiments.
                status = {
                    'completed': sorted(list(completed)),
                    'total': len(combinations)
                }
                with open(status_file_path, 'w') as f:
                    json.dump(status, f, indent=2)     
                    
            except subprocess.CalledProcessError:
                print(f"Failed {i+1}/{len(combinations)}")
                continue
            
        print("All experiments finished.")

        