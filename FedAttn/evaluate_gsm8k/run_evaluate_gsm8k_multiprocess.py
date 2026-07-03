# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:28:30 2025

@author: DengXiumei
"""

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
    """用于比较组合是否相等：确保key顺序一致"""
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

    """ 0.1 超参数组合 """
    # 加载已有组合（如果存在）
    if os.path.exists(os.path.join(script_dir, "experiment_records", combinations_file)):
        with open(os.path.join(script_dir, "experiment_records", combinations_file), 'rb') as f:
            old_combinations = pickle.load(f)
        print(f"Loaded {len(old_combinations)} old combinations from {combinations_file}")
    else:
        old_combinations = []
        print("No previous combinations found, starting fresh.")        
         
    # 加载新组合    
    new_combinations = new_combos_generator(base_grid=hyperparams_grid, sample_grid=sample_hyperparams_grid, other_grid = other_hyperparams_grid.get(comm_policy, None), comm_policy = comm_policy, num_local_forward_map = num_local_forward_map)
    print(f"Generated {len(new_combinations)} new combinations.")
    
    # 组合去重：对旧组合只追加新组合
    old_sorted = [sort_dict(c) for c in old_combinations]
    appended = 0
    for combo in new_combinations:
        if sort_dict(combo) not in old_sorted:
            old_combinations.append(combo)
            appended += 1
    print(f"Appended {appended} new combinations.")
    
    # 保存更新后的组合列表
    with open(os.path.join(script_dir, "experiment_records", combinations_file), 'wb') as f:
        pickle.dump(old_combinations, f)
    print(f"Saved total {len(old_combinations)} combinations to {combinations_file}") 
    
    return old_combinations

def new_combos_generator(base_grid, sample_grid, other_grid, comm_policy, num_local_forward_map):
   
    """
    将 base_grid 和 sample_grid 按 do_sample 条件拼接。
    - 若 do_sample=True，笛卡尔积 sample_grid 并合并；
    - 若 do_sample=False，仅使用 base_grid 参数。
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
                all_combinations.append(base)  # 不添加采样参数（也可显式补上默认值）
        
        all_combinations_all_models = all_combinations_all_models + all_combinations
        
    # 过滤规则 1：当切分策略为 "smart" 时，要求 num_shots >= num_clients。若 num_shots < num_clients（比如样例数不够分到每个 client），则剔除该组合。
    # 过滤规则 2：当切分策略为 "smart_question_last" 时，要求 num_shots >= num_clients - 1。若 num_shots < num_clients - 1（需要为“最后一道题”预留），则剔除该组合。
    # 过滤规则 3：这些策略不允许 0-shot（需要 few-shot 提示）。若 split_way ∈ {"smart", "even_question_last", "smart_question_last"} 且 num_shots == 0，则剔除。
    # 过滤规则 4：这些策略不允许 combo["num_clients"] == 1 and combo["split_way"] in {"smart", "even_question_last", "smart_question_last"}
    # 过滤规则 5：这些策略不允许 combo["num_clients"] == 1 and combo["num_local_forward"] > 1
    # 过滤规则 6：若存在 last_num_local_forward，且 >= 当前 num_local_forward，则剔除
    # 过滤规则 7：若存在 last_num_local_forward，且 >= 当前 num_local_forward，则剔除
    
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
        
    # 这是子进程调用，直接运行单个实验: 判断是否为子进程调用（通过命令行参数传入超参数）
    if len(sys.argv) > 1 and sys.argv[1].startswith('{'):
        hyperparams = json.loads(sys.argv[1])  # 从命令行参数解析JSON格式的超参数
        comm_policy = sys.argv[2]
        write_path = sys.argv[3]
        load_path = sys.argv[4]
        
        """ 1. 设备设置 """ 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 检查是否有可用的GPU，优先使用CUDA设备
        
        """ 2. 加载数据集 """
        config = datasets.DownloadConfig(resume_download=True, max_retries=100) # 配置数据集下载参数，允许断点续传和多次重试 

        try:
            dataset = load_dataset("gsm8k", "main", download_config=config) # 加载GSM8K数学推理数据集  
        except Exception:
            dataset_path_maps = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gsm8k")]
            
            for dataset_path_map in dataset_path_maps:
                try:
                    dataset = load_from_disk(dataset_path_map) # 加载GSM8K数学推理数据集  
                    break
                except Exception:
                    continue         

        test_data = dataset["test"].select(range(len(dataset["test"]))) # 选择测试集的所有数据 
        
        # 根据不同模型大小设置合适的批处理大小  
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
    
        """ 3. 子进程：运行单个实验 """        
        try:
            main(
                hyperparams=hyperparams,           # 传入解析的超参数
                test_data=test_data,               # 测试数据
                device=device,                     # 计算设备
                split_way=hyperparams["split_way"],                 # 数据分割方式
                batch_size=min(model_batch_map[hyperparams['checkpoint_path']][hyperparams['num_shots']], len(test_data)),  # 根据模型选择批次大小
                comm_policy=comm_policy,
                script_dir=load_path,
                results_dir=os.path.join(write_path, "results", comm_policy) 
            )

        except Exception as e:
            print(f"Error: {e}") # 捕获并打印任何异常 
            sys.exit(1)
            
        finally:
            torch.cuda.empty_cache() # 实验结束后清理GPU内存，避免内存泄漏
            gc.collect()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            torch.cuda.synchronize()
        
        sys.exit(0) # 子进程完成任务后退出


    """ 0. 主进程 """
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
        
        """ 0.1 加载任务文件 """ 
        load_path, write_path = create_paths(comm_policy)
        
        combinations_file = comm_policy + "_combinations.pkl" 
        # 构造新超参数组合
        num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid = get_para_dict(comm_policy)
        combinations = task_generator(comm_policy, combinations_file, load_path, num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid)   

        """ 0.2 加载状态文件 """
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

        """ 0.3 遍历每个超参数组合，创建子进程运行实验 """ 
        remaining_combinations = [i for i in range(len(combinations)) if i not in completed]
        print(f"Remaining: {len(remaining_combinations)} experiments")
        
        for i in remaining_combinations:
            hyperparams = combinations[i]

            # 只运行满足条件的实验 for main
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

            # 只运行满足条件的实验 for others
            run_conditions = [
                hyperparams.get('checkpoint_path') in [
                    'Qwen/Qwen2.5-0.5B',
                    'Qwen/Qwen2.5-1.5B',
                    'Qwen/Qwen2.5-3B',
                    'Qwen/Qwen2.5-7B'
                    ],
                ]
            
            if not all(run_conditions):  # 如果不满足任何运行条件就跳过
                print(f"Skipping experiment {i+1}/{len(combinations)}")
                continue 
            print(f"Experiment {i+1}/{len(combinations)} Start")
            
            cmd = [sys.executable, __file__, json.dumps(hyperparams), comm_policy, write_path, load_path]
            try:
                result = subprocess.run(cmd, check=True, env=env)
                print(f"Completed {i+1}/{len(combinations)}")
                completed.add(i)
                
                # 每次更新状态文件（只记录成功）
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

        