# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 16:28:30 2025

@author: DengXiumei
"""

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


""" -------------------------Test Part -------------------------"""
num_local_forward_map = {
    "Qwen/Qwen2.5-0.5B": [2, 3, 4, 6, 8, 12, 18, 24],  # 24
    "Qwen/Qwen2.5-1.5B": [2, 3, 4, 8, 10, 14, 21, 28],  # 28
    "Qwen/Qwen2.5-3B": [2, 3, 4, 9, 12, 18, 27, 36],  # 36
    "Qwen/Qwen2.5-7B": [2, 3, 4, 8, 10, 14, 21, 28]  # 28
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
""" -------------------------Test Part -------------------------"""

if __name__ == "__main__":
    
    """ 0. 主进程 """
    for comm_policy in [
            # "main", 
            # "uni_extra", 
            # "uni_fr", "uni_bk", "inc_gp", "dec_gp", 
            # "sample_uni_comp", "sample_last_comp",
            "sample_uni_comm", 
            # "sample_last_comm",
            # "sample_uni_comp_comm", "sample_last_comp_comm",
            ]:

        
        """ 0.1 加载任务文件 """
        load_path, write_path = create_paths(comm_policy)
        
        combinations_file = comm_policy + "_combinations_test_0921.pkl"  
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
        
        """ -------------------------Test Part -------------------------"""
        # test_data = dataset["test"].select(range(5)) # 选择测试集的所有数据 
        
        # 根据不同模型大小设置合适的批处理大小  
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
        """ -------------------------Test Part -------------------------"""
        
        for i in remaining_combinations:
            hyperparams = combinations[i]
            
            # # 只运行满足条件的实验 for main
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

            # # 只运行满足条件的实验 for others
            # run_conditions = [
            #     hyperparams.get('checkpoint_path') in [
            #         'Qwen/Qwen2.5-0.5B',
            #         'Qwen/Qwen2.5-1.5B',
            #         'Qwen/Qwen2.5-3B',
            #         # 'Qwen/Qwen2.5-7B'
            #         ],
            #     ]
            
            # if not all(run_conditions):  # 如果不满足任何运行条件就跳过
            #     print(f"Skipping experiment {i+1}/{len(combinations)}")
            #     continue 
            # print(f"Experiment {i+1}/{len(combinations)} Start")
            
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
                print(f"Completed {i+1}/{len(combinations)}")
                completed.add(i)
    
                # 每次更新状态文件（只记录成功）
                status = {
                    'completed': sorted(list(completed)),
                    'total': len(combinations)
                }
                with open(status_file_path, 'w') as f:
                    json.dump(status, f, indent=2) 
                     
            except Exception as e:
                print(f"Error: {e}") # 捕获并打印任何异常 
    
        print("All experiments completed.")
