# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:57:02 2025

@author: DengXiumei
"""

from mpl_toolkits.axes_grid1 import Divider, Size
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import json
import os
import re
import jsonlines
import numpy as np
import numbers
import copy

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
    "figure.dpi": 150
})

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess_prompt_and_res import generate_token_chunks_and_splits_file, generate_res_lens_file
from figs_preview import figs_preview
from preprocess_data import get_exp_stats, get_info, get_perf_stat

from utils_get_performance_stats import read_shape, get_model_weight_byte
from utils_plot_main_fig_comms import plot_main_fig_comm, plot_main_fig_comm_bottom_legends
from utils_plot_main_fig_comps import plot_main_fig_comp
from utils_plot_main_fig_mems import plot_main_fig_mem
from utils_plot_main_fig_comps_mem import plot_main_fig_comp_mem

def load_main_data(data, main_keys, data_path, script_dir, comm_policy):
    
    # 筛选数据
    main_keys = {key: value for key, value in main_keys.items() if value is not None}
    print(main_keys)
     
    main_data = []
    for record in data:
        match = True
        for k, v in main_keys.items():
            if record.get(k) != v:
                match = False
                break
        if match:
            main_data.append(get_info(record = record, 
                                      script_dir = script_dir, 
                                      folder_path = os.path.join(data_path, record["comm_policy"])))
            # 补充来自main的数据
            if comm_policy != record["comm_policy"]:
                if comm_policy == "uni_extra": 
                    main_data[-1]["num_local_forwards_last"] = main_data[-1]["num_local_forwards"] 
                elif comm_policy == "sample_last_comp_comm":  
                    main_data[-1].update({'ratio_comp': 1.0, 'ratio_last_comp': 1.0, 'ratio_comm': 1.0, 'ratio_last_comm': 1.0})
                elif comm_policy == "sample_uni_comp_comm":
                    main_data[-1].update({'ratio_comp': 1.0, 'ratio_comm': 1.0})    
                elif comm_policy == "sample_uni_comp":
                    main_data[-1].update({'ratio_comp': 1.0})
                elif comm_policy == "sample_uni_comm":
                    main_data[-1].update({'ratio_comm': 1.0})               
                elif comm_policy == "sample_last_comp": 
                    main_data[-1].update({'ratio_comp': 1.0, 'ratio_last_comp': 1.0})
                elif comm_policy == "sample_last_comm":
                    main_data[-1].update({'ratio_comm': 1.0, 'ratio_last_comm': 1.0})
                    
        else:
            # 补充无损的数据
            if comm_policy == "main":
                if record['num_clients'] == 1 and record.get('num_local_forwards') == 1 and record['num_shots'] ==  main_keys.get('num_shots', None) and record['model'] ==  main_keys['model'] and record['if_do_sample'] ==  main_keys['if_do_sample'] and record['max_new_tokens'] ==  main_keys['max_new_tokens']: 
                    main_data.append(get_info(record = record, 
                                              script_dir = script_dir, 
                                              folder_path = os.path.join(data_path, record["comm_policy"])))
            if comm_policy in ["sample_last_comp", "sample_last_comm"]:
                main_keys_supp = {key: value for key, value in main_keys.items() 
                                  if value is not None and key not in ['ratio_comp', 'ratio_comm', 'ratio_last_comp', 'ratio_last_comm']}
                match_supp = True
                for k, v in main_keys_supp.items():
                    if record.get(k) != v:
                        match_supp = False
                        break
                if match_supp:
                    if record["comm_policy"] == "main":
                        main_data.append(get_info(record = record, 
                                                  script_dir = script_dir, 
                                                  folder_path = os.path.join(data_path, record["comm_policy"])))  
                        if comm_policy == "sample_last_comp":  
                            main_data[-1].update({'ratio_comp': 1.0, 'ratio_last_comp': 1.0})
                        elif comm_policy == "sample_last_comm":
                            main_data[-1].update({'ratio_comm': 1.0, 'ratio_last_comm': 1.0})                    
        
    # 计算cost数据：通信和计算开销 
    model_shape = read_shape(repo_id = "Qwen/" + main_keys["model"])
    model_w_byte = get_model_weight_byte(model_shape)
    
    main_data = get_perf_stat(data = main_data, 
                              model = main_keys['model'], 
                              model_shape = model_shape, 
                              model_w_byte = model_w_byte, 
                              data_path = data_path, 
                              script_dir = script_dir,
                              comm_policy = comm_policy)
         
    return main_data 
        

def extract_and_average(data_list, target_field, client_field):
    """
    提取数字key对应字典中的某个字段并计算
    """
    all_values = []

    for data_dict in data_list:
        # 遍历每个字典中的数字key（排除 'total_num_input_tokens'）
        for key, value in data_dict.items():
            if key in client_field or str(key) in client_field:  # 只处理数字key
                # 按字段路径提取值
                current_value = value
                for field in target_field.split('.'):
                    current_value = current_value.get(field, {})

                if isinstance(current_value, (int, float)):
                    all_values.append(current_value)
    return sum(all_values) / len(all_values)
    # try:
    #     return sum(all_values) / len(all_values) 
    # except Exception as e:
    #     print(f"extract_and_average() 时出错: {e}")


def generate_xy_axis(main_data, x_axis, model_w_byte, comm_policy):
    
    keys = ['All Participants', 
            r'Participant $N$', 
            r'Participants $1, \ldots, N-1$'] 

    """ 0.3 预设置 mem_peak 的计算方式 """ 
    load_model_para_prefill = [
        # "mem_peak_load_all", # 一次性加载全部model，算完一整个inference才释放        
        # "mem_peak_load_per_layer", # 一层一层加载layer，算完这层layer就释放该层layer
        "mem_peak_load_only_trans", # 一层一层加载layer，算完这层layer就释放该层layer，但是仅计算transformer block的mem_peak（vob词表占内存太大了，x轴又对embedding和projectlayer没影响）
        ][0] 
    load_model_para_decode = [
        # "final_step_mem_peak_load_all", # （decoding仅算最后一步和最后一个transformer 因为此时占的内存最大）一次性加载全部model，算完一整个inference才释放        
        "final_step_mem_peak_load_per_layer", # 一层一层加载layer，算完这层layer就释放该层layer
        # "final_step_mem_peak_only_trans", # 一层一层加载layer，算完这层layer就释放该层layer
        ][0]     


    x = []
    y_acc = {
        key: [] for key in keys
        }
     
    y_cost = { 
        key: {
            'Prefilling phase': {
                "FLOPs": [], 
                'Key-Value matrices': [], 
                "Memory usage": [] 
                }, 
            'Decoding phase': { 
                "FLOPs": [], 
                "Memory usage": [] 
                }
            } for key in keys}
    
    for record in main_data:
        
        record_conditions = []

        if comm_policy == "main":
            record_conditions.extend([
                x_axis == "num_clients" and record.get("num_clients") > 8,
                ])  


        if any(record_conditions): 
            continue
        else: 
            x.append(record[x_axis])

        client_field = {
            'All Participants': [str(i) for i in range(record["num_clients"])], 
            r'Participant $N$': [str(record["num_clients"]-1)], 
            r'Participants $1, \ldots, N-1$': [str(i) for i in range(max(record["num_clients"]-1,1))]
            }

        y_acc['All Participants'].append(record["data"].get('avg'))
        y_acc[r'Participant $N$'].append(record["data"].get(client_field[r'Participant $N$'][0]))
        values = [record["data"][k] for k in client_field[r'Participants $1, \ldots, N-1$']]        
        y_acc[r'Participants $1, \ldots, N-1$'].append(max(values))

        for key in keys:
    
            y_cost[key]['Prefilling phase']["FLOPs"].append(extract_and_average(record["main"],'prefill.totals.flops',client_field[key]))
            y_cost[key]['Prefilling phase']["Memory usage"].append(extract_and_average(record["main"],'prefill.totals.'+load_model_para_prefill,client_field[key]))
            y_cost[key]['Prefilling phase']['Key-Value matrices'].append(extract_and_average(record["main"],'prefill.totals.kv_comm',client_field[key]))
            
            y_cost[key]['Decoding phase']["FLOPs"].append(extract_and_average(record["main"],'decode.flops_total',client_field[key]))
            y_cost[key]['Decoding phase']["Memory usage"].append(extract_and_average(record["main"],'decode.'+load_model_para_decode,client_field[key]))
       
    x = np.array(x)
    sort_indices = np.argsort(x)
    x = x[sort_indices]
    
    for key in keys:    
        y_acc[key] = np.array(y_acc[key])[sort_indices]
        y_cost[key]['Prefilling phase']["FLOPs"] = np.array(y_cost[key]['Prefilling phase']["FLOPs"])[sort_indices]
        y_cost[key]['Prefilling phase']["Memory usage"] = np.array(y_cost[key]['Prefilling phase']["Memory usage"])[sort_indices]
        y_cost[key]['Prefilling phase']['Key-Value matrices'] = np.array(y_cost[key]['Prefilling phase']['Key-Value matrices'])[sort_indices]
        y_cost[key]['Decoding phase']["FLOPs"] = np.array(y_cost[key]['Decoding phase']["FLOPs"])[sort_indices]
        y_cost[key]['Decoding phase']["Memory usage"] = np.array(y_cost[key]['Decoding phase']["Memory usage"])[sort_indices]



    """ 4.1 处理 y_cost"""
    if comm_policy in ["main", "sample_uni_comp", "sample_uni_comm", "sample_uni_comp_comm"]:
        y_cost = y_cost['All Participants']
    elif comm_policy in ["uni_extra", "sample_last_comp", "sample_last_comm"]:
        y_cost = y_cost[r'Participant $N$']
        
    y_cost["Decoding phase"]["Memory usage"] = y_cost["Decoding phase"]["Memory usage"] - model_w_byte["trans_layer"]
    y_cost["Prefilling phase"]["Memory usage"] = y_cost["Prefilling phase"]["Memory usage"] - model_w_byte["trans_layer"] 


    return x, y_acc, y_cost  
                          
        
def plot_main(
        model, main_keys, comm_policy, x_axis, data, data_path, script_dir, model_w_byte, utils_plot_main_figs, xlabel, 
        grid = True, save_path = None):  
    
    accs_map = {
        'All Participants': "Average", 
        r'Participant $N$': "Task publisher", 
        r'Participants $1, \ldots, N-1$': "Best collaborator",
        }
    
    metrics_map={ 
        'Prefilling phase': 'Prefill', 
        'Decoding phase': 'Decode'
        }    

    for plot_main_fig_prefix, plot_main_fig in utils_plot_main_figs[x_axis].items(): 
                  
        if grid:                
            """ 创建2x2子图，总大小为单个图大小的2倍 """ 
            if is_legends_bottom:            
                if x_axis == "num_clients":
            
                    fig_len, fig_wid = 3.7, 2.5
                    figsize = (fig_len * 4*1 + 0.4, fig_wid * 1*1.2)
                    fig, axes = plt.subplots(1, 4, figsize=figsize)
                elif x_axis == "num_local_forwards":
            
                    fig_len, fig_wid = 3.7, 2.5
                    figsize = (fig_len * 4*1 + 0.4, fig_wid * 1*1.2)
                    fig, axes = plt.subplots(1, 4, figsize=figsize)                    
            else:
                
                fig_len, fig_wid = 3.7, 2.5
                figsize = (fig_len * 4, fig_wid * 1)
                fig, axes = plt.subplots(1, 4, figsize=figsize)
                
            # 手动加大子图间距
            # plt.subplots_adjust(wspace=0.6, hspace=0.6)   # 默认大概是 0.2，改成 0.6 就明显分开了
            axes = axes.flatten()
            
        else:
            fig_len, fig_wid = 4.3, 3.3
            figsize = (fig_len, fig_wid)
        
        for idx, split_way in enumerate(
                [
                    "even", 
                    "even_question_last", 
                    "smart", 
                    "smart_question_last"
                    ]
                ):
            
            main_keys[comm_policy][x_axis]['model'] = model
            main_keys[comm_policy][x_axis]['split_way'] = split_way
            if 'num_local_forwards' in main_keys[comm_policy][x_axis].keys():
                main_keys[comm_policy][x_axis]['num_local_forwards'] = 9 if model == "Qwen2.5-3B" else 4
            
            main_data = load_main_data(data, main_keys[comm_policy][x_axis], data_path, script_dir, comm_policy)
            x, y_acc, y_cost = generate_xy_axis(
                main_data = main_data, 
                x_axis = x_axis,
                model_w_byte = model_w_byte, 
                comm_policy = comm_policy
                )
    
            """ 4. plot"""                 
            plot_main_fig(
                x=x, y_acc=y_acc, y_cost=y_cost, xlabel=xlabel[x_axis], main_keys=main_keys[comm_policy][x_axis], 
                save_path=save_path, 
                accs_map=accs_map,
                metrics_map=metrics_map,
                figsize=figsize,
                ax=axes[idx] if grid else None
                )    
        
        if grid:         
            # 整体调整
            if is_legends_bottom:
                if x_axis == "num_clients":
                    plt.subplots_adjust(
                    top=0.735,
                    bottom=0.385,
                    left=0.135,
                    right=0.89,
                    hspace=0.6,
                    wspace=0.57)
                    fig.text(0.09, 0.56, model, rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
                    
                elif x_axis == "num_local_forwards":
                    plt.subplots_adjust(
                    top=0.735,
                    bottom=0.385,
                    left=0.135,
                    right=0.89,
                    hspace=0.6,
                    wspace=0.57)  
                    fig.text(0.09, 0.56, model, rotation=90, va='center', ha='center', fontsize=10, fontweight='bold')
            else:
                plt.tight_layout()
            plt.show()
        
            # 保存整张图
            if save_path:
                file_path = os.path.join(save_path, f"{model}_{x_axis}_{plot_main_fig_prefix}.pdf")
                fig.savefig(file_path, format="pdf", bbox_inches="tight")
                plt.close() 


if __name__ == "__main__":
    
    """ 3. 筛选数据"""
    main_keys = {
        
        "main": { 
            "num_local_forwards": { 
                'model': None,
                'file_type': "avg_acc_results",
                'num_shots': 4,
                'num_clients': 4,
                'if_do_sample': False,
                'max_new_tokens': 256,
                'split_way': None 
                }, 
            
            "num_clients": { 
                'model': None,
                'file_type': "avg_acc_results",
                'num_shots': 8,
                'num_local_forwards': None,
                'if_do_sample': False,
                'max_new_tokens': 256,
                'split_way': None
                },                   
            }
         }          

    
    """ 0.1 数据预处理 """ 
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    comm_policy = "main"  
        
    folder_path = os.path.join(data_path, comm_policy)   
    # generate_token_chunks_and_splits_file()
    generate_res_lens_file(folder_path)    
        
    # # """ 0.2. 预览数据 """ 
    # data = get_exp_stats(data_path, "main")
    # figs_preview(data, script_dir)        

    """ 0.4 预设置 画图内容 """ 
    is_legends_bottom = True
    utils_plot_main_figs = {
        "num_local_forwards": {
            "comm": plot_main_fig_comm_bottom_legends if is_legends_bottom else plot_main_fig_comm
            }, 
        "num_clients": {
            "comp_mem": plot_main_fig_comp_mem
            } if is_legends_bottom else {
                "comp": plot_main_fig_comp, 
                "mem": plot_main_fig_mem,
                }
        }   
    
    xlabel = {"num_local_forwards": "Number of local forwards",
              "num_clients": "Number of participants"                 
              } 
    
    
    """
    画图: "main"
    """
        
    """ 1. 加载数据""" 
    folder_path = os.path.join(data_path, comm_policy) 
    data = get_exp_stats(data_path, comm_policy)
    
    for model in [
            "Qwen2.5-0.5B", 
            "Qwen2.5-1.5B", 
            "Qwen2.5-3B", 
            "Qwen2.5-7B"
            ]:
        
        """ 2. 计算 model size""" 
        model_shape = read_shape(repo_id = "Qwen/" + model)
        model_w_byte = get_model_weight_byte(model_shape)
        
        for x_axis in main_keys[comm_policy].keys():  
            save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "main")
            os.makedirs(save_path, exist_ok=True)                   
                
            plot_main(model = model, 
                    main_keys = main_keys, 
                    comm_policy = comm_policy, 
                    x_axis = x_axis, 
                    data = data, 
                    data_path = data_path , 
                    script_dir = script_dir, 
                    model_w_byte = model_w_byte, 
                    utils_plot_main_figs = utils_plot_main_figs, 
                    xlabel = xlabel,
                    save_path = save_path, 
                    grid = True
                    )
                                     
