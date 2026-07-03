# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:57:02 2025

@author: DengXiumei
"""

import json
import os
import re
import jsonlines

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_get_performance_stats import read_shape, summarize_prefill, summarize_decode, get_model_weight_byte, kv_bytes_per_layer
from utils import get_points


def get_para_infos(folder_path, name, record, para_infos, time_stamp_idx, if_do_sample):
 
    if "uni_extra" in folder_path: 
        substring = "llf"
        idx = name.index(substring)
        para_infos_appendix = name[idx:].split('_')
        split_way_idx_end = 0 - len(para_infos_appendix)
        record['split_way'] = '_'.join(para_infos_appendix[1:])                                      
        record['num_local_forwards_last'] = int(para_infos_appendix[0].replace(substring, ''))


    elif "sample" in folder_path:   
        if "_comp" in folder_path: 
            substring = "rp"
           
        elif "_comm" in folder_path:
            substring = "rm"
            
        idx = name.index(substring)
        para_infos_appendix = name[idx:].split('_')
        split_way_idx_end = 0 - len(para_infos_appendix)
        if if_do_sample == 'True':           
            record['split_way'] = '_'.join(para_infos[time_stamp_idx + 12:split_way_idx_end])
        else:             
            record['split_way'] = '_'.join(para_infos[time_stamp_idx + 9:split_way_idx_end])                           

        if "sample_last_comp_comm" in folder_path:   
            record['ratio_comp'] = float(para_infos_appendix[0].replace("rp", '').replace("rm", ''))
            record['ratio_last_comp'] = float(para_infos_appendix[1].replace("rp", '').replace("rm", ''))
            record['ratio_comm'] = float(para_infos_appendix[2].replace("rp", '').replace("rm", ''))
            record['ratio_last_comm'] = float(para_infos_appendix[3].replace("rp", '').replace("rm", '')) 
            record['split_way'] = '_'.join(para_infos_appendix[4:]) 

        elif "sample_uni_comp_comm" in folder_path:
            record['ratio_comp'] = float(para_infos_appendix[0].replace("rp", '').replace("rm", ''))
            record['ratio_comm'] = float(para_infos_appendix[1].replace("rp", '').replace("rm", ''))
            record['split_way'] = '_'.join(para_infos_appendix[2:]) 
   
        elif "sample_uni_comp" in folder_path:
            record['ratio_comp'] = float(para_infos_appendix[0].replace("rp", '').replace("rm", ''))
            record['split_way'] = '_'.join(para_infos_appendix[1:]) 

        elif "sample_uni_comm" in folder_path:
            record['ratio_comm'] = float(para_infos_appendix[0].replace("rp", '').replace("rm", ''))   
            record['split_way'] = '_'.join(para_infos_appendix[1:]) 
          
        elif "sample_last_comp" in folder_path: 
            record['ratio_comp'] = float(para_infos_appendix[0].replace("rp", '').replace("rm", ''))
            record['ratio_last_comp'] = float(para_infos_appendix[1].replace("rp", '').replace("rm", ''))
            record['split_way'] = '_'.join(para_infos_appendix[2:]) 
           
        elif "sample_last_comm" in folder_path:
            record['ratio_comm'] = float(para_infos_appendix[0].replace("rp", '').replace("rm", ''))
            record['ratio_last_comm'] = float(para_infos_appendix[1].replace("rp", '').replace("rm", ''))  
            record['split_way'] = '_'.join(para_infos_appendix[2:]) 

    else:   

        if if_do_sample == 'True':           
            record['split_way'] = '_'.join(para_infos[time_stamp_idx + 12:])
        else:             
            record['split_way'] = '_'.join(para_infos[time_stamp_idx + 9:])
            
    return record

def get_exp_stat(folder_path, comm_policy):
    
    data = []
    for file_name in os.listdir(os.path.join(folder_path, "avg_acc")):
        if file_name.endswith('.json') and "avg_acc_results" in file_name:

            # 读取JSON数据
            with open(os.path.join(folder_path, "avg_acc", file_name), 'r') as f:
                json_data = json.load(f)
            
            # 解析文件名
            name = file_name.replace('.json', '')
            para_infos = name.split('_')
     
            # 找到时间戳位置（8位数字）
            time_stamp_idx = None
            for i, para_info in enumerate(para_infos):
                if len(para_info) == 8 and para_info.isdigit():
                    time_stamp_idx = i
                    break
            time_stamp = f"{para_infos[time_stamp_idx]}_{para_infos[time_stamp_idx + 1]}"  # 20250720_134456            
            
            # 提取信息
            file_type = '_'.join(para_infos[:time_stamp_idx])  # avg_acc_results 或其他带下划线的类型
            model = para_infos[time_stamp_idx + 2]  # Qwen2.5-0.5B
            max_new_tokens = para_infos[time_stamp_idx + 3]  # 256
            num_shots = para_infos[time_stamp_idx + 4].replace('shot', '')  # 8
            num_clients = para_infos[time_stamp_idx + 5].replace('c', '')  # 2
            num_local_forwards = para_infos[time_stamp_idx + 6].replace('lf', '')  # 2
            if_do_sample = para_infos[time_stamp_idx + 8]  # True/False  
                                
            # 构建记录
            record = {
                'model': model,
                'file_type': file_type,
                'num_shots': int(num_shots),
                'num_clients': int(num_clients),
                'num_local_forwards': int(num_local_forwards),
                'if_do_sample': if_do_sample == 'True',
                'max_new_tokens': int(max_new_tokens),
                'time_stamp': time_stamp,
                'file_name': file_name,
                'data': json_data,
                'comm_policy': comm_policy
                }
            
            if if_do_sample == 'True':           
                record['temperature'] = float(para_infos[time_stamp_idx + 9].replace('temp', ''))
                record['top_k'] = int(para_infos[time_stamp_idx + 10].replace('k', ''))
                record['top_p'] = float(para_infos[time_stamp_idx + 11].replace('p', ''))
            else:
                record['temperature'] = None
                record['top_k'] = None
                record['top_p'] = None 

            # 提取剩下的信息
            record = get_para_infos(folder_path, name, record, para_infos, time_stamp_idx, if_do_sample)
                   
            data.append(record)
                
    return data

def get_exp_stats(data_path, comm_policy):
    
    folder_path = os.path.join(data_path, comm_policy)
    data = get_exp_stat(folder_path, comm_policy)

    if comm_policy not in ["main"]:
        folder_path = os.path.join(data_path, "main")
        data.extend(get_exp_stat(folder_path, "main"))
               
    return data


def get_info(record, script_dir, folder_path):   

    data_process_token_split_file_path = os.path.join(os.path.dirname(script_dir), "data_process", "data_process_token_split",
        f"data_process_{record['model']}_shot{record['num_shots']}_c{record['num_clients']}_{record['split_way']}.jsonl"
        )
    
    with jsonlines.open(data_process_token_split_file_path, "r") as reader:
        inputs = list(reader)             

    responses = {}
    
    # 遍历所有符合条件的文件
    for file_name in os.listdir(folder_path):
        if "client" in file_name and record['time_stamp'] in file_name and file_name.endswith(".jsonl"):
            match = re.search(r"client(\d+)\.jsonl$", file_name)
            if match:
                client_id = int(match.group(1))
                file_path = os.path.join(folder_path, file_name)
    
                with jsonlines.open(file_path, "r") as reader:
                    for row in reader:
                        # 存 completion / acc
                        responses.setdefault(client_id, []).append({
                            "completion": row.get("completion"),
                            "acc": row.get("acc")
                        })        
    record["inputs"] = inputs
    record["responses"] = responses

    data_process_responses_file_path = os.path.join(folder_path, "data_process_responses")    
    
    # 遍历所有符合条件的文件
    for file_name in os.listdir(data_process_responses_file_path):
        if record['time_stamp'] in file_name:
            with open(os.path.join(data_process_responses_file_path, file_name), 'r', encoding='utf-8') as f:
                record["len_responses"] = json.load(f)
            break

    return record

def create_exp_name(hyperparams, comm_policy):

    exp_name = f"shot{hyperparams['num_shots']}_c{hyperparams['num_clients']}_lf{hyperparams['num_local_forwards']}_sample_{hyperparams['if_do_sample']}"
    if hyperparams['if_do_sample']:
        exp_name += f"_temp{hyperparams['temperature']}_k{hyperparams['top_k']}_p{hyperparams['top_p']}"   
    exp_name_appendix = {
        "uni_extra": f"_llf{hyperparams.get('num_local_forwards_last')}",
        "sample_uni_comp": f"_rp{hyperparams.get('ratio_comp')}",
        "sample_uni_comm": f"_rm{hyperparams.get('ratio_comm')}",
        "sample_uni_comp_comm": f"_rp{hyperparams.get('ratio_comp')}_rm{hyperparams.get('ratio_comm')}",
        "sample_last_comp": f"_rp{hyperparams.get('ratio_comp')}_{hyperparams.get('ratio_last_comp')}",
        "sample_last_comm": f"_rm{hyperparams.get('ratio_comm')}_{hyperparams.get('ratio_last_comm')}",
        "sample_last_comp_comm": f"_rp{hyperparams.get('ratio_comp')}_{hyperparams.get('ratio_last_comp')}_rm{hyperparams.get('ratio_comm')}_{hyperparams.get('ratio_last_comm')}",
        }          
    exp_name += exp_name_appendix.get(comm_policy,"")  
    
    return f"{hyperparams['max_new_tokens']}_{exp_name}_{hyperparams['split_way']}"


def get_perf_stat(data, model, model_shape, model_w_byte, data_path, script_dir, comm_policy):
    
    file_path = os.path.join(os.path.join(data_path, comm_policy), "collected_data", model)
    os.makedirs(file_path, exist_ok=True)
    
    indices_to_remove = []

    for idx, item in enumerate(data):
        if item.get('model') == model:   
            
            exp_name = create_exp_name(item, comm_policy)          
            output_file = f"main_record_{item['time_stamp']}_{item['model']}_{exp_name}.json"
            
            if os.path.exists(os.path.join(file_path, output_file)):     
                with open(os.path.join(file_path, output_file) , 'r', encoding='utf-8') as f:
                    item["main"] = json.load(f)  
            
            else:
                try:
                    record = get_info(item, script_dir = script_dir, 
                                      folder_path = os.path.join(data_path, item["comm_policy"]))
                except Exception as e:
                    print(f"get_info() {item['time_stamp']} 时出错: {e}")
                    print(f"删除数据 {item['time_stamp']}: idx = {idx}")
                    indices_to_remove.append(idx)
                    continue
                
                main_record = []         
                
                # 获取常用变量
                num_local_forwards = record["num_local_forwards"]
                num_layers = model_shape["L"]
                
                break_flag = False
                for task_id in range(len(record["inputs"])):
                    collected_data = {"total_num_input_tokens": 0}
                    
                    # 收集每个客户端的数据并计算总输入token
                    num_clients = record["num_clients"]
                    for client_id in range(num_clients):
                        num_input_tokens = record["inputs"][task_id]["token_chunks_lens"][client_id]
                        num_output_tokens = record["len_responses"][str(client_id)][task_id]
                        
                        collected_data[client_id] = {
                            "num_input_tokens": num_input_tokens,
                            "num_output_tokens": num_output_tokens
                        }
                        collected_data["total_num_input_tokens"] += num_input_tokens
                    
                    if break_flag:
                        break  
                    
                    # 为每个客户端计算KV缓存和性能指标
                    total_num_input_tokens = collected_data["total_num_input_tokens"]

                    if "sample" in comm_policy:   
                        collected_data["total_num_input_tokens_sample_comp"]= 0
                        collected_data["total_num_input_tokens_sample_comm"]= 0
                        collected_data["ratios_comp"] = [record.get('ratio_comp', 1.0) for _ in range(num_clients-1)] + [record.get('ratio_last_comp', record.get('ratio_comp', 1.0))]
                        collected_data["ratios_comm"] = [record.get('ratio_comm', 1.0) for _ in range(num_clients-1)] + [record.get('ratio_last_comm', record.get('ratio_comm', 1.0))]  
                        
                        for client_id in range(num_clients): 
                            client_data = collected_data[client_id]
                            client_data["num_input_tokens_sample_comp"] = int(client_data["num_input_tokens"] * collected_data["ratios_comp"][client_id])
                            client_data["num_input_tokens_sample_comm"] = int(client_data["num_input_tokens_sample_comp"] * collected_data["ratios_comm"][client_id])
                            collected_data["total_num_input_tokens_sample_comp"] += client_data["num_input_tokens_sample_comp"]
                            collected_data["total_num_input_tokens_sample_comm"] += client_data["num_input_tokens_sample_comm"]
                    
                    for client_id in range(num_clients):
                        client_data = collected_data[client_id]
                        num_input_tokens = client_data["num_input_tokens"]
                        num_output_tokens = client_data["num_output_tokens"]

                        # 计算KV缓存列表&把满足节奏条件的 KV 累加到“通信量”
                        list_num_kv = []
                        kv_comm = 0
                        kv_comm_0 =  kv_bytes_per_layer(num_input_tokens, model_shape)
                                           
                        if comm_policy == "main":                    
                            
                            for layer_idx in range(num_layers):
                                if layer_idx % num_local_forwards == 0:
                                    list_num_kv.append(total_num_input_tokens)
                                    kv_comm += kv_comm_0  
                                else:
                                    list_num_kv.append(num_input_tokens)
                    
                        elif "sample" in comm_policy:   

                            num_input_tokens_sample_comp = client_data["num_input_tokens_sample_comp"]
                            num_input_tokens_sample_comm = client_data["num_input_tokens_sample_comm"]
                            total_num_input_tokens_sample_comm = collected_data["total_num_input_tokens_sample_comm"]

                            for layer_idx in range(num_layers):
                                if layer_idx % num_local_forwards == 0:
                                    list_num_kv.append(total_num_input_tokens_sample_comm - num_input_tokens_sample_comm + num_input_tokens_sample_comp)
                                    kv_comm += kv_bytes_per_layer(num_input_tokens_sample_comm, model_shape)
                                else:
                                    list_num_kv.append(num_input_tokens_sample_comp)
                            num_input_tokens = client_data["num_input_tokens_sample_comp"]
                 
                        elif comm_policy == "uni_extra":  
                            
                            num_local_forwards_last = record.get("num_local_forwards_last")
                            for layer_idx in range(num_layers):
                                if layer_idx % num_local_forwards == 0:
                                    list_num_kv.append(total_num_input_tokens)
                                    kv_comm += kv_comm_0 
                                else:
                                    if layer_idx % num_local_forwards_last == 0:
                                        list_num_kv.append(num_input_tokens + collected_data[num_clients-1]["num_input_tokens"])
                                        if client_id == num_clients-1:
                                            kv_comm += kv_comm_0
                                    else:
                                        list_num_kv.append(num_input_tokens)              
                            
                        elif comm_policy in ["uni_fr", "uni_bk", "inc_gp", "dec_gp"]: 

                            comm_points = get_points(num_layers, num_local_forwards, comm_policy)                     
                            for layer_idx in range(num_layers):
                                if layer_idx in comm_points:
                                    list_num_kv.append(total_num_input_tokens)
                                    kv_comm += kv_comm_0  
                                else:
                                    list_num_kv.append(num_input_tokens)
                            
                        else:
                            raise ValueError(f"不支持的通信策略: {comm_policy}")                                    
                        
                        # 添加计算结果到客户端数据
                        client_data.update({
                            "list_num_kv": list_num_kv,
                            "prefill": summarize_prefill(model_w_byte, kv_comm, model_shape, num_input_tokens, list_num_kv, T_logits=1),
                            "decode": summarize_decode(model_w_byte, model_shape, T0_kv_list=list_num_kv, G=num_output_tokens)
                        })                         

                    main_record.append(collected_data)
                
                if not break_flag:
                    with open(os.path.join(file_path, output_file), "w", encoding="utf-8") as f:
                        json.dump(main_record, f, ensure_ascii=False, indent=2)
                        
                    item["main"] = main_record
    
    # 保留不在删除索引中的元素
    if indices_to_remove:
        data = [item for idx, item in enumerate(data) if idx not in indices_to_remove]  
    
    return data


  