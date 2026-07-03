# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 12:21:28 2025

@author: DengXiumei
"""
def get_para_dict(comm_policy):
    
    # if comm_policy == "main":
    #     # 构造新超参数组合
    #     num_local_forward_map = {
    #         "Qwen/Qwen2.5-0.5B": [2, 3, 4, 6, 8, 12, 18, 24],  # 24
    #         "Qwen/Qwen2.5-1.5B": [2, 3, 4, 8, 10, 14, 21, 28],  # 28
    #         "Qwen/Qwen2.5-3B": [2, 3, 4, 9, 12, 18, 27, 36],  # 36
    #         "Qwen/Qwen2.5-7B": [2, 3, 4, 8, 10, 14, 21, 28]  # 28
    #         }  
    #     sample_hyperparams_grid = {
    #         'temperature': [0.2],
    #         'top_k': [20],
    #         'top_p': [0.9]
    #         }
    #     hyperparams_grid = {
    #         'checkpoint_path': 
    #             [
    #                 'Qwen/Qwen2.5-0.5B',
    #                 'Qwen/Qwen2.5-1.5B',
    #                 'Qwen/Qwen2.5-3B',
    #                 'Qwen/Qwen2.5-7B'
    #                 ],
    #         'num_clients': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 24, 32],            # [2, 4, 8, 16, 32]  [2, 4, 6, 8, 10, 16, 24, 32]             
    #         'num_local_forward': [],
    #         'do_sample': [False, True],             # True, False
    #         'max_new_tokens': [256],                  # 2, 3, 4, 8
    #         'num_shots': [1, 2, 4, 8],                 # 4, 8
    #         'split_way': ["even", "smart", "even_question_last", "smart_question_last"] # ["even", "smart", "even_question_last", "smart_question_last"] 
    #         } 
              
    #     other_hyperparams_grid = {}  
    #     return num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid
    
    
    
   
    if comm_policy == "main":
        # 构造新超参数组合
        num_local_forward_map = {
            "Qwen/Qwen2.5-0.5B": [2, 3, 4, 8],  # 24
            "Qwen/Qwen2.5-1.5B": [2, 3, 4, 8],  # 28
            "Qwen/Qwen2.5-3B": [2, 3, 4, 9],  # 36
            # "Qwen/Qwen2.5-7B": [2, 3, 4, 5, 6]  # 28
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
                    'Qwen/Qwen2.5-1.5B',
                    'Qwen/Qwen2.5-3B',
                    # 'Qwen/Qwen2.5-7B'
                    ],
            'num_clients': [2, 3, 4, 5, 6],            # [2, 4, 8, 16, 32]  [2, 4, 6, 8, 10, 16, 24, 32]             
            'num_local_forward': [],
            'do_sample': [False],             # True, False
            'max_new_tokens': [16, 32, 64, 128, 512, 1024],                 # 2, 3, 4, 8
            'num_shots': [1, 2, 3, 4, 5, 6, 7, 8],                 # 4, 8
            'split_way': ["even", "smart", "even_question_last", "smart_question_last"] # ["even", "smart", "even_question_last", "smart_question_last"] 
            } 
              
        other_hyperparams_grid = {}  
        return num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid    
    
    
    # if comm_policy == "main":
    #     # 构造新超参数组合
    #     num_local_forward_map = {
    #         "Qwen/Qwen2.5-0.5B": [1],  # 24
    #         "Qwen/Qwen2.5-1.5B": [1],  # 28
    #         "Qwen/Qwen2.5-3B": [1],  # 36
    #         "Qwen/Qwen2.5-7B": [1]  # 28
    #         }  
    #     sample_hyperparams_grid = {
    #         'temperature': [0.2],
    #         'top_k': [20],
    #         'top_p': [0.9]
    #         }
    #     hyperparams_grid = {
    #         'checkpoint_path': 
    #             [
    #                 'Qwen/Qwen2.5-0.5B',
    #                 'Qwen/Qwen2.5-1.5B',
    #                 'Qwen/Qwen2.5-3B',
    #                 'Qwen/Qwen2.5-7B'
    #                 ],
    #         'num_clients': [1],            # [2, 4, 8, 16, 32]  [2, 4, 6, 8, 10, 16, 24, 32]             
    #         'num_local_forward': [1],
    #         'do_sample': [False, True],             # True, False
    #         'max_new_tokens': [256],                  # 2, 3, 4, 8
    #         'num_shots': [1, 2, 3, 4, 5, 6, 7, 8],                 # 4, 8
    #         'split_way': ["even"] # ["even", "smart", "even_question_last", "smart_question_last"] 
    #         } 
              
    #     other_hyperparams_grid = {}  
    #     return num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid    
    
    if comm_policy == "uni_extra":
        # 构造新超参数组合
        num_local_forward_map = {
            "Qwen/Qwen2.5-0.5B": [8],  # 24
            "Qwen/Qwen2.5-1.5B": [8],  # 28
            "Qwen/Qwen2.5-3B": [9],  # 36
            "Qwen/Qwen2.5-7B": [8]  # 28
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
                    'Qwen/Qwen2.5-1.5B',
                    'Qwen/Qwen2.5-3B',
                    'Qwen/Qwen2.5-7B'
                    ],
            'num_clients': [4],            # [2, 4, 8, 16, 32]  [2, 4, 6, 8, 10, 16, 24, 32]             
            'num_local_forward': [],
            'do_sample': [False],             # True, False
            'max_new_tokens': [256],                  # 2, 3, 4, 8
            'num_shots': [4],                 # 4, 8
            'split_way': ["even", "smart", "even_question_last", "smart_question_last"] # ["even", "smart", "even_question_last", "smart_question_last"] 
            } 
              
        other_hyperparams_grid = {
            "uni_extra": {'last_num_local_forward': [1, 2, 3, 4, 5, 6, 7, 8, 9]},
            }  
        return num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid
    
    # if "sample" in comm_policy:
    #     # 构造新超参数组合
    #     num_local_forward_map = {
    #         "Qwen/Qwen2.5-0.5B": [8],  # 24
    #         "Qwen/Qwen2.5-1.5B": [2,4,8],  # 28
    #         "Qwen/Qwen2.5-3B": [9],  # 36
    #         "Qwen/Qwen2.5-7B": [8]  # 28
    #         }      
    #     sample_hyperparams_grid = {
    #         'temperature': [0.2],
    #         'top_k': [20],
    #         'top_p': [0.9]
    #         }
    #     hyperparams_grid = {
    #         'checkpoint_path': 
    #             [
    #                 'Qwen/Qwen2.5-0.5B',
    #                 'Qwen/Qwen2.5-1.5B',
    #                 'Qwen/Qwen2.5-3B',
    #                 'Qwen/Qwen2.5-7B'
    #                 ],
    #         'num_clients': [4],            # [2, 4, 8, 16, 32]  [2, 4, 6, 8, 10, 16, 24, 32]             
    #         'num_local_forward': [],
    #         'do_sample': [False],             # True, False
    #         'max_new_tokens': [256],                  # 2, 3, 4, 8
    #         'num_shots': [4],                 # 4, 8
    #         'split_way': ["even", "smart", "even_question_last", "smart_question_last"] # ["even", "smart", "even_question_last", "smart_question_last"] 
    #         } 
              
    #     other_hyperparams_grid = {
    #         "sample_uni_comp": {'ratio_comp': [0.9, 0.7, 0.5, 0.3, 0.1]}, # [0.9, 0.7, 0.5, 0.3, 0.1]
    #         "sample_uni_comm": {'ratio_comm': [0.9, 0.7, 0.5, 0.3, 0.1]}, 
    #         "sample_uni_comp_comm": {'ratio_comp': [0.9, 0.5, 0.1],'ratio_comm': [0.9, 0.5, 0.1]}, 
    #         "sample_last_comp": {'ratio_comp': [0.9, 0.7, 0.5, 0.3, 0.1], 'ratio_last_comp': [1.0]}, 
    #         "sample_last_comm": {'ratio_comm': [1.0], 'ratio_last_comm': [0.9, 0.7, 0.5, 0.3, 0.1]}, 
    #         "sample_last_comp_comm": {'ratio_comp': [0.9, 0.5, 0.1],'ratio_comm': [0.9, 0.5, 0.1], 'ratio_last_comp': [1.0],'ratio_last_comm': [1.0]}, 
    #         }   
    #     return num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid

    if "sample" in comm_policy:
        # 构造新超参数组合
        num_local_forward_map = {
            "Qwen/Qwen2.5-0.5B": [8,3,1,2,4],  # 24
            "Qwen/Qwen2.5-1.5B": [8,2,1,3,4],  # 28
            "Qwen/Qwen2.5-3B": [9,5,1,2,3,4],  # 36        
            # "Qwen/Qwen2.5-7B": [8,
            #                     # 2,
            #                     # 1
            #                     ]  # [8,2,1]
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
                    'Qwen/Qwen2.5-1.5B',
                    'Qwen/Qwen2.5-3B',
                    # 'Qwen/Qwen2.5-7B'
                    ],
            'num_clients': [4],            # [2, 4, 8, 16, 32]  [2, 4, 6, 8, 10, 16, 24, 32]             
            'num_local_forward': [],
            'do_sample': [False],             # True, False
            'max_new_tokens': [256],                  # 2, 3, 4, 8
            'num_shots': [4],                 # 4, 8
            'split_way': ["even", "smart", "even_question_last", "smart_question_last"] # ["even", "smart", "even_question_last", "smart_question_last"] 
            } 
              
        other_hyperparams_grid = {
            "sample_uni_comp": {
                'ratio_comp': [ 
                    # 0.9, 0.7, 0.5, 0.3, 0.1, 
                    # 0.09, 0.07, 0.05, 0.03, 0.01, 
                    0.001,
                    ]
                }, # [0.9, 0.7, 0.5, 0.3, 0.1]
            "sample_uni_comm": {
                'ratio_comm': [
                    0.9, 0.7, 0.5, 0.3, 0.1, 
                    # 0.09, 0.07, 0.05, 0.03, 0.01, 
                    # 0.001,
                    ]
                }, 
            
            # "sample_last_comp": {
            #     'ratio_comp': [1.0],
            #     'ratio_last_comp': [
            #         0.9, 0.7, 0.5, 0.3, 0.1, 
            #         # 0.09, 0.07, 0.05, 0.03, 0.01,
            #         # 0.001
            #         ]                 
            #     },  
            # "sample_last_comm": {
            #     'ratio_comm': [1.0],
            #     'ratio_last_comm': [
            #         0.9, 0.7, 0.5, 0.3, 0.1, 
            #         # 0.09, 0.07, 0.05, 0.03, 0.01,
            #         # 0.001
            #         ]             
            #     }, 
            
            "sample_last_comp": {
                'ratio_comp': [
                    0.9, 0.7, 0.5, 0.3, 0.1, 
                    # 0.09, 0.07, 0.05, 0.03, 0.01,
                    # 0.001
                    ], 
                'ratio_last_comp': [1.0]
                },  
            "sample_last_comm": {
                'ratio_comm': [
                    0.9, 0.7, 0.5, 0.3, 0.1, 
                    # 0.09, 0.07, 0.05, 0.03, 0.01,
                    # 0.001
                    ], 
                'ratio_last_comm': [1.0]
                }, 

  
            
            # "sample_last_comp": {'ratio_comp': [1.0], 'ratio_last_comp': [0.9, 0.7, 0.5, 0.3, 0.1]}, 
            # "sample_last_comm": {'ratio_comm': [1.0], 'ratio_last_comm': [0.9, 0.7, 0.5, 0.3, 0.1]},
            
            # # "sample_last_comp": {'ratio_comp': [0.9, 0.7, 0.5, 0.3, 0.1], 'ratio_last_comp': [1.0]},  
            # "sample_last_comm": {'ratio_comm': [0.9, 0.7, 0.5, 0.3, 0.1], 'ratio_last_comm': [1.0]}, 
                         

            # "sample_uni_comp_comm": {'ratio_comp': [0.9, 0.5, 0.1],'ratio_comm': [0.9, 0.5, 0.1]},             
            # "sample_last_comp_comm": {'ratio_comp': [0.9, 0.5, 0.1],'ratio_comm': [0.9, 0.5, 0.1], 'ratio_last_comp': [1.0],'ratio_last_comm': [1.0]}, 
            }  
        return num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid




    if comm_policy in ["uni_fr", "uni_bk", "inc_gp", "dec_gp"]:
        # 构造新超参数组合
        num_local_forward_map = {
            "Qwen/Qwen2.5-0.5B": [8],  # 24
            "Qwen/Qwen2.5-1.5B": [8],  # 28
            "Qwen/Qwen2.5-3B": [9],  # 36
            "Qwen/Qwen2.5-7B": [8]  # 28
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
                    'Qwen/Qwen2.5-1.5B',
                    'Qwen/Qwen2.5-3B',
                    'Qwen/Qwen2.5-7B'
                    ],
            'num_clients': [4],            # [2, 4, 8, 16, 32]  [2, 4, 6, 8, 10, 16, 24, 32]             
            'num_local_forward': [],
            'do_sample': [False],             # True, False
            'max_new_tokens': [256],                  # 2, 3, 4, 8
            'num_shots': [4],                 # 4, 8
            'split_way': ["even", "smart", "even_question_last", "smart_question_last"] # ["even", "smart", "even_question_last", "smart_question_last"] 
            } 
              
        other_hyperparams_grid = {}  
        return num_local_forward_map, sample_hyperparams_grid, hyperparams_grid, other_hyperparams_grid    
    