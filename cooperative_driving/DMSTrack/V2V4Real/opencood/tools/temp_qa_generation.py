import argparse
import os
import time
import numpy as np

import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils
from opencood.utils import box_utils
from opencood.utils.transformation_utils import x1_to_x2
from opencood.utils.box_utils import project_points_by_matrix_torch

import json
import random

from opencood.tools.inference import generate_3d_grounding_qa_dataset_nq3, generate_3d_grounding_qa_dataset_nq4, generate_3d_grounding_qa_dataset_nq5, generate_3d_grounding_qa_dataset_nq6, generate_3d_grounding_qa_dataset_nq7, generate_3d_grounding_qa_dataset_nq8, generate_3d_grounding_qa_dataset_nq9

import os
from pathlib import Path

# 强制把工作目录切到：temp_qa_generation.py 所在的 V2V4Real 根目录
THIS_FILE = Path(__file__).resolve()
V2V4REAL_ROOT = THIS_FILE.parents[2]  # opencood/tools/temp_qa_generation.py -> 回到 V2V4Real

PROJECT_ROOT  = THIS_FILE.parents[4]          # .../V2V-GoT-main
LLAVA_ROOT    = PROJECT_ROOT / "LLaVA"        # .../V2V-GoT-main/LLaVA

print("[V2V4REAL_ROOT]", V2V4REAL_ROOT)
print("[LLAVA_ROOT]", LLAVA_ROOT)

os.chdir(V2V4REAL_ROOT)

print("[cwd fixed to]", os.getcwd())

def test_parser(argv=None):
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--save_visible_gt_ids', action='store_true',
                        help='whether to save visiable gt ids'
                             'in npy file')
    # Temp QA generation for graph inference
    parser.add_argument('--model', type=str, required=True,
                        help='model')
    parser.add_argument('--ckpt', type=int, required=True,
                        help='ckpt')
    parser.add_argument('--input_qa_dataset',nargs='+', type=str, required=True,
                        help='list input_qa_dataset')
    parser.add_argument('--output_qa_dataset', type=str, required=True,
                        help='output_qa_dataset')
    parser.add_argument('--graph', type=str, default="",
                        help='graph')
    parser.add_argument('--is_training_data', action='store_true',
                        help='whether to generate training data. otherwise generate inference qa')
    parser.add_argument('--split', type=str, default="val",
                        help='split')
    
    # ============ 添加这个参数 ============
    parser.add_argument('--exp_suffix', type=str, default="",
                        help='实验后缀（用于区分不同配置）')
    # ====================================
    
    opt = parser.parse_args(argv)
    return opt


def main(argv=None):
    opt = test_parser(argv)
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion', 'no_fusion_keep_all']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'


    # ===== 直接硬编码（不需要加载数据集）=====
    print('Skipping Dataset Building - Using hardcoded values')
    
    # 创建一个简单的对象来存储 len_record
    class SimpleDataset:
        def __init__(self, val_split=True):
            if val_split:
                # V2V4Real val set: 1993 frames
                self.len_record = [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]
                self.scenario_database = {i: None for i in range(9)}
            else:
                # V2V4Real train set: 7105 frames
                self.len_record = [147, 552, 709, 1953, 2086, 2303, 2425, 2573, 
                                 2983, 3298, 3417, 3524, 3648, 3737, 3817, 3962, 
                                 4255, 4366, 4549, 4726, 5001, 5287, 5516, 5636, 
                                 5804, 6254, 6389, 6532, 6681, 6846, 6997, 7105]
                self.scenario_database = {i: None for i in range(32)}
    
    opencood_dataset = SimpleDataset(val_split=(opt.split != 'train'))
    
    print("opencood_dataset.len_record: ", opencood_dataset.len_record)
    print("opencood_dataset.scenario_database.keys(): ", opencood_dataset.scenario_database.keys())
    # ===== 硬编码结束 =====


    # hypes = yaml_utils.load_yaml(None, opt)

    # print('Dataset Building')
    # opencood_dataset = build_dataset(hypes, visualize=True, train=(opt.split=='train'),
    #                                  isSim=opt.isSim)
    # print(hypes['fusion']['core_method'])
    # # IntermediateFusionDataset

    # print("opencood_dataset.len_record: ", opencood_dataset.len_record)
    # # train set has 32 sequences
    # # [147, 552, 709, 1953, 2086, 2303, 2425, 2573, 2983, 3298, 3417, 3524, 3648, 3737, 3817, 3962, 4255, 4366, 4549, 4726, 5001, 5287, 5516, 5636, 5804, 6254, 6389, 6532, 6681, 6846, 6997, 7105]
    # # test set has 9 sequences
    # # [147, 261, 405, 603, 783, 1093, 1397, 1618, 1993]
    # print("opencood_dataset.scenario_database.keys(): ", opencood_dataset.scenario_database.keys())
    # # odict_keys([0, 1, 2, 3, 4, 5, 6, 7, 8])
    npy_save_path = os.path.join(opt.model_dir, 'npy')
    print('npy_save_path: ', npy_save_path)
    # ./official_models/cobevt/npy

    input_data_list = []
    for input_qa_dataset in opt.input_qa_dataset:

      if opt.is_training_data: # input context is from gt
        input_file = os.path.join(
          opt.model_dir, # './official_models/no_fusion_keep_all/', './official_models/train_no_fusion_keep_all/'  
          'npy/co_llm',
          'v2v4real_3d_grounding_qa_dataset_%s.json' % input_qa_dataset).replace("\\", "/")
        print('input_file: ', input_file)
        with open(input_file, 'r') as f:
          input_data = json.load(f)

      else: # input context is from inference result

        # ============ 添加 exp_suffix 支持 ============
        if opt.exp_suffix:
            input_exp_name = 'v2v4real_3d_grounding_%s_%d_%s_%s_%s' % (
                opt.model, opt.ckpt, opt.graph, input_qa_dataset, opt.exp_suffix)
        else:
            input_exp_name = 'v2v4real_3d_grounding_%s_%d_%s_%s' % (
                opt.model, opt.ckpt, opt.graph, input_qa_dataset)
        # ============================================
        
        input_file = os.path.join(
          LLAVA_ROOT, 'playground/data/eval', 
          input_exp_name,  # ← 使用包含 exp_suffix 的名称
          'answers/%s/llava-v1.5-7b' % opt.split,
          'merge.jsonl').replace("\\", "/")
        print('input_file: ', input_file)

        # ../../LLaVA/playground/data/eval/v2v4real_3d_grounding_allwc_10ep_both_shallow_f2_4330_full_nq2sm3w0d/answers/val/llava-v1.5-7b/merge.jsonl
        # ../../LLaVA/playground/data/eval/v2v4real_3d_grounding_allwc_10ep_both_shallow_f2_4330_spe_nq2sm3w0d/answers/val/llava-v1.5-7b/merge.jsonl
        with open(input_file, 'r') as f:
          input_data = [json.loads(line) for line in f]

      print('input_data[0]: ', input_data[0])
      input_data_list.append(input_data)


    print('opt.output_qa_dataset: ', opt.output_qa_dataset)
    if opt.is_training_data:
      # generate training QA data
      output_folder = os.path.join(
        opt.model_dir, # './official_models/no_fusion_keep_all/', './official_models/train_no_fusion_keep_all/'  
        'npy/co_llm').replace("\\", "/")
      print('output_folder: ', output_folder) 
      os.makedirs(output_folder, exist_ok=True)
      output_file_name = 'in_' + '_'.join(opt.input_qa_dataset) + '_out_' + opt.output_qa_dataset + '.json'
      print('output_file_name: ', output_file_name)
      output_file = os.path.join(output_folder, output_file_name) 

    else:
      # generate inference graph-of-thoughts QA data
      # ============ 添加 exp_suffix ============
      if opt.exp_suffix:
          exp_name = 'v2v4real_3d_grounding_%s_%d_%s_%s_%s' % (
              opt.model, opt.ckpt, opt.graph, opt.output_qa_dataset, opt.exp_suffix)
      else:
          exp_name = 'v2v4real_3d_grounding_%s_%d_%s_%s' % (
              opt.model, opt.ckpt, opt.graph, opt.output_qa_dataset)
      # ========================================
      
      output_folder = os.path.join(
          LLAVA_ROOT, 'playground/data/eval', 
          exp_name,
          'answers/%s/llava-v1.5-7b' % opt.split).replace("\\", "/")

      print('output_folder: ', output_folder) 
      os.makedirs(output_folder, exist_ok=True)
      output_file = os.path.join(output_folder, '%s.json' % opt.output_qa_dataset)

    print('output_file: ', output_file)
    
    # if generating training data, context is from gt json
    # otherwise it is from inference output
    context_list_from_gt = opt.is_training_data


    if opt.output_qa_dataset == 'nq3sm3w0dc':
      generate_3d_grounding_qa_dataset_nq3(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=0, double_cavs=True, with_context=True, context_list=input_data_list, output_file=output_file, context_list_from_gt=context_list_from_gt)
    elif opt.output_qa_dataset == 'nq4sm3w0dc':
      generate_3d_grounding_qa_dataset_nq4(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=0, double_cavs=True, with_context=True, context_list=input_data_list, output_file=output_file, context_list_from_gt=context_list_from_gt)
    elif opt.output_qa_dataset == 'nq5sm3w1dc':
      generate_3d_grounding_qa_dataset_nq5(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=1, double_cavs=True, with_context=True, context_list=input_data_list, output_file=output_file, context_list_from_gt=context_list_from_gt)
    elif opt.output_qa_dataset == 'nq6sm3w1dc':
      generate_3d_grounding_qa_dataset_nq6(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=1, double_cavs=True, with_context=True, context_list=input_data_list, output_file=output_file, context_list_from_gt=context_list_from_gt)
    elif opt.output_qa_dataset == 'nq7sm3w1dc':
      generate_3d_grounding_qa_dataset_nq7(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=1, double_cavs=True, with_context=True, context_list=input_data_list, output_file=output_file, context_list_from_gt=context_list_from_gt)
    elif opt.output_qa_dataset == 'nq8sm3w6dc':
      generate_3d_grounding_qa_dataset_nq8(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=6, double_cavs=True, with_context=True, context_list=input_data_list, output_file=output_file, context_list_from_gt=context_list_from_gt)
    elif opt.output_qa_dataset == 'nq9sm3w6dc':
      generate_3d_grounding_qa_dataset_nq9(opencood_dataset.len_record, npy_save_path, ['ego', '1'], downsample_negatives=False, simplified=True, max_num_answer_objects=3, num_future_waypoints=6, double_cavs=True, with_context=True, context_list=input_data_list, output_file=output_file, context_list_from_gt=context_list_from_gt)
    else:
      print('not implemented')
      assert False
      
    print('QA data generation finished.')
    print(opt.exp_suffix)

if __name__ == '__main__':
    # 默认：读命令行（pipeline 用这个）
    main()

    # 你需要步进 debug 时，临时把上面 main() 注释掉，再用下面这段：
    # main([
    #   "--model_dir", "./official_models/no_fusion_keep_all/",
    #   "--fusion_method", "no_fusion_keep_all",
    #   "--model", "v2vgot_10ep_both_shallow_f2",
    #   "--ckpt", "4330",
    #   "--input_qa_dataset", "nq2sm3w0d",
    #   "--output_qa_dataset", "nq3sm3w0dc",
    #   "--graph", "full",
    #   "--split", "val",
    #   "--exp_suffix", "strided_s16_G2",
    # ])
