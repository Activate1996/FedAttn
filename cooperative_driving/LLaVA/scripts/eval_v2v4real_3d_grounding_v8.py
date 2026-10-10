# -*- coding: utf-8 -*-
"""Per-sample evaluation and configuration-level aggregation for V2V4Real."""

"""Path configuration"""

import sys
import os

# True: evaluate the bundled Q1/H demo; False: evaluate normal inference outputs.
DEMO_MODE = True

# Resolve the script path before changing the working directory.
SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

os.chdir(SCRIPT_DIR)

LLAVA_ROOT = os.path.join(PROJECT_ROOT, "LLaVA")

RAW_OUTPUT_ROOT = os.path.join(
    LLAVA_ROOT,
    "playground",
    "data",
    "eval",
    )

NPY_ROOT = os.path.join(
    PROJECT_ROOT,
    "DMSTrack",
    "V2V4Real",
    "official_models",
    "no_fusion_keep_all",
    "npy",
    )

EVALUATION_ROOT = (
    os.path.join(PROJECT_ROOT, "results", "demo", "q1_evaluation")
    if DEMO_MODE else os.path.join(PROJECT_ROOT, "results", "evaluation")
)

PER_SAMPLE_V2V_LLM_DIR = os.path.join(
    EVALUATION_ROOT,
    "per_sample",
    "v2v_llm",
    )

AGGREGATE_V2V_LLM_DIR = os.path.join(
    EVALUATION_ROOT,
    "aggregate",
    "v2v_llm",
    )

for output_dir in [
    PER_SAMPLE_V2V_LLM_DIR,
    AGGREGATE_V2V_LLM_DIR,
    ]:
    os.makedirs(output_dir, exist_ok=True)


sys.path.append(os.path.join(PROJECT_ROOT, 'DMSTrack').replace("\\", "/"))
sys.path.append(os.path.join(PROJECT_ROOT, 'DMSTrack/AB3DMOT').replace("\\", "/"))
sys.path.append(os.path.join(PROJECT_ROOT, 'DMSTrack/AB3DMOT/Xinshuo_PyToolbox').replace("\\", "/"))
sys.path.append(os.path.join(PROJECT_ROOT, 'DMSTrack/V2V4Real').replace("\\", "/"))

"""Imports""" 
from tqdm import tqdm
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')
import argparse 
import json  
import numpy as np  

"""Serialization utilities"""

def to_jsonable(x):
    # numpy scalar -> python scalar
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    # numpy array -> list
    if isinstance(x, np.ndarray):
        return x.tolist()
    # containers
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x

"""Per-sample metric functions"""
from eval_v2v4real_3d_grounding_v6_utils import ( 
    classify_answer_type,
    parse_multiple_box_parameters,
    evaluate_object_existance,
    evaluate_reason,
    evaluate_multiple_box_accuracy,
    evaluate_future_trajectory_per_sample
    )

"""Q5 aggregation"""
def aggregate_eval_results_Q5(samples):
    """Aggregate trajectory metrics over all samples in one configuration."""    
    n = len(samples)
    if n == 0:
        return {'num_samples': len(samples), 'num_valid': 0}
    
    results = [s['eval_result'] for s in samples]
    
    dict_results = {
        'num_samples': len(samples),
        'num_valid': n,
        'l2_error_avg_1s': np.mean([r['l2_error_1s'] for r in results]),
        'l2_error_avg_2s': np.mean([r['l2_error_2s'] for r in results]),
        'l2_error_avg_3s': np.mean([r['l2_error_3s'] for r in results]),
        'collision_rate_1s': np.mean([r['collision_1s'] for r in results]),
        'collision_rate_2s': np.mean([r['collision_2s'] for r in results]),
        'collision_rate_3s': np.mean([r['collision_3s'] for r in results]),
        'NC': np.mean([r['nc'] for r in results]),
        'TTC': np.mean([r['ttc'] for r in results]),
        'C': np.mean([r['comfort'] for r in results]),
        'PDMS': np.mean([r['pdms'] for r in results])
        }

    dict_results['l2_error_avg_all'] = (dict_results['l2_error_avg_1s'] + dict_results['l2_error_avg_2s'] + dict_results['l2_error_avg_3s']) / 3
    dict_results['collision_rate_avg_all'] = (dict_results['collision_rate_1s'] + dict_results['collision_rate_2s'] + dict_results['collision_rate_3s']) / 3

    return dict_results

"""Q1 and Q4 aggregation"""
def aggregate_eval_results_Q1_4(data_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate object-presence, attribution, and localization statistics."""
    
    # ============================================================
    # 1) Initialize pooled metric accumulators.
    # ============================================================
    
    # Object-presence counts.
    object_existance_accumulator = {
        'TP': 0,   # True positive: present in both the ground truth and prediction.
        'FP': 0,   # False positive: absent in the ground truth, predicted present.
        'FN': 0,   # False negative: present in the ground truth, predicted absent.
        'TN': 0    # True negative: absent in both the ground truth and prediction.
    }
    
    # Detection-attribution counts.
    reason_accumulator = {
        'TP_reason_correct': 0,  # Correct attribution among true positives.
        'TP_reason_count': 0,    # Number of true positives.
        'TN_reason_correct': 0,  # Correct attribution among true negatives.
        'TN_reason_count': 0     # Number of true negatives.
    }
    
    # Localization statistics.
    box_accuracy_accumulator = {
        'num_samples': 0,           # Number of samples.
        'TP_count': 0,              # Number of true positives.
        'num_failed_parsing': 0,    # Number of parsing failures.
        'num_total_gt_boxes': 0,    # Total number of valid ground-truth boxes.
        'num_total_output_boxes': 0,  # Total number of valid predicted boxes.
        'all_dist_errors': [],      # Pooled localization errors.
        'threshold_accumulator': {  # Matching counts at each distance threshold.
            0.5: {'num_matched_output': 0, 'num_matched_gt': 0},
            1.0: {'num_matched_output': 0, 'num_matched_gt': 0},
            2.0: {'num_matched_output': 0, 'num_matched_gt': 0},
            4.0: {'num_matched_output': 0, 'num_matched_gt': 0}
        }
    }
    
    # ============================================================
    # 2) Accumulate statistics across samples.
    # ============================================================
    
    for sample in data_samples:
        # Accumulate object-presence counts.
        if 'object_existance' in sample:
            obj_exist = sample['object_existance']
            object_existance_accumulator['TP'] += obj_exist.get('TP', 0)
            object_existance_accumulator['FP'] += obj_exist.get('FP', 0)
            object_existance_accumulator['FN'] += obj_exist.get('FN', 0)
            object_existance_accumulator['TN'] += obj_exist.get('TN', 0)
        
        # Accumulate attribution counts.
        if 'reason' in sample:
            reason = sample['reason']
            reason_accumulator['TP_reason_correct'] += reason.get('TP_reason_correct', 0)
            reason_accumulator['TP_reason_count'] += reason.get('TP_reason_count', 0)
            reason_accumulator['TN_reason_correct'] += reason.get('TN_reason_correct', 0)
            reason_accumulator['TN_reason_count'] += reason.get('TN_reason_count', 0)
        
        # Accumulate localization statistics.
        if 'multiple_box_accuracy' in sample:
            box_acc = sample['multiple_box_accuracy']
            
            # Sample-level counts.
            box_accuracy_accumulator['num_samples'] += box_acc.get('num_samples', 0)
            box_accuracy_accumulator['TP_count'] += box_acc.get('TP_count', 0)
            box_accuracy_accumulator['num_failed_parsing'] += box_acc.get('num_failed_parsing', 0)
            
            # Object-level counts.
            box_accuracy_accumulator['num_total_gt_boxes'] += box_acc.get('num_total_gt_boxes', 0)
            box_accuracy_accumulator['num_total_output_boxes'] += box_acc.get('num_total_output_boxes', 0)
            
            # Localization errors.
            box_accuracy_accumulator['all_dist_errors'].extend(box_acc.get('dist_errors', []))
            
            # Threshold-specific counts.
            if 'threshold_stats' in box_acc:
                for threshold, stats in box_acc['threshold_stats'].items():
                    # JSON object keys may encode numerical thresholds as strings.
                    threshold_key = float(threshold)
                    if threshold_key in box_accuracy_accumulator['threshold_accumulator']:
                        box_accuracy_accumulator['threshold_accumulator'][threshold_key]['num_matched_output'] += \
                            stats.get('num_matched_output', 0)
                        box_accuracy_accumulator['threshold_accumulator'][threshold_key]['num_matched_gt'] += \
                            stats.get('num_matched_gt', 0)
    
    # ============================================================
    # 3) Compute metrics from pooled counts.
    # ============================================================
    
    aggregated_metrics = {}
    
    # Object-presence metrics.
    TP = object_existance_accumulator['TP']
    FP = object_existance_accumulator['FP']
    FN = object_existance_accumulator['FN']
    TN = object_existance_accumulator['TN']
    
    total_samples = TP + FP + FN + TN
    
    # Accuracy.
    if total_samples > 0:
        accuracy = (TP + TN) / total_samples
    else:
        accuracy = 0.0
    
    # Precision for the object-present class.
    if (TP + FP) > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0
    
    # Recall for the object-present class.
    if (TP + FN) > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0
    
    # F1 score.
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    aggregated_metrics['object_existance'] = {
        'total_samples': int(total_samples),
        'TP': int(TP),
        'FP': int(FP),
        'FN': int(FN),
        'TN': int(TN),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
    
    # Detection-attribution accuracy.
    # Attribution accuracy among true positives.
    if reason_accumulator['TP_reason_count'] > 0:
        TP_reason_accuracy = reason_accumulator['TP_reason_correct'] / reason_accumulator['TP_reason_count']
    else:
        TP_reason_accuracy = 0.0
    
    # Attribution accuracy among true negatives.
    if reason_accumulator['TN_reason_count'] > 0:
        TN_reason_accuracy = reason_accumulator['TN_reason_correct'] / reason_accumulator['TN_reason_count']
    else:
        TN_reason_accuracy = 0.0
    
    aggregated_metrics['reason'] = {
        'TP_reason_correct': int(reason_accumulator['TP_reason_correct']),
        'TP_reason_count': int(reason_accumulator['TP_reason_count']),
        'TP_reason_accuracy': float(TP_reason_accuracy),
        'TN_reason_correct': int(reason_accumulator['TN_reason_correct']),
        'TN_reason_count': int(reason_accumulator['TN_reason_count']),
        'TN_reason_accuracy': float(TN_reason_accuracy)
    }
    
    # Localization metrics.
    # Parsing success rate.
    if box_accuracy_accumulator['TP_count'] > 0:
        parsing_success_rate = (box_accuracy_accumulator['TP_count'] - 
                               box_accuracy_accumulator['num_failed_parsing']) / \
                               box_accuracy_accumulator['TP_count']
    else:
        parsing_success_rate = 0.0
    
    # Localization-error statistics.
    if len(box_accuracy_accumulator['all_dist_errors']) > 0:
        avg_dist_error = float(np.mean(box_accuracy_accumulator['all_dist_errors']))
        median_dist_error = float(np.median(box_accuracy_accumulator['all_dist_errors']))
        std_dist_error = float(np.std(box_accuracy_accumulator['all_dist_errors']))
    else:
        avg_dist_error = 0.0
        median_dist_error = 0.0
        std_dist_error = 0.0
    
    # Precision, recall, and F1 at each distance threshold.
    threshold_metrics = {}
    for threshold, counts in box_accuracy_accumulator['threshold_accumulator'].items():
        # Precision.
        if box_accuracy_accumulator['num_total_output_boxes'] > 0:
            threshold_precision = counts['num_matched_output'] / \
                                 box_accuracy_accumulator['num_total_output_boxes']
        else:
            threshold_precision = 0.0
        
        # Recall.
        if box_accuracy_accumulator['num_total_gt_boxes'] > 0:
            threshold_recall = counts['num_matched_gt'] / \
                              box_accuracy_accumulator['num_total_gt_boxes']
        else:
            threshold_recall = 0.0
        
        # F1 score.
        if (threshold_precision + threshold_recall) > 0:
            threshold_f1 = 2 * threshold_precision * threshold_recall / \
                          (threshold_precision + threshold_recall)
        else:
            threshold_f1 = 0.0
        
        threshold_metrics[float(threshold)] = {
            'num_matched_output': int(counts['num_matched_output']),
            'num_matched_gt': int(counts['num_matched_gt']),
            'precision': float(threshold_precision),
            'recall': float(threshold_recall),
            'f1': float(threshold_f1)
        }
    
    aggregated_metrics['multiple_box_accuracy'] = {
        'num_samples': int(box_accuracy_accumulator['num_samples']),
        'TP_count': int(box_accuracy_accumulator['TP_count']),
        'num_failed_parsing': int(box_accuracy_accumulator['num_failed_parsing']),
        'parsing_success_rate': float(parsing_success_rate),
        'num_total_gt_boxes': int(box_accuracy_accumulator['num_total_gt_boxes']),
        'num_total_output_boxes': int(box_accuracy_accumulator['num_total_output_boxes']),
        'num_dist_errors': len(box_accuracy_accumulator['all_dist_errors']),
        'avg_dist_error': float(avg_dist_error),
        'median_dist_error': float(median_dist_error),
        'std_dist_error': float(std_dist_error),
        'threshold_metrics': threshold_metrics
    }
    
    return aggregated_metrics

def aggregate_all_configs(
    per_sample_file,
    output_file,
    qa_type_id,
    experiment_group,
):
    """Aggregate all samples by complete configuration, without category subdivision."""

    if qa_type_id in (1, 4):
        aggregate_fn = aggregate_eval_results_Q1_4
        required_fields = (
            'object_existance',
            'reason',
            'multiple_box_accuracy',
        )
    elif qa_type_id == 5:
        aggregate_fn = aggregate_eval_results_Q5
        required_fields = ('eval_result',)
    else:
        raise ValueError('qa_type_id must be 1, 4 or 5')

    buckets = {}

    with open(per_sample_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            sample = json.loads(line)

            if sample.get('experiment_group', experiment_group) != experiment_group:
                raise ValueError('Mixed experiment groups in per-sample file')

            if any(field not in sample for field in required_fields):
                raise ValueError(
                    f"Missing per-sample metrics: id={sample.get('id')}"
                )

            setting = sample['fedattn_setting']
            key = json.dumps(
                setting,
                sort_keys=True,
                separators=(',', ':'),
            )

            if key not in buckets:
                buckets[key] = {
                    'fedattn_setting': setting,
                    'samples': [],
                }

            buckets[key]['samples'].append(sample)

    if not buckets:
        raise ValueError(f'No per-sample records: {per_sample_file}')

    result = {
        'task': f'q{qa_type_id}',
        'experiment_group': experiment_group,
        'configs': [],
    }

    for bucket in buckets.values():
        samples = bucket['samples']
        setting = bucket['fedattn_setting']

        # Aggregate all samples belonging to this complete configuration.
        result['configs'].append({
            'fedattn_setting': setting,
            'num_samples': len(samples),
            'all': aggregate_fn(samples),
        })

        print(
            f"{experiment_group} | "
            f"{setting['config_name']} | "
            f"{setting['com_ratio_name']} | "
            f"n={len(samples)}"
        )

    os.makedirs(
        os.path.dirname(os.path.abspath(output_file)),
        exist_ok=True,
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            to_jsonable(result),
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f'Aggregated {len(buckets)} configurations: {output_file}')
    return result

''' ==================== main ==================== '''
if __name__ == "__main__":
  
    # Task-specific checkpoint and dataset settings.
    QUESTION_CONFIGS = {
        'q1': {
            'model_suffix': 'v2vllmq1_10ep_both_shallow_f1',
            'checkpoint': 1387,
            'question_file': 'v2v4real_3d_grounding_qa_dataset_v2vllmq1.json',
            'num_input_frames': 1,
            'batch_size': 9,
        },
        'q4': {
            'model_suffix': 'v2vllmq4_10ep_both_shallow_f1',
            'checkpoint': 490,
            'question_file': 'v2v4real_3d_grounding_qa_dataset_v2vllmq4.json',
            'num_input_frames': 1,
            'batch_size': 9,
        },
        
        'q5': {
            'model_suffix': 'v2vllmq5_10ep_both_shallow_f2',
            'checkpoint': 490,
            'question_file': 'v2v4real_3d_grounding_qa_dataset_v2vllmq5.json',
            'num_input_frames': 2, 
            'batch_size': 9,
        },
    }


    # ============================================================
    # Select the task and experiment group.
    # ============================================================
    QUESTION_ID = 1                  # 1, 4, or 5
    ACTIVE_QUESTION = f'q{QUESTION_ID}'
    EXPERIMENT_GROUP = 'H'          # H, A, B, C, or D

    # ============================================================
    
    # Command-line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--simplified", default=True)
    parser.add_argument("--multiple-output", default=True)
    parser.add_argument("--max-num-answer-objects", type=int, default=3)     
    parser.add_argument("--qa-type-id", type=int, default=QUESTION_ID)
    parser.add_argument("--num-future-waypoints", type=int, default=6)
    parser.add_argument("--npy-save-path", type=str, default=NPY_ROOT)
    args = parser.parse_args()
    QUESTION_ID = args.qa_type_id
    ACTIVE_QUESTION = f'q{QUESTION_ID}'
    config = QUESTION_CONFIGS[ACTIVE_QUESTION]

    if QUESTION_ID in [1,4,5]:
        
        print(f"\n{'='*60}")
        print(f"🎯 Running Question Set: {ACTIVE_QUESTION}")
        print(f"   Model: {config['model_suffix']}")
        print(f"   Checkpoint: {config['checkpoint']}")
        print(f"   Question File: {config['question_file']}")
        print(f"   Input Frames: {config['num_input_frames']}")
        print(f"{'='*60}\n")       
        
        """Per-sample evaluation"""  
        
        # Output directories.
        output_file_eval_per_sample = os.path.join(
            PER_SAMPLE_V2V_LLM_DIR,
            EXPERIMENT_GROUP,
            f"v2v_llm_{ACTIVE_QUESTION}.jsonl",
        )

        output_file_eval_aggregate = os.path.join(
            AGGREGATE_V2V_LLM_DIR,
            EXPERIMENT_GROUP,
            f"v2v_llm_{ACTIVE_QUESTION}.json",
        )

        os.makedirs(
            os.path.dirname(output_file_eval_per_sample),
            exist_ok=True,
        )

        root_dir = os.path.join(
            RAW_OUTPUT_ROOT,
            f"v2v4real_3d_grounding_{config['model_suffix']}_{config['checkpoint']}",
            "answers",
            "val",
            "llava-v1.5-7b",
            EXPERIMENT_GROUP,
        )

        if DEMO_MODE:
            if ACTIVE_QUESTION != 'q1' or EXPERIMENT_GROUP != 'H':
                raise ValueError('DEMO_MODE supports only q1 with EXPERIMENT_GROUP="H"')
            root_dir = os.path.join(PROJECT_ROOT, "demo", "q1_evaluation", "input")

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"找不到推理结果目录: {root_dir}")      
                     
        qa_tag = f"q{args.qa_type_id}".lower()
        
        all_data = []
        selected_files = []
        seen_records = {}
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames.sort()
            for fn in sorted(filenames):
                fn_lower = fn.lower()
                if not fn_lower.endswith(".jsonl"):
                    continue   
                if "completed" in fn_lower:
                    continue                    
                if qa_tag not in fn_lower:
                    continue
        
                path = os.path.join(dirpath, fn)
                selected_files.append(path)
        
                with open(path, "r", encoding="utf-8") as f:
                    for line_number, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try: 
                            # Parse the prediction record.
                            item = json.loads(line)
                            
                            # Extract fields that identify a sample and configuration.
                            uid = item['id']
                            fed_setting = item['fedattn_setting']

                            setting_key = json.dumps(
                                fed_setting,
                                sort_keys=True,
                                separators=(',', ':'),
                            )
                            
                            # Construct the deduplication key.
                            unique_key = (
                                EXPERIMENT_GROUP,
                                uid,
                                setting_key,
                            )
                            
                            # Reject conflicting predictions with the same key.
                            if unique_key in seen_records:
                                if seen_records[unique_key] != item:
                                    raise ValueError(
                                        f"Conflicting duplicate prediction: {path}:{line_number}, "
                                        f"id={uid}, config={fed_setting.get('config_name')}"
                                    )
                                continue

                            seen_records[unique_key] = item.copy()

                            # Record the experiment group for aggregation.
                            item['experiment_group'] = EXPERIMENT_GROUP
                            all_data.append(item)
                                
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"Invalid JSON in prediction file: {path}:{line_number}"
                            ) from exc
        
        print(f"递归读取目录: {root_dir}")
        print(f"匹配到 {len(selected_files)} 个 jsonl 文件（包含 {qa_tag}）")
        print(f"总样本数: {len(all_data)}")         

        if not all_data:
            raise ValueError(f"No prediction records found for {qa_tag}: {root_dir}")

        # Per-sample evaluation.
        if args.qa_type_id in [1, 4]:                 
        
            # Evaluate each sample and write its metrics.
            with open(output_file_eval_per_sample, "w", encoding="utf-8") as fout:
                for data_sample in tqdm(all_data, desc="Evaluating samples"):
                    
                    data = [data_sample]
                    
                    # 1) Parse ground-truth answer types.
                    # gt_answer_types: (num_samples, 2)
                    # [:, 0] = answer subtype (0-7).
                    # [:, 1] = object presence (1: present, 0: absent).
                    gt_answer_types = np.stack([
                        classify_answer_type(d['conversations'][1]['value'])  # Ground-truth answer text.
                        for d in data
                    ], axis=0)
                    
                    
                    
                    # 2) Parse ground-truth object locations.
                    # gt_boxes: (num_samples, max_num_answer_objects, 3)
                    # [:, :, 0] = x coordinate.
                    # [:, :, 1] = z coordinate in the answer format.
                    # [:, :, 2] = validity flag (1: valid, 0: padding).
                    gt_boxes = [
                        parse_multiple_box_parameters(
                            d['conversations'][1]['value'],  # Ground-truth answer text.
                            args.simplified,                  # Parse [x, z] when simplified=True.
                            args.max_num_answer_objects      # Maximum number of answer objects.
                        ) 
                        for d in data
                    ]
                    gt_boxes = np.stack(gt_boxes, axis=0)
                    
                    
                    # 3) Parse predicted answer types.
                    output_answer_types = np.stack([
                        classify_answer_type(d['outputs'])  # Predicted answer text.
                        for d in data
                    ], axis=0)
                    
                    # 4) Parse predicted object locations.
                    output_boxes = [
                        parse_multiple_box_parameters(
                            d['outputs'],                    # Predicted answer text.
                            args.simplified,
                            args.max_num_answer_objects
                        ) 
                        for d in data
                    ]
                    output_boxes = np.stack(output_boxes, axis=0)
                    
                    
                    # 5) Evaluate presence, attribution, and localization.
                    # Binary object-presence classification.
                    object_existance = evaluate_object_existance(gt_answer_types, output_answer_types, flag_return_results = True)
                    
                    # Four-way attribution: both vehicles, ego, CAV 1, or neither.
                    reason = evaluate_reason(gt_answer_types, output_answer_types, flag_return_results = True)
                    
                    # Object-localization accuracy based on center distance.
                    multiple_box_accuracy = evaluate_multiple_box_accuracy(
                        gt_answer_types, 
                        gt_boxes, 
                        output_answer_types, 
                        output_boxes, 
                        args.simplified, 
                        args.max_num_answer_objects,
                        flag_return_results = True
                    )
                    
                    if multiple_box_accuracy == None:
                        continue
                    
                    data_sample["object_existance"] = object_existance
                    data_sample["reason"] = reason
                    data_sample["multiple_box_accuracy"] = multiple_box_accuracy                           
                                
                    fout.write(json.dumps(to_jsonable(data_sample), ensure_ascii=False) + "\n")
            
            print(f"saved: {output_file_eval_per_sample}")
        
        elif args.qa_type_id in [5]:
    
            """Trajectory prediction evaluation"""
            # Evaluate each sample and write its metrics.
            with open(output_file_eval_per_sample, "w", encoding="utf-8") as fout:
                for data_sample in tqdm(all_data, desc="Evaluating samples"):

                    eval_result = evaluate_future_trajectory_per_sample(
                        num_future_waypoints = args.num_future_waypoints,
                        data_sample = data_sample,
                        npy_save_path = args.npy_save_path,
                        qa_type_id = args.qa_type_id
                    )
                    data_sample["eval_result"] = eval_result
        
                    fout.write(json.dumps(data_sample, ensure_ascii=False) + "\n")
            
            print(f"saved: {output_file_eval_per_sample}")  

        # Aggregate all samples by complete experiment configuration.
        aggregate_all_configs(
            per_sample_file=output_file_eval_per_sample,
            output_file=output_file_eval_aggregate,
            qa_type_id=args.qa_type_id,
            experiment_group=EXPERIMENT_GROUP,
        )                  
                            