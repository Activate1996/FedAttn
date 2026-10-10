# -*- coding: utf-8 -*-
"""Per-sample evaluation utilities for cooperative-driving tasks Q1, Q4, and Q5."""
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import numpy as np

# Resolve paths relative to this file; LLaVA and DMSTrack are siblings.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
os.chdir(SCRIPT_DIR)

sys.path.append(os.path.join(PROJECT_ROOT, 'DMSTrack').replace("\\", "/"))
sys.path.append(os.path.join(PROJECT_ROOT, 'DMSTrack/AB3DMOT').replace("\\", "/"))
sys.path.append(os.path.join(PROJECT_ROOT, 'DMSTrack/AB3DMOT/Xinshuo_PyToolbox').replace("\\", "/"))
sys.path.append(os.path.join(PROJECT_ROOT, 'DMSTrack/V2V4Real').replace("\\", "/"))

from AB3DMOT.AB3DMOT_libs.box import Box3D
from AB3DMOT_libs.dist_metrics import iou
from V2V4Real.opencood.utils.transformation_utils import x1_to_x2
from V2V4Real.opencood.utils.box_utils import corner_to_center, project_points_by_matrix_torch


def parse_multiple_box_parameters(text, simplified, max_num_answer_objects, skip_dummy_boxes=False):
    """Parse up to max_num_answer_objects bracketed bounding boxes.

    Simplified boxes use [x, z, valid]; full boxes use
    [h, w, l, x, y, z, a, valid]. Invalid or missing boxes have a zero
    validity flag. With skip_dummy_boxes=True, return only valid boxes
    or None when no box can be parsed."""
    all_boxes = []  # Parsed bounding boxes.
    
    # Locate the first bracketed box.
    start_idx = text.find('[')
    end_idx = text.find(']')
    remaining_text = text[start_idx:]  # Text beginning at the first opening bracket.

    # Parse up to max_num_answer_objects boxes.
    for i in range(max_num_answer_objects):
        # Parse one box.
        box = parse_box_parameters(remaining_text, simplified)
        all_boxes.append(box)

        # Pad the remaining entries after an invalid box.
        if box[-1] == 0:
            remaining_text = ''    
        else:  
            # Advance to the next box.
            start_idx = remaining_text.find('[')
            end_idx = remaining_text.find(']')
            remaining_text = remaining_text[end_idx+1:]  # Discard the parsed text.

    # Stack into shape (max_num_answer_objects, 8 or 3).
    all_boxes = np.stack(all_boxes, axis=0)  

    if not skip_dummy_boxes:
        return all_boxes  # Include invalid padding boxes.
    else:
        # Retain boxes whose validity flag is nonzero.
        final_all_boxes = []
        for i in range(all_boxes.shape[0]):
            if all_boxes[i][-1] != 0:
                final_all_boxes.append(all_boxes[i])    
        if len(final_all_boxes) > 0:
            final_all_boxes = np.stack(final_all_boxes)
            return final_all_boxes
        else:
            return None  # No valid boxes were parsed.


def parse_box_parameters(text, simplified):
    """Parse one bracketed box and append its validity flag.

    Simplified inputs contain [x, z]; full inputs contain
    [h, w, l, x, y, z, a]. Invalid inputs return a padding box with
    a zero validity flag."""
    # Initialize an invalid padding box.
    # Padding coordinates are located at the origin.
    # The final validity flag excludes padding from matching.
    if not simplified:
        num_parameters = 7  # h, w, l, x, y, z, a
        dummy_box = np.array([1, 1, 1, 0, 0, 0, 0.0, 0])  # A zero final flag denotes an invalid box.
    else:  
        num_parameters = 2  # x, z
        dummy_box = np.array([0.0, 0, 0])  # A zero final flag denotes an invalid box.

    # Locate the opening and closing brackets.
    start_idx = text.find('[')
    end_idx = text.find(']')

    # Require both brackets in the expected order.
    if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
        return dummy_box  # Return padding if parsing fails.

    # Split the bracketed values at commas.
    box_str = text[start_idx + 1 : end_idx].strip()
    box_str = box_str.split(',')

    # Check the expected number of parameters.
    if len(box_str) != num_parameters:
        return dummy_box

    # Convert coordinate strings to floating-point values.
    try:
        box_str = [float(s.strip()) for s in box_str]
    except:
        return dummy_box  # Return padding if conversion fails.

    # Append a validity flag of one.
    box = np.array(box_str + [1])

    # Full 3D boxes require positive height, width, and length.
    if not simplified:
        if box[0] <= 0 or box[1] <= 0 or box[2] <= 0:  # Require h, w, and l to be positive.
            return dummy_box
    
    return box


def eval_multiple_center_dist_2d(gt_boxes, output_boxes, simplified, flag_return_results=False): 
    """Evaluate center-distance coverage for true-positive samples.

    Input arrays contain padded boxes and a final validity flag.
    Compute nearest-center errors and precision, recall, and F1 from
    pairs within each distance threshold. Matching uses coverage
    in each direction, without a one-to-one assignment. Set
    flag_return_results=True to return statistics for aggregation."""
    
    # 1) Expand ground-truth boxes for pairwise distance computation.
    # Input shape: (num_samples, max_num_answer_objects, 3).
    # Expanded shape: (num_samples, max_num_answer_objects, 1, 3).
    # Broadcast against the predicted boxes.
    gt_boxes = np.expand_dims(gt_boxes, axis=2)
    
    # Count valid ground-truth boxes across all samples.
    # The final coordinate stores the validity flag.
    # A flag of one denotes a valid box.
    num_total_gt_boxes = np.sum(gt_boxes[:, :, :, -1] == 1)  # Integer scalar.
    

    # 2) Expand predicted boxes for broadcasting.
    # Input shape: (num_samples, max_num_answer_objects, 3).
    # Expanded shape: (num_samples, 1, max_num_answer_objects, 3).
    # The singleton axis broadcasts across ground-truth boxes.
    output_boxes = np.expand_dims(output_boxes, axis=1)
    
    # Count valid predicted boxes.
    num_total_output_boxes = np.sum(output_boxes[:, :, :, -1] == 1)  # Integer scalar.
    

    # 3) Compute pairwise 2D center distances.
    # Broadcasting dimensions:
    #   gt_boxes:     (num_samples, max_num_gt_objects,     1, 3)
    #   output_boxes: (num_samples, 1,                      max_num_output_objects, 3)
    # Result: (num_samples, max_num_gt_objects, max_num_output_objects).
    # Each entry is a ground-truth/prediction center distance.
    
    if not simplified:
        # Full box format: x at index 3 and z at index 5.
        center_dist_2d = np.sqrt(
            (output_boxes[:, :, :, 3] - gt_boxes[:, :, :, 3]) ** 2 +  # Difference in x.
            (output_boxes[:, :, :, 5] - gt_boxes[:, :, :, 5]) ** 2    # Difference in z.
        )
    else:
        # Simplified format: x at index 0 and z at index 1.
        center_dist_2d = np.sqrt(
            (output_boxes[:, :, :, 0] - gt_boxes[:, :, :, 0]) ** 2 +  # Difference in x.
            (output_boxes[:, :, :, 1] - gt_boxes[:, :, :, 1]) ** 2    # Difference in z.
        )
    # center_dist_2d shape: (num_samples, max_num_gt, max_num_output)

    # 4) Identify valid ground-truth/prediction pairs.
    # Both boxes must have nonzero validity flags.
    # valid_mask shape: (num_samples, max_num_gt, max_num_output)
    valid_mask = np.logical_and(
        gt_boxes[:, :, :, -1],      # Ground-truth validity flags.
        output_boxes[:, :, :, -1]   # Prediction validity flags.
    )

    # 5) Exclude invalid pairs from nearest-neighbor distances.
    # Copy the distance matrix.
    center_dist_2d_padded_max = center_dist_2d.copy()
    # Assign a large distance if either box is padding.
    # These pairs cannot determine the nearest valid match.
    center_dist_2d_padded_max[np.logical_not(valid_mask)] = 1e5
    
    # 6) Find each predicted box's nearest ground-truth center.
    dist_errors = []  # Distances for valid predicted boxes.
    
    # Iterate over samples.
    for i in range(center_dist_2d_padded_max.shape[0]):
        # Iterate over predicted boxes.
        for output_idx in range(center_dist_2d_padded_max.shape[2]):
            # Require a valid prediction and a valid first ground-truth box.
            if valid_mask[i, 0, output_idx]:
                # Take the minimum distance across ground-truth boxes.
                # This column contains distances to all ground-truth boxes.
                dist_error = np.min(center_dist_2d_padded_max[i, :, output_idx]).item()
                dist_errors.append(dist_error)
                
                # Reject an output with no valid ground-truth match.
                if dist_error > 1e4:
                    print("输出框有效但找不到有效GT框，数据异常")
                    return None
                    
    
    # Mean nearest-center distance.
    if len(dist_errors) > 0:
        avg_dist_error = sum(dist_errors) / len(dist_errors)
    else:
        avg_dist_error = 0.0
    

    # 7) Compute precision, recall, and F1 at each distance threshold.
    threshold_stats = {}  # Threshold-specific statistics.
    
    for threshold in [0.5, 1, 2, 4]:  # Distance thresholds in meters.
        
        
        # 7.1) Mark pairs within the distance threshold.
        # A pair qualifies when its distance is less than or equal to the threshold.
        distance_less_than_threshold = center_dist_2d <= threshold
        # Retain only pairs of valid boxes.
        distance_less_than_threshold = np.logical_and(
            distance_less_than_threshold, 
            valid_mask
        )
        # shape: (num_samples, max_num_gt, max_num_output)

        # 7.2) Count matched predictions for precision.
        # A prediction matches if at least one ground-truth box is within the threshold.
        # Reduce over the ground-truth dimension.
        output_close_to_at_least_one_gt = np.any(
            distance_less_than_threshold, 
            axis=1  # Ground-truth box dimension.
        )
        # shape: (num_samples, max_num_output)
        
        # Count predictions with at least one match.
        num_matched_output = np.sum(output_close_to_at_least_one_gt)  # Integer scalar.

        # 7.3) Count matched ground-truth boxes for recall.
        # A ground-truth box matches if at least one prediction is within the threshold.
        # Reduce over the prediction dimension.
        gt_close_to_at_least_one_output = np.any(
            distance_less_than_threshold, 
            axis=2  # Predicted-box dimension.
        )
        # shape: (num_samples, max_num_gt)
        
        # Count ground-truth boxes with at least one match.
        num_match_gt = np.sum(gt_close_to_at_least_one_output)  # Integer scalar.

        # 7.4) Compute precision, recall, and F1.
        # Precision = matched predictions / all valid predictions.
        # The fraction of valid predictions covered by ground truth.
        if num_total_output_boxes > 0:
            precision = 1.0 * num_matched_output / num_total_output_boxes
        else:
            precision = 0.0
        
        
        # Recall = matched ground-truth boxes / all valid ground-truth boxes.
        # The fraction of valid ground-truth boxes covered by predictions.
        if num_total_gt_boxes > 0:
            recall = 1.0 * num_match_gt / num_total_gt_boxes
        else:
            recall = 0.0
        
        
        # F1 = 2 * P * R / (P + R)
        # Harmonic mean of precision and recall.
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        
        # Store statistics for this threshold.
        threshold_stats[threshold] = {
            'num_matched_output': int(num_matched_output),  # Number of matched predictions.
            'num_matched_gt': int(num_match_gt),            # Number of matched ground-truth boxes.
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    # 8) Return per-sample statistics for aggregation.
    if flag_return_results:
        results_dict = {
            'num_total_gt_boxes': int(num_total_gt_boxes),
            'num_total_output_boxes': int(num_total_output_boxes),
            'dist_errors': dist_errors,  # list of float
            'avg_dist_error': float(avg_dist_error),
            'threshold_stats': threshold_stats  # dict of dict
        }
        return results_dict
    else:
        print('num_total_gt_boxes: ', num_total_gt_boxes)
        print('num_total_output_boxes: ', num_total_output_boxes)        
        print('avg_dist_error: ', avg_dist_error)
        print('threshold_stats: ', threshold_stats)
    
        return None


def eval_iou_3d(gt_boxes, output_boxes):
    """Print 2D and 3D IoU accuracy for aligned bounding-box pairs.

    Input boxes follow the [h, w, l, x, y, z, a] convention."""
    num_boxes = gt_boxes.shape[0]

    # Add epsilon to avoid numerical failures for nearly identical boxes.
    output_boxes += 1e-7

    # Convert boxes to the AB3DMOT representation.
    # Box3D argument order: (x, y, z, h, w, l, ry, s).
    # Input order: [h, w, l, x, y, z, a].
    gt_boxes = [Box3D(b[3], b[4], b[5], b[0], b[1], b[2], b[6], 1) for b in gt_boxes]
    output_boxes = [Box3D(b[3], b[4], b[5], b[0], b[1], b[2], b[6], 1) for b in output_boxes]

    # Compute 2D and 3D intersection over union.
    for iou_type in ['iou_2d', 'iou_3d']:
        # Compute IoU for each aligned pair.
        iou_nd = [iou(output_boxes[i], gt_boxes[i], iou_type) for i in range(num_boxes)]
        iou_nd = sorted(iou_nd)  # Sort IoU values.
        iou_nd = np.array(iou_nd)
        
        # Count pairs with negligible overlap.
        num_zero_iou = np.sum(iou_nd < 1e-7)
        print('num_zero_iou: ', num_zero_iou)
        
        # Compute accuracy at each IoU threshold.
        for threshold in [0.5, 0.7]:
            num_matches = np.sum(iou_nd > threshold)
            accuracy = 1.0 * num_matches / num_boxes
            print('Accuracy: %.4f @ %s threshold %.1f' % (accuracy, iou_type, threshold))

    return


def classify_answer_type(answer):
    """Parse answer attribution and object presence from task templates.

    Return [qa_sub_type, object_exist]. Positive subtypes 0-3 denote
    both vehicles, ego, CAV 1, and neither vehicle; negative subtypes
    4-7 use the same order. An unrecognized attribution has subtype -1.
    Object presence is zero only when the answer contains "There is no"."""
    # Parse object presence.
    if 'There is no' in answer:  # The answer uses the negative-presence template.
        object_exist = 0  # Object absent.
        
        # Parse the vehicle attribution for the negative answer.
        if 'both connected autonomous vehicles' in answer:
            qa_sub_type = 4  # Both vehicles.
        elif 'connected autonomous vehicle ego' in answer:
            qa_sub_type = 5  # Ego vehicle.
        elif 'connected autonomous vehicle 1' in answer:
            qa_sub_type = 6  # CAV 1.
        elif 'None of the connected autonomous vehicles' in answer:
            qa_sub_type = 7  # Neither vehicle.
        else:
            qa_sub_type = -1  # Unrecognized attribution.
            
    else:  # Answers without the negative template are classified as positive.
        object_exist = 1  # Object present.
        
        # Parse the vehicle attribution for the positive answer.
        if 'Both connected autonomous vehicles' in answer:
            qa_sub_type = 0  # Both vehicles detect the object.
        elif 'Connected autonomous vehicle ego' in answer:
            qa_sub_type = 1  # Only ego detects the object.
        elif 'Connected autonomous vehicle 1' in answer:
            qa_sub_type = 2  # Only CAV 1 detects the object.
        elif 'None of the connected autonomous vehicles' in answer:
            qa_sub_type = 3  # Neither vehicle detects the object.
        else:
            qa_sub_type = -1  # Unrecognized attribution.

    return np.array([qa_sub_type, object_exist])


def evaluate_object_existance(gt_answer_types, output_answer_types, flag_return_results = False):
    """Evaluate binary object presence.

    Each input has shape (num_samples, 2), with presence in column 1.
    Return confusion-matrix counts when flag_return_results=True;
    otherwise print accuracy and positive-class precision, recall, and F1."""
    # Extract object-presence labels.
    gt_object_exist = gt_answer_types[:, 1]      # Ground truth: 1 = present, 0 = absent.
    output_object_exist = output_answer_types[:, 1]  # Prediction: 1 = present, 0 = absent.

    # Compute overall binary accuracy.
    # Binary classification accuracy.
    accuracy = np.sum(gt_object_exist == output_object_exist) / gt_object_exist.shape[0]

    # Compute the confusion-matrix counts.
    # True positives.
    TP = np.sum(np.logical_and(gt_object_exist, output_object_exist))
    
    # False positives.
    FP = np.sum(np.logical_and(np.logical_not(gt_object_exist), output_object_exist))
    
    # False negatives.
    FN = np.sum(np.logical_and(gt_object_exist, np.logical_not(output_object_exist)))
    
    # True negatives.
    TN = np.sum(np.logical_and(np.logical_not(gt_object_exist), np.logical_not(output_object_exist)))

    if flag_return_results:

        return {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN
            }
    
    else:

        print('accuracy: ', accuracy)  # Binary classification accuracy.
        print('TP: ', TP)
        print('FP: ', FP)
        print('FN: ', FN)
        print('TN: ', TN)
            
        # Compute precision, recall, and F1 for the positive class.
        precision = 1.0 * TP / (TP + FP)  # Fraction of positive predictions that are correct.
        recall = 1.0 * TP / (TP + FN)     # Fraction of positive ground-truth labels recovered.
        f1 = 2 * precision * recall / (precision + recall)  # Harmonic mean.
        print('precision: ', precision)
        print('recall: ', recall)
        print('f1: ', f1)
        
        return        


def evaluate_reason(gt_answer_types, output_answer_types, flag_return_results = False):
    """Evaluate vehicle attribution conditional on correct object presence.

    Compare answer subtypes separately among true positives and
    true negatives. Return counts for aggregation when requested."""
    # Extract object-presence labels.
    gt_object_exist = gt_answer_types[:,1]
    output_object_exist = output_answer_types[:,1]

    # Select samples positive in both ground truth and prediction.
    TP_mask = np.logical_and(gt_object_exist, output_object_exist)  # True-positive mask.
    TP_count = np.sum(TP_mask)  # True-positive count.
    
    # Select samples negative in both ground truth and prediction.
    TN_mask = np.logical_and(
        np.logical_not(gt_object_exist), 
        np.logical_not(output_object_exist)
    )  # True-negative mask.
    TN_count = np.sum(TN_mask)  # True-negative count.

    if flag_return_results:
        
        # Compare attribution subtypes among true positives.
        TP_correct_reason_count = np.sum(
            gt_answer_types[TP_mask][:, 0] == output_answer_types[TP_mask][:, 0]
        )
    
        # Compare attribution subtypes among true negatives.
        TN_correct_reason_count = np.sum(
            gt_answer_types[TN_mask][:, 0] == output_answer_types[TN_mask][:, 0]
        )

        return {           
            "TP_reason_correct": int(TP_correct_reason_count),
            "TP_reason_count": int(TP_count),
            "TN_reason_correct": int(TN_correct_reason_count),
            "TN_reason_count": int(TN_count),
            }

    else:
        
        # Compare attribution subtypes among true positives.
        correct_reason_count = np.sum(
            gt_answer_types[TP_mask][:, 0] == output_answer_types[TP_mask][:, 0]
        )
        accuracy = 1.0 * correct_reason_count / TP_count  # Attribution accuracy among true positives.
        print('TP reason accuracy: ', accuracy)
    
    
        # Compare attribution subtypes among true negatives.
        correct_reason_count = np.sum(
            gt_answer_types[TN_mask][:, 0] == output_answer_types[TN_mask][:, 0]
        )
        accuracy = 1.0 * correct_reason_count / TN_count  # Attribution accuracy among true negatives.
        print('TN reason accuracy: ', accuracy)

        return


def evaluate_multiple_box_accuracy(gt_answer_types, gt_boxes, output_answer_types, 
                                   output_boxes, simplified, max_num_answer_objects, 
                                   flag_return_results=False):
    """Evaluate localization for true-positive samples.

    Answer-type arrays have shape (num_samples, 2). Simplified box
    arrays have shape (num_samples, max_num_answer_objects, 3) with
    [x, z, valid] entries. Return parsing and center-distance statistics
    when flag_return_results=True. Full 3D IoU is not enabled here."""
    
    # 1) Select the coordinate representation.
    if not simplified:
        num_parameters = 7  # Full 3D box: [h, w, l, x, y, z, a].
    else:
        num_parameters = 2  # Simplified center: [x, z].
    
    # 2) Select true-positive samples.
    # Column 1 stores the object-presence label.
    gt_object_exist = gt_answer_types[:, 1]       # shape: (num_samples,)
    output_object_exist = output_answer_types[:, 1]  # shape: (num_samples,)
    
    # Both ground truth and prediction must indicate presence.
    TP_mask = np.logical_and(gt_object_exist, output_object_exist)  # Boolean array of shape (num_samples,).
    TP_count = np.sum(TP_mask)  # Number of true-positive samples.
    
    # 3) Count prediction-parsing failures.
    # A true-positive sample fails parsing if all predicted boxes are invalid.
    # output_boxes[TP_mask][:, :, -1]: shape (TP_count, max_num_answer_objects)
    # The last coordinate is the validity flag.
    # Sum validity flags across predicted boxes.
    # A zero sum means that every predicted box is padding.
    num_failed_parsing = np.sum(
        np.sum(output_boxes[TP_mask][:, :, -1], axis=1) == 0
    )  # Number of samples with no valid predicted box.
    
    # Parsing success = (true positives - parsing failures) / true positives.
    # This rate is defined at the sample level.
    if TP_count > 0:  # Handle an empty true-positive set.
        parsing_success_rate = 1.0 * (TP_count - num_failed_parsing) / TP_count
    else:
        parsing_success_rate = 0.0
    
    # 4) Extract boxes for true-positive samples.
    # Apply the true-positive mask to both box arrays.
    gt_boxes_tp = gt_boxes[TP_mask]       # shape: (TP_count, max_num_answer_objects, 3)
    output_boxes_tp = output_boxes[TP_mask]  # shape: (TP_count, max_num_answer_objects, 3)
    
    # 5) Evaluate center distances and threshold-based matching.
    # The center-distance evaluator returns:
    # 1. Mean nearest-center distance.
    # 2. Precision, recall, and F1 at 0.5, 1, 2, and 4 meters.
    results = eval_multiple_center_dist_2d(
        gt_boxes_tp,      # Ground-truth boxes for true-positive samples.
        output_boxes_tp,  # Predicted boxes for true-positive samples.
        simplified,       # Coordinate representation.
        flag_return_results  # Return statistics for aggregation.
    )
    if results == None:
        return None
    
    # 6) Full 3D IoU evaluation is not enabled in this path.
    if not simplified:
        print('not implemented')
        assert False
        
        eval_iou_3d(gt_boxes_tp, output_boxes_tp)
    
    # 7) Return per-sample statistics.
    if flag_return_results:
        # Construct the result dictionary.
        results_dict = {
            # Sample-level counts.
            'num_samples': len(gt_answer_types),  # Number of samples in the batch.
            'TP_count': int(TP_count),            # Number of true positives.
            'num_failed_parsing': int(num_failed_parsing),  # Number of parsing failures.
            
            # Object-level counts from the center-distance evaluator.
            'num_total_gt_boxes': results['num_total_gt_boxes'],      # Number of valid ground-truth boxes.
            'num_total_output_boxes': results['num_total_output_boxes'],  # Number of valid predicted boxes.
            
            # Nearest-center distance errors.
            'dist_errors': results['dist_errors'],  # One distance per valid predicted box.
            
            # Matching statistics at each threshold.
            'threshold_stats': results['threshold_stats']  # Dictionary indexed by distance threshold.
            # Format: {0.5: {'num_matched_output': x, 'num_matched_gt': y}, ...}.
        }
        return results_dict
    else:
    
        print('num_failed_parsing: ', num_failed_parsing)        
        print('parsing_success_rate: ', parsing_success_rate)  
        print('gt_boxes[TP_mask].shape: ', gt_boxes_tp.shape)
        print('output_boxes[TP_mask].shape: ', output_boxes_tp.shape)  
        
        return None


def check_has_collision(future_cav_location_in_current_ego_coordinate, global_timestamp_index, 
                        future_global_timestamp_index, npy_save_path, asker_cav_id):
    """Test whether a future CAV position intersects a ground-truth box.

    Transform the position from the current ego frame into the future
    ego frame, compare 3D boxes, and exclude the querying CAV itself.
    Pose and ground-truth arrays are loaded from npy_save_path."""
    # Load current and future ego LiDAR poses.
    current_ego_lidar_pose = np.load(os.path.join(npy_save_path, 'ego', 
                                      '%04d_lidar_pose.npy' % global_timestamp_index))
    future_ego_lidar_pose = np.load(os.path.join(npy_save_path, 'ego', 
                                     '%04d_lidar_pose.npy' % future_global_timestamp_index))

    # Transform the current ego frame into the future ego frame.
    current_ego_pose_in_future_coordinate = x1_to_x2(current_ego_lidar_pose, future_ego_lidar_pose)
    future_ego_pose_in_future_coordinate = current_ego_pose_in_future_coordinate.copy()
    future_ego_pose_in_future_coordinate[0, 3] += future_cav_location_in_current_ego_coordinate[0]
    future_ego_pose_in_future_coordinate[1, 3] += future_cav_location_in_current_ego_coordinate[1]
    
    # Use fixed CAV dimensions.
    future_ego_box = np.array([
        1.5, 2, 4,  # h, w, l
        future_ego_pose_in_future_coordinate[0, 3], 
        future_ego_pose_in_future_coordinate[1, 3], 0,  # x, y, z
        0  # Yaw angle.
    ])
    future_cav_box = future_ego_box

    # Transform the future CAV position into the future ego frame.
    transformation_matrix = x1_to_x2(current_ego_lidar_pose, future_ego_lidar_pose)
    future_cav_location_in_current_ego_coordinate = np.expand_dims(
        future_cav_location_in_current_ego_coordinate, axis=0)
    # Append a zero z coordinate.
    future_cav_location_in_current_ego_coordinate = np.pad(
        future_cav_location_in_current_ego_coordinate, [[0, 0], [0, 1]])
    # Apply the coordinate transformation.
    future_cav_location_in_future_ego_coordinate = project_points_by_matrix_torch(
        future_cav_location_in_current_ego_coordinate, transformation_matrix)
    future_cav_location_in_future_ego_coordinate = future_cav_location_in_future_ego_coordinate[0, :2]
    
    # Construct the future CAV box.
    future_cav_box = np.array([
        1.5, 2, 4,  # h, w, l
        future_cav_location_in_future_ego_coordinate[0], 
        future_cav_location_in_future_ego_coordinate[1], 0,  # x, y, z
        0  # a
    ])

    # Locate the querying CAV at the future time to exclude self-collisions.
    
    future_asker_lidar_pose = np.load(os.path.join(npy_save_path, asker_cav_id, 
                                       '%04d_lidar_pose.npy' % future_global_timestamp_index))
    future_asker_to_ego_transformation_matrix = x1_to_x2(future_asker_lidar_pose, future_ego_lidar_pose)
    cav_asker_gt_future_in_future_ego_coordinate = np.array([
        future_asker_to_ego_transformation_matrix[0, 3], 
        future_asker_to_ego_transformation_matrix[1, 3]])

    # Load future ground-truth boxes.
    # V2V4Real axes: x forward, y right, and z upward.
    gt_boxes_in_future_coordinate = np.load(os.path.join(npy_save_path, 
                                             '%04d_gt.npy' % future_global_timestamp_index))
    # Convert corners to (N, 7) centers: [x, y, z, h, w, l, theta].
    gt_boxes_in_future_coordinate = corner_to_center(gt_boxes_in_future_coordinate, order='hwl')
    # Reorder to [h, w, l, x, y, z, rot_y].
    gt_boxes_in_future_coordinate = np.concatenate([
        gt_boxes_in_future_coordinate[:, 3:6], 
        gt_boxes_in_future_coordinate[:, 0:3], 
        gt_boxes_in_future_coordinate[:, 6:7]], 
        axis=1)
    
    # Detect collisions from positive 3D IoU.
    # Convert to AB3DMOT Box3D objects.
    gt_boxes_ab3dmot = [Box3D(b[3], b[4], b[5], b[0], b[1], b[2], b[6], 1) 
                        for b in gt_boxes_in_future_coordinate]
    future_cav_box_ab3dmot = [Box3D(b[3], b[4], b[5], b[0], b[1], b[2], b[6], 1) 
                              for b in [future_cav_box]]
    
    for i in range(gt_boxes_in_future_coordinate.shape[0]):
        iou_type = 'iou_3d'  
        iou_nd = iou(future_cav_box_ab3dmot[0], gt_boxes_ab3dmot[i], iou_type)
        if iou_nd > 0:
            # Exclude the ego vehicle's own ground-truth box.
            if asker_cav_id == 'ego':
                if np.linalg.norm([gt_boxes_in_future_coordinate[i][3], 
                                   gt_boxes_in_future_coordinate[i][4]]) < 2:
                    continue  # Skip the querying vehicle itself.
                else:
                    return True  # Collision with another object.
            
            # Exclude the querying collaborator's own ground-truth box.
            if asker_cav_id == '1' or asker_cav_id == '2':
                if np.linalg.norm([gt_boxes_in_future_coordinate[i][3] - 
                                   cav_asker_gt_future_in_future_ego_coordinate[0], 
                                   gt_boxes_in_future_coordinate[i][4] - 
                                   cav_asker_gt_future_in_future_ego_coordinate[1]]) < 2:
                    continue  # Skip the querying vehicle itself.
                else:
                    return True

            assert False  # Unsupported querying CAV.

    return False  # No collision.


def evaluate_future_trajectory_per_sample(num_future_waypoints, data_sample, npy_save_path, qa_type_id):
    """Evaluate one predicted trajectory.

    Return mean waypoint L2 errors, cumulative collision indicators,
    a time-to-collision indicator, comfort components, and the local
    composite driving score. The evaluation uses six future waypoints
    sampled at 0.5 second intervals."""
    
    # Comfort thresholds for the trajectory metrics.
    # Reference: https://github.com/motional/nuplan-devkit/blob/d60b4cd2071de9bb041509c43f5226dd22f248c0/docs/metrics_description.md
    min_lon_accel = -4.05  # Minimum longitudinal acceleration (m/s^2).
    max_lon_accel = 2.40   # Maximum longitudinal acceleration (m/s^2).
    max_abs_lat_accel = 4.89  # Maximum absolute lateral acceleration (m/s^2).
    max_abs_yaw_accel = 1.93  # Maximum absolute yaw acceleration (rad/s^2).
    max_abs_yaw_rate = 0.95   # Maximum absolute yaw rate (rad/s).
    max_abs_lon_jerk = 4.13   # Maximum absolute longitudinal jerk (m/s^3).
    max_abs_mag_jerk = 8.37   # Maximum jerk magnitude (m/s^3).
    
    result = {}
    
    # Use ego when the querying CAV is unspecified.
    if 'asker_cav_id' not in data_sample:
        data_sample['asker_cav_id'] = 'ego'  

    gt_answer = data_sample['conversations'][1]['value']  
    # Parse ground-truth future waypoints.
    gt_future_waypoints = parse_planned_future_trajectory(gt_answer, num_future_waypoints, 
                                                          prepend_current_position=False) 
    # Parse predicted future waypoints.
    output_future_waypoints = parse_planned_future_trajectory(data_sample['outputs'], 
                                                               num_future_waypoints, 
                                                               prepend_current_position=False)

    # For a collaborator query, transform Q5 waypoints into the ego frame.
    if qa_type_id == 5 and 'asker_cav_id' in data_sample and \
       (data_sample['asker_cav_id'] == '1' or data_sample['asker_cav_id'] == '2'):
        gt_future_waypoints = transform_waypoints_from_1_to_ego(gt_future_waypoints, data_sample)
        output_future_waypoints = transform_waypoints_from_1_to_ego(output_future_waypoints, data_sample)


    # Mean waypoint L2 errors over the 1, 2, and 3 second horizons.
    l2_error = np.linalg.norm(output_future_waypoints - gt_future_waypoints, axis=1)
    result['l2_error_1s'] = float(np.average(l2_error[:2]))
    result['l2_error_2s'] = float(np.average(l2_error[:4]))
    result['l2_error_3s'] = float(np.average(l2_error))
    result['l2_error_per_waypoint'] = l2_error.tolist()
    

    # Trajectory safety and comfort components.
    # Express the querying CAV's current position in the ego frame.
    current_position_in_cav_asker = np.array([[0, 0]])
    if 'asker_cav_id' in data_sample and (data_sample['asker_cav_id'] != 'ego'):
        current_position_in_cav_ego = transform_waypoints_from_1_to_ego(
            current_position_in_cav_asker, data_sample)
    else:  
        current_position_in_cav_ego = current_position_in_cav_asker  

    # Prepend the current position to waypoints already expressed in the ego frame.
    ttc_output_future_waypoints = np.concatenate([
        current_position_in_cav_ego,
        output_future_waypoints  # Waypoints in the ego coordinate frame.
    ], axis=0)
    
    
    
    
    
    

    # Comfort computation.
    # Estimate velocities using the 0.5 second waypoint interval.
    velocity0 = (ttc_output_future_waypoints[1] - ttc_output_future_waypoints[0]) / 0.5
    velocity1 = (ttc_output_future_waypoints[2] - ttc_output_future_waypoints[1]) / 0.5
    velocity2 = (ttc_output_future_waypoints[3] - ttc_output_future_waypoints[2]) / 0.5
    
    # Estimate accelerations.
    accel0 = (velocity1 - velocity0) / 0.5
    accel1 = (velocity2 - velocity1) / 0.5
    
    # Estimate jerk.
    jerk = (accel1 - accel0) / 0.5

    # Separate longitudinal and lateral components.
    lon_accel = accel0[0]  # Longitudinal x component.
    lat_accel = accel0[1]  # Lateral y component.
    lon_jerk = jerk[0]
    mag_jerk = np.linalg.norm(jerk)

    # Estimate heading, yaw rate, and yaw acceleration.
    angle0 = np.arctan2(velocity0[1], velocity0[0])
    angle1 = np.arctan2(velocity1[1], velocity1[0])
    angle2 = np.arctan2(velocity2[1], velocity2[0])
    yaw_rate0 = (angle1 - angle0) / 0.5
    yaw_rate1 = (angle2 - angle1) / 0.5
    yaw_rate = yaw_rate0
    yaw_accel = (yaw_rate1 - yaw_rate0) / 0.5
    
    # Require all comfort constraints to hold.
    if (lon_accel >= min_lon_accel and lon_accel <= max_lon_accel and 
        np.abs(lat_accel) <= max_abs_lat_accel and 
        np.abs(yaw_accel) <= max_abs_yaw_accel and 
        np.abs(yaw_rate) <= max_abs_yaw_rate and 
        np.abs(lon_jerk) <= max_abs_lon_jerk and 
        np.abs(mag_jerk) <= max_abs_mag_jerk):
        result['comfort'] = 1
    else:
        result['comfort'] = 0

    # Retain the numerical comfort components.
    result['lon_accel'] = float(lon_accel)
    result['lat_accel'] = float(lat_accel)
    result['yaw_rate'] = float(yaw_rate)
    result['yaw_accel'] = float(yaw_accel)
    result['lon_jerk'] = float(lon_jerk)
    result['mag_jerk'] = float(mag_jerk)        

    # Time-to-collision indicator.
    # Project the initial velocity at constant speed.
    # Projection times: 0.3, 0.6, and 0.9 seconds.
    origin = ttc_output_future_waypoints[0]
    velocity = (ttc_output_future_waypoints[1] - ttc_output_future_waypoints[0]) / 0.5
    projected_future_waypoints = np.stack([
        origin + velocity * 0.3, 
        origin + velocity * 0.6, 
        origin + velocity * 0.9 
    ], axis=0)

    local_ttc_collision_count = 0
    for i in range(projected_future_waypoints.shape[0]):
        global_timestamp_index = data_sample['global_timestamp_index']  
        future_global_timestamp_index = global_timestamp_index + 3 * (i+1)  # Three frames correspond to 0.3 seconds.
        future_cav_location_in_current_ego_coordinate = projected_future_waypoints[i]
        has_collision = check_has_collision(future_cav_location_in_current_ego_coordinate, 
                                             global_timestamp_index, 
                                             future_global_timestamp_index, 
                                             npy_save_path, 
                                             data_sample['asker_cav_id'])
        if has_collision:
            local_ttc_collision_count = 1
            break
    result['ttc_collision'] = local_ttc_collision_count
    result['ttc'] = 1 - local_ttc_collision_count


    # Cumulative collision indicators.

    collision_1s = 0
    collision_2s = 0
    collision_3s = 0    
    
    for i in range(num_future_waypoints):
        global_timestamp_index = data_sample['global_timestamp_index']  
        # Map each waypoint to a future frame within the 30-frame horizon.
        future_global_timestamp_index = global_timestamp_index + int(30 / num_future_waypoints) * (i+1)
        future_cav_location_in_current_ego_coordinate = output_future_waypoints[i]
        has_collision = check_has_collision(future_cav_location_in_current_ego_coordinate, 
                                             global_timestamp_index, 
                                             future_global_timestamp_index, 
                                             npy_save_path, 
                                             data_sample['asker_cav_id'])
        if has_collision:    
            if i < 2:  # Collision within 1 second.
                collision_1s = 1
                collision_2s = 1
                collision_3s = 1
                break
            elif i < 4:  # Collision within 2 seconds.
                collision_2s = 1
                collision_3s = 1
                break
            else:  # Collision within 3 seconds.
                collision_3s = 1
                break

    result['collision_1s'] = collision_1s
    result['collision_2s'] = collision_2s
    result['collision_3s'] = collision_3s
    result['nc'] = 1 - collision_3s

    # Local composite driving score.
    # PDMS = NC * (TTC * 5 + C * 2) / 7
    # NC: No Collision, TTC: Time to Collision metric, C: Comfort
    result['pdms'] = result['nc'] * \
                 (result['ttc'] * 5.0 + result['comfort'] * 2.0) / 7.0
     
    return result


def parse_planned_future_trajectory(question, num_future_waypoints, prepend_current_position, 
                                     current_position=np.array([[0, 0]]), return_has_parsing_error=False):
    """Parse a bracketed future trajectory with optional current position."""
    if num_future_waypoints == 0:
        if return_has_parsing_error:  
            return None, False
        else:
            return None

    # Locate the bracketed trajectory.
    start_idx = question.find('[')
    end_idx = question.find(']')

    waypoint_str = question[start_idx+1: end_idx]
    # Convert waypoint parentheses to brackets.
    waypoint_str = waypoint_str.replace('(', '[')
    waypoint_str = waypoint_str.replace(')', ']')

    # Parse the requested number of waypoints.
    planned_future_trajectory = parse_multiple_box_parameters(waypoint_str, True, num_future_waypoints)
    # Check whether any waypoint failed parsing.
    if np.any(planned_future_trajectory[:, 2] == 0):
        has_parsing_error = True
    else:  
        has_parsing_error = False
    planned_future_trajectory = planned_future_trajectory[:, :2]  # Retain x and y coordinates.

    # Optionally prepend the current position.
    if prepend_current_position:
        planned_future_trajectory = np.concatenate([
            current_position,
            planned_future_trajectory
        ], axis=0)

    if not return_has_parsing_error:   
        return planned_future_trajectory
    else:
        return planned_future_trajectory, has_parsing_error


def transform_waypoints_from_1_to_ego(waypoints, data_sample):
    """Transform waypoints from the querying CAV frame into the ego frame."""
    cav_ego_lidar_pose = data_sample['cav_ego_lidar_pose'] 
    # Accept either flattened 4-by-4 matrices or six-element poses.
    if len(cav_ego_lidar_pose) == 16:
        cav_ego_lidar_pose = np.array(cav_ego_lidar_pose).reshape([4, 4])

    # Read the querying CAV's pose.
    if data_sample['asker_cav_id'] == 'ego':
        cav_1_lidar_pose = data_sample['cav_ego_lidar_pose'] 
    elif data_sample['asker_cav_id'] == '1':
        cav_1_lidar_pose = data_sample['cav_1_lidar_pose'] 
    elif data_sample['asker_cav_id'] == '2':
        cav_1_lidar_pose = data_sample['cav_2_lidar_pose'] 
    else:  
        print('assert False asker_cav_id: ', data_sample['asker_cav_id'])  
        assert False

    if len(cav_1_lidar_pose) == 16:
        cav_1_lidar_pose = np.array(cav_1_lidar_pose).reshape([4, 4])
    
    # Compute the frame transformation.
    transformation_matrix = x1_to_x2(cav_1_lidar_pose, cav_ego_lidar_pose)

    # Append a zero z coordinate.
    waypoints = np.pad(waypoints, [[0, 0], [0, 1]])
    # Apply the transformation.
    waypoints = project_points_by_matrix_torch(waypoints, transformation_matrix)
    waypoints = waypoints[:, :2]  # Retain x and y coordinates.

    return waypoints

