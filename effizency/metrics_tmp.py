from typing import Tuple, List, Union, Dict

import numpy as np
import tensorflow as tf
from code_loader.helpers.detection.yolo.utils import reshape_output_list

from code_loader.contract.responsedataclasses import BoundingBox

# todo

def get_argmax_map_and_separate_masks(image, bbs, masks):
    image_size = image.shape[:2]
    argmax_map = np.zeros(image_size, dtype=np.uint8)
    cats_dict = {}
    separate_masks = []
    for bb, mask in zip(bbs, masks):
        if mask.shape != image_size:
            resize_mask = tf.image.resize(mask[..., None], image_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
            if not isinstance(resize_mask, np.ndarray):
                resize_mask = resize_mask.numpy()
        else:
            resize_mask = mask
        resize_mask = resize_mask.astype(bool)
        label = bb.label
        instance_number = cats_dict.get(label, 0)
        # update counter if reach max instances we treat the last objects as one
        cats_dict[label] = instance_number + 1 if instance_number < CONFIG["MAX_INSTANCES_PER_CLASS"] else instance_number
        label_index = CONFIG["CATEGORIES"].index(label) * CONFIG["MAX_INSTANCES_PER_CLASS"] + cats_dict[label]
        if label == 'Tote':
            empty = argmax_map == 0
            tote = (argmax_map >= CONFIG["CATEGORIES"].index(label) * CONFIG["MAX_INSTANCES_PER_CLASS"]) &\
                   (argmax_map < CONFIG["CATEGORIES"].index(label) * (CONFIG["MAX_INSTANCES_PER_CLASS"]+1))
            argmax_map[(empty | tote) & resize_mask] = label_index
        else:
            argmax_map[resize_mask] = label_index
        if bb.label == 'Object':
            separate_masks.append(resize_mask)
    argmax_map[argmax_map == 0] = len(CONFIG['INSTANCES']) + 1
    argmax_map -= 1
    return {"argmax_map": argmax_map, "separate_masks": separate_masks}

def get_mask_list(data, masks, is_gt):
    is_inference = CONFIG["MODEL_FORMAT"] == "inference"
    if is_gt:
        bb_object, mask_list = bb_array_to_object(data, iscornercoded=False, bg_label=CONFIG["BACKGROUND_LABEL"],
                                                  is_gt=True,
                                                  masks=masks)
    else:
        from_logits = not is_inference
        decoded = is_inference
        class_list_reshaped, loc_list_reshaped = reshape_output_list(
            np.reshape(data, (1, *data.shape)), decoded=decoded, image_size=CONFIG["IMAGE_SIZE"])
        outputs = DECODER(loc_list_reshaped,
                          class_list_reshaped,
                          DEFAULT_BOXES,
                          from_logits=from_logits,
                          decoded=decoded,
                          )
        bb_object, mask_list = bb_array_to_object(outputs[0], iscornercoded=True, bg_label=CONFIG["BACKGROUND_LABEL"],
                                                  masks=masks)
    return bb_object, mask_list


def ioa_mask(mask_containing, mask_contained):
    """
    Calculates the Intersection over Area (IOA) between two binary masks.

    Args:
        mask_containing (ndarray or Tensor): Binary mask representing the containing object.
        mask_contained (ndarray or Tensor): Binary mask representing the contained object.

    Returns:
        float: The IOA (Intersection over Area) value between the two masks.

    Note:
        - The input masks should have compatible shapes.
        - The function performs a bitwise AND operation between the 'mask_containing' and 'mask_contained' masks to obtain
          the intersection mask.
        - It calculates the number of True values in the intersection mask to determine the intersection area.
        - The area of the contained object is computed as the number of True values in the 'mask_contained' mask.
        - If the area of the contained object is 0, the IOA is defined as 0.
        - The IOA value is calculated as the ratio of the intersection area to the maximum of the area of the contained
          object or 1.
    """

    intersection_mask = mask_containing & mask_contained
    intersection = len(intersection_mask[intersection_mask])
    area = len(mask_contained[mask_contained])
    return intersection / max(area, 1)


def segmentation_metrics_dict(image: tf.Tensor, y_pred_bb: tf.Tensor, y_pred_mask: tf.Tensor, bb_gt: tf.Tensor,
                              mask_gt: tf.Tensor) -> Dict[str, Union[int, float]]:
    bs = bb_gt.shape[0]
    bb_mask_gt = [get_mask_list(bb_gt[i, ...], mask_gt[i, ...], is_gt=True) for i in range(bs)]
    bb_mask_pred = [get_mask_list(y_pred_bb[i, ...], y_pred_mask[i, ...], is_gt=False) for i in range(bs)]
    sep_mask_pred = [get_argmax_map_and_separate_masks(image[i, ...], bb_mask_pred[i][0],
                                                       bb_mask_pred[i][1])['separate_masks'] for i in range(bs)]
    sep_mask_gt = [get_argmax_map_and_separate_masks(image[i, ...], bb_mask_gt[i][0],
                                                     bb_mask_gt[i][1])['separate_masks'] for i in range(bs)]
    pred_gt_ioas = [np.array([[ioa_mask(pred_mask, gt_mask) for gt_mask in sep_mask_gt[i]]
                              for pred_mask in sep_mask_pred[i]]) for i in range(bs)]
    gt_pred_ioas = [np.array([[ioa_mask(gt_mask, pred_mask) for gt_mask in sep_mask_gt[i]]
                              for pred_mask in sep_mask_pred[i]]) for i in range(bs)]
    gt_pred_ioas_t = [arr.transpose() for arr in gt_pred_ioas]
    over_seg_bool, over_seg_count, avg_segments_over, _, over_conf = \
        over_under_segmented_metrics(gt_pred_ioas_t, get_avg_confidence=True, bb_mask_object_list=bb_mask_pred)
    under_seg_bool, under_seg_count, avg_segments_under, under_small_bb, _ = \
        over_under_segmented_metrics(pred_gt_ioas, count_small_bbs=True, bb_mask_object_list=bb_mask_gt)
    res = {
        "Over_Segmented_metric": over_seg_bool,
        "Under_Segmented_metric": under_seg_bool,
        "Small_BB_Under_Segmtented": under_small_bb,
        "Over_Segmented_Instances_count": over_seg_count,
        "Under_Segmented_Instances_count": under_seg_count,
        "Average_segments_num_Over_Segmented": avg_segments_over,
        "Average_segments_num_Under_Segmented": avg_segments_under,
        "Over_Segment_confidences": over_conf
    }
    return res



def over_under_segmented_metrics(batched_ioas_list: List[np.ndarray], count_small_bbs=False, get_avg_confidence=False,
                                 bb_mask_object_list: List[Union[List[BoundingBox], List[np.ndarray]]] = None):
    th = 0.8
    segmented_arr = [0.]*len(batched_ioas_list)
    segmented_arr_count = [0.]*len(batched_ioas_list)
    average_segments_amount = [0.]*len(batched_ioas_list)
    conf_arr = [0.]*len(batched_ioas_list)
    has_small_bbs = [0.]*len(batched_ioas_list)
    for batch in range(len(batched_ioas_list)):
        ioas = batched_ioas_list[batch]
        if len(ioas) > 0:
            th_arr = ioas > th
            matches_count = th_arr.astype(int).sum(axis=-1)
            is_over_under_segmented = float(len(matches_count[matches_count > 1]) > 0)
            over_under_segmented_count = float(len(matches_count[matches_count > 1]))
            if over_under_segmented_count > 0:
                average_segments_num_over_under = float(matches_count[matches_count > 1].mean())
            else:
                average_segments_num_over_under = 0.
            average_segments_amount[batch] = average_segments_num_over_under
            segmented_arr_count[batch] = over_under_segmented_count
            segmented_arr[batch] = is_over_under_segmented
            if count_small_bbs or get_avg_confidence:
                relevant_bbs = np.argwhere(matches_count > 1)[..., 0]  # [Indices of bbs]
                relevant_gts = np.where(np.any(th_arr[relevant_bbs], axis=0))[0]  # [Indices of gts]
                if count_small_bbs:
                    new_gt_objects = remove_label_from_bbs(bb_mask_object_list[batch][0], "Tote", "gt")
                    new_bb_array = [new_gt_objects[i] for i in relevant_gts]
                    for j in range(len(new_bb_array)):
                        if new_bb_array[j].width * new_bb_array[j].height < CONFIG["SMALL_BBS_TH"]:
                            has_small_bbs[batch] = 1.
                if get_avg_confidence:
                    new_bb_pred_object = remove_label_from_bbs(bb_mask_object_list[batch][0], "Tote", "pred")
                    new_bb_array = [new_bb_pred_object[j] for j in relevant_gts]
                    if len(new_bb_array) > 0:
                        avg_conf = np.array([new_bb_array[j].confidence for j in range(len(new_bb_array))]).mean()
                    else:
                        avg_conf = 0.
                    conf_arr[batch] = avg_conf
    return tf.convert_to_tensor(segmented_arr), tf.convert_to_tensor(segmented_arr_count),\
           tf.convert_to_tensor(average_segments_amount), tf.convert_to_tensor(has_small_bbs),\
           tf.convert_to_tensor(conf_arr)
