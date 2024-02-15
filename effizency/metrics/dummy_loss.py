import tensorflow as tf


def predictions_listing(pred_boxes, scores, class_ids, pred_masks):
    pred_boxes = tf.cast(tf.reduce_sum(pred_boxes), dtype=tf.float32)
    scores = tf.cast(tf.reduce_sum(scores), dtype=tf.float32)
    class_ids = tf.cast(tf.reduce_sum(class_ids), dtype=tf.float32)
    pred_masks = tf.cast(tf.reduce_sum(pred_masks), dtype=tf.float32)
    z = tf.convert_to_tensor(0.0, dtype=tf.float32)
    return pred_boxes * scores * class_ids * pred_masks * z


def zero_loss(gt_boxes: tf.Tensor, gt_polygons: tf.Tensor,
              pred_objectness_logits: tf.Tensor, pred_anchor_deltas: tf.Tensor,
              cls_loss_predictions: tf.Tensor, box_loss_predictions: tf.Tensor,
              proposal_boxes: tf.Tensor, proposal_logits: tf.Tensor, mask_features: tf.Tensor) -> tf.Tensor:
    return tf.convert_to_tensor(tf.reduce_sum(proposal_logits) * 0.0, dtype=tf.float32)
