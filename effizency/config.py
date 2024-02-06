import os
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import tensorflow as tf
from detectron2.config import get_cfg
from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator, build_anchor_generator


def generate_anchors(feature_maps):
    detectron_cfg = get_cfg()
    detectron_cfg.merge_from_file('model/config.yml')
    anchor_generator = build_anchor_generator(detectron_cfg, input_shape=[ShapeSpec(channels=256, stride=2 ** i)
                                                                          for i in range(2, 7)])

    anchors = anchor_generator([np.random.random((1, 256, *fm)) for fm in feature_maps])
    anchors = tf.concat([a.tensor for a in anchors], axis=0)
    return anchors


def load_od_config() -> Dict[str, Any]:
    root = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(root, 'project_config.yaml')
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    config['anchor_matcher']['thresholds'] = [float(i) for i in config['anchor_matcher']['thresholds']]
    config['proposal_matcher']['thresholds'] = [float(i) for i in config['proposal_matcher']['thresholds']]

    config['id_to_label_name'] = {v: k for k, v in config['label_name_to_id'].items()}
    config['anchors'] = generate_anchors(config['feature_maps'])
    return config


CONFIG = load_od_config()
