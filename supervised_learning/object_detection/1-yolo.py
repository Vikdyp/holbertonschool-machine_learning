#!/usr/bin/env python3
"""YOLO v3 object detection and image processing."""

import numpy as np
from tensorflow import keras as K


class Yolo:
    """Decode and display detections from a Darknet model."""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Load the detector and retain its classes, thresholds and anchors."""
        self.model = K.models.load_model(model_path)
        with open(classes_path, encoding='utf-8') as classes_file:
            self.class_names = [line.strip() for line in classes_file]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Decode model logits into original-image boxes and probabilities."""
        input_h, input_w = self.model.input_shape[1:3]
        image_h, image_w = image_size
        boxes, box_confidences, box_class_probs = [], [], []
        for index, output in enumerate(outputs):
            grid_h, grid_w = output.shape[:2]
            grid_x = np.arange(grid_w)[None, :, None]
            grid_y = np.arange(grid_h)[:, None, None]
            center_x = (1 / (1 + np.exp(-output[..., 0])) + grid_x) / grid_w
            center_y = (1 / (1 + np.exp(-output[..., 1])) + grid_y) / grid_h
            width = (np.exp(output[..., 2]) * self.anchors[index, :, 0]
                     / input_w)
            height = (np.exp(output[..., 3]) * self.anchors[index, :, 1]
                      / input_h)
            box = np.empty_like(output[..., :4])
            box[..., 0] = (center_x - width / 2) * image_w
            box[..., 1] = (center_y - height / 2) * image_h
            box[..., 2] = (center_x + width / 2) * image_w
            box[..., 3] = (center_y + height / 2) * image_h
            boxes.append(box)
            box_confidences.append(1 / (1 + np.exp(-output[..., 4:5])))
            box_class_probs.append(1 / (1 + np.exp(-output[..., 5:])))
        return boxes, box_confidences, box_class_probs
