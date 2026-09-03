#!/usr/bin/env python3
"""YOLO v3 object detection and image processing."""

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
