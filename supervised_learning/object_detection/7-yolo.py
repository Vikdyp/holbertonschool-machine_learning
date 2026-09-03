#!/usr/bin/env python3
"""YOLO v3 object detection and image processing."""

import os

import cv2
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
        input_w, input_h = self.model.input_shape[1:3]
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Keep boxes whose best class score reaches the threshold."""
        filtered_boxes, box_classes, box_scores = [], [], []
        for box, confidence, probabilities in zip(
                boxes, box_confidences, box_class_probs):
            scores = confidence * probabilities
            classes = np.argmax(scores, axis=-1)
            scores = np.max(scores, axis=-1)
            keep = scores >= self.class_t
            filtered_boxes.append(box[keep])
            box_classes.append(classes[keep])
            box_scores.append(scores[keep])
        return (np.concatenate(filtered_boxes), np.concatenate(box_classes),
                np.concatenate(box_scores))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Suppress overlapping boxes separately for each class."""
        selected = []
        for class_index in np.unique(box_classes):
            indices = np.flatnonzero(box_classes == class_index)
            indices = indices[np.argsort(-box_scores[indices], kind='stable')]
            while indices.size:
                best = indices[0]
                selected.append(best)
                indices = indices[1:]
                if not indices.size:
                    break
                box = filtered_boxes[best]
                others = filtered_boxes[indices]
                upper_left = np.maximum(box[:2], others[:, :2])
                lower_right = np.minimum(box[2:], others[:, 2:])
                overlap = np.maximum(lower_right - upper_left, 0)
                intersection = overlap[:, 0] * overlap[:, 1]
                area = np.prod(np.maximum(box[2:] - box[:2], 0))
                areas = np.prod(np.maximum(others[:, 2:] - others[:, :2], 0),
                                axis=1)
                union = area + areas - intersection
                iou = np.divide(intersection, union,
                                out=np.zeros_like(intersection, dtype=float),
                                where=union > 0)
                indices = indices[iou < self.nms_t]
        selected = np.array(selected, dtype=int)
        return (filtered_boxes[selected], box_classes[selected],
                box_scores[selected])

    @staticmethod
    def load_images(folder_path):
        """Load readable images and their corresponding paths from a folder."""
        images, image_paths = [], []
        for file_name in sorted(os.listdir(folder_path)):
            path = os.path.join(folder_path, file_name)
            if not os.path.isfile(path):
                continue
            image = cv2.imread(path)
            if image is not None:
                images.append(image)
                image_paths.append(path)
        return images, image_paths

    def preprocess_images(self, images):
        """Resize images with cubic interpolation and normalize pixels."""
        input_h, input_w = self.model.input_shape[1:3]
        pimages, image_shapes = [], []
        for image in images:
            image_shapes.append(image.shape[:2])
            resized = cv2.resize(image, (input_w, input_h),
                                 interpolation=cv2.INTER_CUBIC)
            pimages.append(resized / 255.0)
        pimages = np.asarray(pimages).reshape((-1, input_h, input_w, 3))
        image_shapes = np.asarray(image_shapes, dtype=int).reshape((-1, 2))
        return pimages, image_shapes

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Display detections, saving them only when the s key is pressed."""
        image = image.copy()
        for box, class_index, score in zip(boxes, box_classes, box_scores):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = '{} {:.2f}'.format(self.class_names[class_index], score)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(file_name, image)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            os.makedirs('detections', exist_ok=True)
            path = os.path.join('detections', os.path.basename(file_name))
            cv2.imwrite(path, image)
        cv2.destroyWindow(file_name)

    def predict(self, folder_path):
        """Detect and display objects in each image, preserving path order."""
        images, image_paths = self.load_images(folder_path)
        if not images:
            return [], image_paths
        pimages, image_shapes = self.preprocess_images(images)
        outputs = self.model.predict(pimages)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        predictions = []
        for index, image in enumerate(images):
            image_outputs = [output[index] for output in outputs]
            boxes, confidences, probabilities = self.process_outputs(
                image_outputs, image_shapes[index])
            filtered = self.filter_boxes(boxes, confidences, probabilities)
            prediction = self.non_max_suppression(*filtered)
            predictions.append(prediction)
            self.show_boxes(image, *prediction,
                            os.path.basename(image_paths[index]))
        return predictions, image_paths
