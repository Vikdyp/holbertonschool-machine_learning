#!/usr/bin/env python3
"""Test real NumPy/OpenCV operations with the model loader isolated."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / 'supervised_learning'


class YoloTests(unittest.TestCase):
    """Exercise geometry and image operations without loading TensorFlow."""

    @classmethod
    def setUpClass(cls):
        """Load the class, substituting only the model-loading boundary."""
        filename = PROJECT / 'object_detection/7-yolo.py'
        spec = importlib.util.spec_from_file_location('yolo_tested', filename)
        module = importlib.util.module_from_spec(spec)
        cls.load_model = Mock()
        keras = SimpleNamespace(models=SimpleNamespace(
            load_model=cls.load_model))
        with patch.dict('sys.modules', {
                'tensorflow': SimpleNamespace(keras=keras)}):
            spec.loader.exec_module(module)
        cls.Yolo = module.Yolo

    def setUp(self):
        """Use rectangular model inputs to expose swapped dimensions."""
        self.yolo = self.Yolo.__new__(self.Yolo)
        self.yolo.model = SimpleNamespace(input_shape=(None, 40, 80, 3))
        self.yolo.anchors = np.array([[[20, 10]], [[40, 20]]])
        self.yolo.class_t = 0.25
        self.yolo.nms_t = 0.5

    def test_constructor_reads_names_and_keeps_configuration(self):
        """Constructor data must come from the supplied files and arguments."""
        with TemporaryDirectory() as folder:
            classes = Path(folder) / 'classes.txt'
            classes.write_text('person\nbicycle\n', encoding='utf8')
            yolo = self.Yolo('model.h5', str(classes), .3, .6,
                             self.yolo.anchors)
        self.load_model.assert_called_with('model.h5')
        self.assertEqual(yolo.class_names, ['person', 'bicycle'])
        self.assertEqual((yolo.class_t, yolo.nms_t), (.3, .6))
        self.assertIs(yolo.anchors, self.yolo.anchors)

    def test_decode_rectangular_grid_and_multiple_outputs(self):
        """Decode with the exercise's width-first model-dimension contract."""
        outputs = [np.zeros((2, 4, 1, 7)), np.zeros((1, 1, 1, 7))]
        boxes, confidences, probabilities = self.yolo.process_outputs(
            outputs, np.array([160, 320]))
        np.testing.assert_allclose(boxes[0][0, 0, 0], [-40, 30, 120, 50])
        np.testing.assert_allclose(boxes[0][1, 3, 0],
                                   [200, 110, 360, 130])
        np.testing.assert_allclose(boxes[1][0, 0, 0], [0, 60, 320, 100])
        np.testing.assert_allclose(confidences[0], .5)
        np.testing.assert_allclose(probabilities[1], .5)
        outputs[0][0, 0, 0, 2:4] = np.log([2, 3])
        boxes, _, _ = self.yolo.process_outputs(outputs, [160, 320])
        np.testing.assert_allclose(boxes[0][0, 0, 0], [-120, 10, 200, 70])

    def test_filter_threshold_and_empty_result_shapes(self):
        """Scores multiply object confidence by the strongest class score."""
        boxes = [np.arange(12).reshape(1, 3, 1, 4)]
        confidence = [np.array([.5, .5, .8]).reshape(1, 3, 1, 1)]
        probabilities = [np.array([[.5, .4], [.3, .6], [.1, .2]])
                         .reshape(1, 3, 1, 2)]
        filtered, classes, scores = self.yolo.filter_boxes(
            boxes, confidence, probabilities)
        np.testing.assert_array_equal(filtered, np.arange(8).reshape(2, 4))
        np.testing.assert_array_equal(classes, [0, 1])
        np.testing.assert_allclose(scores, [.25, .3])
        self.yolo.class_t = 1.0
        empty = self.yolo.filter_boxes(boxes, confidence, probabilities)
        self.assertEqual([x.shape for x in empty], [(0, 4), (0,), (0,)])

    def test_nms_is_per_class_and_orders_scores_within_class(self):
        """Overlapping boxes in different classes must survive suppression."""
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10],
                          [1, 1, 11, 11], [20, 20, 30, 30]], dtype=float)
        classes = np.array([1, 0, 0, 0])
        scores = np.array([.9, .7, .8, .6])
        result, categories, confidence = self.yolo.non_max_suppression(
            boxes, classes, scores)
        np.testing.assert_array_equal(result, boxes[[2, 3, 0]])
        np.testing.assert_array_equal(categories, [0, 0, 1])
        np.testing.assert_allclose(confidence, [.8, .6, .9])
        empty = self.yolo.non_max_suppression(
            np.empty((0, 4)), np.empty(0, dtype=int), np.empty(0))
        self.assertEqual([x.shape for x in empty], [(0, 4), (0,), (0,)])

    def test_loading_and_rectangular_image_preprocessing(self):
        """Real OpenCV data must stay paired with paths and original shapes."""
        with TemporaryDirectory() as folder:
            first = np.full((3, 7, 3), [0, 128, 255], dtype=np.uint8)
            second = np.full((9, 4, 3), [255, 0, 128], dtype=np.uint8)
            cv2.imwrite(str(Path(folder) / 'a.png'), first)
            cv2.imwrite(str(Path(folder) / 'b.png'), second)
            images, paths = self.Yolo.load_images(folder)
            self.assertEqual({Path(p).name for p in paths}, {'a.png', 'b.png'})
            for image, path in zip(images, paths):
                np.testing.assert_array_equal(image, cv2.imread(path))
            resized, shapes = self.yolo.preprocess_images(images)
        self.assertEqual(resized.shape, (2, 40, 80, 3))
        for index, image in enumerate(images):
            np.testing.assert_array_equal(shapes[index], image.shape[:2])
            np.testing.assert_allclose(resized[index, 20, 40],
                                       image[0, 0] / 255.0)

    def test_prediction_pairs_model_outputs_with_original_images(self):
        """A single-output model produces one result per original image."""
        images = [np.zeros((20, 40, 3), dtype=np.uint8),
                  np.zeros((40, 80, 3), dtype=np.uint8)]
        paths = ['samples/a.png', 'samples/b.png']
        self.yolo.load_images = Mock(return_value=(images, paths))
        self.yolo.show_boxes = Mock()
        self.yolo.model.predict = Mock(return_value=np.zeros((2, 1, 1, 1, 7)))
        predictions, returned_paths = self.yolo.predict('samples')
        self.assertEqual(returned_paths, paths)
        np.testing.assert_allclose(predictions[0][0], [[10, 8.75, 30, 11.25]])
        np.testing.assert_allclose(predictions[1][0], [[20, 17.5, 60, 22.5]])
        self.assertEqual(self.yolo.show_boxes.call_count, 2)
        self.assertEqual([call.args[-1] for call in
                          self.yolo.show_boxes.call_args_list],
                         ['a.png', 'b.png'])


if __name__ == '__main__':
    unittest.main()
