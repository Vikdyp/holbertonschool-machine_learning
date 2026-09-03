#!/usr/bin/env python3
"""Check configurable feature selection with the Keras boundary isolated."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class Layer:
    """Represent a named layer while recording feature traversal."""

    def __init__(self, name):
        """Retain the layer name."""
        self.name = name

    def __call__(self, value):
        """Append this layer to the feature path."""
        return value + '>' + self.name


class FeatureSelectionTests(unittest.TestCase):
    """Feature selection must be independent of content-layer position."""

    def test_early_content_layer_preserves_later_style_outputs(self):
        """A content tap must not terminate extraction of later style taps."""
        base = SimpleNamespace(input='input', layers=[
            Layer('input'), Layer('early'), Layer('late')])
        keras = SimpleNamespace(
            applications=SimpleNamespace(VGG19=lambda **kwargs: base),
            layers=SimpleNamespace(MaxPooling2D=type('MaxPooling2D', (), {})),
            Model=lambda inputs, outputs: SimpleNamespace(
                inputs=inputs, outputs=outputs))
        path = Path(__file__).resolve().parents[1] / 'supervised_learning'
        path = path / 'neural_style_transfer/10-neural_style.py'
        spec = importlib.util.spec_from_file_location('tested_nst', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {
                'tensorflow': SimpleNamespace(keras=keras)}):
            spec.loader.exec_module(module)
        nst = module.NST.__new__(module.NST)
        nst.style_layers = ['late']
        nst.content_layer = 'early'
        nst.load_model()
        self.assertEqual(nst.model.outputs,
                         ['input>early>late', 'input>early'])
        self.assertEqual(nst.model.inputs, 'input')
        self.assertFalse(nst.model.trainable)
        self.assertFalse(base.trainable)


if __name__ == '__main__':
    unittest.main()
