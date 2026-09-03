#!/usr/bin/env python3
"""Neural style transfer with a frozen VGG19 feature extractor."""

import numpy as np
import tensorflow as tf


class NST:
    """Optimize an image to match reference content and style features."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Validate references and prepare the features for optimization."""
        for name, image in [('style_image', style_image),
                            ('content_image', content_image)]:
            if (not isinstance(image, np.ndarray) or image.ndim != 3
                    or image.shape[-1] != 3):
                raise TypeError(name + ' must be a numpy.ndarray with '
                                'shape (h, w, 3)')
        for name, weight in [('alpha', alpha), ('beta', beta)]:
            if not isinstance(weight, (int, float)) or weight < 0:
                raise TypeError(name + ' must be a non-negative number')
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Resize the longest side to 512 pixels and scale into [0, 1]."""
        if (not isinstance(image, np.ndarray) or image.ndim != 3
                or image.shape[-1] != 3):
            raise TypeError('image must be a numpy.ndarray with shape '
                            '(h, w, 3)')
        height, width = image.shape[:2]
        scale = 512 / max(height, width)
        size = (int(height * scale), int(width * scale))
        resized = tf.image.resize(image, size, method='bicubic')
        return tf.expand_dims(tf.clip_by_value(resized / 255, 0, 1), 0)
