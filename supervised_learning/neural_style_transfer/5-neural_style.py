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
        self.load_model()
        self.generate_features()

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

    def load_model(self):
        """Extract VGG19 features with average pooling and frozen weights."""
        base = tf.keras.applications.VGG19(include_top=False,
                                           weights='imagenet')
        base.trainable = False
        output = base.input
        features = {}
        for layer in base.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                output = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size, strides=layer.strides,
                    padding=layer.padding, name=layer.name)(output)
            else:
                output = layer(output)
            if layer.name in self.style_layers + [self.content_layer]:
                features[layer.name] = output
            if layer.name == self.content_layer:
                break
        outputs = [features[name] for name in self.style_layers]
        outputs.append(features[self.content_layer])
        self.model = tf.keras.Model(base.input, outputs)
        self.model.trainable = False

    @staticmethod
    def gram_matrix(input_layer):
        """Average channel correlations over the spatial dimensions."""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or input_layer.shape.rank != 4):
            raise TypeError('input_layer must be a tensor of rank 4')
        shape = tf.shape(input_layer)
        features = tf.reshape(input_layer, (shape[0], -1, shape[3]))
        gram = tf.matmul(features, features, transpose_a=True)
        return gram / tf.cast(shape[1] * shape[2], input_layer.dtype)

    def generate_features(self):
        """Cache the style Gram matrices and reference content features."""
        preprocess = tf.keras.applications.vgg19.preprocess_input
        style_outputs = self.model(preprocess(self.style_image * 255))
        content_outputs = self.model(preprocess(self.content_image * 255))
        self.gram_style_features = [self.gram_matrix(output)
                                    for output in style_outputs[:-1]]
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Measure the mean squared discrepancy between Gram matrices."""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or style_output.shape.rank != 4):
            raise TypeError('style_output must be a tensor of rank 4')
        channels = style_output.shape[-1]
        target_shape = [1, channels, channels]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != target_shape):
            raise TypeError('gram_target must be a tensor of shape '
                            + str(target_shape))
        gram = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram - gram_target))

    def style_cost(self, style_outputs):
        """Average the style discrepancy across all configured layers."""
        count = len(self.style_layers)
        if not isinstance(style_outputs, list) or len(style_outputs) != count:
            raise TypeError('style_outputs must be a list with a length of '
                            + str(count))
        costs = [self.layer_style_cost(output, target)
                 for output, target in zip(style_outputs,
                                           self.gram_style_features)]
        return tf.add_n(costs) / count
