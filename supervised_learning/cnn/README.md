# Convolutional Neural Networks

Forward and backward propagation for convolutional and pooling layers, plus a
modified LeNet-5 image classifier.

| File | Purpose |
| --- | --- |
| `0-conv_forward.py` | Convolution, bias, and activation on NHWC image batches |
| `1-pool_forward.py` | Maximum and average pooling |
| `2-conv_backward.py` | Gradients with respect to inputs, kernels, and biases |
| `3-pool_backward.py` | Accumulated gradients through pooling windows |
| `5-lenet5.py` | Compiled Keras classifier for 28 by 28 grayscale images |

The NumPy exercises use symmetric zero padding for `same` convolutions, following
the preceding convolution exercises, and no padding for `valid` convolutions.
Pooling uses only complete windows. All tensors use channels-last layout.

LeNet-5 uses two convolution and max-pooling stages followed by dense layers with
120, 84, and 10 units. Hidden activations are ReLU, the output uses softmax, and
each kernel has its own He-normal initializer seeded with zero. Training uses
Adam and categorical cross-entropy.

Target environment: Python 3.9, NumPy 1.25.2, TensorFlow 2.15, and
pycodestyle 2.11.1. The functional and numerical-gradient tests can be run from
the repository root with `python tests/test_cnn.py`. The Keras test is skipped
when TensorFlow is unavailable.
