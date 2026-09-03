# Deep Convolutional Architectures

Keras implementations of Inception, ResNet-50 and DenseNet-121.

| File | Function | Purpose |
| --- | --- | --- |
| `0-inception_block.py` | `inception_block` | Parallel convolution and pooling branches |
| `1-inception_network.py` | `inception_network` | Inception image classifier |
| `2-identity_block.py` | `identity_block` | Bottleneck with an identity shortcut |
| `3-projection_block.py` | `projection_block` | Bottleneck with a learned shortcut |
| `4-resnet50.py` | `resnet50` | ResNet-50 with stage depths 3, 4, 6, 3 |
| `5-dense_block.py` | `dense_block` | DenseNet bottleneck feature concatenation |
| `6-transition_layer.py` | `transition_layer` | Channel compression and downsampling |
| `7-densenet121.py` | `densenet121` | DenseNet with block depths 6, 12, 24, 16 |

The models use channels-last tensors and ReLU activations. ResNet and DenseNet
use batch normalization and He normal kernel initialization with seed 0.
Inception uses Keras' default kernel initialization and 40% dropout before its
classifier. The three classifiers accept 224x224 RGB images and return class
probabilities with shape `(None, 1, 1, 1000)`.

Target environment: Python 3.9, NumPy 1.25.2 and TensorFlow 2.15.
