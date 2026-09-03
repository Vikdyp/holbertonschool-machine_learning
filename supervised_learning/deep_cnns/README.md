# Deep Convolutional Architectures

Keras implementations of residual network building blocks and ResNet-50.

| File | Function | Purpose |
| --- | --- | --- |
| `2-identity_block.py` | `identity_block` | Bottleneck with an identity shortcut |
| `3-projection_block.py` | `projection_block` | Bottleneck with a learned shortcut |
| `4-resnet50.py` | `resnet50` | ResNet-50 with stage depths 3, 4, 6, 3 |

The models use channels-last tensors, batch normalization, ReLU activations,
and He normal kernel initialization with seed 0. ResNet-50 accepts 224x224 RGB
images and returns class probabilities with shape `(None, 1, 1, 1000)`.

Target environment: Python 3.9, NumPy 1.25.2 and TensorFlow 2.15.
