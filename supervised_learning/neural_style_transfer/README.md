# Neural Style Transfer

The eleven progressive `NST` classes implement image scaling, frozen VGG19
features with average pooling, normalized Gram matrices, content/style losses,
image gradients, Adam optimization and total variation regularization.

Only NumPy and TensorFlow are imported. The exercises target TensorFlow 2.15
and NumPy 1.25.2. VGG19 uses pretrained ImageNet weights. The returned generated
image contains the pixels associated with the lowest observed cost.

The [TensorFlow style transfer tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)
documents the feature correlations, mean-square losses and image optimization
operations used here. Project-specific pooling, interfaces and validation rules
follow the Holberton instructions.

TensorFlow execution requires that dependency and the pretrained model weights.
Syntax and style can be checked without running the model:

```sh
python -m compileall -q supervised_learning/neural_style_transfer
python -m pycodestyle supervised_learning/neural_style_transfer
```
