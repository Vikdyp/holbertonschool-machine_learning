# Regularization

This project implements regularization techniques for neural networks using
NumPy and TensorFlow/Keras.

## Topics

- L2-regularized cost and gradient descent
- Keras dense layers with L2 regularization
- Inverted dropout during forward and backward propagation
- Keras dropout layers
- Early stopping based on validation cost

## Files

- `0-l2_reg_cost.py`: adds an L2 penalty to a NumPy network cost.
- `1-l2_reg_gradient_descent.py`: performs L2-regularized backpropagation.
- `2-l2_reg_cost.py`: combines a Keras cost with per-layer losses.
- `3-l2_reg_create_layer.py`: creates an L2-regularized dense layer.
- `4-dropout_forward_prop.py`: performs forward propagation with dropout.
- `5-dropout_gradient_descent.py`: performs backpropagation with dropout.
- `6-dropout_create_layer.py`: creates a Keras dense and dropout layer.
- `7-early_stopping.py`: updates an early-stopping patience counter.
