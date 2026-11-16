# Neural Network (Multi-Layer Perceptron)

Files: [nn.py](../nn.py) and [libnn.py](../libnn.py)

## Idea

A feed-forward neural network is a stack of layers. Each layer multiplies its input by a weight matrix, adds a bias, and passes the result through a non-linear activation function. Stacking layers lets the network learn non-linear decision boundaries that a single logistic regression cannot.

## Forward propagation

For layer $l = 1, \dots, L$, with $A^{[0]} = X$ (the input, one column per sample):

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \qquad A^{[l]} = g^{[l]}\big(Z^{[l]}\big)$$

The output layer uses a sigmoid so the result is a probability:

$$\hat{y} = A^{[L]} = \sigma\big(Z^{[L]}\big)$$

Activation functions available in the code (set by the `activation` variable for the hidden layers):

$$\text{sigmoid: } \frac{1}{1+e^{-z}}, \qquad \tanh(z), \qquad \text{ReLU: } \max(0, z), \qquad \text{leaky ReLU: } \max(0.001 z, z)$$

## Loss

Binary cross entropy averaged over the $m$ samples:

$$J = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) \Big]$$

## Backpropagation

Starting at the output, where the sigmoid and cross entropy combine nicely:

$$dZ^{[L]} = A^{[L]} - Y$$

For each layer going backwards:

$$dW^{[l]} = \frac{1}{m} dZ^{[l]} \, A^{[l-1]T}, \qquad db^{[l]} = \frac{1}{m} \sum_{i} dZ^{[l]}_{i}$$

$$dZ^{[l-1]} = W^{[l]T} dZ^{[l]} \odot g'\big(Z^{[l-1]}\big)$$

where $\odot$ is element-wise multiplication and $g'$ is the derivative of the activation: $a(1-a)$ for sigmoid, $1 - a^2$ for tanh, 1 or 0 for ReLU, and 1 or 0.001 for leaky ReLU. Then gradient descent updates every layer:

$$W^{[l]} \leftarrow W^{[l]} - \alpha \, dW^{[l]}, \qquad b^{[l]} \leftarrow b^{[l]} - \alpha \, db^{[l]}$$

## What the code does

- Data: `mushrooms.csv`, one-hot encoded and standardised. Edible = 0, poisonous = 1. 80/20 split with `random_state=42`.
- Architecture: input layer, hidden layers of 3 and 2 neurons (the `hidden` list), and 1 output neuron. Sigmoid activation in the hidden layers.
- Weights are initialised from a normal distribution scaled by $\sqrt{1 / n_{in}}$, biases start at zero, seed 42.
- Full-batch gradient descent, learning rate 1, 1000 epochs. The loss is printed every 100 epochs and plotted.
- Predictions use the 0.5 threshold on the output probability.

## Library version

`libnn.py` uses scikit-learn's `MLPClassifier` with the same architecture, `activation="logistic"`, `solver="sgd"`, learning rate 1, 1000 iterations, full batch, no momentum and no L2 penalty, so that it trains the same way as the code version. The loss per epoch comes from `loss_curve_`. The random starting weights are different (scikit-learn uses Glorot initialisation), so the curves will not match exactly.
