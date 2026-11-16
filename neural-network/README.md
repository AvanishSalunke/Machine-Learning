# Neural Network (Multi-Layer Perceptron)

Files in this folder:

- [nn.py](nn.py): a neural network with forward and backward propagation, step by step with numpy.
- [libnn.py](libnn.py): library version with scikit-learn's `MLPClassifier`.

## In simple words

A neural network is a chain of small calculators called **neurons**. Each neuron takes some numbers, multiplies each by a weight, adds them up, and passes the sum through a bend (an **activation function**). The neurons are arranged in **layers**: the output of one layer is the input of the next.

A single neuron can only draw a straight dividing line (it is basically [logistic regression](../logistic-regression/README.md)). But when neurons are stacked in layers, the bends add up, and the network can draw very curved boundaries.

Training is the same idea as in the earlier models: make a prediction, measure the error, and nudge all the weights a little to reduce it. The clever part is **backpropagation**, a way to work out the nudge for every weight in every layer efficiently, by sending the error backwards through the network.

## Key words

- **Layer**: a group of neurons that work side by side.
- **Hidden layer**: a layer between the input and the output.
- **Activation function**: the bend applied to each neuron's sum. Without it, many layers would collapse into one straight line.
- **Forward propagation**: computing the output from the input, layer by layer.
- **Backpropagation**: computing the gradients, from the output layer backwards.
- **Epoch**: one full pass of "forward, backward, update".

## The maths, step by step

Layers are numbered $l = 1, \dots, L$. The input is $A^{[0]} = X$, with one column per sample.

**1. Forward propagation.** For every layer, first the weighted sum, then the activation:

$$Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}, \qquad A^{[l]} = g\big(Z^{[l]}\big)$$

The last layer uses a sigmoid, so the output is a probability that the mushroom is poisonous:

$$\hat{y} = A^{[L]} = \sigma\big(Z^{[L]}\big)$$

**2. Activation functions** you can choose for the hidden layers (the variable `activation` in the code):

$$\text{sigmoid: } \frac{1}{1+e^{-z}} \qquad \tanh(z) \qquad \text{ReLU: } \max(0, z) \qquad \text{leaky ReLU: } \max(0.001 z,\; z)$$

**3. The loss**, the binary cross entropy over $m$ samples:

$$J = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) \Big]$$

**4. Backpropagation.** It uses the chain rule from calculus to pass the blame backwards. At the output layer, because the sigmoid and the cross entropy fit together, the error signal is simply

$$dZ^{[L]} = A^{[L]} - Y$$

For each layer, going from the last to the first, the gradients of its weights and bias are

$$dW^{[l]} = \frac{1}{m}\, dZ^{[l]}\, A^{[l-1]T}, \qquad db^{[l]} = \frac{1}{m} \sum_{i} dZ^{[l]}_{i}$$

and the error signal for the layer before it is

$$dZ^{[l-1]} = \big(W^{[l]T}\, dZ^{[l]}\big) \odot g'\big(Z^{[l-1]}\big)$$

($\odot$ means multiply element by element, and $g'$ is the derivative of the activation: $a(1-a)$ for sigmoid, $1 - a^2$ for tanh, 1 or 0 for ReLU, and 1 or 0.001 for leaky ReLU.) In words: the blame for a neuron is the blame of the neurons it feeds, weighted by the connections, times how sensitive the neuron is.

**5. The update** for every layer, with learning rate $\alpha$:

$$W^{[l]} \leftarrow W^{[l]} - \alpha\, dW^{[l]}, \qquad b^{[l]} \leftarrow b^{[l]} - \alpha\, db^{[l]}$$

**6. Starting weights.** All weights must **not** start at zero, or all neurons in a layer would stay identical. The code draws them from a normal distribution scaled by $\sqrt{1 / n_{in}}$, where $n_{in}$ is the number of inputs of the layer. This keeps the signals at a healthy size from layer to layer. Biases start at zero.

## A small worked example

One neuron with two inputs $\mathbf{x} = (1, 2)$, weights $\mathbf{w} = (0.5, -0.5)$, bias $b = 0.1$, and true label $y = 1$.

- Forward: $z = 0.5 \cdot 1 - 0.5 \cdot 2 + 0.1 = -0.4$ and $\hat{y} = \sigma(-0.4) = 0.401$.
- Loss: $-\ln 0.401 = 0.913$.
- Error signal: $dz = \hat{y} - y = -0.599$.
- Gradients: $dw = dz \cdot \mathbf{x} = (-0.599,\ -1.197)$ and $db = -0.599$.
- With $\alpha = 1$: new weights $(1.099,\ 0.697)$ and $b = 0.699$. The next forward pass gives $z = 1.099 + 1.394 + 0.699 = 3.19$, so $\hat{y} = 0.96$: much closer to the label 1.

In a real network, the same steps happen in every layer, with the error signal passed back by the formula in step 4.

## How the code works

1. **Load** `../datasets/mushrooms.csv`, one-hot encode and standardise the features. Edible = 0, poisonous = 1. **Split** 80/20 with `random_state=42`. The data is arranged with one **column** per sample.
2. **Build the network.** `hidden = [3, 2]`, so the layer sizes are: input (number of features), 3, 2, 1. Weights are random (seed 42) and scaled by $\sqrt{1/n_{in}}$, biases are zeros.
3. **Train** for 1000 epochs of full-batch gradient descent: forward pass through all layers (saving $Z$ and $A$), backward pass with the formulas above, update all weights, compute and store the loss (printed every 100 epochs).
4. **Predict.** A forward pass on the test data, and the 0.5 threshold on the output.
5. **Report** accuracy and the confusion matrix and plot the loss.

Settings: hidden layers of 3 and 2 neurons, sigmoid activation, learning rate 1, 1000 epochs.

## Library version

[libnn.py](libnn.py) uses `MLPClassifier` with the same layer sizes, `activation="logistic"` (sigmoid), `solver="sgd"`, learning rate 1, 1000 iterations, one batch with all rows, and with momentum and L2 penalty switched off, so that it learns in the same way as the numpy code. The loss per epoch comes from `loss_curve_`. The starting weights are different (scikit-learn uses a different recipe), so the two loss curves will not be identical.

## Strengths and weaknesses

- Can learn very complicated patterns, given enough neurons and data.
- Sensitive to settings: learning rate, number of layers and starting weights. Sigmoid in many layers can make the gradients tiny (the "vanishing gradient" problem), which is why ReLU is popular.
- A black box: hard to say why it gave an answer.

## Things to try

- Change `activation` to `"relu"` or `"tanh"` and watch the loss curve.
- Try `hidden = [8, 4]` and compare the final loss.
- Lower the learning rate from 1 to 0.1 and see how much slower it learns.
