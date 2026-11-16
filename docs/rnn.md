# Recurrent Neural Network (Simple RNN)

Files: [rnn.py](../rnn.py) and [librnn.py](../librnn.py)

## Idea

A recurrent neural network processes a sequence one element at a time while carrying a hidden state $h_t$ from one step to the next. The hidden state is the network's memory of what it has seen so far. The same weights are reused at every step.

## Forward pass

For input $\mathbf{x}_t$ at step $t$, starting from $\mathbf{h}_0 = \mathbf{0}$:

$$\mathbf{h}_t = \tanh\big(W_i \mathbf{x}_t + W_h \mathbf{h}_{t-1} + \mathbf{b}\big)$$

For classification, only the last hidden state is used:

$$\hat{y} = \sigma\big(\mathbf{w}_o^T \mathbf{h}_T + b_o\big)$$

and the loss is the binary cross entropy

$$J = -\frac{1}{m} \sum_i \Big[ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) \Big]$$

## Backpropagation through time (BPTT)

The network is unrolled over the $T$ steps and ordinary backpropagation is applied, going from the last step to the first. At the output, $dz = (\hat{y} - y)/m$ and the gradient flowing into the last hidden state is $d\mathbf{h}_T = dz \cdot \mathbf{w}_o$. Then for $t = T, \dots, 1$:

$$d\mathbf{a}_t = d\mathbf{h}_t \odot \big(1 - \mathbf{h}_t^2\big)$$

$$dW_i \mathrel{+}= d\mathbf{a}_t \, \mathbf{x}_t^T, \qquad dW_h \mathrel{+}= d\mathbf{a}_t \, \mathbf{h}_{t-1}^T, \qquad d\mathbf{b} \mathrel{+}= d\mathbf{a}_t$$

$$d\mathbf{h}_{t-1} = W_h^T \, d\mathbf{a}_t$$

The gradients of the shared weights are summed over all time steps. Because the gradient is multiplied by $W_h^T$ and a $\tanh$ derivative at every step, it tends to shrink (vanishing gradient) or blow up (exploding gradient) over long sequences. That is the weakness the [LSTM](lstm.md) and [GRU](gru.md) were designed to fix.

## What the code does

An RNN needs a sequence, so each mushroom is treated as a sequence of its attribute values, one attribute per time step, in the column order of the data.

- Data: `mushrooms.csv`, `?` replaced by the mode. Each attribute value is converted to an integer code and then one-hot encoded, so $\mathbf{x}_t$ is a one-hot vector over all possible symbols. Edible = 0, poisonous = 1. 80/20 split with `random_state=42`.
- Hidden size 8, learning rate 0.5, 300 epochs, full batch plain gradient descent. Seed 42, small random starting weights.
- Loss is printed every 50 epochs and plotted.
- Prediction runs the test sequences through the network and thresholds the output at 0.5.

## Library version

`librnn.py` builds the same model in Keras: `SimpleRNN(8, activation="tanh")` followed by `Dense(1, activation="sigmoid")`, trained with `SGD` at learning rate 0.5, binary cross entropy, 300 epochs and a batch containing all training rows.
