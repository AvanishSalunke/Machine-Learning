# Long Short-Term Memory (LSTM)

Files: [lstm.py](lstm.py) and [liblstm.py](liblstm.py)

## Idea

An LSTM is a recurrent network designed to remember information over long sequences. Besides the hidden state $\mathbf{h}_t$ it keeps a separate **cell state** $\mathbf{c}_t$, a conveyor belt that information can travel along with only small, controlled changes. Three gates, each a sigmoid layer producing numbers between 0 and 1, decide what is forgotten, what is added and what is shown.

## Equations

Let $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ be the previous hidden state joined with the current input. Each gate has its own weights and bias:

$$\mathbf{f}_t = \sigma\big(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\big) \quad \text{(forget gate)}$$

$$\mathbf{i}_t = \sigma\big(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i\big) \quad \text{(input gate)}$$

$$\tilde{\mathbf{c}}_t = \tanh\big(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c\big) \quad \text{(candidate memory)}$$

$$\mathbf{o}_t = \sigma\big(W_q [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_q\big) \quad \text{(output gate)}$$

The cell state keeps part of the old memory and adds part of the new candidate:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

The hidden state is a filtered view of the cell state:

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

Final output after the last step, with the log loss as in the [RNN](../rnn/README.md):

$$\hat{y} = \sigma\big(\mathbf{w}_o^T \mathbf{h}_T + b_o\big)$$

## Why it helps

The cell state is updated by addition, $\mathbf{f}_t \odot \mathbf{c}_{t-1} + \dots$, instead of being squashed through a matrix and a $\tanh$ at every step. When the forget gate is close to 1 the gradient flows back along the cell state almost unchanged, which greatly reduces the vanishing gradient problem.

## Backpropagation through time

The code computes the gradients by hand while walking backwards through the steps. With $d\mathbf{h}_t$ and $d\mathbf{c}_t$ coming from the future:

$$d\mathbf{c}_t \mathrel{+}= d\mathbf{h}_t \odot \mathbf{o}_t \odot \big(1 - \tanh^2(\mathbf{c}_t)\big)$$

$$d\mathbf{a}_o = d\mathbf{h}_t \odot \tanh(\mathbf{c}_t) \odot \mathbf{o}_t (1 - \mathbf{o}_t), \qquad d\mathbf{a}_f = d\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \mathbf{f}_t (1 - \mathbf{f}_t)$$

$$d\mathbf{a}_i = d\mathbf{c}_t \odot \tilde{\mathbf{c}}_t \odot \mathbf{i}_t (1 - \mathbf{i}_t), \qquad d\mathbf{a}_c = d\mathbf{c}_t \odot \mathbf{i}_t \odot \big(1 - \tilde{\mathbf{c}}_t^2\big)$$

The weight gradients are $d W_\ast \mathrel{+}= d\mathbf{a}_\ast [\mathbf{h}_{t-1}, \mathbf{x}_t]^T$ and the cell gradient passed to the previous step is $d\mathbf{c}_{t-1} = d\mathbf{c}_t \odot \mathbf{f}_t$.

## What the code does

Each mushroom is treated as a sequence of its attribute values (one attribute per time step), one-hot encoded.

- Data: `mushrooms.csv` with `?` replaced by the mode. Edible = 0, poisonous = 1. 80/20 split with `random_state=42`.
- Hidden size 8, learning rate 0.3, 500 epochs, full batch gradient descent, seed 42. The forget gate bias starts at 1, which is a common trick so that the network remembers by default early in training.
- The loss is printed every 50 epochs and plotted. Predictions threshold the final output at 0.5.

## Library version

`liblstm.py` uses Keras: `LSTM(8)` followed by `Dense(1, activation="sigmoid")`, trained with `SGD` at learning rate 0.3, binary cross entropy, 500 epochs and full batches.
