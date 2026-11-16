# Gated Recurrent Unit (GRU)

Files: [gru.py](../gru.py) and [libgru.py](../libgru.py)

## Idea

A GRU is a simpler cousin of the [LSTM](lstm.md). It has no separate cell state and only two gates, so it has fewer parameters and trains faster, while still being able to keep information over long sequences.

- The **update gate** $\mathbf{z}_t$ decides how much of the old state to keep and how much to replace with new information.
- The **reset gate** $\mathbf{r}_t$ decides how much of the old state is used when computing the new candidate.

## Equations

$$\mathbf{z}_t = \sigma\big(W_z \mathbf{x}_t + U_z \mathbf{h}_{t-1} + \mathbf{b}_z\big) \quad \text{(update gate)}$$

$$\mathbf{r}_t = \sigma\big(W_r \mathbf{x}_t + U_r \mathbf{h}_{t-1} + \mathbf{b}_r\big) \quad \text{(reset gate)}$$

$$\tilde{\mathbf{h}}_t = \tanh\big(W_n \mathbf{x}_t + U_n (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_n\big) \quad \text{(candidate state)}$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

The last line is a blend: when $\mathbf{z}_t$ is near 0 the old state is copied forward unchanged, and when it is near 1 the state is replaced by the candidate. After the last step the output is

$$\hat{y} = \sigma\big(\mathbf{w}_o^T \mathbf{h}_T + b_o\big)$$

trained with binary cross entropy.

## Backpropagation through time

Going backwards with the gradient $d\mathbf{h}_t$ coming from the future:

$$d\mathbf{z}_t = d\mathbf{h}_t \odot (\tilde{\mathbf{h}}_t - \mathbf{h}_{t-1}), \qquad d\tilde{\mathbf{h}}_t = d\mathbf{h}_t \odot \mathbf{z}_t$$

$$d\mathbf{a}_n = d\tilde{\mathbf{h}}_t \odot (1 - \tilde{\mathbf{h}}_t^2), \qquad d\mathbf{a}_z = d\mathbf{z}_t \odot \mathbf{z}_t (1 - \mathbf{z}_t)$$

The reset gate gets its gradient through the candidate, $d\mathbf{r}_t = (U_n^T d\mathbf{a}_n) \odot \mathbf{h}_{t-1}$, followed by $d\mathbf{a}_r = d\mathbf{r}_t \odot \mathbf{r}_t(1 - \mathbf{r}_t)$. The gradient sent to the previous hidden state is the sum of four paths:

$$d\mathbf{h}_{t-1} = d\mathbf{h}_t \odot (1 - \mathbf{z}_t) + (U_n^T d\mathbf{a}_n) \odot \mathbf{r}_t + U_z^T d\mathbf{a}_z + U_r^T d\mathbf{a}_r$$

Weight gradients are accumulated over the time steps, e.g. $dW_z \mathrel{+}= d\mathbf{a}_z \mathbf{x}_t^T$ and $dU_z \mathrel{+}= d\mathbf{a}_z \mathbf{h}_{t-1}^T$.

## What the code does

Each mushroom is treated as a sequence of its attribute values, one attribute per step, one-hot encoded.

- Data: `mushrooms.csv` with `?` replaced by the mode. Edible = 0, poisonous = 1. 80/20 split with `random_state=42`.
- Hidden size 8, learning rate 0.1, 500 epochs, full batch gradient descent, seed 42. The loss is printed every 50 epochs and plotted.
- Predictions threshold the final output at 0.5.

## Library version

`libgru.py` uses Keras: `GRU(8, reset_after=False)` followed by `Dense(1, activation="sigmoid")`, with `SGD` at learning rate 0.1, binary cross entropy, 500 epochs and full batches. `reset_after=False` makes Keras apply the reset gate before the matrix multiplication, $U_n(\mathbf{r}_t \odot \mathbf{h}_{t-1})$, which matches the equations above.
