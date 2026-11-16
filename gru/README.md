# Gated Recurrent Unit (GRU)

Files in this folder:

- [gru.py](gru.py): a GRU with backpropagation through time, step by step with numpy.
- [libgru.py](libgru.py): library version with Keras `GRU`.

## In simple words

A GRU is a leaner version of the [LSTM](../lstm/README.md). It also uses gates to protect its memory, but it has only two gates and no separate notebook: just one hidden state that is both memory and output.

- The **update gate** decides, for each part of the memory, whether to keep the old value or replace it with a new one. Like a dial between "stay" and "change".
- The **reset gate** decides how much of the old memory to look at when working out what the new value should be.

With fewer parts, a GRU is faster and has fewer weights, and in many tasks it works about as well as an LSTM.

## Key words

- **Hidden state** $\mathbf{h}_t$: the memory and the output at step $t$.
- **Update gate** $\mathbf{z}_t$: how much of the memory to replace.
- **Reset gate** $\mathbf{r}_t$: how much of the old memory to use when proposing a new value.
- **Candidate state** $\tilde{\mathbf{h}}_t$: the proposed new memory.

## The maths, step by step

Start with $\mathbf{h}_0 = \mathbf{0}$. At each step $t$ with input $\mathbf{x}_t$ ($\sigma$ is the sigmoid, $\odot$ multiplies element by element):

**1. The update gate:**

$$\mathbf{z}_t = \sigma\big(W_z\, \mathbf{x}_t + U_z\, \mathbf{h}_{t-1} + \mathbf{b}_z\big)$$

**2. The reset gate:**

$$\mathbf{r}_t = \sigma\big(W_r\, \mathbf{x}_t + U_r\, \mathbf{h}_{t-1} + \mathbf{b}_r\big)$$

**3. The candidate state.** The reset gate scales down the old memory before it is used:

$$\tilde{\mathbf{h}}_t = \tanh\big(W_n\, \mathbf{x}_t + U_n\, (\mathbf{r}_t \odot \mathbf{h}_{t-1}) + \mathbf{b}_n\big)$$

If $\mathbf{r}_t$ is near 0, the candidate ignores the past and looks only at the new input.

**4. The new hidden state** is a blend between the old state and the candidate:

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

If $\mathbf{z}_t$ is near 0, the old state is copied forward unchanged. If it is near 1, the state is replaced by the candidate. That copy path is what lets gradients flow through many steps without fading.

**5. The output and loss**, after the last step:

$$\hat{y} = \sigma\big(\mathbf{w}_o^T\, \mathbf{h}_T + b_o\big), \qquad J = -\frac{1}{m} \sum_i \Big[ y_i \ln \hat{y}_i + (1 - y_i) \ln (1 - \hat{y}_i) \Big]$$

**6. Backpropagation through time.** Going backwards with the blame $d\mathbf{h}_t$ from the future:

$$d\mathbf{z}_t = d\mathbf{h}_t \odot (\tilde{\mathbf{h}}_t - \mathbf{h}_{t-1}), \qquad d\tilde{\mathbf{h}}_t = d\mathbf{h}_t \odot \mathbf{z}_t$$

$$d\mathbf{a}_n = d\tilde{\mathbf{h}}_t \odot \big(1 - \tilde{\mathbf{h}}_t^2\big), \qquad d\mathbf{a}_z = d\mathbf{z}_t \odot \mathbf{z}_t (1 - \mathbf{z}_t)$$

The reset gate gets its blame through the candidate: $d\mathbf{r}_t = (U_n^T d\mathbf{a}_n) \odot \mathbf{h}_{t-1}$, then $d\mathbf{a}_r = d\mathbf{r}_t \odot \mathbf{r}_t (1 - \mathbf{r}_t)$. The blame sent to the previous hidden state is the sum of four paths:

$$d\mathbf{h}_{t-1} = d\mathbf{h}_t \odot (1 - \mathbf{z}_t) + (U_n^T d\mathbf{a}_n) \odot \mathbf{r}_t + U_z^T d\mathbf{a}_z + U_r^T d\mathbf{a}_r$$

The first path is the direct "copy" route. The weight gradients are collected over all steps, for example $dW_z \mathrel{+}= d\mathbf{a}_z\, \mathbf{x}_t^T$ and $dU_z \mathrel{+}= d\mathbf{a}_z\, \mathbf{h}_{t-1}^T$.

## A small worked example

One memory unit. The old state is $h_{t-1} = 1.0$, the update gate is $z = 0.25$ and the candidate is $\tilde{h} = 0.2$.

$$h_t = (1 - 0.25) \cdot 1.0 + 0.25 \cdot 0.2 = 0.75 + 0.05 = 0.80$$

The update gate was mostly closed (0.25), so 75% of the old memory stayed and only 25% was replaced by the new candidate.

## How the code works

1. **Load** `../datasets/mushrooms.csv`, replace `?` with the most common value, turn each attribute value into an integer code and then a one-hot vector. Each mushroom is a sequence of 22 steps. Edible = 0, poisonous = 1. **Split** 80/20 with `random_state=42`.
2. **Weights.** Three groups $(W_z, U_z, \mathbf{b}_z)$, $(W_r, U_r, \mathbf{b}_r)$, $(W_n, U_n, \mathbf{b}_n)$ and an output layer, started as small random numbers (seed 42).
3. **Forward pass** over all 22 steps for all training rows at once, saving the gates and states.
4. **Loss and backward pass** with the formulas above.
5. **Update** all weights. Repeat for 500 epochs (the error is printed every 50).
6. **Predict** by running the test sequences forward and thresholding at 0.5. Report accuracy and the confusion matrix and plot the error.

Settings: 8 hidden units, learning rate 0.1, 500 epochs, full batch.

## Library version

[libgru.py](libgru.py) uses Keras: `GRU(8, reset_after=False)` followed by `Dense(1, activation="sigmoid")`, trained with `SGD` (learning rate 0.1), binary cross entropy, 500 epochs, one batch with all rows. The option `reset_after=False` makes Keras apply the reset gate **before** the matrix multiplication, $U_n(\mathbf{r}_t \odot \mathbf{h}_{t-1})$, which is the formula above.

## Strengths and weaknesses

- Fewer weights than an LSTM (three groups instead of four), so it trains faster, and still handles long-range memory well.
- On very long or very hard sequences an LSTM can sometimes be better.

## Things to try

- Compare the loss curve and the accuracy with [LSTM](../lstm/README.md) and the [simple RNN](../rnn/README.md).
- Change the number of hidden units to 4 or 16.
