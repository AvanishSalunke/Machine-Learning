# Long Short-Term Memory (LSTM)

Files in this folder:

- [lstm.py](lstm.py): an LSTM with backpropagation through time, step by step with numpy.
- [liblstm.py](liblstm.py): library version with Keras `LSTM`.

## In simple words

A [simple RNN](../rnn/README.md) rewrites its whole memory at every step, so old information gets overwritten and fades. An LSTM is built like a notebook with a **pen, an eraser and a window**:

- the **forget gate** decides what to erase from the notebook,
- the **input gate** decides what new notes to write,
- the **output gate** decides which part of the notebook to show to the outside world at this step.

The notebook itself is the **cell state**. Because the cell state is changed only by small, controlled edits (not completely rewritten), important information can survive for many steps.

## Key words

- **Cell state** $\mathbf{c}_t$: the long-term memory, the notebook.
- **Hidden state** $\mathbf{h}_t$: the short-term output of the cell at step $t$.
- **Gate**: a layer with a sigmoid that outputs numbers from 0 (closed) to 1 (open), used to multiply other values.
- **Forget, input and output gates**: $\mathbf{f}_t$, $\mathbf{i}_t$, $\mathbf{o}_t$.
- **Candidate memory** $\tilde{\mathbf{c}}_t$: the new information that could be written.

## The maths, step by step

At each step, the previous hidden state and the new input are joined into one vector $[\mathbf{h}_{t-1}, \mathbf{x}_t]$. Four small layers look at it.

**1. The three gates** (each has its own weights, and $\sigma$ is the sigmoid):

$$\mathbf{f}_t = \sigma\big(W_f\, [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f\big) \qquad \text{(forget: how much of the old memory to keep)}$$

$$\mathbf{i}_t = \sigma\big(W_i\, [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i\big) \qquad \text{(input: how much of the new note to write)}$$

$$\mathbf{o}_t = \sigma\big(W_q\, [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_q\big) \qquad \text{(output: how much memory to show)}$$

**2. The candidate memory**, with values between $-1$ and $1$:

$$\tilde{\mathbf{c}}_t = \tanh\big(W_c\, [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c\big)$$

**3. Update the cell state.** Keep part of the old memory and add part of the new:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

($\odot$ multiplies element by element.)

**4. Compute the hidden state:**

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

**5. The output and loss**, after the last step, as in the [RNN](../rnn/README.md):

$$\hat{y} = \sigma\big(\mathbf{w}_o^T\, \mathbf{h}_T + b_o\big), \qquad J = -\frac{1}{m} \sum_i \Big[ y_i \ln \hat{y}_i + (1 - y_i) \ln (1 - \hat{y}_i) \Big]$$

**6. Why the gradient survives.** The cell state changes by $\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \dots$. When going backwards, the blame for $\mathbf{c}_{t-1}$ is the blame for $\mathbf{c}_t$ multiplied by $\mathbf{f}_t$. If the forget gate is close to 1, the blame passes back almost unchanged, step after step. This is why LSTMs can learn long-range connections, and why the vanishing gradient problem is much smaller.

**7. Backpropagation through time.** The code goes from the last step to the first. With $d\mathbf{h}_t$ and $d\mathbf{c}_t$ coming from the future:

$$d\mathbf{c}_t \mathrel{+}= d\mathbf{h}_t \odot \mathbf{o}_t \odot \big(1 - \tanh^2(\mathbf{c}_t)\big)$$

$$d\mathbf{a}_o = d\mathbf{h}_t \odot \tanh(\mathbf{c}_t) \odot \mathbf{o}_t (1 - \mathbf{o}_t), \qquad d\mathbf{a}_f = d\mathbf{c}_t \odot \mathbf{c}_{t-1} \odot \mathbf{f}_t (1 - \mathbf{f}_t)$$

$$d\mathbf{a}_i = d\mathbf{c}_t \odot \tilde{\mathbf{c}}_t \odot \mathbf{i}_t (1 - \mathbf{i}_t), \qquad d\mathbf{a}_c = d\mathbf{c}_t \odot \mathbf{i}_t \odot \big(1 - \tilde{\mathbf{c}}_t^2\big)$$

Each weight gradient is $dW_\ast \mathrel{+}= d\mathbf{a}_\ast\, [\mathbf{h}_{t-1}, \mathbf{x}_t]^T$, summed over all steps. The blame passed to the previous cell state is $d\mathbf{c}_{t-1} = d\mathbf{c}_t \odot \mathbf{f}_t$, and the blame passed to the previous hidden state comes from all four paths through the weight matrices.

## A small worked example

One memory cell at one step. The previous cell state is $c_{t-1} = 2.0$. The gates are $f = 0.9$, $i = 0.5$, $o = 0.8$ and the candidate is $\tilde{c} = 0.4$.

- New cell state: $c_t = 0.9 \cdot 2.0 + 0.5 \cdot 0.4 = 1.8 + 0.2 = 2.0$
- Hidden state: $h_t = 0.8 \cdot \tanh(2.0) = 0.8 \cdot 0.964 = 0.771$

The forget gate was almost open (0.9), so most of the old memory (2.0) survived. A new small note (0.2) was added, and the output gate showed 80% of the (squeezed) memory.

## How the code works

1. **Load** `../datasets/mushrooms.csv`, replace `?` with the most common value, convert each attribute value to an integer code and then to a one-hot vector. Each mushroom is a sequence of 22 steps. Edible = 0, poisonous = 1. **Split** 80/20 with `random_state=42`.
2. **Weights.** Four matrices $W_f, W_i, W_c, W_q$, each with the input and the previous hidden state joined together, plus biases and an output layer. Small random start (seed 42). The forget bias starts at 1, so that the network remembers by default at the start of training (a common trick).
3. **Forward pass** over all 22 steps for all training rows at once, saving gates, cell states and hidden states.
4. **Loss and backward pass** with the formulas above.
5. **Update** all weights. Repeat for 500 epochs (the error is printed every 50).
6. **Predict** by running the test sequences forward and thresholding at 0.5. Report accuracy and the confusion matrix and plot the error.

Settings: 8 hidden units, learning rate 0.3, 500 epochs, full batch.

## Library version

[liblstm.py](liblstm.py) uses Keras: `LSTM(8)` followed by `Dense(1, activation="sigmoid")`, trained with `SGD` (learning rate 0.3), binary cross entropy, 500 epochs, and one batch containing all rows. Keras does the backpropagation through time for you.

## Strengths and weaknesses

- Remembers over long sequences far better than a simple RNN.
- Has four times as many weights as a simple RNN with the same hidden size, so it is slower.
- Still reads the sequence one step at a time.

## Things to try

- Compare the loss curve with the [simple RNN](../rnn/README.md) and the [GRU](../gru/README.md).
- Set the forget bias to 0 and see if training changes.
