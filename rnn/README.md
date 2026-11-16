# Recurrent Neural Network (Simple RNN)

Files in this folder:

- [rnn.py](rnn.py): a simple RNN with backpropagation through time, step by step with numpy.
- [librnn.py](librnn.py): library version with Keras `SimpleRNN`.

## In simple words

Read the sentence "The cat sat on the ..." and you can guess the next word because you **remember** the words before it. An ordinary neural network has no memory: it sees all its inputs at once, with no sense of order. A **recurrent** neural network reads a sequence **one item at a time** and keeps a small note, the **hidden state**, that it updates with each new item. The note from the previous step is fed back in with the new input, so information can travel along the sequence.

In this code, each mushroom is treated as a sequence: its 22 attributes (cap shape, cap colour, odor, ...) are read one after another, and the network decides at the end whether the mushroom is edible or poisonous.

## Key words

- **Sequence**: a list of inputs in order, $\mathbf{x}_1, \dots, \mathbf{x}_T$.
- **Hidden state** $\mathbf{h}_t$: the network's memory after step $t$.
- **Time step**: one position in the sequence.
- **One-hot vector**: a vector of zeros with a single 1, used to represent a category (for example, odor = "foul" is $(0, 0, 1, 0, \dots)$).
- **Backpropagation through time (BPTT)**: backpropagation on the network "unrolled" over all time steps.

## The maths, step by step

**1. The input.** Each attribute value is turned into a one-hot vector $\mathbf{x}_t$ over all possible symbols.

**2. The recurrence.** The same weights are used at every step. Start with $\mathbf{h}_0 = \mathbf{0}$ and for $t = 1, \dots, T$:

$$\mathbf{h}_t = \tanh\big(W_i\, \mathbf{x}_t + W_h\, \mathbf{h}_{t-1} + \mathbf{b}\big)$$

$W_i$ turns the new input into the state space, $W_h$ carries the old memory forward, and $\tanh$ squeezes every value into the range $-1$ to $1$ so that the state does not explode.

**3. The output.** After the last step, only the final hidden state is used for the decision:

$$\hat{y} = \sigma\big(\mathbf{w}_o^T\, \mathbf{h}_T + b_o\big)$$

**4. The loss** is the binary cross entropy:

$$J = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i) \Big]$$

**5. Backpropagation through time.** Imagine copying the network $T$ times, one copy per step, with the state flowing from one copy to the next. Then ordinary backpropagation applies, going from the last copy to the first. At the output: $dz = (\hat{y} - y)/m$ and the blame sent to the last state is $d\mathbf{h}_T = dz \cdot \mathbf{w}_o$. For $t = T, \dots, 1$:

$$d\mathbf{a}_t = d\mathbf{h}_t \odot \big(1 - \mathbf{h}_t^2\big)$$

$$dW_i \mathrel{+}= d\mathbf{a}_t\, \mathbf{x}_t^T, \qquad dW_h \mathrel{+}= d\mathbf{a}_t\, \mathbf{h}_{t-1}^T, \qquad d\mathbf{b} \mathrel{+}= d\mathbf{a}_t$$

$$d\mathbf{h}_{t-1} = W_h^T\, d\mathbf{a}_t$$

(The term $1 - \mathbf{h}_t^2$ is the derivative of $\tanh$, and "$\mathrel{+}=$" means the gradients from all steps are added up, because the weights are shared.)

**6. Update** all weights with learning rate $\alpha$: $W \leftarrow W - \alpha\, dW$.

**7. The weakness.** Going back one step multiplies the blame by $W_h^T$ and by a $\tanh$ derivative (which is at most 1). Over many steps this repeated multiplication makes the blame **shrink towards zero** (vanishing gradient) or **grow without limit** (exploding gradient). So the network forgets what happened long ago. [LSTM](../lstm/README.md) and [GRU](../gru/README.md) were invented to fix this.

## A small worked example

A network with one hidden unit: $W_i = 0.5$, $W_h = 0.5$, $b = 0$. The inputs are $x_1 = 1$ and $x_2 = 1$.

- $h_0 = 0$
- $h_1 = \tanh(0.5 \cdot 1 + 0.5 \cdot 0) = \tanh(0.5) = 0.462$
- $h_2 = \tanh(0.5 \cdot 1 + 0.5 \cdot 0.462) = \tanh(0.731) = 0.624$

The second state contains information about both inputs: it is bigger than the first, because the first input's memory (0.462) added to the push of the second input. That is the memory at work.

## How the code works

1. **Load** `../datasets/mushrooms.csv`, replace `?` with the most common value, and turn each attribute value into an integer code. Each code is then one-hot encoded, so a mushroom becomes a sequence of 22 one-hot vectors. Edible = 0, poisonous = 1. **Split** 80/20 with `random_state=42`.
2. **Start weights** as small random numbers (seed 42): $W_i$, $W_h$, $\mathbf{w}_o$. Biases are zeros.
3. **Forward pass.** For all training mushrooms at once, loop over the 22 steps and compute the hidden states, saving every one. Then compute the output probability and the loss.
4. **Backward pass** (BPTT) with the formulas above, looping from the last step to the first.
5. **Update** the weights. Repeat for 300 epochs (the error is printed every 50).
6. **Predict.** Run the test sequences through the network and threshold the output at 0.5. Report accuracy and the confusion matrix and plot the error.

Settings: 8 hidden units, learning rate 0.5, 300 epochs, full batch.

## Library version

[librnn.py](librnn.py) builds the same network in Keras: `SimpleRNN(8, activation="tanh")` followed by `Dense(1, activation="sigmoid")`, trained with `SGD` (learning rate 0.5), binary cross entropy, 300 epochs, with one batch holding all rows. Keras does the backpropagation through time for you.

## Strengths and weaknesses

- Handles sequences of any length with a fixed number of weights.
- Forgets long-range information because of the vanishing gradient problem.
- Cannot be computed in parallel along the sequence, because each step needs the one before.

## Things to try

- Change the hidden size to 4 and to 16.
- Compare the loss curve with [LSTM](../lstm/README.md) and [GRU](../gru/README.md) on the same data.
