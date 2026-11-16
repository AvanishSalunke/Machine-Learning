# HMM: Viterbi Algorithm

Files in this folder:

- [hmmviterbi.py](hmmviterbi.py): the Viterbi algorithm on a small example, step by step with numpy.
- [datasethmmviterbi.py](datasethmmviterbi.py): the same, with the model learned by counting from a dataset.
- [libhmmviterbi.py](libhmmviterbi.py): library version with `hmmlearn`.

## In simple words

Go back to the friend whose weather you cannot see, but whose daily activities you are told about. The [forward-backward algorithm](../hmm-forward-backward/README.md) answers "how likely is this list of activities?". Viterbi answers a different question: **"What was the single most likely sequence of weather days that explains these activities?"**

The idea is a bit like finding the best route on a map with stops. At each day, for each possible weather, remember only the **best way to have arrived there**. Because of the Markov property (tomorrow depends only on today), the best route to a place never needs to be rethought later. At the end, pick the best final weather and **walk backwards** along the saved choices.

## Key words

- **$\delta_t(j)$ (delta)**: the probability of the best path that ends in state $j$ at time $t$ and explains the first $t$ observations.
- **$\psi_t(j)$ (psi)**: the previous state on that best path (the "pointer").
- **Backtracking**: reading the pointers from the end to the start to recover the path.
- **Dynamic programming**: solving a big problem by saving the answers to small pieces.

## The maths, step by step

The model is $\lambda = (\pi, A, B)$ as described in [forward-backward](../hmm-forward-backward/README.md), and the observations are $O = o_1, \dots, o_T$. We want

$$Q^* = \arg\max_{q_1, \dots, q_T} P(q_1, \dots, q_T, O \mid \lambda)$$

The forward algorithm **adds** over all paths. Viterbi takes the **maximum** instead, and remembers which path won.

**1. Start.**

$$\delta_1(j) = \pi_j\, B_j(o_1), \qquad \psi_1(j) = 0$$

**2. Recursion**, for $t = 2, \dots, T$. To arrive in state $j$, look at every previous state $i$, take the best, and multiply by the chance of seeing $o_t$:

$$\delta_t(j) = \Big[ \max_{i}\; \delta_{t-1}(i)\, A_{ij} \Big]\, B_j(o_t)$$

$$\psi_t(j) = \arg\max_{i}\; \delta_{t-1}(i)\, A_{ij}$$

**3. End.** The probability of the best full path, and its last state:

$$P^* = \max_j \delta_T(j), \qquad q_T^* = \arg\max_j \delta_T(j)$$

**4. Backtrack**, for $t = T-1, \dots, 1$:

$$q_t^* = \psi_{t+1}\big(q_{t+1}^*\big)$$

**5. Vectorised.** In the code, `delta[t-1][:, None] * A` builds a table of all "previous state to next state" scores at once. `np.max(..., axis=0)` takes the best previous state for each next state, and `np.argmax` stores the pointer.

**6. A practical note.** For long sequences the products become tiny. Real implementations work with logarithms and replace products by sums.

## A small worked example

Two weather states, `S` (state 0) and `R` (state 1), and two possible observations (0 and 1).

$$\pi = (0.667,\; 0.333), \qquad A = \begin{pmatrix} 0.8 & 0.2 \\ 0.4 & 0.6 \end{pmatrix}, \qquad B = \begin{pmatrix} 0.8 & 0.2 \\ 0.4 & 0.6 \end{pmatrix}, \qquad O = (0, 0, 1, 1, 1, 0)$$

**Step 1.** $\delta_1 = \pi \odot B[:, 0] = (0.667 \cdot 0.8,\; 0.333 \cdot 0.4) = (0.5336,\; 0.1332)$.

**Step 2** (observation 0 again):

- Arrive in `S`: from `S`: $0.5336 \cdot 0.8 = 0.4269$; from `R`: $0.1332 \cdot 0.4 = 0.0533$. The best is from `S`, so $\psi_2(S) = S$ and $\delta_2(S) = 0.4269 \cdot 0.8 = 0.3415$.
- Arrive in `R`: from `S`: $0.5336 \cdot 0.2 = 0.1067$; from `R`: $0.1332 \cdot 0.6 = 0.0799$. The best is from `S`, so $\psi_2(R) = S$ and $\delta_2(R) = 0.1067 \cdot 0.4 = 0.0427$.

Repeating this for all six observations and then backtracking from the largest final $\delta$ gives the best weather sequence. The program prints the full $\delta$ table, the $\psi$ table, the path, and its probability.

## How the code works

**[hmmviterbi.py](hmmviterbi.py)**

1. Define $\pi$, $A$, $B$, the observations, and the state names `["S", "R"]`.
2. `from_start` decides how the sequence begins: 0 means the first observation comes straight from $\pi$, 1 means one transition is applied to $\pi$ first.
3. Fill the `delta` and `psi` tables, one row per time step.
4. Pick the best last state, and backtrack through `psi` to get the path.
5. Print `delta`, `psi`, the best path (with names) and its probability.

**[datasethmmviterbi.py](datasethmmviterbi.py)** learns $\pi$, $A$ and $B$ from a `dataset.csv` with a `state` column and an `obs` column, by counting with add-one smoothing (see [forward-backward](../hmm-forward-backward/README.md)), then decodes the first 10 observations. The file `dataset.csv` is not included in this repository, so add your own to run it.

## Library version

[libhmmviterbi.py](libhmmviterbi.py) builds a `hmmlearn.hmm.CategoricalHMM` with the same parameters and calls `model.decode(X, algorithm="viterbi")`. It returns the log probability of the best path and the path itself. The code takes `exp` of the log probability to compare with the numpy result.

## Strengths and weaknesses

- Exact: it always finds the best path, in $N^2 T$ steps.
- It finds the single best path, which is not always the same as picking the most likely state at each time separately (that is what $\gamma$ from forward-backward gives).

## Things to try

- Change the observation list and see how the best path changes.
- Make the transition matrix "sticky" (large diagonal) and see paths get longer runs of the same state.
