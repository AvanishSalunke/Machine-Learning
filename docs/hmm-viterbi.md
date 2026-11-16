# HMM: Viterbi Algorithm

Files: [hmmviterbi.py](../hmmviterbi.py), [datasethmmviterbi.py](../datasethmmviterbi.py) and [libhmmviterbi.py](../libhmmviterbi.py)

## The problem

Given a [hidden Markov model](hmm-forward-backward.md) $\lambda = (\pi, A, B)$ and an observation sequence $O = o_1, \dots, o_T$, the decoding problem asks for the single most likely sequence of hidden states:

$$Q^* = \arg\max_{q_1, \dots, q_T} P(q_1, \dots, q_T, O \mid \lambda)$$

The forward algorithm *sums* over all state paths. Viterbi takes the *maximum* instead and remembers which path gave it.

## Algorithm

$\delta_t(j)$ is the probability of the best path that ends in state $j$ at time $t$ and explains the first $t$ observations. $\psi_t(j)$ stores the previous state on that best path.

**Initialisation**

$$\delta_1(j) = \pi_j \, B_j(o_1), \qquad \psi_1(j) = 0$$

**Recursion**, for $t = 2, \dots, T$:

$$\delta_t(j) = \Big[ \max_{i} \delta_{t-1}(i) \, A_{ij} \Big] B_j(o_t)$$

$$\psi_t(j) = \arg\max_{i} \; \delta_{t-1}(i) \, A_{ij}$$

**Termination**

$$P^* = \max_j \delta_T(j), \qquad q_T^* = \arg\max_j \delta_T(j)$$

**Backtracking**, for $t = T-1, \dots, 1$:

$$q_t^* = \psi_{t+1}\big(q_{t+1}^*\big)$$

## What the code does

[hmmviterbi.py](../hmmviterbi.py) works on a small two-state example (states named `S` and `R`) with $\pi = (0.667, 0.333)$, $A = \begin{pmatrix} 0.8 & 0.2 \\ 0.4 & 0.6 \end{pmatrix}$, $B = \begin{pmatrix} 0.8 & 0.2 \\ 0.4 & 0.6 \end{pmatrix}$ and the observation sequence `[0, 0, 1, 1, 1, 0]`. The $\delta$ table is built with one vectorised step per time slot, the $\psi$ table stores the arg max, and the best path is read back by backtracking. It prints $\delta$, $\psi$, the best path and its probability.

As in the forward-backward code, the flag `from_start` decides whether the first observation is emitted from $\pi$ directly (0) or after one transition from $\pi$ (1).

[datasethmmviterbi.py](../datasethmmviterbi.py) estimates $\pi$, $A$ and $B$ from `dataset.csv` (columns `state` and `obs`) by counting with add-one smoothing, as described in [hmm-forward-backward.md](hmm-forward-backward.md), and then decodes the first 10 observations. `dataset.csv` is not included in this repository.

For long sequences the products of small probabilities can underflow, so real implementations work with log probabilities and replace the products by sums.

## Library version

[libhmmviterbi.py](../libhmmviterbi.py) builds a `hmmlearn.hmm.CategoricalHMM` with the same parameters and calls `model.decode(X, algorithm="viterbi")`, which returns the log probability of the best path and the path itself.
