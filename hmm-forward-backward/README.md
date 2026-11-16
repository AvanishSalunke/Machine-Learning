# HMM: Forward-Backward Algorithm

Files: [hmmfb.py](../hmmfb.py), [datasethmmfb.py](../datasethmmfb.py) and [libhmmfb.py](../libhmmfb.py)

## Hidden Markov Model basics

An HMM describes a system that moves between hidden states, where each state produces an observation that we can see. It is defined by three things:

- the initial distribution $\pi_i = P(q_1 = i)$
- the transition matrix $A_{ij} = P(q_{t+1} = j \mid q_t = i)$
- the emission matrix $B_j(k) = P(o_t = k \mid q_t = j)$

The model is written $\lambda = (\pi, A, B)$. Given an observation sequence $O = o_1, \dots, o_T$, the evaluation problem asks: how likely is this sequence under the model, $P(O \mid \lambda)$? Adding up over all $N^T$ possible state paths is far too slow, and the forward-backward algorithm does it in $O(N^2 T)$ with dynamic programming.

## Forward pass

$\alpha_t(j)$ is the probability of seeing the first $t$ observations and being in state $j$ at time $t$:

$$\alpha_1(j) = \pi_j \, B_j(o_1)$$

$$\alpha_{t+1}(j) = \Big[ \sum_{i} \alpha_t(i) A_{ij} \Big] B_j(o_{t+1})$$

$$P(O \mid \lambda) = \sum_{j} \alpha_T(j)$$

## Backward pass

$\beta_t(i)$ is the probability of seeing the remaining observations after time $t$, given that the state at time $t$ is $i$:

$$\beta_T(i) = 1$$

$$\beta_t(i) = \sum_{j} A_{ij} \, B_j(o_{t+1}) \, \beta_{t+1}(j)$$

Starting from the beginning, the same probability can be recovered as $P(O \mid \lambda) = \sum_j \pi_j B_j(o_1) \beta_1(j)$. Getting the same number from both directions is a good check.

## State probabilities

Combining both passes gives the probability of being in state $j$ at time $t$, given the whole sequence:

$$\gamma_t(j) = P(q_t = j \mid O, \lambda) = \frac{\alpha_t(j) \, \beta_t(j)}{P(O \mid \lambda)}$$

## What the code does

[hmmfb.py](../hmmfb.py) uses a small example with 2 states and 3 possible observations, $\pi = (0.6, 0.4)$, a $2 \times 2$ matrix $A$ and a $2 \times 3$ matrix $B$, and the observation sequence `[1, 2, 0, 1]`. It prints $\alpha$, $\beta$, the forward and backward probabilities and $\gamma$.

The variable `from_start` controls where the sequence begins. If it is 1, the code first applies one transition to $\pi$ (the first observation is emitted after a step, $\pi A$). If it is 0, the first observation is emitted directly from $\pi$.

[datasethmmfb.py](../datasethmmfb.py) runs the same algorithm on parameters estimated from data. It reads `dataset.csv` with a `state` column and an `obs` column, and counts with add-one smoothing:

$$\pi_i = \frac{\text{count}(q_1 = i) + 1}{\ldots}, \qquad A_{ij} = \frac{\text{count}(i \to j) + 1}{\sum_{j'} \big(\text{count}(i \to j') + 1\big)}, \qquad B_j(k) = \frac{\text{count}(j, k) + 1}{\sum_{k'} \big(\text{count}(j, k') + 1\big)}$$

It then evaluates the first `n = 10` observations. Note that `dataset.csv` is not stored in this repository, so you need to supply it to run this script.

## Library version

[libhmmfb.py](../libhmmfb.py) uses `hmmlearn.hmm.CategoricalHMM` with the same $\pi$, $A$ and $B$. `model.score` gives the log of the forward probability and `predict_proba` gives $\gamma$.
