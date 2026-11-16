# HMM: Baum-Welch Algorithm

Files: [hmmbaumwelch.py](../hmmbaumwelch.py), [datasethmmbaumwelch.py](../datasethmmbaumwelch.py) and [libhmmbaumwelch.py](../libhmmbaumwelch.py)

## The problem

Given only an observation sequence $O$ and the number of hidden states, learn the model $\lambda = (\pi, A, B)$ that makes the sequence most likely. This is the learning problem for [hidden Markov models](hmm-forward-backward.md). The states are hidden, so the parameters cannot just be counted. Baum-Welch is the Expectation-Maximisation (EM) algorithm for HMMs: it alternates between guessing the hidden states softly and re-estimating the parameters, and each round never decreases $P(O \mid \lambda)$.

## E-step

Run the forward and backward passes ($\alpha$ and $\beta$, see [hmm-forward-backward.md](hmm-forward-backward.md)) with the current parameters. Then compute two quantities.

The probability of being in state $i$ at time $t$:

$$\gamma_t(i) = \frac{\alpha_t(i) \, \beta_t(i)}{P(O \mid \lambda)}$$

The probability of being in state $i$ at time $t$ and in state $j$ at time $t+1$:

$$\xi_t(i, j) = \frac{\alpha_t(i) \, A_{ij} \, B_j(o_{t+1}) \, \beta_{t+1}(j)}{P(O \mid \lambda)}$$

## M-step

Re-estimate the parameters from these expected counts:

$$\hat{\pi}_i = \gamma_1(i)$$

$$\hat{A}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

$$\hat{B}_j(k) = \frac{\sum_{t \,:\, o_t = k} \gamma_t(j)}{\sum_{t=1}^{T} \gamma_t(j)}$$

The numerator of $\hat{B}$ only adds up the time steps where symbol $k$ was observed.

## Convergence

EM converges to a local optimum that depends on the starting values. The code tracks the negative log likelihood

$$\text{error} = -\ln P(O \mid \lambda)$$

which should go down at every epoch.

## What the code does

[hmmbaumwelch.py](../hmmbaumwelch.py) learns a model with 2 hidden states and 3 symbols from the sequence `[0, 1, 2, 0, 1, 1, 2, 0, 0, 1, 2, 1, 0, 2, 2, 0]`.

- $\pi$, $A$ and $B$ start as random numbers (seed 42) normalised so each distribution sums to 1.
- 50 epochs of E-step and M-step. The error is printed every 10 epochs and plotted, and the learned $\pi$, $A$ and $B$ are printed.

[datasethmmbaumwelch.py](../datasethmmbaumwelch.py) does the same, but takes the observation sequence from the `obs` column of `dataset.csv` (the first 50 values) and works out the number of symbols from the data. `dataset.csv` is not included in this repository.

## Library version

[libhmmbaumwelch.py](../libhmmbaumwelch.py) uses `hmmlearn.hmm.CategoricalHMM` with the same random starting parameters and the same number of epochs. It calls `fit` for one EM iteration at a time (`n_iter=1`) so that the error can be recorded after every epoch with `-model.score(X)`.
