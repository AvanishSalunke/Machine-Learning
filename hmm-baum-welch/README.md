# HMM: Baum-Welch Algorithm

Files in this folder:

- [hmmbaumwelch.py](hmmbaumwelch.py): Baum-Welch learning on a short sequence, step by step with numpy.
- [datasethmmbaumwelch.py](datasethmmbaumwelch.py): the same, reading the observations from a dataset.
- [libhmmbaumwelch.py](libhmmbaumwelch.py): library version with `hmmlearn`.

## In simple words

So far the [forward-backward](../hmm-forward-backward/README.md) and [Viterbi](../hmm-viterbi/README.md) algorithms needed a model $(\pi, A, B)$ that somebody gave us. Where does the model come from? If we could see the hidden states, we could just count. But the states are hidden, so we can't.

Baum-Welch solves this with a trick of repeated guessing, a method called **expectation-maximisation (EM)**:

1. Start with a random guess of the model.
2. **E-step (expectation):** using the current model, work out how likely each hidden state was at each time. These are "soft counts", fractions instead of whole numbers.
3. **M-step (maximisation):** build a new model by counting with these soft counts, as if they were real.
4. Repeat. Each round, the model explains the observations at least as well as before.

It is like sorting a pile of coins by sound without knowing the types: guess a sorting, work out how each type sounds, re-sort with that knowledge, and repeat until nothing changes.

## Key words

- **Expectation-maximisation (EM)**: the "soft counting, then re-estimating" loop.
- **$\gamma_t(i)$**: probability of being in state $i$ at time $t$.
- **$\xi_t(i,j)$**: probability of being in state $i$ at time $t$ and state $j$ at time $t+1$.
- **Local optimum**: EM stops at a good solution that depends on the starting guess, not necessarily the best one.
- **Log likelihood**: $\ln P(O \mid \lambda)$, how well the model explains the data.

## The maths, step by step

**1. E-step.** Run the forward and backward passes with the current model to get $\alpha$ and $\beta$ (see [forward-backward](../hmm-forward-backward/README.md)). Then:

$$\gamma_t(i) = \frac{\alpha_t(i)\, \beta_t(i)}{P(O \mid \lambda)}$$

$$\xi_t(i, j) = \frac{\alpha_t(i)\, A_{ij}\, B_j(o_{t+1})\, \beta_{t+1}(j)}{P(O \mid \lambda)}$$

Reading $\xi$: $\alpha_t(i)$ is everything before time $t$ ending in $i$, then the step $i \to j$ ($A_{ij}$), then state $j$ produces $o_{t+1}$ ($B_j(o_{t+1})$), then everything after ($\beta_{t+1}(j)$), divided by the total.

**2. M-step.** New parameters are ratios of expected counts.

The starting distribution is the probability of each state at time 1:

$$\hat{\pi}_i = \gamma_1(i)$$

Transitions: expected number of moves $i \to j$ divided by the expected number of times we left $i$:

$$\hat{A}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

Emissions: expected number of times state $j$ produced symbol $k$, divided by the expected number of times we were in $j$:

$$\hat{B}_j(k) = \frac{\sum_{t:\, o_t = k} \gamma_t(j)}{\sum_{t=1}^{T} \gamma_t(j)}$$

The numerator only adds the time steps where the symbol $k$ was observed.

**3. Progress measure.** The code tracks the negative log likelihood

$$\text{error} = -\ln P(O \mid \lambda)$$

It should go down (or stay level) at every epoch. This is guaranteed by the theory of EM.

## A small worked example

Suppose the E-step found, for state 1, these expected counts over the sequence: it moved to state 1 about 3.0 times and to state 2 about 1.0 times. Then

$$\hat{A}_{11} = \frac{3.0}{3.0 + 1.0} = 0.75, \qquad \hat{A}_{12} = \frac{1.0}{3.0 + 1.0} = 0.25$$

and the row sums to 1, as a probability row should. For the emissions, suppose state 1 has total expected presence $\sum_t \gamma_t(1) = 8$, of which 5 happened at times where symbol 0 was seen. Then $\hat{B}_1(0) = 5/8 = 0.625$.

## How the code works

**[hmmbaumwelch.py](hmmbaumwelch.py)**

1. The observation list is a sequence of 16 symbols (0, 1 or 2). The model has 2 hidden states.
2. $\pi$, $A$ and $B$ start as random numbers (seed 42), each row divided by its sum so it is a proper distribution.
3. **50 epochs.** In each: run forward and backward passes, compute $P(O)$ and the error $-\ln P(O)$, compute $\gamma$ and $\xi$ (the $\xi$ is summed over time), and then re-estimate $\pi$, $A$ and $B$ with the formulas above.
4. Print the error every 10 epochs, then the learned $\pi$, $A$, $B$, and plot the error curve.

**[datasethmmbaumwelch.py](datasethmmbaumwelch.py)** does the same, but takes the first 50 values of the `obs` column of `dataset.csv` as the sequence and finds the number of symbols from the data. The file `dataset.csv` is not included in this repository, so add your own to run it.

## Library version

[libhmmbaumwelch.py](libhmmbaumwelch.py) uses `hmmlearn.hmm.CategoricalHMM`, starting from the same random $\pi$, $A$, $B$. To record the error after every epoch, it calls `fit` for exactly one EM round at a time (`n_iter=1`) and measures `-model.score(X)` before each. The final parameters should be very close to the numpy result.

## Strengths and weaknesses

- Learns a model from data with no labels for the hidden states.
- The result depends on the random start, and it can get stuck in a poor local optimum. Run several starts and keep the best.
- The learned states have no names: "state 1" might mean "sunny" or "rainy". You must interpret them afterwards.

## Things to try

- Change the seed and see how the final model and the final error change.
- Try 3 hidden states instead of 2.
- Plot the error curve and see how quickly it flattens.
