# HMM: Forward-Backward Algorithm

Files in this folder:

- [hmmfb.py](hmmfb.py): forward and backward passes on a small example, step by step with numpy.
- [datasethmmfb.py](datasethmmfb.py): the same, with the model learned by counting from a dataset.
- [libhmmfb.py](libhmmfb.py): library version with `hmmlearn`.

## In simple words

Suppose a friend lives far away and you cannot see the weather there. But each day they tell you what they did: walked, shopped or cleaned. The weather (sunny, rainy) is **hidden**. What they did is **observed**. Weather tends to stay the same from one day to the next, and some activities are more likely in some weather.

A **hidden Markov model (HMM)** is exactly this setup. The forward-backward algorithm answers the question: **"How likely is it that I would see this exact list of activities?"** It also tells you, for each day, how likely it was sunny or rainy given everything you were told.

Checking every possible weather sequence would take far too long (with $N$ states and $T$ days there are $N^T$ sequences). The forward-backward algorithm reuses partial results to do it in $N^2 T$ steps.

## Key words

- **State**: a hidden condition (sunny, rainy).
- **Observation**: what we can see (walk, shop, clean).
- **Initial distribution** $\pi$: how likely each state is at the start.
- **Transition matrix** $A$: $A_{ij}$ is the chance of moving from state $i$ to state $j$.
- **Emission matrix** $B$: $B_j(k)$ is the chance that state $j$ produces observation $k$.
- **Markov property**: tomorrow depends only on today's state, not on the past before it.

The model is written $\lambda = (\pi, A, B)$.

## The maths, step by step

We observe $O = o_1, o_2, \dots, o_T$.

**1. The forward variable.** $\alpha_t(j)$ is the probability of seeing the first $t$ observations **and** being in state $j$ at time $t$.

Start: the first state is chosen by $\pi$, and then it produces the first observation.

$$\alpha_1(j) = \pi_j\, B_j(o_1)$$

Step: to be in state $j$ at time $t+1$, we could have come from any state $i$. Add up over all of them, then multiply by the chance that state $j$ produces the new observation:

$$\alpha_{t+1}(j) = \Big[ \sum_{i} \alpha_t(i)\, A_{ij} \Big]\, B_j(o_{t+1})$$

Finish: add up the last column.

$$P(O \mid \lambda) = \sum_{j} \alpha_T(j)$$

**2. The backward variable.** $\beta_t(i)$ is the probability of seeing the **rest** of the observations (after time $t$), given that we are in state $i$ at time $t$. It works from the end to the start.

$$\beta_T(i) = 1$$

$$\beta_t(i) = \sum_{j} A_{ij}\, B_j(o_{t+1})\, \beta_{t+1}(j)$$

The same total probability can be found from this side: $P(O \mid \lambda) = \sum_j \pi_j\, B_j(o_1)\, \beta_1(j)$. Getting the same answer from both directions is a good check of the code.

**3. State probabilities.** Multiplying what came before ($\alpha$) with what comes after ($\beta$) gives the probability of being in state $j$ at time $t$, given the whole observation list:

$$\gamma_t(j) = P(q_t = j \mid O, \lambda) = \frac{\alpha_t(j)\, \beta_t(j)}{P(O \mid \lambda)}$$

**4. Matrix form.** If $\boldsymbol{\alpha}_t$ is the row vector of the $\alpha_t(j)$, one forward step is `(alpha @ A) * B[:, o]`, which is how the code does it.

## A small worked example

Two states and three observation symbols (0, 1, 2).

$$\pi = (0.6,\; 0.4), \qquad A = \begin{pmatrix} 0.7 & 0.3 \\ 0.4 & 0.6 \end{pmatrix}, \qquad B = \begin{pmatrix} 0.1 & 0.4 & 0.5 \\ 0.7 & 0.2 & 0.1 \end{pmatrix}, \qquad O = (1, 2, 0, 1)$$

In this example the first observation is produced after one transition from $\pi$ (the variable `from_start = 1`), so the starting distribution is $\pi A = (0.58,\; 0.42)$.

- $\alpha_1 = (0.58 \cdot 0.4,\; 0.42 \cdot 0.2) = (0.232,\; 0.084)$
- $\alpha_1 A = (0.232\cdot0.7 + 0.084\cdot0.4,\; 0.232\cdot0.3 + 0.084\cdot0.6) = (0.196,\; 0.120)$, and multiplying by $B[:, 2] = (0.5, 0.1)$ gives $\alpha_2 = (0.098,\; 0.012)$
- Continuing in the same way: $\alpha_3 = (0.00734,\; 0.02562)$ and $\alpha_4 = (0.00615,\; 0.00351)$
- $P(O \mid \lambda) = 0.00615 + 0.00351 = 0.00967$

So there is about a 1% chance that this model produces the sequence 1, 2, 0, 1.

## How the code works

**[hmmfb.py](hmmfb.py)**

1. Define $\pi$, $A$, $B$ and the observation list $O$ as above.
2. `from_start` decides how the sequence begins. If it is 1, one transition is applied to $\pi$ first (`start = pi @ A`). If it is 0, the first observation is produced straight from $\pi$.
3. **Forward:** fill the `alpha` table row by row. `forward_prob` is the sum of the last row.
4. **Backward:** fill the `beta` table from the last row up. `backward_prob` is computed from the first row and should equal `forward_prob`.
5. **Gamma:** `alpha * beta / forward_prob`. Every row sums to 1.
6. Print all tables.

**[datasethmmfb.py](datasethmmfb.py)**

Instead of typing the model, it learns $\pi$, $A$ and $B$ from a file `dataset.csv` that has a `state` column and an `obs` column, by **counting with add-one smoothing**:

$$A_{ij} = \frac{\text{count}(i \to j) + 1}{\sum_{j'} \big(\text{count}(i \to j') + 1\big)}, \qquad B_j(k) = \frac{\text{count}(\text{state } j \text{ shows } k) + 1}{\sum_{k'} \big(\text{count}(\text{state } j \text{ shows } k') + 1\big)}$$

and $\pi$ from how often each state occurs. Then it runs the same forward and backward passes on the first 10 observations. The file `dataset.csv` is not included in this repository, so add your own to run this script.

## Library version

[libhmmfb.py](libhmmfb.py) builds a `hmmlearn.hmm.CategoricalHMM` with the same $\pi$, $A$, $B$. `model.score(X)` gives the logarithm of the forward probability, so the code takes `exp` to get $P(O \mid \lambda)$. `model.predict_proba(X)` gives $\gamma$.

## Strengths and weaknesses

- Exact and fast: no guessing.
- For long sequences the probabilities become tiny and can underflow to zero. Real implementations scale each step or work with logarithms.

## Things to try

- Change one entry of $B$ and see how $P(O \mid \lambda)$ changes.
- Check that `forward_prob` and `backward_prob` match.
- Look at $\gamma$ to see on which days the model thinks state 1 was most likely.
