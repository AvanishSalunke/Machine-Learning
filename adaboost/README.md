# AdaBoost

Files in this folder:

- [adaboost.py](adaboost.py): AdaBoost with decision stumps, step by step with numpy.
- [libadaboost.py](libadaboost.py): library version with scikit-learn's `AdaBoostClassifier`.

## In simple words

Imagine studying for an exam with a friend. First you both take a practice test and see which questions you got wrong. Then you spend the next session mostly on those questions. After each session you test again and focus on what is still wrong. At the end, you combine what you learnt, giving more trust to the sessions that taught you the most.

AdaBoost (adaptive boosting) does this with very simple models. It trains a **weak learner** (here a tree with just one question), finds which samples it got wrong, **raises the weight of those samples**, and trains the next weak learner, which is therefore forced to pay attention to the hard cases. In the end, all weak learners vote, and the better ones get a louder voice.

## Key words

- **Weak learner**: a model only a little better than guessing. Here it is a **decision stump**: a tree with one split.
- **Sample weight** $w_i$: how much sample $i$ matters. All start equal.
- **Weighted error** $\varepsilon$: the total weight of the samples the stump got wrong.
- **$\alpha$ (alpha)**: the "amount of say" of a stump in the final vote.
- **Boosting**: building models one after another, each correcting the last.

## The maths, step by step

Labels are written $y_i \in \{-1, +1\}$. A stump is described by a feature $j$, a threshold $\theta$ and a side $s \in \{+1, -1\}$:

$$h(\mathbf{x}) = \begin{cases} s & \text{if } x_j \le \theta \\ -s & \text{otherwise} \end{cases}$$

**1. Start with equal weights:** $w_i = 1/m$ for every sample.

**2. For rounds $t = 1, \dots, T$:**

- Choose the stump that has the smallest weighted error:

$$\varepsilon_t = \sum_{i=1}^{m} w_i \cdot \mathbb{1}\big[h_t(\mathbf{x}_i) \ne y_i\big]$$

($\mathbb{1}[\cdot]$ is 1 when the statement is true and 0 otherwise.)

- Compute the stump's amount of say:

$$\alpha_t = \frac{1}{2} \ln \frac{1 - \varepsilon_t}{\varepsilon_t}$$

If $\varepsilon = 0.5$ (pure guessing), then $\alpha = 0$: no say. If $\varepsilon$ is small, then $\alpha$ is large and positive. If $\varepsilon > 0.5$ the stump is worse than guessing and $\alpha$ is negative, which flips its vote.

- Update the sample weights, then divide by their sum $Z_t$ so they add up to 1 again:

$$w_i \leftarrow \frac{w_i \, \exp\!\big(-\alpha_t\, y_i\, h_t(\mathbf{x}_i)\big)}{Z_t}$$

When the stump is right, $y_i h_t(\mathbf{x}_i) = +1$ and the weight is multiplied by $e^{-\alpha}$ (smaller). When the stump is wrong, $y_i h_t = -1$ and the weight is multiplied by $e^{+\alpha}$ (bigger).

**3. The final classifier** is a weighted vote:

$$H(\mathbf{x}) = \mathrm{sign}\left( \sum_{t=1}^{T} \alpha_t\, h_t(\mathbf{x}) \right)$$

## A small worked example

Five samples, all with weight $0.2$. The first stump gets one of them wrong.

- $\varepsilon = 0.2$
- $\alpha = \tfrac12 \ln(0.8 / 0.2) = \tfrac12 \ln 4 = 0.693$
- The wrong sample: $0.2 \times e^{0.693} = 0.2 \times 2 = 0.4$
- Each of the four right samples: $0.2 \times e^{-0.693} = 0.2 \times 0.5 = 0.1$
- The sum is $0.4 + 4 \times 0.1 = 0.8$, so divide by $0.8$: the wrong sample now has weight $0.5$ and each right one has $0.125$.

The sample that was wrong now carries half of the total weight, so the next stump will almost certainly try to get it right.

## How the code works

1. **Load** `../datasets/mushrooms.csv`. All features are letters, so they are turned into 0/1 columns (`get_dummies(drop_first=True)`) and standardised. Edible = 0 and poisonous = 1, converted internally to $-1$ and $+1$. **Split** 80/20 with `random_state=42`.
2. **Weights** start at $1/m$.
3. **For each of 10 rounds:** try every feature, every distinct value of that feature as a threshold, and both sides. For each combination compute the weighted error and keep the best stump. Compute $\alpha$ (a tiny number `1e-9` is added inside the logarithm to avoid dividing by zero). Update and normalise the weights. Save the stump and its $\alpha$. If a stump has zero error the loop stops early.
4. **Predict.** For a test row, add up $\alpha_t \cdot h_t(\mathbf{x})$ over all stumps. If the sum is $\ge 0$ the mushroom is classified as poisonous.
5. **Report** accuracy, precision, recall, F1 and the confusion matrix, and plot the weighted error per round.

## Library version

[libadaboost.py](libadaboost.py) uses `AdaBoostClassifier` with `DecisionTreeClassifier(max_depth=1)` as the stump and 10 estimators. For two classes the scikit-learn weight for a stump is $\ln\frac{1-\varepsilon}{\varepsilon}$, which is exactly twice the $\alpha$ above. Since every stump's weight is doubled, the final sign is the same, so the predictions agree. The per-round errors are taken from `estimator_errors_`.

## Strengths and weaknesses

- Simple idea, strong results, and it needs very few settings.
- It can be sensitive to noisy labels and outliers, because it keeps raising the weight of samples it cannot get right.
- Training is sequential: a round cannot start before the previous one ends.

## Things to try

- Change the number of rounds from 10 to 2, 5 and 50, and watch the weighted error.
- Print the stumps' features to see which features AdaBoost finds most useful.
