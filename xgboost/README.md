# XGBoost

Files in this folder:

- [xgboost.py](xgboost.py): XGBoost-style boosting with stumps, step by step with numpy.
- [libxgboost.py](libxgboost.py): library version with the `xgboost` package.

## In simple words

[Gradient boosting](../gradient-boosting/README.md) looks at the **slope** of the loss to decide which way to move each score. XGBoost (extreme gradient boosting) also looks at the **curvature**: how quickly the slope itself changes. Think of walking down a hill in fog. The slope tells you the direction. The curvature tells you how big a step is safe: on a gentle, curved bowl you can take a bigger step, on a steep, bending slope you should be careful.

On top of that, XGBoost **penalises complicated trees**, so it prefers small, simple trees with modest leaf values. This makes it less likely to overfit, and a big reason why it is so popular.

## Key words

- **Gradient** $g_i$: the first derivative of the loss for sample $i$ (the slope).
- **Hessian** $h_i$: the second derivative (the curvature).
- **$\lambda$ (lambda)**: a penalty on large leaf values. Larger means more cautious trees.
- **$\gamma$ (gamma)**: a penalty for every extra leaf. A split must gain more than $\gamma$ to be worth it.
- **Leaf weight** $w$: the value a leaf adds to the score.
- **Learning rate** $\nu$: how much of each tree is added.

## The maths, step by step

**1. The objective.** At each round we add a tree $f_t$. The goal is to lower the loss plus a penalty on the tree:

$$\mathcal{L} = \sum_i L\big(y_i,\; \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\big) + \gamma\, T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

($T$ is the number of leaves and $w_j$ the value of leaf $j$.)

**2. A second-order approximation.** Instead of using the loss directly, XGBoost replaces it by a parabola built from the slope and curvature at the current prediction. For the log loss with $p_i = \sigma(\hat{y}_i)$:

$$g_i = p_i - y_i, \qquad h_i = p_i (1 - p_i)$$

**3. The best value for a leaf.** For the set of samples $I$ in one leaf, let $G = \sum_{i \in I} g_i$ and $H = \sum_{i \in I} h_i$. Minimising the parabola plus the penalty gives

$$w^* = -\frac{G}{H + \lambda}$$

In words: move against the total slope, divided by the total curvature (plus a safety amount $\lambda$).

**4. How good is a leaf?** Putting $w^*$ back into the objective gives a quality score for the leaf: $\dfrac{G^2}{H + \lambda}$. Bigger is better.

**5. The gain of a split.** Splitting a node into left and right parts, the gain is the sum of the two new leaf scores minus the old one, minus the price of one more leaf:

$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

The tree takes the split with the highest gain. If the best gain is negative, the split is not worth making.

**6. The prediction.** The score is the starting score plus all trees, each scaled by $\nu$, then through the sigmoid:

$$\hat{y}(\mathbf{x}) = \sum_{t=1}^{T} \nu\, f_t(\mathbf{x}), \qquad p = \sigma(\hat{y})$$

## A small worked example

Four samples with labels $y = (1, 1, 0, 0)$. At the start the score is 0, so every $p = 0.5$.

- $g = p - y = (-0.5,\ -0.5,\ 0.5,\ 0.5)$ and every $h = 0.25$.
- Split the first two from the last two, with $\lambda = 1$ and $\gamma = 0$:
  - Left: $G_L = -1$, $H_L = 0.5$. Score $= 1 / 1.5 = 0.667$.
  - Right: $G_R = 1$, $H_R = 0.5$. Score $= 0.667$.
  - Before the split: $G = 0$, $H = 1$, score $= 0 / 2 = 0$.
- Gain $= \tfrac12 (0.667 + 0.667 - 0) = 0.667$.
- Leaf values: left $w = -(-1)/1.5 = +0.667$, right $w = -(1)/1.5 = -0.667$.
- With $\nu = 0.1$ the scores change to $+0.067$ and $-0.067$.

The tree pushes the first two samples up and the last two down, a little.

## How the code works

1. **Load** `../datasets/mushrooms.csv`, one-hot encode and standardise. Edible = 0, poisonous = 1. **Split** 80/20 with `random_state=42`.
2. **Start** with a score of 0 for every sample.
3. **For each of 100 rounds:**
   - compute $p$, then $g = p - y$ and $h = p(1-p)$;
   - try every feature and every distinct threshold. For each, compute $G_L, H_L, G_R, H_R$ and the gain from the formula above. Keep the split with the highest gain;
   - the two leaf values are $-G/(H + \lambda)$ for each side;
   - add $\nu$ times the stump to the scores and compute the log loss (printed every 10 rounds).
4. **Predict.** Add up the stumps for each test row, apply the sigmoid and use the 0.5 threshold.
5. **Report** accuracy and the confusion matrix and plot the loss.

Settings: $\nu = 0.1$, $\lambda = 1$, $\gamma = 0$, 100 rounds, stumps.

## Library version

[libxgboost.py](libxgboost.py) uses `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, eval_metric="logloss")` and plots the training log loss. The library also uses $\lambda = 1$ and $\gamma = 0$ by default, so the main settings match. Details such as the starting score and the way it searches for splits can differ, so small differences in the numbers are normal.

## Strengths and weaknesses

- Very accurate and fast on table data, with built-in protection against overfitting.
- Many settings to tune ($\nu$, depth, $\lambda$, $\gamma$, number of rounds).
- Less readable than a single tree.

## Things to try

- Change $\lambda$ to 0 and to 10. Larger $\lambda$ makes leaf values smaller.
- Set $\gamma$ to 1 and see which splits are rejected.
- Compare the loss curve with [gradient boosting](../gradient-boosting/README.md).
