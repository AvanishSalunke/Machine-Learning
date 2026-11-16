# XGBoost

Files: [xgboost.py](xgboost.py) and [libxgboost.py](libxgboost.py)

## Idea

XGBoost (extreme gradient boosting) is a regularised version of [gradient boosting](../gradient-boosting/README.md). Where plain gradient boosting uses only the first derivative of the loss, XGBoost uses a second order Taylor approximation, so it knows both the slope and the curvature. It also adds penalties on the size of the trees.

## Objective

At round $t$ the model adds a tree $f_t$ with leaf weights $w_1, \dots, w_T$ (here $T$ is the number of leaves). The regularised objective is

$$\mathcal{L} = \sum_i L\big(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\big) + \gamma T + \frac{1}{2}\lambda \sum_{j} w_j^2$$

$\gamma$ penalises each extra leaf and $\lambda$ penalises large leaf weights.

## Gradients and Hessians

Expanding the loss to second order around the current prediction gives, for each sample, a gradient $g_i$ and a Hessian $h_i$. For the log loss with $p_i = \sigma(\hat{y}_i)$:

$$g_i = p_i - y_i, \qquad h_i = p_i (1 - p_i)$$

## Optimal leaf weight and split gain

For a leaf containing the set of samples $I$, let $G = \sum_{i \in I} g_i$ and $H = \sum_{i \in I} h_i$. The best leaf weight and its score are

$$w^* = -\frac{G}{H + \lambda}, \qquad \text{score} = \frac{G^2}{H + \lambda}$$

When a node is split into a left and right part, the gain of the split is

$$\text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma$$

The split with the highest gain is chosen. If the gain is negative, the split is not worth it.

## Prediction

$$\hat{y}(\mathbf{x}) = \sum_{t=1}^{T} \nu \, f_t(\mathbf{x}), \qquad p = \sigma(\hat{y})$$

with learning rate $\nu$. Class 1 is predicted when $p \ge 0.5$.

## What the code does

- Data: `mushrooms.csv`, one-hot encoded and standardised, edible = 0 and poisonous = 1, 80/20 split with `random_state=42`.
- Starts from score 0 (probability 0.5). 100 rounds, $\nu = 0.1$, $\lambda = 1$, $\gamma = 0$. Each tree is a stump (one split). At every round the code computes $g$ and $h$, tries every feature and every distinct threshold, and keeps the one with the highest gain. The leaf values are $-G/(H+\lambda)$ on each side.
- The training log loss is printed every 10 rounds and plotted.

## Library version

`libxgboost.py` uses `XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=1, eval_metric="logloss")` from the `xgboost` package and plots the logloss recorded on the training data. The library also applies its default $\lambda = 1$ and $\gamma = 0$, so the main settings match. Details such as the starting score and the split search can differ, so small differences in the numbers are normal.
