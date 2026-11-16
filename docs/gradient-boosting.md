# Gradient Boosting

Files: [gradientboosting.py](../gradientboosting.py) and [libgradientboosting.py](../libgradientboosting.py)

## Idea

Gradient boosting builds a model in stages. Each new small tree is trained to correct the mistakes the current model still makes. "Mistakes" are measured as the negative gradient of the loss function with respect to the current predictions, which turns boosting into gradient descent in function space.

$$F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \nu \, h_t(\mathbf{x})$$

where $\nu$ is the learning rate (shrinkage) and $h_t$ is the new tree.

## Binary classification

The model outputs a score $F(\mathbf{x})$ (the log-odds). The probability is the sigmoid of the score:

$$p = \sigma(F) = \frac{1}{1 + e^{-F}}$$

The loss is the log loss:

$$L(y, F) = -\big[\, y \ln p + (1 - y)\ln(1 - p) \,\big]$$

Its derivative with respect to the score is $p - y$, so the negative gradient (the residual) is

$$r_i = y_i - p_i$$

## Algorithm

1. **Initial model**: the constant that minimises the loss is the log-odds of the positive class rate $\bar{y}$:

$$F_0 = \ln \frac{\bar{y}}{1 - \bar{y}}$$

2. For $t = 1, \dots, T$:
   - compute the probabilities $p_i = \sigma(F_{t-1}(\mathbf{x}_i))$ and the residuals $r_i = y_i - p_i$
   - fit a regression stump $h_t$ to the residuals by choosing the feature and threshold that minimise the squared error $\sum (r_i - h_t(\mathbf{x}_i))^2$. The value in each of its two leaves is the mean residual of the samples that fall in it
   - update $F_t = F_{t-1} + \nu h_t$

3. Predict class 1 if $\sigma(F_T(\mathbf{x})) \ge 0.5$.

## What the code does

- Data: `mushrooms.csv`, one-hot encoded and standardised, edible = 0 and poisonous = 1, 80/20 split with `random_state=42`.
- 100 rounds, learning rate 0.1, depth-1 trees (stumps). Every feature and every distinct threshold is tried at each round.
- The log loss on the training set is printed every 10 rounds and plotted.
- The code uses the mean residual as the leaf value. Scikit-learn goes one step further and rescales the leaf values with a Newton step, $\sum r_i / \sum p_i(1-p_i)$. That difference is why the two versions will not give identical numbers.

## Library version

`libgradientboosting.py` uses scikit-learn's `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1)`. The training loss curve comes from `train_score_`.

See [xgboost.md](xgboost.md) for the second order version of this idea.
