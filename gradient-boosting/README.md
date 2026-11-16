# Gradient Boosting

Files in this folder:

- [gradientboosting.py](gradientboosting.py): gradient boosting with stumps for classification, step by step with numpy.
- [libgradientboosting.py](libgradientboosting.py): library version with scikit-learn's `GradientBoostingClassifier`.

## In simple words

Think of a golfer. The first shot goes in the right direction but lands short. The second shot is not aimed at the hole again from the start: it is aimed at **the remaining distance**. A third shot fixes what is still left, and so on. Every shot corrects the mistake of the shots before it.

Gradient boosting builds a model the same way. It starts with a very simple guess. Then it measures how wrong that guess is for every sample (the **residual**), and trains a small tree to predict those errors. It adds a small part of that tree to the guess. Then it measures the errors again and repeats. Each tree is a small correction.

## Key words

- **Score** $F(\mathbf{x})$: the model's raw output (the log-odds). It becomes a probability through the sigmoid.
- **Residual** $r$: what is still wrong, here $y - p$.
- **Weak learner**: a small tree. We use a stump (one split).
- **Learning rate** (shrinkage) $\nu$: how much of each new tree we add. A small value makes learning slower but safer.
- **Boosting round**: one step of "measure the errors, fit a tree, add it".

## The maths, step by step

**1. From score to probability.** The model works with a score $F$ and turns it into a probability with the sigmoid:

$$p = \sigma(F) = \frac{1}{1 + e^{-F}}$$

**2. The loss** is the log loss, for a true label $y \in \{0, 1\}$:

$$L(y, F) = -\big[\, y \ln p + (1 - y)\ln(1 - p) \,\big]$$

**3. "Gradient" boosting.** We want to lower the loss, so we should move each score a little against the slope of the loss. The slope of the log loss with respect to the score is $p - y$. So the **negative gradient**, the direction to move, is

$$r_i = y_i - p_i$$

This is simply "the true label minus the current probability", the residual. If $y = 1$ and $p = 0.3$, $r = 0.7$: push the score up. If $y = 0$ and $p = 0.8$, $r = -0.8$: push it down.

**4. The starting model.** The best constant score is the log-odds of the overall positive rate $\bar{y}$:

$$F_0 = \ln \frac{\bar{y}}{1 - \bar{y}}$$

**5. Each round $t = 1, \dots, T$:**

1. Compute probabilities $p_i = \sigma(F_{t-1}(\mathbf{x}_i))$ and residuals $r_i = y_i - p_i$.
2. Fit a stump $h_t$ to the residuals by choosing the feature and threshold that minimise the squared error $\sum_i (r_i - h_t(\mathbf{x}_i))^2$. The value of each leaf is the **mean residual** of the samples in it.
3. Add it with the learning rate:

$$F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \nu\, h_t(\mathbf{x})$$

**6. Prediction.** Class 1 if $\sigma(F_T(\mathbf{x})) \ge 0.5$.

## A small worked example

Four samples with labels $y = (1, 1, 0, 0)$.

- $\bar{y} = 0.5$, so $F_0 = \ln(1) = 0$ and every $p = 0.5$.
- Residuals: $r = (0.5,\ 0.5,\ -0.5,\ -0.5)$.
- Suppose a stump splits the first two samples from the last two. Mean residual on the left $= 0.5$, on the right $= -0.5$. The squared error is 0, a perfect fit of the residuals.
- With $\nu = 0.1$: $F_1 = (0.05,\ 0.05,\ -0.05,\ -0.05)$.
- New probabilities: $\sigma(0.05) = 0.5125$ for the first two, $\sigma(-0.05) = 0.4875$ for the last two.

Both groups moved a little toward their correct label. Repeating this 100 times moves them far.

## How the code works

1. **Load** `../datasets/mushrooms.csv`, one-hot encode the features (`get_dummies(drop_first=True)`) and standardise them. Edible = 0, poisonous = 1. **Split** 80/20 with `random_state=42`.
2. **Start** with $F_0$ for every sample, from the log-odds of the training labels.
3. **For each of 100 rounds:**
   - compute probabilities and residuals;
   - try every feature and every distinct threshold; for each, set the left and right leaf values to the mean residual on each side and compute the squared error; keep the best split;
   - update the scores: $F \leftarrow F + 0.1 \times$ the stump's output;
   - compute and store the log loss (printed every 10 rounds).
4. **Predict.** Start every test row at $F_0$, add the contribution of every stump, apply the sigmoid and use the 0.5 threshold.
5. **Report** accuracy and the confusion matrix and plot the loss.

Settings: learning rate 0.1, 100 rounds, stumps (depth 1).

One difference from the full textbook method: the leaf values here are the plain mean residual. scikit-learn goes one step further and rescales each leaf value with a Newton step $\sum r_i / \sum p_i (1 - p_i)$, which is why the two versions do not give identical numbers.

## Library version

[libgradientboosting.py](libgradientboosting.py) uses `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1)`. The loss curve comes from `train_score_`.

## Strengths and weaknesses

- Often one of the most accurate methods for table data, and flexible: it works with many loss functions.
- Trees are built one after another, so it cannot be parallelised across rounds.
- Too many rounds can overfit. The learning rate and the number of rounds should be tuned together: a smaller rate needs more rounds.

## Things to try

- Change the learning rate to 1 and to 0.01 and watch the loss curve.
- Use more than 100 rounds and see where the loss stops improving.
- Read the [XGBoost](../xgboost/README.md) folder to see the second-order version of this idea.
