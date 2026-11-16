# AdaBoost

Files: [adaboost.py](adaboost.py) and [libadaboost.py](libadaboost.py)

## Idea

AdaBoost (adaptive boosting) builds a strong classifier from many weak ones. The weak learners are trained one after another. After each round, the training samples that were classified wrongly get a larger weight, so the next learner is forced to concentrate on the hard cases. The final prediction is a weighted vote of all learners, where better learners get a bigger say.

The weak learner here is a **decision stump**: a tree with just one split, $h(\mathbf{x}) = s$ if $x_j \le \theta$ and $-s$ otherwise, where $s \in \{+1, -1\}$ is the side.

## Algorithm

Labels are $y_i \in \{-1, +1\}$. Start with equal weights $w_i = 1/m$. For rounds $t = 1, \dots, T$:

1. Pick the stump (feature $j$, threshold $\theta$, side $s$) with the lowest weighted error

$$\varepsilon_t = \sum_{i} w_i \cdot \mathbb{1}\big[h_t(\mathbf{x}_i) \ne y_i\big]$$

2. Compute the stump's weight (its "amount of say"):

$$\alpha_t = \frac{1}{2} \ln \frac{1 - \varepsilon_t}{\varepsilon_t}$$

   A stump with error 0.5 (random guessing) gets $\alpha = 0$, and a more accurate stump gets a larger $\alpha$.

3. Update the sample weights and normalise:

$$w_i \leftarrow \frac{w_i \, e^{-\alpha_t y_i h_t(\mathbf{x}_i)}}{Z_t}$$

   Correctly classified samples ($y_i h_t = +1$) shrink, wrongly classified ones ($y_i h_t = -1$) grow. $Z_t$ is the sum of the new weights, so they add to 1.

Final classifier:

$$H(\mathbf{x}) = \mathrm{sign}\left( \sum_{t=1}^{T} \alpha_t h_t(\mathbf{x}) \right)$$

## What the code does

- Data: `mushrooms.csv`, one-hot encoded with `get_dummies(drop_first=True)` and standardised. Edible = 0, poisonous = 1, internally converted to $-1$ and $+1$. 80/20 split with `random_state=42`.
- For each of 10 rounds it tries every feature, every distinct value of the feature as the threshold, and both sides, and keeps the stump with the smallest weighted error. A tiny constant (`1e-9`) inside the logarithm avoids dividing by zero when the error is 0. If a round reaches zero error the loop stops early.
- The error and $\alpha$ of every round are printed, and the weighted error per round is plotted.
- Prediction sums $\alpha_t h_t(\mathbf{x})$ over the stumps and classifies as poisonous when the sum is $\ge 0$.

## Library version

`libadaboost.py` uses scikit-learn's `AdaBoostClassifier` with `DecisionTreeClassifier(max_depth=1)` as the stump and 10 estimators. For two classes the scikit-learn weight is $\ln\frac{1-\varepsilon}{\varepsilon}$, which is exactly twice the $\alpha$ above, so the predicted classes are the same. The per-round weighted errors come from `estimator_errors_`.
