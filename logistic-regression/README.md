# Logistic Regression

Files in this folder:

- [logisticregression.py](logisticregression.py): gradient descent written step by step with numpy
- [liblogisticregression.py](liblogisticregression.py): the same task with scikit-learn

## In simple words

Despite its name, logistic regression is used to **classify**: is this mushroom edible or poisonous? It works like linear regression first: it adds up weighted features into one score. But a raw score can be any number, like $-7$ or $42$. We want a probability between 0 and 1. So we pass the score through an S-shaped curve, the **sigmoid**, which turns very negative scores into values near 0 and very positive scores into values near 1. A score of 0 becomes exactly 0.5.

Then the rule is simple: if the probability is at least 0.5, say "poisonous", otherwise "edible".

## Key words

- **Score** (or logit) $z$: the weighted sum of the features plus the bias.
- **Sigmoid** $\sigma$: the function that squeezes any number into the range 0 to 1.
- **Probability** $\hat{p}$: the model's belief that the sample belongs to class 1.
- **Threshold**: the cut-off, 0.5 here, where we switch from predicting class 0 to class 1.
- **Cross entropy**: the loss that punishes confident wrong answers heavily.

## The maths, step by step

**1. The score and the probability.**

$$z = \mathbf{w}^T \mathbf{x} + b$$

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \hat{p} = P(y = 1 \mid \mathbf{x}) = \sigma(z)$$

Some values to get a feeling: $\sigma(-4) \approx 0.018$, $\sigma(0) = 0.5$, $\sigma(4) \approx 0.982$. A positive score means class 1 is more likely, so "$\hat{p} \ge 0.5$" and "$z \ge 0$" are the same test.

**2. The loss.** If the true class is 1, we want $\hat{p}$ close to 1, so we use $-\ln \hat{p}$: it is 0 when $\hat{p} = 1$ and grows without limit as $\hat{p} \to 0$. If the true class is 0, we use $-\ln(1 - \hat{p})$. One formula covers both cases, and then we average over all $m$ samples. This is the **binary cross entropy** (or log loss):

$$J = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i) \Big]$$

If $y_i = 1$ only the first term stays, if $y_i = 0$ only the second. The code adds a tiny number (`1e-9`) inside each logarithm so that $\ln 0$ can never appear.

Why not use squared error as in linear regression? With a sigmoid inside, squared error gives a bumpy loss surface with flat areas where learning stalls. Cross entropy gives a smooth bowl with one minimum.

**3. The gradient.** Working through the chain rule, the sigmoid's derivative cancels nicely against the cross entropy and we are left with the same simple shape as in linear regression:

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} X^T (\hat{\mathbf{p}} - \mathbf{y}), \qquad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_i (\hat{p}_i - y_i)$$

**4. The update**, with learning rate $\alpha$:

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}, \qquad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

## A small worked example

One sample with feature $x = 2$ and true class $y = 1$. Start with $w = 0$, $b = 0$, $\alpha = 0.1$.

- $z = 0$, so $\hat{p} = 0.5$ and the loss is $-\ln 0.5 = 0.693$.
- Error $\hat{p} - y = -0.5$.
- $\partial J / \partial w = -0.5 \times 2 = -1$ and $\partial J / \partial b = -0.5$.
- Update: $w = 0.1$ and $b = 0.05$.
- New score $z = 0.1 \cdot 2 + 0.05 = 0.25$, so $\hat{p} = 0.562$ and the loss is $0.576$.

The model became more confident that this sample is class 1, and the loss went down.

## How the code works

1. **Load and prepare.** Read `../datasets/mushrooms.csv`. The target `class` becomes `e` (edible) = 0 and `p` (poisonous) = 1. All feature columns are letters, so they are one-hot encoded into 0/1 columns.
2. **Split** 80/20 with `random_state=42`, then **standardise** the features using the training statistics.
3. **Train.** Start from zero weights. For 1000 epochs: compute scores, probabilities, gradients, update the weights, and record the loss.
4. **Predict.** Compute the probability of each test row and use the 0.5 threshold.
5. **Report** the measures below and plot the loss curve.

Settings: learning rate 0.1, 1000 epochs, full batch.

## How good is the model?

From the confusion matrix we count true positives (TP), false positives (FP), true negatives (TN) and false negatives (FN):

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN} \qquad Precision = \frac{TP}{TP + FP} \qquad Recall = \frac{TP}{TP + FN} \qquad F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$

- **Precision**: of the mushrooms we called poisonous, how many really are?
- **Recall**: of the really poisonous mushrooms, how many did we catch?

For mushrooms, recall matters most: missing a poisonous one is much worse than a false alarm.

## Library version

`liblogisticregression.py` uses `LogisticRegression(max_iter=1000)` from scikit-learn. It minimises the same log loss, but with a smarter optimiser (L-BFGS) than plain gradient descent, and it adds a mild L2 penalty on the weights by default (strength $C = 1$). The results are usually very close to those of the numpy version.

## Strengths and weaknesses

- Fast, simple, gives probabilities, and the weights show which features matter.
- The decision boundary is a straight line (or flat plane). It cannot capture curved boundaries by itself.
- Needs scaled features for gradient descent to work well.

## Things to try

- Change the threshold from 0.5 to 0.3 and see how precision and recall move in opposite directions.
- Try learning rates 0.01 and 1.
