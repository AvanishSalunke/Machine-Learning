# Logistic Regression

Files: [logisticregression.py](logisticregression.py) and [liblogisticregression.py](liblogisticregression.py)

## Idea

Logistic regression is used for classification. It computes a linear score like linear regression, then squashes the score into a probability between 0 and 1 with the sigmoid function.

$$z = \mathbf{w}^T \mathbf{x} + b$$

$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \hat{p} = P(y = 1 \mid \mathbf{x}) = \sigma(z)$$

The predicted class is 1 when $\hat{p} \ge 0.5$ (which is the same as $z \ge 0$), otherwise 0.

## Loss function

Binary cross entropy (log loss):

$$J = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \Big]$$

The code adds a tiny constant (`small = 1e-9`) inside each logarithm so that $\log(0)$ never happens.

## Gradient descent

Because the sigmoid and the cross entropy fit together, the gradient has the same neat form as in linear regression:

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} X^T (\hat{\mathbf{p}} - \mathbf{y}), \qquad \frac{\partial J}{\partial b} = \frac{1}{m} \sum (\hat{p}_i - y_i)$$

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}, \qquad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

## What the code does

- Data: `mushrooms.csv`. The target `class` is mapped as edible = 0 and poisonous = 1.
- All feature columns are categorical, so they are one-hot encoded with `get_dummies(drop_first=True)` and then standardised.
- 80/20 train and test split with `random_state=42`.
- Weights and bias start at zero. Full-batch gradient descent with learning rate 0.1 for 1000 epochs. The loss is stored every epoch and plotted.
- Prediction on the test set uses the 0.5 threshold.

## Evaluation

With TP, FP, TN, FN taken from the confusion matrix:

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}, \qquad Precision = \frac{TP}{TP + FP}$$

$$Recall = \frac{TP}{TP + FN}, \qquad F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}$$

## Library version

`liblogisticregression.py` uses scikit-learn's `LogisticRegression(max_iter=1000)`. It minimises the same log loss, but with the L-BFGS optimiser instead of plain gradient descent, and it adds L2 regularisation by default ($C = 1$). Results are usually very close to the gradient descent version.
