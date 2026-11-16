# Linear Regression

Files: [linearregression.py](../linearregression.py) and [liblinearregression.py](../liblinearregression.py)

## Idea

Linear regression predicts a continuous number as a weighted sum of the input features. It finds the weights that make the predictions as close as possible to the real values.

For one sample with features $x_1, \dots, x_n$:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = \mathbf{w}^T \mathbf{x} + b$$

For the whole training set with $m$ rows, in matrix form:

$$\hat{\mathbf{y}} = X\mathbf{w} + b$$

## Loss function

The code uses the mean squared error with a factor of one half, which makes the derivative cleaner:

$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

## Gradient descent

The gradients of the loss with respect to the weights and the bias are:

$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} X^T (\hat{\mathbf{y}} - \mathbf{y})$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$

Each epoch moves the parameters a small step against the gradient, with learning rate $\alpha$:

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}, \qquad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

## What the code does

- Data: `electricity_cost_dataset.csv`. The target column is `electricity cost`.
- Categorical columns (like structure type) are one-hot encoded with `pd.get_dummies(drop_first=True)`.
- The data is split 80/20 into train and test with `random_state=42`.
- Features are standardised with `StandardScaler`, so each feature has mean 0 and standard deviation 1: $x' = (x - \mu) / \sigma$. The scaler is fitted on the training data only and then applied to the test data. This keeps gradient descent stable.
- Weights and bias start at zero. Training is full-batch gradient descent with learning rate 0.01 for 1000 epochs. The loss of every epoch is stored and plotted.

## Evaluation

$$MAE = \frac{1}{m}\sum |y_i - \hat{y}_i|, \qquad MSE = \frac{1}{m}\sum (y_i - \hat{y}_i)^2, \qquad RMSE = \sqrt{MSE}$$

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

$R^2$ close to 1 means the model explains most of the variation in the target.

## Library version

`liblinearregression.py` uses scikit-learn's `LinearRegression`. It does not run gradient descent. It solves the least squares problem directly, which gives the weights that minimise the same squared error:

$$\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$$

The weights from gradient descent should end up very close to this exact solution when training has converged, so the two files are a good way to check each other.
