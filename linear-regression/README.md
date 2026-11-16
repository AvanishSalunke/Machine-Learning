# Linear Regression

Files in this folder:

- [linearregression.py](linearregression.py): gradient descent written step by step with numpy
- [liblinearregression.py](liblinearregression.py): the same task with scikit-learn

## In simple words

Imagine you want to guess the electricity bill of a building from its size, how many people live there, and so on. Linear regression says: each fact pushes the bill up or down by a fixed amount. Add up all the pushes, add a starting amount, and you have your guess.

The "learning" part is finding the right push for each fact. The computer starts with all pushes at zero, checks how wrong its guesses are, then nudges every push a little in the direction that makes the guesses less wrong. It repeats this many times. This nudging method is called **gradient descent**.

## Key words

- **Feature**: one fact about a sample, like site area. We write the features as $x_1, x_2, \dots, x_n$.
- **Weight**: how strongly a feature pushes the answer. One weight $w_j$ per feature.
- **Bias**: the starting amount $b$, the answer when all features are zero.
- **Target**: the real value we want to predict, $y$.
- **Loss**: one number that says how wrong the predictions are. Smaller is better.
- **Learning rate**: the size of each nudge, written $\alpha$.
- **Epoch**: one full round of "predict, measure error, nudge".

## The maths, step by step

**1. The prediction.** For one sample, multiply every feature by its weight and add the bias:

$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = \mathbf{w}^T \mathbf{x} + b$$

($\hat{y}$ is read "y hat" and means "the predicted y".) For all $m$ training rows at once, stack the rows into a matrix $X$ with $m$ rows and $n$ columns:

$$\hat{\mathbf{y}} = X\mathbf{w} + b$$

**2. The loss.** We measure the error of one sample as $(\hat{y}_i - y_i)$. We square it so that errors above and below the real value do not cancel, and so that big mistakes count much more than small ones. Then we average over all samples. The extra $\tfrac12$ is only there to make the derivative neat:

$$J(\mathbf{w}, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2$$

**3. The gradient.** The gradient tells us how the loss changes if we change a weight a tiny bit. Using the chain rule on the loss above:

$$\frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)\, x_{ij} \qquad\Longrightarrow\qquad \frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} X^T (\hat{\mathbf{y}} - \mathbf{y})$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)$$

In words: the push for weight $j$ is the average of "error times feature value". If the model guesses too high on rows where the feature is large, that weight should go down.

**4. The update.** Move each parameter a small step against its gradient, because the gradient points uphill and we want to go downhill:

$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial J}{\partial \mathbf{w}}, \qquad b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$

Repeat for many epochs. The loss falls quickly at first and then flattens out when the model is close to the best line.

**5. Feature scaling.** If one feature is measured in thousands (area) and another in tens (people), the loss surface becomes a long thin valley, and gradient descent zig-zags slowly. So each feature is standardised:

$$x' = \frac{x - \mu}{\sigma}$$

where $\mu$ is the mean and $\sigma$ the standard deviation of that feature **in the training set**. The same $\mu$ and $\sigma$ are reused for the test set, so the test data never leaks into training.

## A small worked example

Three training points: $(x, y) = (1, 2), (2, 3), (3, 5)$. Start with $w = 0$, $b = 0$ and $\alpha = 0.1$.

- Predictions are all 0, so the errors $\hat{y} - y$ are $-2, -3, -5$.
- Loss: $\frac{1}{2 \cdot 3}(4 + 9 + 25) = 6.33$.
- $\frac{\partial J}{\partial w} = \frac{1}{3}\big(1\cdot(-2) + 2\cdot(-3) + 3\cdot(-5)\big) = -7.67$
- $\frac{\partial J}{\partial b} = \frac{1}{3}(-2 - 3 - 5) = -3.33$
- Update: $w = 0 - 0.1 \cdot (-7.67) = 0.767$ and $b = 0 - 0.1\cdot(-3.33) = 0.333$.
- New predictions: $1.10,\ 1.87,\ 2.63$. New loss: about $1.28$.

One step cut the loss from 6.33 to 1.28. More epochs bring it close to the best line, which is $\hat{y} = 1.5x + 0.33$.

## How the code works

1. **Load and prepare.** Read `../datasets/electricity_cost_dataset.csv`. The column `electricity cost` is the target $y$. Text columns, like the structure type, are turned into 0/1 columns with one-hot encoding (`get_dummies(drop_first=True)`). One category per column is dropped, because it can be worked out from the others.
2. **Split.** 80% of the rows for training, 20% for testing, with `random_state=42` so the split is the same every time.
3. **Scale.** `StandardScaler` is fitted on the training part and applied to both parts.
4. **Train.** Weights and bias start at zero. For 1000 epochs: predict, compute the loss, compute `dw` and `db`, update. The loss of every epoch is saved; it is printed every 100 epochs and plotted at the end.
5. **Evaluate on the test set** with the measures below.

Settings: learning rate 0.01, 1000 epochs, full batch (all rows are used in every step).

## How good is the model?

$$MAE = \frac{1}{m}\sum_i |y_i - \hat{y}_i| \qquad MSE = \frac{1}{m}\sum_i (y_i - \hat{y}_i)^2 \qquad RMSE = \sqrt{MSE}$$

- **MAE** is the average miss, in the same unit as the target.
- **RMSE** is similar but punishes big misses more.
- **R squared** compares the model with the dumbest model, which always guesses the average $\bar{y}$:

$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

$R^2 = 1$ is perfect, $R^2 = 0$ is no better than guessing the average.

## Library version

`liblinearregression.py` uses `LinearRegression` from scikit-learn. It does not do any nudging. It finds the best weights in one go by solving the least squares problem exactly (the "normal equation"):

$$\mathbf{w} = (X^T X)^{-1} X^T \mathbf{y}$$

If your gradient descent has run long enough, its weights and bias should be almost the same as the library's. If they differ a lot, try more epochs or a larger learning rate.

## Strengths and weaknesses

- Simple, fast and easy to explain: every weight has a meaning.
- It can only draw straight-line (flat) relationships. Curved relationships need extra features, like squares of the features.
- A single wild outlier can pull the line, because errors are squared.

## Things to try

- Change the learning rate to 0.1 and to 0.001 and watch the loss curve.
- Remove the scaling and see how much slower the loss falls.
- Compare the printed weights with the library's weights.
