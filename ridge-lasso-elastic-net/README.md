# Ridge, Lasso and Elastic Net

Files in this folder:

- [ridgelassoelasticnet.py](ridgelassoelasticnet.py): linear regression with a penalty, trained by gradient descent with numpy. It runs four models: no penalty, ridge, lasso and elastic net.
- [libridgelassoelasticnet.py](libridgelassoelasticnet.py): the same four models with scikit-learn's `LinearRegression`, `Ridge`, `Lasso` and `ElasticNet`.

## In simple words

Plain [linear regression](../linear-regression/README.md) tries only to make the training error small. If it has many features, or features that are almost copies of each other, it may do this by using **huge weights** that cancel each other out. The model fits the training data very closely, but the weights are unstable, and small changes in the data can change the predictions a lot. This is overfitting.

**Regularisation** adds a rule to the game: "make the error small, but also keep the weights small, or you pay a fine". The fine is a penalty added to the loss. There are three popular ways to set the fine:

- **Ridge** fines the **squares** of the weights. It makes all the weights smaller, but none of them exactly zero. Like asking everybody in a team to work a bit less.
- **Lasso** fines the **absolute values** of the weights. It pushes the weak features all the way to **exactly zero**, so it also **selects features**. Like sending the least useful team members home.
- **Elastic Net** uses **both** fines, so it gets some of each behaviour.

## Key words

- **Penalty** (or regularisation term): the extra cost added to the loss for large weights.
- **$\alpha$ (alpha)**: how strong the penalty is. $\alpha = 0$ gives plain linear regression. A bigger $\alpha$ shrinks the weights more.
- **L2 norm** $\lVert \mathbf{w} \rVert_2^2 = \sum_j w_j^2$: the sum of squared weights (ridge).
- **L1 norm** $\lVert \mathbf{w} \rVert_1 = \sum_j |w_j|$: the sum of absolute weights (lasso).
- **L1 ratio** $\rho$: in elastic net, how much of the penalty is L1. $\rho = 1$ is lasso, $\rho = 0$ is ridge.
- **Sparse model**: a model in which many weights are exactly zero.
- **Shrinkage**: making weights smaller.

## The maths, step by step

The prediction is the same as in linear regression, $\hat{y} = \mathbf{w}^T \mathbf{x} + b$. Only the loss changes. The bias $b$ is never penalised.

**1. The combined loss** (the form used by scikit-learn's `ElasticNet`, and by the code):

$$J(\mathbf{w}, b) = \underbrace{\frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2}_{\text{error}} \;+\; \underbrace{\alpha \rho \sum_{j} |w_j|}_{\text{L1 penalty}} \;+\; \underbrace{\frac{1}{2}\, \alpha (1 - \rho) \sum_{j} w_j^2}_{\text{L2 penalty}}$$

The three models are special cases:

- **Ridge**: $\rho = 0$, so $J = \text{error} + \tfrac12 \alpha \sum w_j^2$
- **Lasso**: $\rho = 1$, so $J = \text{error} + \alpha \sum |w_j|$
- **Elastic net**: any $\rho$ between 0 and 1
- **No penalty**: $\alpha = 0$

**2. The smooth part: error + L2.** These two parts have an ordinary gradient:

$$\frac{\partial}{\partial \mathbf{w}} = \frac{1}{m} X^T (\hat{\mathbf{y}} - \mathbf{y}) + \alpha (1 - \rho)\, \mathbf{w}$$

The L2 term simply adds a pull towards zero that gets stronger for bigger weights. So a gradient descent step is $\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot (\text{that gradient})$, with learning rate $\eta$.

**3. The L1 part has a corner.** $|w|$ has a sharp point at 0 and no ordinary derivative there. A common fix is the **proximal gradient** method (also called ISTA). After each ordinary step, apply the **soft-threshold** to each weight:

$$w_j \leftarrow \operatorname{sign}(w_j)\, \max\big(|w_j| - \eta\, \alpha \rho,\; 0\big)$$

In words: pull every weight towards zero by a fixed amount $\eta \alpha \rho$, and if that would cross zero, stop at **exactly zero**. This is what makes lasso weights become exactly 0. (The code applies this step after every gradient step. For ridge, $\rho = 0$, so the step does nothing.)

**4. Why L1 gives zeros and L2 does not.** The L2 pull is proportional to the weight: a small weight feels only a tiny pull, so it gets small but never reaches 0. The L1 pull is a **constant** push, no matter how small the weight. A weak feature, whose usefulness is less than the push, is driven all the way to 0 and stays there.

**5. Scaling is required.** The penalty treats every weight equally, so the features must be on the same scale. Otherwise a feature measured in tiny units would need a large weight and be punished unfairly. The code standardises the features first.

**6. Choosing $\alpha$.** Too small and you get plain linear regression. Too large and all weights go to zero and the model only predicts the average (underfitting). The right value is usually found by [cross-validation](../k-fold-cross-validation/README.md).

**7. The scikit-learn conversion for ridge.** `Ridge` minimises $\sum (\hat{y} - y)^2 + a \lVert \mathbf{w} \rVert^2$ without the factor $\frac{1}{2m}$. To get the same model as the loss above, use $a = \alpha \cdot m$. The library file does this with `Ridge(alpha=0.1 * rows)`. `Lasso` and `ElasticNet` use the same form as the loss above, so their `alpha` is used as it is.

## A small worked example

One feature with the data $x = (1, 2, 3)$ and $y = (2, 4, 6)$, so the perfect line is $y = 2x$. Here $m = 3$, $\sum xy = 28$ and $\sum x^2 = 14$. We take $\rho = 0$ for ridge and $\rho = 1$ for lasso, and no bias.

**Ridge.** Setting the derivative to zero gives $w = \dfrac{\sum xy}{\sum x^2 + m \alpha}$.

- $\alpha = 0$: $w = 28 / 14 = 2.00$ (plain regression)
- $\alpha = 1$: $w = 28 / 17 = 1.65$
- $\alpha = 5$: $w = 28 / 29 = 0.97$
- $\alpha = 100$: $w = 28 / 314 = 0.09$

The weight shrinks smoothly towards zero, but it never reaches it.

**Lasso.** Setting the (sub)derivative to zero gives $w = \max\!\Big(\dfrac{\sum xy / m - \alpha}{\sum x^2 / m},\, 0\Big)$, where $\sum xy / m = 9.33$ and $\sum x^2/m = 4.67$.

- $\alpha = 0$: $w = 9.33 / 4.67 = 2.00$
- $\alpha = 2$: $w = (9.33 - 2) / 4.67 = 1.57$
- $\alpha = 9.33$ or more: $w = 0$ exactly

So lasso reaches zero for a large enough penalty and ridge does not. With many features, this is how lasso drops the weak ones.

## How the code works

1. **Load** `../datasets/electricity_cost_dataset.csv`. The target is `electricity cost`. The text column `structure type` is one-hot encoded, giving 10 features in total. **Split** 80/20 with `random_state=42` and **standardise** with the training statistics, exactly as in [linear regression](../linear-regression/README.md).
2. **List the models.** Each entry has a name, the penalty strength $\alpha$ and the L1 ratio $\rho$:
   - no penalty: $\alpha = 0$
   - ridge: $\alpha = 0.1$, $\rho = 0$
   - lasso: $\alpha = 30$, $\rho = 1$
   - elastic net: $\alpha = 20$, $\rho = 0.99$
3. **Train each model** with gradient descent from zero weights: predict, compute the loss (error plus both penalties), compute the gradient of the smooth part, take a step, then apply the soft-threshold.
4. **Evaluate** on the test set with MAE, MSE, RMSE and $R^2$, as in linear regression. Print the weights, the bias and the number of weights that are exactly zero.
5. **Plot** the weights of the four models side by side as bars, one group of bars per feature.

Settings: learning rate 0.05 and 5000 epochs for every model. The $\alpha$ values are different for each model, because the penalties work on different scales. The electricity cost is in thousands, so the weights are in the hundreds, and an L2 penalty of even 1 would shrink them a lot. That is why ridge uses a small $\alpha$, and why the elastic net is almost all L1 ($\rho = 0.99$).

## What it finds on this data

The features in order are: site area, water consumption, recycling rate, utilisation rate, air quality index, issue resolution time, resident count, and the structure types Industrial, Mixed-use and Residential.

- **No penalty:** $R^2 = 0.922$. The biggest weights are site area (793) and structure type Mixed-use (-360).
- **Ridge** ($\alpha = 0.1$): all weights shrink. Site area goes from 793 to 683 and the Mixed-use weight from -360 to -300. $R^2 = 0.914$, a little lower, but the model is more stable. No weight is exactly zero.
- **Lasso** ($\alpha = 30$): three weights become exactly zero: recycling rate, air quality index and structure type Residential. These are the features that add least. $R^2 = 0.916$, almost as good with 3 fewer features.
- **Elastic net** ($\alpha = 20$, $\rho = 0.99$): one weight is exactly zero, and the others shrink between ridge and lasso. $R^2 = 0.891$.

The penalty costs a little accuracy on the training-like data, in return for simpler and more stable models.

## Library version

[libridgelassoelasticnet.py](libridgelassoelasticnet.py) fits the same four models with scikit-learn. Plain `LinearRegression` solves the least squares problem exactly. `Ridge` also has an exact formula. `Lasso` and `ElasticNet` use **coordinate descent**: they improve one weight at a time, with the soft-threshold, until nothing changes. The printed final loss uses the same formula as the numpy file, so the numbers can be compared. On this data the two files give the same weights, the same zero weights and the same $R^2$ (up to the last few decimals).

## Strengths and weaknesses

- Reduce overfitting and make unstable models stable, especially when features are correlated or numerous.
- Lasso and elastic net select features automatically.
- Lasso can pick only one feature out of a group of correlated ones, and which one it picks may change with small changes in the data. Elastic net handles groups better.
- $\alpha$ (and $\rho$) must be tuned, and the features must be scaled.

## Things to try

- Change the $\alpha$ of the lasso to 10, 50 and 100 and watch more weights become zero and $R^2$ fall.
- Increase the ridge $\alpha$ to 1 and see how much the weights shrink.
- Use `LassoCV` or `RidgeCV` in the library version to choose $\alpha$ by cross-validation.
