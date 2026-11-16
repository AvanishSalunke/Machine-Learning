# Linear SVM

Files in this folder:

- [libsvmlinear.py](libsvmlinear.py): library version with scikit-learn's `SVC(kernel="linear")`.
- [svmlinear.py](svmlinear.py): currently empty. The linear case of the numpy gradient descent code is available in [svmpolynomial.py](../svm-polynomial/svmpolynomial.py) with `degree = 1`, because a degree-1 polynomial kernel is a linear kernel. See [svm-polynomial](../svm-polynomial/README.md).

## In simple words

Picture red and blue dots on a sheet of paper. Many straight lines could separate them. A support vector machine picks the line that stays **as far as possible from both colours**. Think of laying the widest possible road between the two groups, and putting the dividing line down the middle of the road. A wide road means the line is not sitting right next to any dot, so small changes or noise are less likely to put a new dot on the wrong side.

The dots that touch the edge of the road (or sit inside it) are the **support vectors**. They are the only dots that decide where the road goes. If you removed all the other dots, the road would not move.

## Key words

- **Hyperplane**: a line in 2D, a flat sheet in 3D, and the same idea in more dimensions. It is the dividing boundary.
- **Margin**: the width of the road between the classes.
- **Support vectors**: the samples on or inside the margin.
- **Slack** $\xi$: how far a sample has gone into the road, or over to the wrong side.
- **C**: how strongly we punish samples that break the margin.
- **Hinge loss**: the penalty a sample pays for being on the wrong side of the margin.

## The maths, step by step

**1. The boundary and the labels.** We label the two classes $y = -1$ and $y = +1$. The boundary is the set of points where

$$\mathbf{w}^T \mathbf{x} + b = 0$$

A new point is classified by the sign of $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$. The two edges of the road are where $f = +1$ and $f = -1$.

**2. The margin.** The distance between the two edges is

$$\text{margin} = \frac{2}{\lVert \mathbf{w} \rVert}$$

so **a short $\mathbf{w}$ means a wide road**. Maximising the margin is the same as minimising $\tfrac12 \lVert \mathbf{w} \rVert^2$.

**3. The hard rule.** A perfectly separated sample satisfies $y_i f(\mathbf{x}_i) \ge 1$: it is on the correct side and outside the road.

**4. The soft rule.** Real data usually overlaps, so we allow breaking the rule at a price. The slack $\xi_i \ge 0$ says how much sample $i$ breaks it:

$$\min_{\mathbf{w}, b, \xi} \; \frac{1}{2} \lVert \mathbf{w} \rVert^2 + C \sum_{i=1}^{m} \xi_i \quad \text{subject to} \quad y_i \big(\mathbf{w}^T \mathbf{x}_i + b\big) \ge 1 - \xi_i$$

The first part wants a wide road, the second part wants few violations. $C$ sets the balance: a big $C$ means "violations are very costly", giving a narrow road that fits the training data closely. A small $C$ gives a wide, forgiving road.

**5. The same thing with hinge loss.** The best slack for each sample is $\xi_i = \max(0,\, 1 - y_i f(\mathbf{x}_i))$, which is the hinge loss:

$$J = \frac{1}{2} \lVert \mathbf{w} \rVert^2 + C \sum_{i} \max\big(0,\; 1 - y_i f(\mathbf{x}_i)\big)$$

Examples with $y = +1$: if $f = 1.5$ the loss is 0 (safe), if $f = 0.3$ the loss is 0.7 (inside the road), if $f = -0.5$ the loss is 1.5 (wrong side).

**6. The dual form** (what the library solves). Using Lagrange multipliers $\alpha_i$, the solution can be written as

$$\mathbf{w} = \sum_{i} \alpha_i y_i \mathbf{x}_i, \qquad 0 \le \alpha_i \le C$$

Only support vectors have $\alpha_i > 0$. The training data enters only through dot products $\mathbf{x}_i^T \mathbf{x}_j$. This is the door that lets us replace the dot product by a **kernel** to get curved boundaries: see [svm-polynomial](../svm-polynomial/README.md) and [svm-kernel](../svm-kernel/README.md).

## A small worked example

One-dimensional data: class $-1$ at $x = 1$ and $x = 2$, class $+1$ at $x = 4$ and $x = 5$. The widest road lies between the closest points, $x = 2$ and $x = 4$. The boundary is in the middle at $x = 3$, and the road edges are at 2 and 4.

- $f(x) = w x + b$ with $f(2) = -1$ and $f(4) = +1$ gives $w = 1$ and $b = -3$.
- Margin: $2 / \lVert w \rVert = 2$, matching the road width from 2 to 4.
- Support vectors: $x = 2$ and $x = 4$.
- A new point $x = 3.5$: $f = 0.5 > 0$, so class $+1$.

## How the code works

[libsvmlinear.py](libsvmlinear.py):

1. **Load** `../datasets/heart_cleveland_upload.csv`. The target `condition` (0 or 1) is changed to $-1$ and $+1$. All features are numbers, so no encoding is needed.
2. **Split** 80/20 with `random_state=42` and **standardise** the features.
3. **Fit** `SVC(kernel="linear", C=C / rows)` with $C = 10$. The $C$ is divided by the number of training rows to match the scripts that average the hinge loss over the rows instead of summing it.
4. **Print** the weights $\mathbf{w}$, the bias $b$, the support vectors and how many there are.
5. **Predict** the test rows and report accuracy, precision, recall, F1 and the confusion matrix.

## Strengths and weaknesses

- Works well when the number of features is large, and finds a robust boundary.
- Only a few support vectors are needed to describe the model.
- Needs scaled features, and a straight boundary cannot separate data that is curved in nature. That is what kernels are for.

## Things to try

- Change $C$ to 1 and to 1000 and watch the number of support vectors change.
- Compare with the kernel versions on the same data.
