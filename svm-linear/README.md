# Linear SVM

Files: [libsvmlinear.py](libsvmlinear.py) (library version) and [svmlinear.py](svmlinear.py) (currently empty)

## Idea

A linear SVM finds the straight line (or hyperplane in higher dimensions) that separates two classes with the largest margin. With labels $y \in \{-1, +1\}$ the boundary is

$$\mathbf{w}^T \mathbf{x} + b = 0$$

and the two margin lines are $\mathbf{w}^T \mathbf{x} + b = \pm 1$. The distance between them is

$$\text{margin} = \frac{2}{\lVert \mathbf{w} \rVert}$$

so maximising the margin is the same as minimising $\lVert \mathbf{w} \rVert$.

## Soft margin objective

To tolerate overlap between classes, each point gets a slack $\xi_i \ge 0$:

$$\min_{\mathbf{w}, b, \xi} \; \frac{1}{2} \lVert \mathbf{w} \rVert^2 + C \sum_{i=1}^{m} \xi_i \quad \text{subject to} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1 - \xi_i$$

Equivalently, with the hinge loss:

$$J = \frac{1}{2} \lVert \mathbf{w} \rVert^2 + C \sum_i \max\big(0, \; 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b)\big)$$

Large $C$ means few violations are accepted, small $C$ means a wider margin with more violations. The points on or inside the margin are the support vectors.

## Dual form

Solving the problem through its dual gives $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ with $0 \le \alpha_i \le C$, and only support vectors have $\alpha_i > 0$. The data enters only through dot products $\mathbf{x}_i^T \mathbf{x}_j$, which is what makes kernels possible. See [svm-kernel.md](../svm-kernel/README.md) and [svm-polynomial.md](../svm-polynomial/README.md).

## What the code does

[libsvmlinear.py](libsvmlinear.py) uses scikit-learn's `SVC(kernel="linear")` on `heart_cleveland_upload.csv`, predicting `condition` (0 becomes $-1$, 1 stays $+1$) with standardised features and an 80/20 split with `random_state=42`. `C` is set to `10 / number_of_training_rows` so that it matches the averaged hinge loss used in the kernel scripts. It prints $\mathbf{w}$, $b$, the support vectors and their count, and the usual metrics.

[svmlinear.py](svmlinear.py) is empty at the moment. The linear case of the gradient descent implementation is available in [svmpolynomial.py](../svm-polynomial/svmpolynomial.py) with `degree = 1`, which uses a linear kernel (see [svm-polynomial.md](../svm-polynomial/README.md)).
