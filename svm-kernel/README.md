# SVM with Gaussian (RBF) Kernel

Files: [svmkernel.py](svmkernel.py) and [libsvmkernel.py](libsvmkernel.py)

## Idea

A support vector machine separates two classes with the boundary that has the widest possible margin. Labels are written as $y \in \{-1, +1\}$ and the decision function is

$$f(\mathbf{x}) = \mathrm{sign}\big(\mathbf{w}^T \mathbf{x} + b\big)$$

Points that lie on or inside the margin ($y f(\mathbf{x}) \le 1$) are the support vectors. Only they decide where the boundary goes.

## Soft margin and hinge loss

Real data is rarely perfectly separable, so the soft margin SVM allows some points to break the margin and pays a penalty for it. The penalty is the hinge loss:

$$\ell(y, f) = \max(0, \; 1 - y f(\mathbf{x}))$$

The parameter $C$ controls the trade-off: large $C$ punishes violations strongly, small $C$ prefers a wider margin.

## The kernel trick

A kernel computes an inner product in a higher dimensional space without going there. The Gaussian (RBF) kernel is

$$K(\mathbf{x}, \mathbf{z}) = \exp\!\left( -\frac{\lVert \mathbf{x} - \mathbf{z} \rVert^2}{2\sigma^2} \right)$$

It is close to 1 for nearby points and close to 0 for distant ones, and $\sigma$ sets how far the influence of a point reaches. With a kernel, the decision function is written as a combination of kernel values to the training points:

$$f(\mathbf{x}) = \sum_{j=1}^{m} w_j \, K(\mathbf{x}_j, \mathbf{x}) + b$$

## Training in the code

The code builds the kernel matrix $K$ ($m \times m$ for training, test rows by training rows for testing) and then trains by sub-gradient descent on

$$J = \frac{1}{2} \sum_j w_j^2 + C \cdot \frac{1}{m} \sum_{i=1}^{m} \max\big(0, \; 1 - y_i f(\mathbf{x}_i)\big)$$

Let $B$ be the set of "bad" points with $y_i f(\mathbf{x}_i) < 1$. The sub-gradients are

$$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{w} - \frac{C}{m} \sum_{i \in B} y_i \, K_{i,:}, \qquad \frac{\partial J}{\partial b} = -\frac{C}{m} \sum_{i \in B} y_i$$

The learning rate decays over time: $\alpha_t = \alpha_0 / (1 + d \cdot t)$.

Settings: $\sigma = 3$, $C = 10$, $\alpha_0 = 0.01$, decay $d = 0.01$, 1000 epochs. After training, points with margin $\le 1$ are reported as support vectors. A test point is classified as $+1$ if its score is $\ge 0$.

## Data

`heart_cleveland_upload.csv`, predicting the `condition` column (0 becomes $-1$, 1 stays $+1$). Features are standardised with `StandardScaler`, using an 80/20 split with `random_state=42`. The code prints the number of support vectors, accuracy, precision, recall, F1 and the confusion matrix, and plots the loss.

## Library version

`libsvmkernel.py` uses scikit-learn's `SVC(kernel="rbf")`. SVC solves the dual problem exactly instead of using gradient descent. To match the code version, `gamma = 1 / (2 sigma^2)` and `C` is divided by the number of training rows, because the code averages the hinge loss over the rows while SVC sums it.

See also [svm-polynomial.md](../svm-polynomial/README.md) and [svm-linear.md](../svm-linear/README.md).
