# SVM with Gaussian (RBF) Kernel

Files in this folder:

- [svmkernel.py](svmkernel.py): a kernel SVM trained with gradient descent on the hinge loss, with numpy.
- [libsvmkernel.py](libsvmkernel.py): library version with scikit-learn's `SVC(kernel="rbf")`.

## In simple words

The Gaussian kernel (also called RBF, "radial basis function") measures **how close** two samples are. Two samples that are almost the same get a similarity near 1. Two samples that are far apart get a similarity near 0. Like a bell-shaped hill around each training sample.

The SVM then decides a new sample's class by letting every training sample "vote" with a strength that depends on how close it is. Samples nearby shout loudly, samples far away whisper. This allows very flexible, curvy boundaries, including islands of one class inside another.

## Key words

- **Kernel** $K(\mathbf{x}, \mathbf{z})$: a similarity score between two samples.
- **$\sigma$ (sigma)**: the width of the bell. A small $\sigma$ means only very close samples count, which gives a wiggly boundary. A big $\sigma$ means a wide influence and a smoother boundary.
- **$C$**: how strongly samples that break the margin are punished.
- **Hinge loss**: the penalty for being inside the margin or on the wrong side.
- **Support vectors**: samples with margin $\le 1$.

## The maths, step by step

**1. The Gaussian kernel:**

$$K(\mathbf{x}, \mathbf{z}) = \exp\!\left( -\frac{\lVert \mathbf{x} - \mathbf{z} \rVert^2}{2\sigma^2} \right)$$

The top part is the squared distance between the samples. Same sample: distance 0, so $K = e^0 = 1$. Far away: the exponent is a large negative number, so $K \approx 0$. In scikit-learn the same thing is written with $\gamma = \dfrac{1}{2\sigma^2}$.

**2. The decision function.** Every training sample contributes a similarity, weighted by $w_j$:

$$f(\mathbf{x}) = \sum_{j=1}^{m} w_j\, K(\mathbf{x}_j, \mathbf{x}) + b$$

Predict $+1$ if $f(\mathbf{x}) \ge 0$, else $-1$.

**3. The loss.** The same soft margin loss as in [svm-polynomial](../svm-polynomial/README.md) with the hinge loss averaged over the samples:

$$J = \frac{1}{2} \sum_{j} w_j^2 + C \cdot \frac{1}{m} \sum_{i=1}^{m} \max\big(0,\; 1 - y_i f(\mathbf{x}_i)\big)$$

**4. The gradient**, where $B$ is the set of samples with $y_i f(\mathbf{x}_i) < 1$:

$$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{w} - \frac{C}{m} \sum_{i \in B} y_i\, \mathbf{K}_{i,:}, \qquad \frac{\partial J}{\partial b} = -\frac{C}{m} \sum_{i \in B} y_i$$

**5. The update with a shrinking step:**

$$\alpha_t = \frac{\alpha_0}{1 + d_{decay} \cdot t}, \qquad \mathbf{w} \leftarrow \mathbf{w} - \alpha_t \frac{\partial J}{\partial \mathbf{w}}, \qquad b \leftarrow b - \alpha_t \frac{\partial J}{\partial b}$$

## A small worked example

Two samples $\mathbf{x} = (0, 0)$ and $\mathbf{z} = (1, 1)$. The squared distance is $1 + 1 = 2$.

- With $\sigma = 1$: $K = \exp(-2 / 2) = e^{-1} = 0.368$
- With $\sigma = 3$: $K = \exp(-2 / 18) = 0.895$

With the wide bell ($\sigma = 3$) these two samples are seen as very similar. With the narrow bell ($\sigma = 1$) they are only slightly similar. This is why a small $\sigma$ gives a boundary that hugs the data and can overfit, and a large $\sigma$ behaves almost like a smooth, simple boundary.

## How the code works

1. **Load** `../datasets/heart_cleveland_upload.csv`; the target `condition` becomes $-1$ or $+1$. **Split** 80/20 with `random_state=42` and **standardise** the features (important here, because the kernel uses distances).
2. **Kernel matrices** with two loops: `K_train` is $m \times m$, and `K_test` is test rows $\times$ $m$.
3. **Train** with gradient descent as described above. Every 100 epochs it prints the loss and how many samples are "bad".
4. **Support vectors.** Samples with margin $\le 1$ are printed and counted.
5. **Predict** with the sign of `K_test @ w + b`. Report accuracy, precision, recall, F1 and the confusion matrix, and plot the loss.

Settings: $\sigma = 3$, $C = 10$, $\alpha_0 = 0.01$, decay 0.01, 1000 epochs.

## Library version

[libsvmkernel.py](libsvmkernel.py) uses `SVC(kernel="rbf", C=C / rows, gamma=1 / (2 * sigma ** 2))`. It does not use gradient descent: it solves the dual problem exactly. So that it matches the numpy code, $\gamma$ is computed from $\sigma$, and $C$ is divided by the number of training rows (the numpy loss averages the hinge loss, while scikit-learn sums it). The results should be close, but not identical.

## Strengths and weaknesses

- Very flexible, and works well on many problems with few features.
- Two settings ($C$ and $\sigma$) must be chosen well. Too much flexibility overfits, too little underfits.
- The kernel matrix is $m \times m$, so training is slow on large data.
- Needs scaled features.

## Things to try

- Change $\sigma$ to 0.5, 1 and 10. Look at accuracy and the number of support vectors.
- Compare with the [linear SVM](../svm-linear/README.md) on the same data.
