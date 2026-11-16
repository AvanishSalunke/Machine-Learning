# SVM with Polynomial Kernel

File: [svmpolynomial.py](svmpolynomial.py)

## Idea

This is the same soft margin SVM described in [svm-kernel.md](../svm-kernel/README.md), with a different kernel. A polynomial kernel lets the SVM draw curved boundaries by implicitly using all products of the features up to a given degree.

## Kernel

$$K(\mathbf{x}, \mathbf{z}) = \left( \frac{\mathbf{x}^T \mathbf{z}}{n} + 1 \right)^{d}$$

where $n$ is the number of features and $d$ is the degree. Dividing the dot product by $n$ keeps the kernel values at a reasonable size when there are many features.

- Degree 1 gives $K = \mathbf{x}^T\mathbf{z}/n + 1$, which is a linear kernel (up to scaling and a constant), so the boundary is a straight hyperplane.
- Degree 2 allows quadratic boundaries, degree 3 cubic ones, and so on. Higher degrees fit more complex shapes and overfit more easily.

## Training

Decision function and loss are the same as for the Gaussian kernel:

$$f(\mathbf{x}) = \sum_{j=1}^{m} w_j K(\mathbf{x}_j, \mathbf{x}) + b$$

$$J = \frac{1}{2} \sum_j w_j^2 + C \cdot \frac{1}{m} \sum_{i} \max\big(0, \; 1 - y_i f(\mathbf{x}_i)\big)$$

$$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{w} - \frac{C}{m} \sum_{i \in B} y_i K_{i,:}, \qquad \frac{\partial J}{\partial b} = -\frac{C}{m} \sum_{i \in B} y_i$$

where $B$ is the set of points with margin below 1. Learning rate decays as $\alpha_0 / (1 + d_{decay}\, t)$.

Settings in the file: degree 1, $C = 10$, $\alpha_0 = 0.001$, decay 0.01, 2000 epochs. The degree is a single variable at the top of the file and can be changed.

## Data

`heart_cleveland_upload.csv`, target `condition` mapped to $-1$ and $+1$, standardised features, 80/20 split with `random_state=42`. It prints the support vector count, accuracy, precision, recall, F1 and the confusion matrix, and plots the loss.

## Library version

There is no separate library file for the polynomial kernel. With scikit-learn it would be `SVC(kernel="poly", degree=d)`.
