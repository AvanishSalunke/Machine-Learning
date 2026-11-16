# SVM with Polynomial Kernel

Files in this folder:

- [svmpolynomial.py](svmpolynomial.py): a kernel SVM trained with gradient descent on the hinge loss, with numpy.

There is no library file for this kernel. With scikit-learn it would be `SVC(kernel="poly", degree=d)`. The Gaussian kernel has a library version in [svm-kernel](../svm-kernel/README.md).

## In simple words

A straight line cannot separate every set of dots. Think of red dots in the middle of a circle of blue dots: no straight line works. But if we add a new measurement for each dot, such as its distance squared from the centre, the red dots become "low" and the blue dots "high", and a flat cut separates them.

A **kernel** is a shortcut that lets the SVM behave as if it had added many such new measurements, without ever computing them. The **polynomial kernel** acts as if we added all products of the features up to some degree: squares, cubes, and combinations like $x_1 x_2$.

## Key words

- **Kernel** $K(\mathbf{x}, \mathbf{z})$: a number that says how similar two samples are, computed as if in a bigger feature space.
- **Degree** $d$: the highest power of the features that the kernel can use.
- **Kernel matrix**: the table of $K$ values between all pairs of training samples.
- **Support vectors**: the samples with margin $\le 1$, which define the boundary.

## The maths, step by step

**1. The polynomial kernel** used in the code:

$$K(\mathbf{x}, \mathbf{z}) = \left( \frac{\mathbf{x}^T \mathbf{z}}{n} + 1 \right)^{d}$$

where $n$ is the number of features. The "+1" lets lower powers take part too, and dividing the dot product by $n$ keeps the numbers from exploding when there are many features.

- Degree 1: $K = \mathbf{x}^T\mathbf{z}/n + 1$. This is a **linear** kernel (up to a constant and a scale), so the boundary is a straight hyperplane.
- Degree 2: boundaries can be curves such as circles and parabolas.
- Degree 3 and more: more wiggly boundaries, with a greater risk of overfitting.

**2. The decision function.** With a kernel, the model is a weighted sum of similarities to the training samples:

$$f(\mathbf{x}) = \sum_{j=1}^{m} w_j\, K(\mathbf{x}_j, \mathbf{x}) + b$$

A new point is class $+1$ if $f(\mathbf{x}) \ge 0$ and class $-1$ otherwise. (The $m$ weights $w_j$ play the role of $\alpha_j y_j$ in the dual form from [svm-linear](../svm-linear/README.md).)

**3. The loss that is minimised**, with the hinge loss averaged over the samples:

$$J = \frac{1}{2} \sum_{j} w_j^2 + C \cdot \frac{1}{m} \sum_{i=1}^{m} \max\big(0,\; 1 - y_i f(\mathbf{x}_i)\big)$$

The first part keeps the weights small (simple model), the second punishes samples that are inside the margin or misclassified.

**4. The gradient.** The hinge loss is flat for samples that are safely outside the margin, and slopes for the "bad" ones. Let $B$ be the set of bad samples, those with $y_i f(\mathbf{x}_i) < 1$. Then

$$\frac{\partial J}{\partial \mathbf{w}} = \mathbf{w} - \frac{C}{m} \sum_{i \in B} y_i\, \mathbf{K}_{i,:}, \qquad \frac{\partial J}{\partial b} = -\frac{C}{m} \sum_{i \in B} y_i$$

where $\mathbf{K}_{i,:}$ is row $i$ of the kernel matrix. Only the bad samples pull on the weights.

**5. The update with a shrinking step.** The learning rate gets smaller as training goes on, so the steps are big at the start and fine at the end:

$$\alpha_t = \frac{\alpha_0}{1 + d_{decay} \cdot t}, \qquad \mathbf{w} \leftarrow \mathbf{w} - \alpha_t \frac{\partial J}{\partial \mathbf{w}}, \qquad b \leftarrow b - \alpha_t \frac{\partial J}{\partial b}$$

## A small worked example

Two points with $n = 2$ features: $\mathbf{x} = (1, 2)$ and $\mathbf{z} = (3, 1)$. The dot product is $1\cdot 3 + 2 \cdot 1 = 5$, so $5/2 + 1 = 3.5$.

- Degree 1: $K = 3.5$
- Degree 2: $K = 3.5^2 = 12.25$
- Degree 3: $K = 3.5^3 = 42.9$

Higher degree makes big similarities much bigger, which is how the model becomes more flexible and more sensitive.

## How the code works

1. **Load** `../datasets/heart_cleveland_upload.csv`; the target `condition` becomes $-1$ or $+1$. **Split** 80/20 with `random_state=42` and **standardise**.
2. **Kernel matrices.** `K_train` ($m \times m$) holds the kernel between all training pairs. `K_test` (test rows $\times$ $m$) holds the kernel between each test row and each training row.
3. **Train.** Weights $\mathbf{w}$ (one per training sample) and $b$ start at zero. For each epoch: compute scores `K_train @ w + b`, find the bad samples (margin below 1), compute the loss and the gradients, and update with the decayed learning rate.
4. **Support vectors.** After training, samples with margin $\le 1$ are listed and counted.
5. **Predict** with `K_test @ w + b` and the sign rule. Report accuracy, precision, recall, F1 and the confusion matrix, and plot the loss.

Settings: degree 1, $C = 10$, $\alpha_0 = 0.001$, decay 0.01, 2000 epochs. The degree is a single variable at the top, so you can try 2 or 3.

## Strengths and weaknesses

- Can capture curved boundaries without building the extra features.
- The kernel matrix has $m \times m$ entries, so it becomes slow and large for big datasets.
- A high degree overfits easily. Try a few degrees and compare on the test set.

## Things to try

- Set `degree = 2` and `degree = 3` and watch the test accuracy.
- Compare the result with [the Gaussian kernel](../svm-kernel/README.md) and [the linear SVM](../svm-linear/README.md).
