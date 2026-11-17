# Principal Component Analysis (PCA)

Files in this folder:

- [pca.py](pca.py): PCA written step by step with numpy (covariance matrix and eigenvectors).
- [libpca.py](libpca.py): the same task with scikit-learn's `PCA`.

## In simple words

Suppose you describe people with 13 numbers each. Many of those numbers move together: older people tend to have a higher blood pressure and a lower maximum heart rate, and so on. So the 13 numbers repeat each other a lot, and the data really lives in a smaller space than 13 dimensions.

PCA finds new **directions** that describe the data better:

- The first direction is the one along which the data **varies the most**. It is the single best "summary axis".
- The second direction is the next most varying one, at a right angle to the first.
- And so on.

Then we can keep only the first few directions and throw the rest away. We lose a little information but gain a much smaller, simpler dataset that is also easy to plot. Think of the shadow of a 3D object on a wall: choosing the angle of the light well gives a shadow that still shows the shape.

## Key words

- **Variance**: how spread out the values are. PCA treats spread as information.
- **Covariance matrix** $C$: a table that says how every pair of features moves together.
- **Eigenvector**: a direction that the covariance matrix only stretches, never turns. Here it is a principal component.
- **Eigenvalue** $\lambda$: how much stretching, which equals the variance of the data along that direction.
- **Principal component (PC)**: one of the new directions.
- **Projection**: the coordinates of the data in the new directions.
- **Explained variance ratio**: the share of the total variance that a component carries.

## The maths, step by step

**1. Centre and scale the data.** Each feature gets mean 0 and standard deviation 1 (`StandardScaler`). Centring is required, because PCA looks at spread around the middle. Scaling is important because otherwise a feature measured in large numbers (like cholesterol) would look like it has the most variance simply because of its unit.

**2. The covariance matrix.** With the centred data matrix $X$ ($m$ rows, $n$ features):

$$C = \frac{1}{m - 1}\, X^T X$$

The entry $C_{jk}$ is the covariance of features $j$ and $k$. The diagonal entries are the variances. Since the features were standardised, the diagonal is about 1 and the whole matrix is the correlation matrix.

**3. Eigen-decomposition.** Find the directions $\mathbf{v}$ and numbers $\lambda$ that satisfy

$$C\, \mathbf{v} = \lambda\, \mathbf{v}$$

Because $C$ is symmetric, the eigenvectors are at right angles to each other, and the eigenvalues are real numbers. The code uses `np.linalg.eigh`, the routine made for symmetric matrices.

**4. Sort.** Put the eigenvalues in descending order, and the eigenvectors in the same order. Eigenvector number 1 (the biggest $\lambda$) is the first principal component.

**5. The meaning of the eigenvalue.** The variance of the data along eigenvector $\mathbf{v}_j$ is exactly $\lambda_j$. The sum of all eigenvalues is the total variance (here 13, because each of the 13 standardised features has variance 1). The share of component $j$ is

$$\text{explained variance ratio}_j = \frac{\lambda_j}{\sum_{k=1}^{n} \lambda_k}$$

**6. Choose how many components to keep**, $p$. A common rule is to keep enough components to reach about 90% or 95% of the total variance, using the cumulative sum of the ratios. A scree plot (the bars of the ratios) helps: look for the "elbow" where the bars become flat.

**7. Project.** Keep the first $p$ eigenvectors as the columns of a matrix $W$ ($n \times p$). The new data is

$$Z = X\, W$$

Row $i$ of $Z$ is sample $i$ described by $p$ numbers instead of $n$.

**8. Reconstruct** (to measure the information that was lost). Go back to the original space with

$$\hat{X} = Z\, W^T$$

and compute the average squared difference between $X$ and $\hat{X}$. It equals the average of the variance that was thrown away.

**9. The sign problem.** An eigenvector $\mathbf{v}$ and $-\mathbf{v}$ are equally correct. To get a repeatable answer, the code flips each eigenvector so that its entry with the largest absolute value is positive. This is the same rule scikit-learn uses, so the two files agree.

## A small worked example

Four points in two dimensions, already centred (their means are 0):

$$(3, 3), \quad (-3, -3), \quad (1, -1), \quad (-1, 1)$$

- Variances: $\frac{9 + 9 + 1 + 1}{3} = 6.67$ for both $x$ and $y$. Covariance: $\frac{9 + 9 - 1 - 1}{3} = 5.33$.
- Covariance matrix: $C = \begin{pmatrix} 6.67 & 5.33 \\ 5.33 & 6.67 \end{pmatrix}$
- Eigenvalues: $6.67 + 5.33 = 12$ and $6.67 - 5.33 = 1.33$.
- Eigenvectors: $\frac{1}{\sqrt 2}(1, 1)$ for 12, and $\frac{1}{\sqrt 2}(1, -1)$ for 1.33.
- Explained variance ratios: $12 / 13.33 = 90\%$ and $10\%$.

Keep one component. The projections are $\frac{3 + 3}{\sqrt 2} = 4.24$ for the first point, $0$ for $(1, -1)$ and $(-1, 1)$, and $-4.24$ for the last. Reconstructing, the points $(3, 3)$ and $(-3, -3)$ come back exactly, and the other two come back as $(0, 0)$. That is a loss, but a small one, and it matches the 10% of variance that was dropped.

## How the code works

1. **Load** `../datasets/heart_cleveland_upload.csv`. The `condition` column is not used to compute PCA. It is only used to colour the plot. The 13 features are standardised.
2. **Covariance matrix** with one line, `X.T @ X / (rows - 1)`.
3. **Eigen-decomposition** with `np.linalg.eigh`, then sort in descending order, and fix the sign of each eigenvector.
4. **Explained variance.** The ratio and the cumulative ratio are printed, together with the number of components needed to reach 90%.
5. **Project** onto the first 2 components, and reconstruct to measure the error.
6. **Plot** a scree plot (bars for the ratios, a red line for the cumulative sum), and a scatter plot of the two new coordinates, coloured by `condition`.

Settings: 2 components kept.

What it finds on this data: the first component carries 23.6% of the variance and the second 12.3%, so two components keep only 35.9%. Reaching 90% needs 10 of the 13 components. This tells you that the heart data has little redundancy: its features are mostly independent, so PCA cannot compress it much. The first component has positive weights on chest pain type, ST depression (`oldpeak`), slope, number of vessels and thalassaemia, and a negative weight on the maximum heart rate (`thalach`). So it works like a "severity" axis, and indeed the patients with heart disease have a higher average PC1 (about 1.28) than those without (about -1.10). The reconstruction error with two components is 0.64.

## Library version

[libpca.py](libpca.py) uses `PCA(n_components=2)`. It finds the components with the singular value decomposition (SVD), which is a more stable way to get the same eigenvectors. `explained_variance_` equals the eigenvalues, `components_` holds the eigenvectors as rows, and `inverse_transform` does the reconstruction. On this data the two files print the same eigenvalues, ratios, components, projected rows and reconstruction error.

## Strengths and weaknesses

- Reduces the number of features while keeping most of the variance, removes correlation, and makes plotting possible.
- It only finds **straight-line** directions, so curved structure is missed.
- The new axes are mixtures of the original features, so they are harder to explain.
- Variance is not always what you want: a direction with small variance can still be the one that separates the classes.
- Needs centred and (usually) scaled data.

## Things to try

- Set `components = 3` and compare the reconstruction error.
- Remove the scaling and look at how the first component becomes dominated by the features with large units.
- Use the two projected features as the input of [kNN](../knn/README.md) and compare the accuracy with the full data.
