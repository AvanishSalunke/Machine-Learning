# K-Means Clustering

Files in this folder:

- [kmeans.py](kmeans.py): K-means written step by step with numpy.
- [libkmeans.py](libkmeans.py): the same task with scikit-learn's `KMeans`. It also draws the elbow curve.

## In simple words

Imagine a hall full of people and you must place two tables with snacks so that everybody walks as little as possible. A simple way to do it:

1. Put the two tables anywhere.
2. Everyone walks to the nearer table.
3. Each table moves to the **middle of the people** who came to it.
4. Repeat steps 2 and 3 until the tables stop moving.

That is K-means. The "tables" are the **centroids** and the people are the data points. It is **unsupervised**: it never sees the true labels. It just groups the samples so that samples in the same group are close to each other. The $K$ in the name is the number of groups, and you choose it.

## Key words

- **Cluster**: a group of similar samples.
- **Centroid** $\boldsymbol{\mu}_j$: the centre (mean) of cluster $j$.
- **Inertia**: the total squared distance from every sample to the centroid of its cluster. Smaller means tighter clusters.
- **Unsupervised learning**: learning without target labels.
- **Purity** and **silhouette**: two ways to judge the clusters (explained below).

## The maths, step by step

We have $m$ samples $\mathbf{x}_1, \dots, \mathbf{x}_m$ and want $K$ clusters.

**1. The goal.** Find the assignment of samples to clusters, and the centroids, that make the inertia as small as possible:

$$J = \sum_{i=1}^{m} \big\lVert \mathbf{x}_i - \boldsymbol{\mu}_{c(i)} \big\rVert^2$$

Here $c(i)$ is the cluster of sample $i$.

**2. Start.** Pick $K$ different samples at random and use them as the first centroids.

**3. Assignment step.** Put each sample in the cluster of its nearest centroid:

$$c(i) = \arg\min_{j} \big\lVert \mathbf{x}_i - \boldsymbol{\mu}_j \big\rVert^2$$

**4. Update step.** Move every centroid to the mean of its samples. With $S_j$ the set of samples in cluster $j$:

$$\boldsymbol{\mu}_j = \frac{1}{|S_j|} \sum_{i \in S_j} \mathbf{x}_i$$

**5. Repeat** steps 3 and 4 until no centroid moves. If a cluster ever becomes empty, the code leaves its centroid where it was.

**6. Why it always stops.** Each step can only lower $J$ or leave it the same. The assignment step gives every sample the closest centre, so $J$ cannot go up. The update step uses the mean, which is the point with the smallest total squared distance to a set of samples, so $J$ cannot go up either. There are only a finite number of ways to assign samples, so the process must end. But it ends at a **local** minimum, which depends on the starting centroids. A bad start can give a worse result, so people often run K-means several times and keep the best.

**7. Scaling.** Distances are used, so every feature is standardised first, otherwise large-valued features would dominate.

**8. Judging the result.**

- **Purity** (needs true labels, used here only for checking). For every cluster, count the most common true class. Add these counts up and divide by the number of samples:

$$\text{purity} = \frac{1}{m} \sum_{j} \max_{c} \big| S_j \cap \text{class } c \big|$$

- **Silhouette** (needs no labels). For a sample, let $a$ be its average distance to the others in its own cluster and $b$ its average distance to the nearest other cluster. Then

$$s = \frac{b - a}{\max(a, b)}$$

It is near 1 for well-separated clusters, near 0 when clusters overlap, and negative when the sample probably sits in the wrong cluster. The score reported is the average over all samples.

## A small worked example

One-dimensional data: $1, 2, 3, 10, 11, 12$, with $K = 2$. We start with a deliberately poor guess: centroids at $1$ and $2$.

- **Round 1.** Samples 1 goes to the first centroid (distance 0). Sample 2 goes to the second centroid (distance 0). Sample 3 is nearer to centroid 2 (distance 1, against 2). Samples 10, 11, 12 are all nearer to centroid 2. Clusters: $\{1\}$ and $\{2, 3, 10, 11, 12\}$. New centroids: $1$ and $(2 + 3 + 10 + 11 + 12)/5 = 7.6$.
- **Round 2.** Sample 2 is now nearer to centroid 1 (distance 1 against 5.6), and so is sample 3 (2 against 4.6). Samples 10, 11, 12 stay with centroid 2. Clusters: $\{1, 2, 3\}$ and $\{10, 11, 12\}$. New centroids: $2$ and $11$.
- **Round 3.** Nobody changes cluster, so the centroids do not move and the algorithm stops.

The inertia is $(1 + 0 + 1) + (1 + 0 + 1) = 4$. Even from a bad start, the algorithm found the two natural groups.

## How the code works

1. **Load** `../datasets/heart_cleveland_upload.csv`. The label column `condition` is kept aside: K-means does not use it for clustering, it is only used afterwards to check the clusters. The 13 features are standardised. All 297 rows are clustered, because there is nothing to train and test.
2. **Start.** With a fixed seed (42), `np.random.choice` picks $K = 2$ different rows as the first centroids.
3. **Loop** (at most 100 epochs):
   - the table of squared distances from every sample to every centroid is computed in one line with broadcasting, and `argmin` gives the cluster of each sample;
   - the inertia is recorded;
   - each centroid becomes the mean of its samples (or stays if its cluster is empty);
   - it stops when the centroids move by exactly zero.
4. **Report** the number of epochs, the final inertia, the cluster sizes, a table of clusters against the real `condition`, the purity, the silhouette score and the centroids.
5. **Plot** the inertia per epoch, and a scatter plot of two features (age and maximum heart rate) coloured by cluster, with the centroids marked as red crosses.

Settings: $K = 2$, up to 100 epochs, seed 42.

What it finds on this data: it needs 10 epochs, and the inertia falls from 6674 in the first epoch to 3200. The clusters have 112 and 185 patients. One cluster contains 99 patients with heart disease and 13 without. The other has 147 without and 38 with. The purity is about 0.83, so the two groups found with **no labels** match the real diagnosis quite well. The silhouette score is low (about 0.17), a sign that the groups overlap a lot, which is normal for real patient data.

## Library version

[libkmeans.py](libkmeans.py) uses `KMeans(n_clusters=2, init=X[start], n_init=1, tol=0, algorithm="lloyd")`. It is given the same starting rows, so it follows the same steps. `lloyd` is the plain algorithm described above. On this data the two files print the same inertia, cluster sizes, centroids, purity and silhouette. Usually you would let the library choose the start with its smarter method (k-means++) and repeat it 10 times, which is the default.

The library file also draws the **elbow plot**: it runs K-means for $K = 1$ to $10$ and plots the inertia against $K$. The inertia always falls as $K$ grows (at $K = m$ it is 0), so look for the "elbow", the point after which more clusters give only a small gain.

## Strengths and weaknesses

- Simple, fast, and works well on groups that are round and of similar size.
- You must choose $K$ in advance.
- Depends on the starting centroids.
- Bad at long, curved or very different-sized groups, and sensitive to outliers because it uses means.
- Needs scaled numeric features.

## Things to try

- Change $K$ to 3 and 4 and see how inertia, purity and silhouette change.
- Change the seed in `np.random.seed(42)` and see if the result changes.
- Look at the elbow plot and decide which $K$ looks best.
