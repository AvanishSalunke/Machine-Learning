# k-Nearest Neighbours (kNN)

Files in this folder:

- [knn.py](knn.py): kNN written step by step with numpy. It also tries every $k$ from 1 to 20.
- [libknn.py](libknn.py): the same task with scikit-learn's `KNeighborsClassifier`.

## In simple words

You move to a new street and want to know if it is a safe place. You do not need a formula. You simply ask the **five nearest neighbours**, and if most of them say "yes, safe", you believe it.

kNN classifies a new sample exactly like that. It looks at the $k$ training samples that are **closest** to the new one and takes a **vote** of their classes. There is no training step at all: the model is just the stored training data. All the work happens when a new sample arrives, which is why kNN is called a **lazy learner**.

## Key words

- **Neighbour**: a training sample that is close to the new sample.
- **$k$**: how many neighbours get a vote. It is chosen by you.
- **Distance**: a number saying how far apart two samples are.
- **Majority vote**: the class that most of the $k$ neighbours have.
- **Lazy learner**: a method that stores the data and does the work only when asked for a prediction.

## The maths, step by step

**1. Distance.** The usual choice is the Euclidean distance, the straight-line distance. For two samples $\mathbf{x}$ and $\mathbf{z}$ with $n$ features:

$$d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{j=1}^{n} (x_j - z_j)^2}$$

**2. Find the neighbours.** For a new sample $\mathbf{x}$, compute its distance to every training sample, sort them, and keep the $k$ smallest. Call the set of these samples $N_k(\mathbf{x})$.

**3. Vote.** Count how many neighbours belong to each class $c$ and choose the biggest count:

$$\hat{y}(\mathbf{x}) = \arg\max_{c} \sum_{i \in N_k(\mathbf{x})} \mathbb{1}\big[y_i = c\big]$$

($\mathbb{1}[\cdot]$ is 1 when the statement is true and 0 otherwise.) If two classes tie, which can happen when $k$ is even, the code picks class 0, the same as scikit-learn picks the smaller label.

**4. Why scaling is a must.** Distance adds up the differences of all features. If one feature is in the hundreds (cholesterol) and another is 0 or 1 (sex), the big one decides everything. So each feature is standardised first: $x' = (x - \mu)/\sigma$, with $\mu$ and $\sigma$ taken from the training set only.

**5. Choosing $k$.**

- $k = 1$: follows every single neighbour, even noisy ones. The boundary is very jagged (overfitting).
- A large $k$: smooths a lot, and in the extreme, with $k$ = the number of training samples, it always predicts the most common class (underfitting).
- A good $k$ is in between. It is found by trying several values and looking at the test error, as the code does for $k = 1, \dots, 20$. For two classes an odd $k$ avoids ties.

## A small worked example

Training points with two features and a class:

- A at $(1, 2)$, class 0
- B at $(2, 1)$, class 1
- C at $(4, 4)$, class 0
- D at $(0, 0)$, class 1
- E at $(3, 1)$, class 1

A new point at $(1, 1)$. Its distances:

- to A: $\sqrt{0^2 + 1^2} = 1.00$
- to B: $\sqrt{1^2 + 0^2} = 1.00$
- to D: $\sqrt{1^2 + 1^2} = 1.41$
- to E: $\sqrt{2^2 + 0^2} = 2.00$
- to C: $\sqrt{3^2 + 3^2} = 4.24$

With $k = 3$ the neighbours are A (class 0), B (class 1) and D (class 1). The vote is 2 to 1 for class 1, so the new point is class 1. With $k = 1$ the nearest are A and B at the same distance, so the answer would depend on the tie rule. That is one more reason to avoid very small $k$.

## How the code works

1. **Load** `../datasets/heart_cleveland_upload.csv`. The target `condition` is 0 (no heart disease) or 1 (heart disease). All features are numbers, so no encoding is needed.
2. **Split** 80/20 with `random_state=42` and **standardise** using the training statistics.
3. **Distances.** One line builds the whole table of distances between every test row and every training row, using numpy broadcasting. `np.argsort` then sorts each row of the table, so the first columns are the nearest neighbours.
4. **Look at one example.** The nearest rows, their distances and their labels are printed for the first test row.
5. **Vote for each $k$ from 1 to 20.** Take the first $k$ columns of the sorted table, count how many neighbours are class 1, and predict class 1 only if they are the majority. The test error for each $k$ is stored.
6. **Report** accuracy, precision, recall, F1 and the confusion matrix for $k = 5$, print the $k$ with the lowest test error, and plot test error against $k$.

Settings: $k = 5$, Euclidean distance, plain majority vote.

## Library version

[libknn.py](libknn.py) uses `KNeighborsClassifier(n_neighbors=5, metric="euclidean")`. `fit` only stores the data, and `kneighbors` returns the nearest rows and distances. The library can use faster search structures (KD-trees and ball trees) so it also works on bigger data, but the answers are the same. On this dataset the two files print identical neighbours, distances, metrics and best $k$.

## Strengths and weaknesses

- Very simple, no training, and it can follow any shape of boundary.
- Slow at prediction time on big data, because it must measure the distance to every stored sample.
- Suffers when there are many features: in high dimensions all points look about equally far apart, and the idea of "near" loses meaning.
- Needs scaled features and a good choice of $k$.

## Things to try

- Look at the error curve: is the best $k$ small or large for this data?
- Remove the scaling and see how much the accuracy drops.
- Change the distance to Manhattan, $\sum_j |x_j - z_j|$, by setting `metric="manhattan"` in the library version.
- Weight each neighbour's vote by $1/d$ so that closer neighbours count more (`weights="distance"` in the library).
