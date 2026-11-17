import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_csv("../datasets/heart_cleveland_upload.csv")
X = data.drop("condition", axis=1)
y = data["condition"].values
names = list(X.columns)
X = X.values.astype(float)
sc = StandardScaler()

X = sc.fit_transform(X)

rows, cols = X.shape

k = 2
epochs = 100
errors = []

np.random.seed(42)
start = np.random.choice(rows, k, replace=False)
centroids = X[start].copy()

print("starting rows", start)

for i in range(epochs):

    dist = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dist, axis=1)

    error = np.sum(dist[np.arange(rows), labels])
    errors.append(error)

    new_centroids = centroids.copy()

    for j in range(k):
        if np.sum(labels == j) > 0:
            new_centroids[j] = np.mean(X[labels == j], axis=0)

    moved = np.sum((new_centroids - centroids) ** 2)
    centroids = new_centroids

    print("epoch", i + 1, "inertia =", error, "moved =", moved)

    if moved == 0:
        break

dist = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
labels = np.argmin(dist, axis=1)
inertia = np.sum(dist[np.arange(rows), labels])

sizes = np.bincount(labels, minlength=k)
table = np.zeros((k, 2), dtype=int)

for j in range(k):
    for c in range(2):
        table[j, c] = np.sum((labels == j) & (y == c))

purity = np.sum(np.max(table, axis=1)) / rows
score = silhouette_score(X, labels)

print("\nK-Means - Scratch")
print("clusters:", k)
print("epochs used:", len(errors))
print("Inertia:", inertia)
print("Cluster sizes:", sizes)
print("Cluster vs condition (rows = cluster, columns = condition 0, 1):")
print(table)
print("Purity:", purity)
print("Silhouette Score:", score)
print("Centroids:")
print(centroids)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(errors) + 1), errors)
plt.xlabel("Epoch")
plt.ylabel("Inertia")
plt.title("K-Means - Scratch")
plt.grid()
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(X[:, 0], X[:, 7], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 7], c="red", marker="x", s=200)
plt.xlabel(names[0] + " (scaled)")
plt.ylabel(names[7] + " (scaled)")
plt.title("K-Means Clusters - Scratch")
plt.grid()
plt.show()
