import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

np.random.seed(42)
start = np.random.choice(rows, k, replace=False)

print("starting rows", start)

model = KMeans(n_clusters=k, init=X[start], n_init=1, max_iter=epochs, tol=0, algorithm="lloyd", random_state=42)

model.fit(X)

labels = model.labels_
centroids = model.cluster_centers_
inertia = model.inertia_

sizes = np.bincount(labels, minlength=k)
table = np.zeros((k, 2), dtype=int)

for j in range(k):
    for c in range(2):
        table[j, c] = np.sum((labels == j) & (y == c))

purity = np.sum(np.max(table, axis=1)) / rows
score = silhouette_score(X, labels)

print("\nK-Means - Library")
print("clusters:", k)
print("epochs used:", model.n_iter_)
print("Inertia:", inertia)
print("Cluster sizes:", sizes)
print("Cluster vs condition (rows = cluster, columns = condition 0, 1):")
print(table)
print("Purity:", purity)
print("Silhouette Score:", score)
print("Centroids:")
print(centroids)

errors = []

for i in range(1, 11):

    temp = KMeans(n_clusters=i, n_init=10, random_state=42)
    temp.fit(X)
    errors.append(temp.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), errors, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("K-Means Elbow - Library")
plt.grid()
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(X[:, 0], X[:, 7], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 7], c="red", marker="x", s=200)
plt.xlabel(names[0] + " (scaled)")
plt.ylabel(names[7] + " (scaled)")
plt.title("K-Means Clusters - Library")
plt.grid()
plt.show()
