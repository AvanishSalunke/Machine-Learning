import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../datasets/heart_cleveland_upload.csv")
X = data.drop("condition", axis=1)
y = data["condition"].values
names = list(X.columns)
X = X.values.astype(float)
sc = StandardScaler()

X = sc.fit_transform(X)

rows, cols = X.shape

components = 2

cov = np.dot(X.T, X) / (rows - 1)

values, vectors = np.linalg.eigh(cov)

order = np.argsort(values)[::-1]
values = values[order]
vectors = vectors[:, order]

for j in range(cols):
    if vectors[np.argmax(np.abs(vectors[:, j])), j] < 0:
        vectors[:, j] = -vectors[:, j]

ratio = values / np.sum(values)
cumulative = np.cumsum(ratio)

W = vectors[:, :components]
Z = np.dot(X, W)

back = np.dot(Z, W.T)
error = np.mean((X - back) ** 2)

print("PCA - Scratch")
print("Covariance Matrix shape:", cov.shape)
print("Eigenvalues:", values)
print("Explained Variance Ratio:", ratio)
print("Cumulative Variance:", cumulative)
print("components for 90% variance:", np.argmax(cumulative >= 0.9) + 1)
print("components kept:", components)
print("Components:")
print(W.T)
print("first 3 projected rows:")
print(Z[:3])
print("Reconstruction Error:", error)

plt.figure(figsize=(8,5))
plt.bar(range(1, cols + 1), ratio)
plt.plot(range(1, cols + 1), cumulative, marker="o", color="red")
plt.xlabel("Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot - Scratch")
plt.grid()
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(Z[:, 0], Z[:, 1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection - Scratch")
plt.grid()
plt.show()
