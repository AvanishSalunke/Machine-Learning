import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv("../datasets/heart_cleveland_upload.csv")
X = data.drop("condition", axis=1)
y = data["condition"].values
names = list(X.columns)
X = X.values.astype(float)
sc = StandardScaler()

X = sc.fit_transform(X)

rows, cols = X.shape

components = 2

full = PCA(n_components=cols)

full.fit(X)

values = full.explained_variance_
ratio = full.explained_variance_ratio_
cumulative = np.cumsum(ratio)

model = PCA(n_components=components)

Z = model.fit_transform(X)

back = model.inverse_transform(Z)
error = np.mean((X - back) ** 2)

print("PCA - Library")
print("Eigenvalues:", values)
print("Explained Variance Ratio:", ratio)
print("Cumulative Variance:", cumulative)
print("components for 90% variance:", np.argmax(cumulative >= 0.9) + 1)
print("components kept:", components)
print("Components:")
print(model.components_)
print("first 3 projected rows:")
print(Z[:3])
print("Reconstruction Error:", error)

plt.figure(figsize=(8,5))
plt.bar(range(1, cols + 1), ratio)
plt.plot(range(1, cols + 1), cumulative, marker="o", color="red")
plt.xlabel("Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Scree Plot - Library")
plt.grid()
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(Z[:, 0], Z[:, 1], c=y)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection - Library")
plt.grid()
plt.show()
