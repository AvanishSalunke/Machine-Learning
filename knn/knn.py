import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("../datasets/heart_cleveland_upload.csv")
X = data.drop("condition", axis=1)
y = data["condition"].values
X = X.values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rows, cols = X_train.shape
test_rows = X_test.shape[0]

k = 5
max_k = 20

dist = np.sqrt(np.sum((X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=2))
order = np.argsort(dist, axis=1)

print("first test row")
print("nearest rows", order[0, :k])
print("distances", dist[0, order[0, :k]])
print("labels", y_train[order[0, :k]])

errors = []

for i in range(1, max_k + 1):

    nearest = order[:, :i]
    ones = np.sum(y_train[nearest], axis=1)
    zeros = i - ones

    pred = np.where(ones > zeros, 1, 0)

    errors.append(np.mean(pred != y_test))

    if i == k:
        predicted = pred

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("\nkNN - Scratch")
print("k:", k)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(matrix)
print("best k:", np.argmin(errors) + 1)

plt.figure(figsize=(8,5))
plt.plot(range(1, max_k + 1), errors)
plt.xlabel("k")
plt.ylabel("Test Error")
plt.title("kNN - Scratch")
plt.grid()
plt.show()
