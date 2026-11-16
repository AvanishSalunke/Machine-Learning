import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("../datasets/mushrooms.csv")
X = data.drop("class", axis=1)
y = data["class"].map({"e": 0, "p": 1}).values
X = pd.get_dummies(X, drop_first=True)
X = X.values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train).T
X_test = sc.transform(X_test).T
y_train = y_train.reshape(1, -1)

n, m = X_train.shape

hidden = [3, 2]
activation = "sigmoid"

layers = [n] + hidden + [1]
L = len(layers) - 1

learning_rate = 1
epochs = 1000
small = 1e-9
errors = []

np.random.seed(42)

W = {}
b = {}

for l in range(1, L + 1):
    W[l] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(1 / layers[l - 1])
    b[l] = np.zeros((layers[l], 1))

for i in range(epochs):

    A = {0: X_train}
    Z = {}

    for l in range(1, L + 1):

        Z[l] = np.dot(W[l], A[l - 1]) + b[l]

        if l == L or activation == "sigmoid":
            A[l] = 1 / (1 + np.exp(-Z[l]))
        elif activation == "tanh":
            A[l] = np.tanh(Z[l])
        elif activation == "relu":
            A[l] = np.maximum(0, Z[l])
        else:
            A[l] = np.maximum(0.001 * Z[l], Z[l])

    pred = A[L]

    dW = {}
    db = {}

    dZ = pred - y_train

    for l in range(L, 0, -1):

        dW[l] = np.dot(dZ, A[l - 1].T) / m
        db[l] = np.sum(dZ, axis=1, keepdims=True) / m

        if l > 1:

            if activation == "sigmoid":
                slope = A[l - 1] * (1 - A[l - 1])
            elif activation == "tanh":
                slope = 1 - A[l - 1] ** 2
            elif activation == "relu":
                slope = (Z[l - 1] > 0).astype(float)
            else:
                slope = np.where(Z[l - 1] > 0, 1, 0.001)

            dZ = np.dot(W[l].T, dZ) * slope

    for l in range(1, L + 1):
        W[l] = W[l] - learning_rate * dW[l]
        b[l] = b[l] - learning_rate * db[l]

    error = -np.mean(y_train * np.log(pred + small) + (1 - y_train) * np.log(1 - pred + small))

    errors.append(error)

    if i % 100 == 0:
        print("epoch", i, "loss =", error)

A = X_test

for l in range(1, L + 1):

    Z = np.dot(W[l], A) + b[l]

    if l == L or activation == "sigmoid":
        A = 1 / (1 + np.exp(-Z))
    elif activation == "tanh":
        A = np.tanh(Z)
    elif activation == "relu":
        A = np.maximum(0, Z)
    else:
        A = np.maximum(0.001 * Z, Z)

prob = A.flatten()
predicted = (prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("\nNeural Network - Scratch")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(matrix)

plt.figure(figsize=(8,5))
plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Neural Network - Scratch")
plt.grid()
plt.show()
