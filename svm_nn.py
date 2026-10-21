import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("heart_cleveland_upload.csv")

X = data.drop("condition", axis = 1)
y = data["condition"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

sc = StandardScaler()

X_train = sc.fit_transform(X_train).T
X_test = sc.transform(X_test).T
y_train = y_train.reshape(1, -1)

n, m = X_train.shape

layers = [n, 3, 2, 1]
L = len(layers) - 1

learning_rate = 0.1
epochs = 1000
errors = []

np.random.seed(42)

W = {}
b = {}

for l in range(1, L + 1):
    W[l] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(1 / layers[l - 1])
    b[l] = np.zeros((layers[l], 1))

for i in range(epochs):
    A = {0: X_train}

    for l in range(1, L + 1):
        Z = np.dot(W[l], A[l - 1]) + b[l]
        A[l] = 1 / (1 + np.exp(-Z))

    pred = A[L]

    dW = {}
    db = {}

    dZ = pred - y_train

    for l in range(L, 0, -1):
        dW[l] = np.dot(dZ, A[l - 1].T) / m
        db[l] = np.sum(dZ, axis = 1, keepdims = True) / m

        if l > 1:
            dZ = np.dot(W[l].T, dZ) * A[l - 1] * (1 - A[l - 1])

    for l in range(1, L + 1):
        W[l] -= learning_rate * dW[l]
        b[l] -= learning_rate * db[l]

    error = -np.mean(y_train * np.log(pred) + (1 - y_train) * np.log(1 - pred))
    errors.append(error)

    if i % 100 == 0:
        print("epoch", i, "loss =", error)

A = X_test

for l in range(1, L + 1):
    Z = np.dot(W[l], A) + b[l]
    A = 1 / (1 + np.exp(-Z))

prob = A.flatten()
predicted = (prob >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, predicted))
print("Precision:", precision_score(y_test, predicted))
print("Recall:", recall_score(y_test, predicted))
print("F1 Score:", f1_score(y_test, predicted))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))

A = X_train

for l in range(1, L + 1):
    Z = np.dot(W[l], A) + b[l]
    A = 1 / (1 + np.exp(-Z))

predicted_train = (A.flatten() >= 0.5).astype(int)
print("Train Accuracy:", accuracy_score(y_train.flatten(), predicted_train))

plt.figure(figsize = (8, 5))
plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Neural Network Loss")
plt.grid()
plt.show()