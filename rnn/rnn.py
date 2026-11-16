import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("mushrooms.csv")
data = data.replace("?", np.nan)
data = data.fillna(data.mode().iloc[0])
X = data.drop("class", axis=1)
y = data["class"].map({"e": 0, "p": 1}).values
for col in X.columns:
    X[col] = X[col].astype("category").cat.codes
X = X.values.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rows, cols = X_train.shape
symbols = X.max() + 1
X_train = np.eye(symbols)[X_train]
X_test = np.eye(symbols)[X_test]

hidden = 8
epochs = 300
learning_rate = 0.5
small = 0.000001

np.random.seed(42)

Wi = np.random.randn(hidden, symbols) * 0.1
Wh = np.random.randn(hidden, hidden) * 0.3
Wo = np.random.randn(hidden) * 0.1
b = np.zeros(hidden)
bo = 0

errors = []

for epoch in range(epochs):

    h = np.zeros((cols + 1, rows, hidden))

    for t in range(cols):
        h[t + 1] = np.tanh(X_train[:, t, :] @ Wi.T + h[t] @ Wh.T + b)

    z = h[cols] @ Wo + bo
    prob = 1 / (1 + np.exp(-z))
    error = -np.mean(y_train * np.log(prob + small) + (1 - y_train) * np.log(1 - prob + small))
    errors.append(error)

    dz = (prob - y_train) / rows
    dWo = h[cols].T @ dz
    dbo = np.sum(dz)
    dh = np.outer(dz, Wo)

    dWi = np.zeros_like(Wi)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)

    for t in range(cols - 1, -1, -1):
        da = dh * (1 - h[t + 1] ** 2)
        dWi = dWi + da.T @ X_train[:, t, :]
        dWh = dWh + da.T @ h[t]
        db = db + np.sum(da, axis=0)
        dh = da @ Wh

    Wi = Wi - learning_rate * dWi
    Wh = Wh - learning_rate * dWh
    Wo = Wo - learning_rate * dWo
    b = b - learning_rate * db
    bo = bo - learning_rate * dbo

    if (epoch + 1) % 50 == 0:
        print("epoch", epoch + 1, "error", error)

h = np.zeros((len(X_test), hidden))

for t in range(cols):
    h = np.tanh(X_test[:, t, :] @ Wi.T + h @ Wh.T + b)

z = h @ Wo + bo
prob = 1 / (1 + np.exp(-z))
predicted = (prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("\nNaive RNN - Scratch")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(matrix)

plt.figure(figsize=(8,5))
plt.plot(range(1, epochs + 1), errors)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Naive RNN - Scratch")
plt.grid()
plt.show()
