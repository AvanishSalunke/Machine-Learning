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
epochs = 500
learning_rate = 0.1
small = 0.000001

np.random.seed(42)

Wz = np.random.randn(hidden, symbols) * 0.1
Uz = np.random.randn(hidden, hidden) * 0.3
bz = np.zeros(hidden)
Wr = np.random.randn(hidden, symbols) * 0.1
Ur = np.random.randn(hidden, hidden) * 0.3
br = np.zeros(hidden)
Wn = np.random.randn(hidden, symbols) * 0.1
Un = np.random.randn(hidden, hidden) * 0.3
bn = np.zeros(hidden)
Wo = np.random.randn(hidden) * 0.1
bo = 0

errors = []

for epoch in range(epochs):

    h = np.zeros((cols + 1, rows, hidden))
    zg = np.zeros((cols, rows, hidden))
    rg = np.zeros((cols, rows, hidden))
    ng = np.zeros((cols, rows, hidden))

    for t in range(cols):
        x = X_train[:, t, :]
        zg[t] = 1 / (1 + np.exp(-(x @ Wz.T + h[t] @ Uz.T + bz)))
        rg[t] = 1 / (1 + np.exp(-(x @ Wr.T + h[t] @ Ur.T + br)))
        ng[t] = np.tanh(x @ Wn.T + (rg[t] * h[t]) @ Un.T + bn)
        h[t + 1] = (1 - zg[t]) * h[t] + zg[t] * ng[t]

    z = h[cols] @ Wo + bo
    prob = 1 / (1 + np.exp(-z))
    error = -np.mean(y_train * np.log(prob + small) + (1 - y_train) * np.log(1 - prob + small))
    errors.append(error)

    dz = (prob - y_train) / rows
    dWo = h[cols].T @ dz
    dbo = np.sum(dz)
    dh = np.outer(dz, Wo)

    dWz = np.zeros_like(Wz)
    dUz = np.zeros_like(Uz)
    dbz = np.zeros_like(bz)
    dWr = np.zeros_like(Wr)
    dUr = np.zeros_like(Ur)
    dbr = np.zeros_like(br)
    dWn = np.zeros_like(Wn)
    dUn = np.zeros_like(Un)
    dbn = np.zeros_like(bn)

    for t in range(cols - 1, -1, -1):
        x = X_train[:, t, :]

        dzg = dh * (ng[t] - h[t])
        dng = dh * zg[t]
        dh_prev = dh * (1 - zg[t])

        da_n = dng * (1 - ng[t] ** 2)
        dWn = dWn + da_n.T @ x
        dUn = dUn + da_n.T @ (rg[t] * h[t])
        dbn = dbn + np.sum(da_n, axis=0)
        d_rh = da_n @ Un
        drg = d_rh * h[t]
        dh_prev = dh_prev + d_rh * rg[t]

        da_z = dzg * zg[t] * (1 - zg[t])
        dWz = dWz + da_z.T @ x
        dUz = dUz + da_z.T @ h[t]
        dbz = dbz + np.sum(da_z, axis=0)
        dh_prev = dh_prev + da_z @ Uz

        da_r = drg * rg[t] * (1 - rg[t])
        dWr = dWr + da_r.T @ x
        dUr = dUr + da_r.T @ h[t]
        dbr = dbr + np.sum(da_r, axis=0)
        dh_prev = dh_prev + da_r @ Ur

        dh = dh_prev

    Wz = Wz - learning_rate * dWz
    Uz = Uz - learning_rate * dUz
    bz = bz - learning_rate * dbz
    Wr = Wr - learning_rate * dWr
    Ur = Ur - learning_rate * dUr
    br = br - learning_rate * dbr
    Wn = Wn - learning_rate * dWn
    Un = Un - learning_rate * dUn
    bn = bn - learning_rate * dbn
    Wo = Wo - learning_rate * dWo
    bo = bo - learning_rate * dbo

    if (epoch + 1) % 50 == 0:
        print("epoch", epoch + 1, "error", error)

h = np.zeros((len(X_test), hidden))

for t in range(cols):
    x = X_test[:, t, :]
    zg_t = 1 / (1 + np.exp(-(x @ Wz.T + h @ Uz.T + bz)))
    rg_t = 1 / (1 + np.exp(-(x @ Wr.T + h @ Ur.T + br)))
    ng_t = np.tanh(x @ Wn.T + (rg_t * h) @ Un.T + bn)
    h = (1 - zg_t) * h + zg_t * ng_t

z = h @ Wo + bo
prob = 1 / (1 + np.exp(-z))
predicted = (prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("\nGRU - Scratch")
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
plt.title("GRU - Scratch")
plt.grid()
plt.show()
