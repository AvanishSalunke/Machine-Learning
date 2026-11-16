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
learning_rate = 0.3
small = 0.000001

np.random.seed(42)

Wf = np.random.randn(hidden, symbols + hidden) * 0.1
bf = np.ones(hidden)
Wi = np.random.randn(hidden, symbols + hidden) * 0.1
bi = np.zeros(hidden)
Wc = np.random.randn(hidden, symbols + hidden) * 0.1
bc = np.zeros(hidden)
Wq = np.random.randn(hidden, symbols + hidden) * 0.1
bq = np.zeros(hidden)
Wo = np.random.randn(hidden) * 0.1
bo = 0

errors = []

for epoch in range(epochs):

    h = np.zeros((cols + 1, rows, hidden))
    c = np.zeros((cols + 1, rows, hidden))
    xh = np.zeros((cols, rows, symbols + hidden))
    fg = np.zeros((cols, rows, hidden))
    ig = np.zeros((cols, rows, hidden))
    cg = np.zeros((cols, rows, hidden))
    og = np.zeros((cols, rows, hidden))

    for t in range(cols):
        xh[t] = np.hstack((X_train[:, t, :], h[t]))
        fg[t] = 1 / (1 + np.exp(-(xh[t] @ Wf.T + bf)))
        ig[t] = 1 / (1 + np.exp(-(xh[t] @ Wi.T + bi)))
        cg[t] = np.tanh(xh[t] @ Wc.T + bc)
        og[t] = 1 / (1 + np.exp(-(xh[t] @ Wq.T + bq)))
        c[t + 1] = fg[t] * c[t] + ig[t] * cg[t]
        h[t + 1] = og[t] * np.tanh(c[t + 1])

    z = h[cols] @ Wo + bo
    prob = 1 / (1 + np.exp(-z))
    error = -np.mean(y_train * np.log(prob + small) + (1 - y_train) * np.log(1 - prob + small))
    errors.append(error)

    dz = (prob - y_train) / rows
    dWo = h[cols].T @ dz
    dbo = np.sum(dz)
    dh = np.outer(dz, Wo)
    dc = np.zeros((rows, hidden))

    dWf = np.zeros_like(Wf)
    dbf = np.zeros_like(bf)
    dWi = np.zeros_like(Wi)
    dbi = np.zeros_like(bi)
    dWc = np.zeros_like(Wc)
    dbc = np.zeros_like(bc)
    dWq = np.zeros_like(Wq)
    dbq = np.zeros_like(bq)

    for t in range(cols - 1, -1, -1):

        tc = np.tanh(c[t + 1])
        dc = dc + dh * og[t] * (1 - tc ** 2)

        da_q = dh * tc * og[t] * (1 - og[t])
        da_f = dc * c[t] * fg[t] * (1 - fg[t])
        da_i = dc * cg[t] * ig[t] * (1 - ig[t])
        da_c = dc * ig[t] * (1 - cg[t] ** 2)

        dWf = dWf + da_f.T @ xh[t]
        dbf = dbf + np.sum(da_f, axis=0)
        dWi = dWi + da_i.T @ xh[t]
        dbi = dbi + np.sum(da_i, axis=0)
        dWc = dWc + da_c.T @ xh[t]
        dbc = dbc + np.sum(da_c, axis=0)
        dWq = dWq + da_q.T @ xh[t]
        dbq = dbq + np.sum(da_q, axis=0)

        dxh = da_f @ Wf + da_i @ Wi + da_c @ Wc + da_q @ Wq
        dh = dxh[:, symbols:]
        dc = dc * fg[t]

    Wf = Wf - learning_rate * dWf
    bf = bf - learning_rate * dbf
    Wi = Wi - learning_rate * dWi
    bi = bi - learning_rate * dbi
    Wc = Wc - learning_rate * dWc
    bc = bc - learning_rate * dbc
    Wq = Wq - learning_rate * dWq
    bq = bq - learning_rate * dbq
    Wo = Wo - learning_rate * dWo
    bo = bo - learning_rate * dbo

    if (epoch + 1) % 50 == 0:
        print("epoch", epoch + 1, "error", error)

h = np.zeros((len(X_test), hidden))
c = np.zeros((len(X_test), hidden))

for t in range(cols):
    xh_t = np.hstack((X_test[:, t, :], h))
    fg_t = 1 / (1 + np.exp(-(xh_t @ Wf.T + bf)))
    ig_t = 1 / (1 + np.exp(-(xh_t @ Wi.T + bi)))
    cg_t = np.tanh(xh_t @ Wc.T + bc)
    og_t = 1 / (1 + np.exp(-(xh_t @ Wq.T + bq)))
    c = fg_t * c + ig_t * cg_t
    h = og_t * np.tanh(c)

z = h @ Wo + bo
prob = 1 / (1 + np.exp(-z))
predicted = (prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("\nLSTM - Scratch")
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
plt.title("LSTM - Scratch")
plt.grid()
plt.show()
