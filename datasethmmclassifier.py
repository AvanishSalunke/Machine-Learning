import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("dataset.csv")
target = "class"
data = data.replace("?", np.nan)
data = data.fillna(data.mode().iloc[0])
X = data.drop(target, axis=1)
y = data[target].astype("category").cat.codes.values
for col in X.columns:
  X[col] = X[col].astype("category").cat.codes
X = X.values.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rows, cols = X_train.shape
symbols = X.max() + 1
classes = len(np.unique(y))

states = 3
epochs = 20
small = 0.000001

np.random.seed(42)

prior = np.array([np.mean(y_train == c) for c in range(classes)])
pis = []
As = []
Bs = []
errors = np.zeros(epochs)

for c in range(classes):

  seqs = X_train[y_train == c]
  n = len(seqs)

  pi = np.random.rand(states)
  pi = pi / np.sum(pi)
  A = np.random.rand(states, states)
  A = A / A.sum(axis=1, keepdims=True)
  B = np.random.rand(states, symbols)
  B = B / B.sum(axis=1, keepdims=True)

  for epoch in range(epochs):

    alpha = np.zeros((n, cols, states))
    beta = np.ones((n, cols, states))
    alpha[:, 0, :] = pi * B[:, seqs[:, 0]].T

    for t in range(1, cols):
      alpha[:, t, :] = (alpha[:, t - 1, :] @ A) * B[:, seqs[:, t]].T

    for t in range(cols - 2, -1, -1):
      beta[:, t, :] = (B[:, seqs[:, t + 1]].T * beta[:, t + 1, :]) @ A.T

    prob = np.sum(alpha[:, cols - 1, :], axis=1)
    errors[epoch] = errors[epoch] - np.mean(np.log(prob + small))

    gamma = alpha * beta / prob[:, None, None]
    xi = np.zeros((states, states))

    for t in range(cols - 1):
      nxt = B[:, seqs[:, t + 1]].T * beta[:, t + 1, :]
      xi = xi + np.sum(alpha[:, t, :, None] * A[None, :, :] * nxt[:, None, :] / prob[:, None, None], axis=0)

    pi = np.mean(gamma[:, 0, :], axis=0)
    A = xi / (np.sum(gamma[:, :cols - 1, :], axis=(0, 1))[:, None] + small)
    A = A / A.sum(axis=1, keepdims=True)

    for k in range(symbols):
      B[:, k] = np.sum(gamma[seqs == k], axis=0) / (np.sum(gamma, axis=(0, 1)) + small)

    B = B + small
    B = B / B.sum(axis=1, keepdims=True)

    if (epoch + 1) % 5 == 0:
      print("class", c, "epoch", epoch + 1)

  pis.append(pi)
  As.append(A)
  Bs.append(B)

scores = np.zeros((len(X_test), classes))

for c in range(classes):

  alpha = pis[c] * Bs[c][:, X_test[:, 0]].T

  for t in range(1, cols):
    alpha = (alpha @ As[c]) * Bs[c][:, X_test[:, t]].T

  scores[:, c] = np.log(np.sum(alpha, axis=1) + small) + np.log(prior[c])

predicted = np.argmax(scores, axis=1)

average = "binary" if classes == 2 else "macro"
accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted, average=average, zero_division=0)
recall = recall_score(y_test, predicted, average=average, zero_division=0)
f1 = f1_score(y_test, predicted, average=average, zero_division=0)
matrix = confusion_matrix(y_test, predicted)

print("\nHMM Classifier - Dataset")
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
plt.title("HMM Classifier - Dataset")
plt.grid()
plt.show()