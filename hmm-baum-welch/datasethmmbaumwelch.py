import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset.csv")
obs_col = "obs"
states = 2
epochs = 50
n = 50

obs_names = sorted(data[obs_col].unique())
obs = data[obs_col].map({v: i for i, v in enumerate(obs_names)}).values[:n]
symbols = len(obs_names)
length = len(obs)

np.random.seed(42)

pi = np.random.rand(states)
pi = pi / np.sum(pi)
A = np.random.rand(states, states)
A = A / A.sum(axis=1, keepdims=True)
B = np.random.rand(states, symbols)
B = B / B.sum(axis=1, keepdims=True)

errors = []

for epoch in range(epochs):

  alpha = np.zeros((length, states))
  beta = np.ones((length, states))
  alpha[0] = pi * B[:, obs[0]]

  for t in range(1, length):
    alpha[t] = (alpha[t - 1] @ A) * B[:, obs[t]]

  for t in range(length - 2, -1, -1):
    beta[t] = A @ (B[:, obs[t + 1]] * beta[t + 1])

  prob = np.sum(alpha[length - 1])
  error = -np.log(prob)
  errors.append(error)

  gamma = alpha * beta / prob
  xi = np.zeros((states, states))

  for t in range(length - 1):
    xi = xi + alpha[t][:, None] * A * (B[:, obs[t + 1]] * beta[t + 1])[None, :] / prob

  pi = gamma[0]
  A = xi / np.sum(gamma[:length - 1], axis=0)[:, None]

  for k in range(symbols):
    B[:, k] = np.sum(gamma[obs == k], axis=0) / np.sum(gamma, axis=0)

  if (epoch + 1) % 10 == 0:
    print("epoch", epoch + 1, "error", error)

print("\nBaum Welch - Dataset")
print("symbols", obs_names)
print("pi")
print(np.round(pi, 4))
print("A")
print(np.round(A, 4))
print("B")
print(np.round(B, 4))

plt.figure(figsize=(8,5))
plt.plot(range(1, epochs + 1), errors)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Baum Welch - Dataset")
plt.grid()
plt.show()