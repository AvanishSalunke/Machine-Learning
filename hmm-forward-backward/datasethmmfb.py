import numpy as np
import pandas as pd

data = pd.read_csv("dataset.csv")
state_col = "state"
obs_col = "obs"
n = 10
from_start = 0

state_names = sorted(data[state_col].unique())
obs_names = sorted(data[obs_col].unique())
s = data[state_col].map({v: i for i, v in enumerate(state_names)}).values
o = data[obs_col].map({v: i for i, v in enumerate(obs_names)}).values

states = len(state_names)
symbols = len(obs_names)

pi = np.bincount(s, minlength=states) + 1.0
pi = pi / np.sum(pi)
A = np.ones((states, states))
B = np.ones((states, symbols))

for t in range(len(s) - 1):
  A[s[t], s[t + 1]] += 1

for t in range(len(s)):
  B[s[t], o[t]] += 1

A = A / A.sum(axis=1, keepdims=True)
B = B / B.sum(axis=1, keepdims=True)
O = list(o[:n])
length = len(O)

print("estimated from dataset")
print("pi")
print(np.round(pi, 4))
print("A")
print(np.round(A, 4))
print("B")
print(np.round(B, 4))

if from_start == 1:
  start = pi @ A
else:
  start = pi

alpha = np.zeros((length, states))
alpha[0] = start * B[:, O[0]]

for t in range(1, length):
  alpha[t] = (alpha[t - 1] @ A) * B[:, O[t]]

forward_prob = np.sum(alpha[length - 1])

beta = np.ones((length, states))

for t in range(length - 2, -1, -1):
  beta[t] = A @ (B[:, O[t + 1]] * beta[t + 1])

backward_prob = np.sum(start * B[:, O[0]] * beta[0])

gamma = alpha * beta / forward_prob

print("\nforward and backward - dataset")
print("alpha")
print(np.round(alpha, 7))
print("beta")
print(np.round(beta, 7))
print("forward prob", forward_prob)
print("backward prob", backward_prob)
print("state prob at each time")
print(np.round(gamma, 4))