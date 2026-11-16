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

delta = np.zeros((length, states))
psi = np.zeros((length, states), dtype=int)
delta[0] = start * B[:, O[0]]

for t in range(1, length):
  scores = delta[t - 1][:, None] * A
  psi[t] = np.argmax(scores, axis=0)
  delta[t] = np.max(scores, axis=0) * B[:, O[t]]

path = np.zeros(length, dtype=int)
path[length - 1] = np.argmax(delta[length - 1])

for t in range(length - 2, -1, -1):
  path[t] = psi[t + 1][path[t + 1]]

best_prob = np.max(delta[length - 1])

print("\nviterbi - dataset")
print("delta")
print(np.round(delta, 5))
print("psi")
print(psi)
print("best path", [state_names[i] for i in path])
print("true path", [state_names[i] for i in s[:n]])
print("path accuracy", np.mean(path == s[:n]))
print("best prob", best_prob)