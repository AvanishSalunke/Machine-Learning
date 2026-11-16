import numpy as np

pi = np.array([0.667, 0.333])
A = np.array([[0.8, 0.2], [0.4, 0.6]])
B = np.array([[0.8, 0.2], [0.4, 0.6]])
O = [0, 0, 1, 1, 1, 0]
names = ["S", "R"]
from_start = 0

states = len(pi)
length = len(O)


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

print("viterbi - scratch")
print("delta")
print(np.round(delta, 5))
print("psi")
print(psi)
print("best path", [names[i] for i in path])
print("best prob", best_prob)