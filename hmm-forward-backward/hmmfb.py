import numpy as np

pi = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.1, 0.4, 0.5], [0.7, 0.2, 0.1]])
O = [1, 2, 0, 1]
from_start = 1

states = len(pi)
length = len(O)

start = pi @ A

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

print("forward and backward - scratch")
print("alpha")
print(np.round(alpha, 7))
print("beta")
print(np.round(beta, 7))
print("forward prob", forward_prob)
print("backward prob", backward_prob)
print("state prob at each time")
print(np.round(gamma, 4))