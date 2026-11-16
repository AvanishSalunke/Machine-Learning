import numpy as np
from hmmlearn import hmm

pi = np.array([0.667, 0.333])
A = np.array([[0.8, 0.2], [0.4, 0.6]])
B = np.array([[0.8, 0.2], [0.4, 0.6]])
O = [0, 0, 1, 1, 1, 0]
names = ["S", "R"]
from_start = 0

states = len(pi)
symbols = B.shape[1]

if from_start == 1:
  start = pi @ A
else:
  start = pi

model = hmm.CategoricalHMM(n_components=states, n_features=symbols)
model.startprob_ = start
model.transmat_ = A
model.emissionprob_ = B

X = np.array(O).reshape(-1, 1)

log_prob, path = model.decode(X, algorithm="viterbi")
best_prob = np.exp(log_prob)

print("viterbi - library")
print("best path", [names[i] for i in path])
print("best prob", best_prob)