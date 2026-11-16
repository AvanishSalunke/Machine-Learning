import numpy as np
from hmmlearn import hmm

pi = np.array([0.6, 0.4])
A = np.array([[0.7, 0.3], [0.4, 0.6]])
B = np.array([[0.1, 0.4, 0.5], [0.7, 0.2, 0.1]])
O = [1, 2, 0, 1]
from_start = 1

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

log_prob = model.score(X)
forward_prob = np.exp(log_prob)
gamma = model.predict_proba(X)

print("forward and backward - library")
print("forward prob", forward_prob)
print("state prob at each time")
print(np.round(gamma, 4))