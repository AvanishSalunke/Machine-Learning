import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

O = [0, 1, 2, 0, 1, 1, 2, 0, 0, 1, 2, 1, 0, 2, 2, 0]
states = 2
symbols = 3
epochs = 50

obs = np.array(O)
X = obs.reshape(-1, 1)

np.random.seed(42)

pi = np.random.rand(states)
pi = pi / np.sum(pi)
A = np.random.rand(states, states)
A = A / A.sum(axis=1, keepdims=True)
B = np.random.rand(states, symbols)
B = B / B.sum(axis=1, keepdims=True)

model = hmm.CategoricalHMM(n_components=states, n_features=symbols, n_iter=1, tol=0, init_params="", params="ste")
model.startprob_ = pi
model.transmat_ = A
model.emissionprob_ = B

errors = []

for epoch in range(epochs):

  error = -model.score(X)
  errors.append(error)
  model.fit(X)

  if (epoch + 1) % 10 == 0:
    print("epoch", epoch + 1, "error", error)

print("\nBaum Welch - Library")
print("pi")
print(np.round(model.startprob_, 4))
print("A")
print(np.round(model.transmat_, 4))
print("B")
print(np.round(model.emissionprob_, 4))

plt.figure(figsize=(8,5))
plt.plot(range(1, epochs + 1), errors)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Baum Welch - Library")
plt.grid()
plt.show()