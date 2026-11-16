import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

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

states = 3
epochs = 20
small = 0.000001

prior = np.array([np.mean(y_train == 0), np.mean(y_train == 1)])
models = []
errors = np.zeros(epochs)

for c in range(2):

  seqs = X_train[y_train == c]
  n = len(seqs)

  model = hmm.CategoricalHMM(n_components=states, n_features=symbols, n_iter=1, tol=0, random_state=42)

  for epoch in range(epochs):

    model.fit(seqs.reshape(-1, 1), lengths=[cols] * n)
    model.init_params = ""
    errors[epoch] = errors[epoch] - model.score(seqs.reshape(-1, 1), lengths=[cols] * n) / n

    if (epoch + 1) % 5 == 0:
      print("class", c, "epoch", epoch + 1)

  emission = model.emissionprob_ + small
  model.emissionprob_ = emission / emission.sum(axis=1, keepdims=True)
  models.append(model)

scores = np.zeros((len(X_test), 2))

for c in range(2):

  for r in range(len(X_test)):
    scores[r, c] = models[c].score(X_test[r].reshape(-1, 1)) + np.log(prior[c])

predicted = np.argmax(scores, axis=1)

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("\nHMM Classifier - Library")
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
plt.title("HMM Classifier - Library")
plt.grid()
plt.show()