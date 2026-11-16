import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("../datasets/mushrooms.csv")
data = data.replace("?", np.nan)
data = data.fillna(data.mode().iloc[0])
X = data.drop("class", axis=1)
y = data["class"].map({"e": 0, "p": 1}).values
names = list(X.columns)

for col in names:
    X[col] = X[col].astype("category").cat.codes

X = X.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

trees = 10

model = RandomForestClassifier(n_estimators=trees, criterion="entropy", max_features="sqrt", bootstrap=True, oob_score=True, random_state=42)

model.fit(X_train, y_train)

predicted = model.predict(X_test)

print("oob error", 1 - model.oob_score_)

votes_test = np.zeros((len(X_test), 2))
errors = []

for tree in model.estimators_:
    pred = tree.predict(X_test).astype(int)
    for r in range(len(X_test)):
        votes_test[r, pred[r]] = votes_test[r, pred[r]] + 1
    errors.append(np.mean(np.argmax(votes_test, axis=1) != y_test))

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("\nRandom Forest - Library")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(matrix)

plt.figure(figsize=(8,5))
plt.plot(range(1, trees + 1), errors)
plt.xlabel("Trees")
plt.ylabel("Test Error")
plt.title("Random Forest - Library")
plt.grid()
plt.show()
