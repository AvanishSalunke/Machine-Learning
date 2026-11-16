import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("../datasets/mushrooms.csv")
data = data.replace("?", np.nan)
data = data.fillna(data.mode().iloc[0])
X = data.drop("class", axis=1)
y = data["class"].map({"e": 0, "p": 1}).values
names = list(X.columns)
X = X.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rows, cols = X_train.shape

classes = [0, 1]

class_count = [np.sum(y_train == 0), np.sum(y_train == 1)]
prior = [class_count[0] / rows, class_count[1] / rows]

print("prior edible", prior[0])
print("prior poisonous", prior[1])

cond = {}

for j in range(cols):

    values = np.unique(X[:, j])
    k = len(values)

    for v in values:
        for c in classes:

            nc = np.sum((X_train[:, j] == v) & (y_train == c))
            cond[(j, v, c)] = (nc + 1) / (class_count[c] + k)

            print("P(", names[j], "=", v, "| class =", c, ") =", cond[(j, v, c)])

predicted = []

for r in range(len(X_test)):

    scores = []

    for c in classes:

        score = prior[c]

        for j in range(cols):
            score = score * cond[(j, X_test[r, j], c)]

        scores.append(score)

    if r < 3:
        print("row", r, "scores", scores)

    predicted.append(np.argmax(scores))

predicted = np.array(predicted)

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("accuracy", accuracy)
print("precision", precision)
print("recall", recall)
print("f1", f1)
print("error", 1 - accuracy)
print(matrix)
