import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

trees = 10
p = int(np.sqrt(cols))

np.random.seed(42)

forest_attr = []
forest_label = []
forest_children = []

votes = np.zeros((rows, 2))

for t in range(trees):

    sample = np.random.randint(0, rows, rows)
    oob = np.setdiff1d(np.arange(rows), sample)

    node_rows = [sample]
    node_attrs = [list(range(cols))]
    node_attr = [-1]
    node_label = [-1]
    node_children = [{}]

    n = 0

    while n < len(node_rows):

        idx = node_rows[n]
        attrs = node_attrs[n]
        labels = y_train[idx]

        counts = np.bincount(labels, minlength=2)
        node_label[n] = np.argmax(counts)

        if np.max(counts) == len(idx) or len(attrs) == 0:
            n = n + 1
            continue

        q = counts[counts > 0] / len(idx)
        entropy = -np.sum(q * np.log2(q))

        chosen = np.random.choice(attrs, size=min(p, len(attrs)), replace=False)

        best_gain = -1

        for j in chosen:

            weighted = 0

            for v in np.unique(X_train[idx, j]):

                sub = labels[X_train[idx, j] == v]
                c = np.bincount(sub, minlength=2)
                c = c[c > 0] / len(sub)
                weighted = weighted + len(sub) / len(idx) * (-np.sum(c * np.log2(c)))

            gain = entropy - weighted

            if gain > best_gain:
                best_gain = gain
                best_col = j

        node_attr[n] = best_col

        for v in np.unique(X_train[idx, best_col]):

            node_rows.append(idx[X_train[idx, best_col] == v])
            node_attrs.append([a for a in attrs if a != best_col])
            node_attr.append(-1)
            node_label.append(-1)
            node_children.append({})
            node_children[n][v] = len(node_rows) - 1

        n = n + 1

    forest_attr.append(node_attr)
    forest_label.append(node_label)
    forest_children.append(node_children)

    for r in oob:

        m = 0

        while node_attr[m] != -1:

            v = X_train[r, node_attr[m]]

            if v not in node_children[m]:
                break

            m = node_children[m][v]

        votes[r, node_label[m]] = votes[r, node_label[m]] + 1

    print("tree", t + 1, "nodes", len(node_rows), "oob rows", len(oob))

seen = votes.sum(axis=1) > 0
oob_error = np.mean(np.argmax(votes[seen], axis=1) != y_train[seen])

print("oob error", oob_error)

votes_test = np.zeros((len(X_test), 2))
errors = []

for t in range(trees):

    for r in range(len(X_test)):

        m = 0

        while forest_attr[t][m] != -1:

            v = X_test[r, forest_attr[t][m]]

            if v not in forest_children[t][m]:
                break

            m = forest_children[t][m][v]

        votes_test[r, forest_label[t][m]] = votes_test[r, forest_label[t][m]] + 1

    errors.append(np.mean(np.argmax(votes_test, axis=1) != y_test))

predicted = np.argmax(votes_test, axis=1)

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("\nRandom Forest - Scratch")
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
plt.title("Random Forest - Scratch")
plt.grid()
plt.show()
