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

node_rows = [np.arange(rows)]
node_attrs = [list(range(cols))]
node_depth = [0]
node_parent = [-1]
node_value = [""]
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

    p = counts[counts > 0] / len(idx)
    entropy = -np.sum(p * np.log2(p))

    print(n, "depth", node_depth[n], "samples", len(idx), "entropy", entropy)

    if entropy == 0 or len(attrs) == 0:
        n = n + 1
        continue

    best_gain = -1

    for j in attrs:

        weighted = 0

        for v in np.unique(X_train[idx, j]):

            sub = labels[X_train[idx, j] == v]
            c = np.bincount(sub, minlength=2)
            c = c[c > 0] / len(sub)
            weighted = weighted + len(sub) / len(idx) * (-np.sum(c * np.log2(c)))

        gain = entropy - weighted

        print(names[j], gain)

        if gain > best_gain:
            best_gain = gain
            best_col = j

    print("best", names[best_col])

    node_attr[n] = best_col

    for v in np.unique(X_train[idx, best_col]):

        node_rows.append(idx[X_train[idx, best_col] == v])
        node_attrs.append([a for a in attrs if a != best_col])
        node_depth.append(node_depth[n] + 1)
        node_parent.append(n)
        node_value.append(v)
        node_attr.append(-1)
        node_label.append(-1)
        node_children.append({})
        node_children[n][v] = len(node_rows) - 1

    n = n + 1

print("rules")

for n in range(len(node_rows)):

    if node_attr[n] == -1:

        rule = []
        m = n

        while m != 0:
            rule.append(names[node_attr[node_parent[m]]] + " = " + node_value[m])
            m = node_parent[m]

        if node_label[n] == 0:
            cls = "edible"
        else:
            cls = "poisonous"

        print("if", " and ".join(rule[::-1]), "then", cls)

predicted = []

for r in range(len(X_test)):

    n = 0

    while node_attr[n] != -1:

        v = X_test[r, node_attr[n]]

        if v not in node_children[n]:
            break

        n = node_children[n][v]

    predicted.append(node_label[n])

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
