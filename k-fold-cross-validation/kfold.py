import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = pd.read_csv("../datasets/heart_cleveland_upload.csv")
X = data.drop("condition", axis=1)
y = data["condition"].values
X = X.values.astype(float)

rows, cols = X.shape

folds = 5

index = np.arange(rows)
rng = np.random.RandomState(42)
rng.shuffle(index)

sizes = np.full(folds, rows // folds)
sizes[:rows % folds] = sizes[:rows % folds] + 1

accuracies = []
precisions = []
recalls = []
f1s = []

start = 0

for i in range(folds):

    stop = start + sizes[i]

    test_index = index[start:stop]
    train_index = np.concatenate((index[:start], index[stop:]))

    start = stop

    X_train = X[train_index]
    X_test = X[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train, y_train)

    predicted = model.predict(X_test)

    accuracies.append(accuracy_score(y_test, predicted))
    precisions.append(precision_score(y_test, predicted))
    recalls.append(recall_score(y_test, predicted))
    f1s.append(f1_score(y_test, predicted))

    print("fold", i + 1, "train rows", len(train_index), "test rows", len(test_index), "accuracy =", accuracies[i])

print("\nK-Fold Cross Validation - Scratch")
print("Folds:", folds)
print("Fold Accuracies:", np.round(accuracies, 4))
print("Mean Accuracy:", np.mean(accuracies))
print("Std Accuracy:", np.std(accuracies))
print("Mean Precision:", np.mean(precisions))
print("Mean Recall:", np.mean(recalls))
print("Mean F1 Score:", np.mean(f1s))

plt.figure(figsize=(8,5))
plt.bar(range(1, folds + 1), accuracies)
plt.axhline(np.mean(accuracies), color="red", label="mean")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("K-Fold Cross Validation - Scratch")
plt.legend()
plt.grid()
plt.show()
