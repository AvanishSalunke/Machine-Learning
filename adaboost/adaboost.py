import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data = pd.read_csv("../datasets/mushrooms.csv")
X = data.drop("class", axis=1)
y = data["class"].map({"e": 0, "p": 1}).values
X = pd.get_dummies(X, drop_first=True)
X = X.values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


rows, cols = X_train.shape

y_signed = np.where(y_train == 1, 1, -1)

w = np.ones(rows) / rows

epochs = 10
small = 1e-9
errors = []

stump_col = []
stump_cut = []
stump_side = []
stump_alpha = []


for i in range(epochs):

    best_error = 1

    for j in range(cols):
        for k in np.unique(X_train[:, j]):
            for side in [1, -1]:

                pred = np.where(X_train[:, j] <= k, side, -side)

                wrong = (pred != y_signed)
                error = np.sum(w * wrong)

                if error < best_error:
                    best_error = error
                    best_col = j
                    best_cut = k
                    best_side = side

    alpha = 0.5 * np.log((1 - best_error + small) / (best_error + small))

    pred = np.where(X_train[:, best_col] <= best_cut, best_side, -best_side)

    w = w * np.exp(-alpha * y_signed * pred)
    w = w / np.sum(w)

    stump_col.append(best_col)
    stump_cut.append(best_cut)
    stump_side.append(best_side)
    stump_alpha.append(alpha)
    errors.append(best_error)

    print("epoch", i + 1, "error =", best_error, "alpha =", alpha)

    if best_error == 0:
        break


z_test = np.zeros(len(X_test))

for i in range(len(stump_alpha)):

    pred = np.where(X_test[:, stump_col[i]] <= stump_cut[i], stump_side[i], -stump_side[i])
    z_test = z_test + stump_alpha[i] * pred

predicted = (z_test >= 0).astype(int)


accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)


print("\nAdaBoost - Scratch")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(matrix)


plt.figure(figsize=(8,5))
plt.plot(errors)

plt.xlabel("Epoch")
plt.ylabel("Weighted Error")
plt.title("AdaBoost Error per Epoch")
plt.grid()

plt.show()
