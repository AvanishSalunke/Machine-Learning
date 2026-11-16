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

z = np.zeros(rows)
prob = 1 / (1 + np.exp(-z))

learning_rate = 0.1
epochs = 100
reg_lambda = 1
gamma = 0
small = 1e-9
errors = []

stump_col = []
stump_cut = []
stump_left = []
stump_right = []


for i in range(epochs):

    g = prob - y_train
    h = prob * (1 - prob)

    best_gain = -np.inf

    for j in range(cols):
        for k in np.unique(X_train[:, j])[:-1]:

            left = X_train[:, j] <= k

            G_left = np.sum(g[left])
            H_left = np.sum(h[left])
            G_right = np.sum(g[~left])
            H_right = np.sum(h[~left])

            gain = 0.5 * (G_left ** 2 / (H_left + reg_lambda) + G_right ** 2 / (H_right + reg_lambda) - (G_left + G_right) ** 2 / (H_left + H_right + reg_lambda)) - gamma

            if gain > best_gain:
                best_gain = gain
                best_col = j
                best_cut = k
                best_left = -G_left / (H_left + reg_lambda)
                best_right = -G_right / (H_right + reg_lambda)

    pred = np.where(X_train[:, best_col] <= best_cut, best_left, best_right)

    z = z + learning_rate * pred
    prob = 1 / (1 + np.exp(-z))

    stump_col.append(best_col)
    stump_cut.append(best_cut)
    stump_left.append(best_left)
    stump_right.append(best_right)

    error = -np.mean(y_train * np.log(prob + small) + (1 - y_train) * np.log(1 - prob + small))

    errors.append(error)

    if i % 10 == 0:
        print("epoch", i, "loss =", error)


z_test = np.zeros(len(X_test))

for i in range(epochs):

    pred = np.where(X_test[:, stump_col[i]] <= stump_cut[i], stump_left[i], stump_right[i])
    z_test = z_test + learning_rate * pred

prob = 1 / (1 + np.exp(-z_test))

predicted = (prob >= 0.5).astype(int)


accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)


print("\nXGBoost - Scratch")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(matrix)


plt.figure(figsize=(8,5))
plt.plot(errors)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("XGBoost Loss")
plt.grid()

plt.show()
