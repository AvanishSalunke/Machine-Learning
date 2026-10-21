import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("heart_cleveland_upload.csv")

X = data.drop("condition", axis = 1)
y = data["condition"].values
y = np.where(y == 0 , -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rows, cols = X_train.shape
test_rows = X_test.shape[0]

sigma = 3

K_train = np.zeros((rows, rows))

for i in range(rows):
    for j in range(rows):
        K_train[i][j] = np.exp(-np.sum((X_train[i] - X_train[j])**2) / (2 * sigma**2))

K_test = np.zeros((test_rows, rows))

for i in range(test_rows):
    for j in range(rows):
        K_test[i][j] = np.exp(-np.sum((X_test[i] - X_train[j])**2) / (2 * sigma**2))

w = np.zeros(rows)
b = 0

C = 10
learning_rate = 0.01
decay = 0.01
epochs = 2000
errors = []

for i in range(epochs):
    lr = learning_rate / (1 + decay * i)

    score = np.dot(K_train, w) + b
    margin = y_train * score
    bad = margin < 1

    loss = 0.5 * np.sum(w**2) + C * np.mean(np.maximum(0, 1 - margin))
    errors.append(loss)

    dw = w - C * np.dot(K_train[bad].T, y_train[bad]) / rows
    db = -C * np.sum(y_train[bad]) / rows

    w = w - lr * dw
    b = b - lr * db

    if i % 200 == 0:
        print("epoch ", i, "loss = ", loss, "bad = ", bad.sum())

print("b = ", b)

score = np.dot(K_train, w) + b
margin = y_train * score

sv = margin <= 1
support_vectors = X_train[sv]

print("no of support vectors = ", sv.sum())
print(support_vectors)

cols_names = X.columns
sv_real = pd.DataFrame(sc.inverse_transform(support_vectors), columns = cols_names)
sv_real["label"] = y_train[sv]
print(sv_real.head(10))

score_test = np.dot(K_test, w) + b
predicted = np.where(score_test >= 0, 1, -1)

print("Accuracy:", accuracy_score(y_test, predicted))
print("Precision:", precision_score(y_test, predicted))
print("Recall:", recall_score(y_test, predicted))
print("F1 Score:", f1_score(y_test, predicted))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))

plt.plot(errors)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("gaussian kernel svm loss")
plt.grid()
plt.show()