import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


data = pd.read_csv("mushrooms.csv")
X = data.drop("class", axis=1)
y = data["class"].map({"e": 0, "p": 1}).values
X = pd.get_dummies(X, drop_first=True)
X = X.values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


rows, cols = X_train.shape

w = np.zeros(cols)
b = 0

learning_rate = 0.1
epochs = 1000
errors = []


for i in range(epochs):

    z = np.dot(X_train, w) + b

    # sigmoid
    pred = 1 / (1 + np.exp(-z))

    dw = np.dot(X_train.T, (pred - y_train)) / rows
    db = np.sum(pred - y_train) / rows

    w -= learning_rate * dw
    b -= learning_rate * db

    small = 1e-9
    error = -np.mean(
        y_train * np.log(pred + small) +
        (1 - y_train) * np.log(1 - pred + small)
    )

    errors.append(error)

    if i % 100 == 0:
        print("epoch", i, "loss =", error)


z_test = np.dot(X_test, w) + b
prob = 1 / (1 + np.exp(-z_test))

predicted = (prob >= 0.5).astype(int)


accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)


print("\nLogistic Regression - Scratch")
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
plt.title("Logistic Regression Loss")
plt.grid()

plt.show()