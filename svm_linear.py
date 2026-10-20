import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("heart_cleveland_upload.csv")

# print(data.shape)
# print(data.head())
# print(data.isnull().sum())
# print(data["condition"].value_counts())

X = data.drop("condition", axis = 1)
y = data["condition"].values
y = np.where(y == 0 , -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 50)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# print(X_train.shape)
# print(X_test.shape)


rows, cols = X_train.shape

w = np.zeros(cols)
# w = np.random.rand(cols)
b = 0

C= 10
learning_rate = 0.01
epochs = 1000
errors = []

for i in range(epochs):
    score = np.dot(X_train, w)+ b
    margin = y_train * score
    bad = margin<1

    # print(score)
    # print(margin)
    # print(bad.sum())

    loss = 0.5 * np.sum(w**2) + C*np.mean(np.maximum(0, 1 - margin))
    errors.append(loss)


    dw = w - C * np.dot(X_train[bad].T , y_train[bad])/rows
    db = -C * np.sum(y_train[bad])/rows
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # print(w)
    # print(b)


    if i%100 == 0:
        print("epoch ", i, "loss = ", loss, "bad = ", bad.sum())

print(w)
print(b)

score = np.dot(X_train, w) + b
margin = y_train * score

sv = margin<= 1
support_vectors = X_train[sv]
print("no of support vectors = ", sv.sum())
print(support_vectors)

score_test = np.dot(X_test, w) + b
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
plt.title("SVM loss")
plt.grid()
plt.show()