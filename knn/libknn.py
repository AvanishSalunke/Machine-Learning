import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("../datasets/heart_cleveland_upload.csv")
X = data.drop("condition", axis=1)
y = data["condition"].values
X = X.values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

k = 5
max_k = 20

model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")

model.fit(X_train, y_train)

distances, neighbours = model.kneighbors(X_test[:1])

print("first test row")
print("nearest rows", neighbours[0])
print("distances", distances[0])
print("labels", y_train[neighbours[0]])

predicted = model.predict(X_test)

errors = []

for i in range(1, max_k + 1):

    temp = KNeighborsClassifier(n_neighbors=i, metric="euclidean")
    temp.fit(X_train, y_train)

    errors.append(np.mean(temp.predict(X_test) != y_test))

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("\nkNN - Library")
print("k:", k)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(matrix)
print("best k:", np.argmin(errors) + 1)

plt.figure(figsize=(8,5))
plt.plot(range(1, max_k + 1), errors)
plt.xlabel("k")
plt.ylabel("Test Error")
plt.title("kNN - Library")
plt.grid()
plt.show()
