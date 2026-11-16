import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("../datasets/heart_cleveland_upload.csv")
X = data.drop("condition", axis=1)
y = data["condition"].values
y = np.where(y == 0, -1, 1)
X = X.values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rows, cols = X_train.shape

C = 10

model = SVC(kernel="linear", C=C / rows)

model.fit(X_train, y_train)

print("w =", model.coef_[0])
print("b =", model.intercept_[0])

support_vectors = model.support_vectors_

print("no of support vectors =", len(model.support_))
print(support_vectors)

predicted = model.predict(X_test)

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("\nSVM Linear - Library")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(matrix)
