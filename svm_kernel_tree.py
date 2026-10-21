import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC

data = pd.read_csv("heart_cleveland_upload.csv")

X = data.drop("condition", axis = 1)
y = data["condition"].values
y = np.where(y == 0 , -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rows, cols = X_train.shape

sigma = 3

clf = SVC(kernel = "rbf", gamma = 1 / (2 * sigma**2))
clf.fit(X_train, y_train)

print("b = ", clf.intercept_[0])

support_vectors = clf.support_vectors_

print("no of support vectors = ", len(clf.support_))
print(support_vectors)

cols_names = X.columns
sv_real = pd.DataFrame(sc.inverse_transform(support_vectors), columns = cols_names)
sv_real["label"] = y_train[clf.support_]
print(sv_real.head(10))

predicted = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predicted))
print("Precision:", precision_score(y_test, predicted))
print("Recall:", recall_score(y_test, predicted))
print("F1 Score:", f1_score(y_test, predicted))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predicted))
