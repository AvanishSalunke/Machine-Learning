import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("../datasets/mushrooms.csv")
data = data.replace("?", np.nan)
data = data.fillna(data.mode().iloc[0])
X = data.drop("class", axis=1)
y = data["class"].map({"e": 0, "p": 1}).values
names = list(X.columns)

for col in names:
    X[col] = X[col].astype("category").cat.codes

X = X.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CategoricalNB(alpha=1, min_categories=X.max(axis=0) + 1)

model.fit(X_train, y_train)

predicted = model.predict(X_test)

print("prior", np.exp(model.class_log_prior_))

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
