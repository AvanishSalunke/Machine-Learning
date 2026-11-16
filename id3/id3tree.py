import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
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

model = DecisionTreeClassifier(criterion="entropy", random_state=42)

model.fit(X_train, y_train)

predicted = model.predict(X_test)

print(export_text(model, feature_names=names))

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
print("depth", model.get_depth())
print("leaves", model.get_n_leaves())
