import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
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


learning_rate = 0.1
epochs = 100

model = GradientBoostingClassifier(n_estimators=epochs, learning_rate=learning_rate, max_depth=1, random_state=42)

model.fit(X_train, y_train)

predicted = model.predict(X_test)
errors = model.train_score_


accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)


print("\nGradient Boosting - Library")
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
plt.title("Gradient Boosting Loss")
plt.grid()

plt.show()