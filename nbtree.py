import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


df = pd.read_csv("mushrooms.csv")

X = df.drop("class", axis=1)
y = df["class"].map({"e": 0, "p": 1})


encoder = OrdinalEncoder()
X = encoder.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


model = CategoricalNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred, labels=[1, 0]))

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy", accuracy)
print("Precision", precision)
print("Recall", recall)
print("F1 score", f1)
print("Error rate", 1 - accuracy)