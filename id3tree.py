## this is for comparing made id3 algorithm with sklearn's decision tree classifier

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

df = pd.read_csv("mushrooms.csv")

df.replace("?", np.nan, inplace=True)

for col in df.columns:
    df[col] = df[col].fillna(df[col].mode()[0])

encoders = {}
encoded_df = df.copy()
for col in encoded_df.columns:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col])
    encoders[col] = le

X = encoded_df.drop("class", axis=1)
y = encoded_df["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Decision Tree Rules")
print(export_text(clf, feature_names=list(X.columns)))

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Confusion Matrix")
print(cm)

print("Accuracy", accuracy)
print("Precision", precision)
print("Recall", recall)
print("F1 score", f1)
print("Error rate", 1 - accuracy)

print("Tree depth", clf.get_depth())
print("Number of leaves", clf.get_n_leaves())