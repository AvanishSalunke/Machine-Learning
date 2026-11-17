import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("../datasets/heart_cleveland_upload.csv")
X = data.drop("condition", axis=1)
y = data["condition"].values
X = X.values.astype(float)

rows, cols = X.shape

folds = 5

splitter = KFold(n_splits=folds, shuffle=True, random_state=42)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))

result = cross_validate(model, X, y, cv=splitter, scoring=["accuracy", "precision", "recall", "f1"])

accuracies = result["test_accuracy"]

for i, (train_index, test_index) in enumerate(splitter.split(X)):
    print("fold", i + 1, "train rows", len(train_index), "test rows", len(test_index), "accuracy =", accuracies[i])

print("\nK-Fold Cross Validation - Library")
print("Folds:", folds)
print("Fold Accuracies:", np.round(accuracies, 4))
print("Mean Accuracy:", np.mean(accuracies))
print("Std Accuracy:", np.std(accuracies))
print("Mean Precision:", np.mean(result["test_precision"]))
print("Mean Recall:", np.mean(result["test_recall"]))
print("Mean F1 Score:", np.mean(result["test_f1"]))

plt.figure(figsize=(8,5))
plt.bar(range(1, folds + 1), accuracies)
plt.axhline(np.mean(accuracies), color="red", label="mean")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("K-Fold Cross Validation - Library")
plt.legend()
plt.grid()
plt.show()
