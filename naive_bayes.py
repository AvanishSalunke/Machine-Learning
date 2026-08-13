import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def train_naive_bayes(data, target):
    classes = data[target].unique()
    attributes = [col for col in data.columns if col != target]

    priors = {}

    for c in classes:
        priors[c] = len(data[data[target] == c]) / len(data)

    probabilities = {}

    for attr in attributes:
        probabilities[attr] = {}

        for value in data[attr].unique():
            probabilities[attr][value] = {}

            for c in classes:
                class_data = data[data[target] == c]
                count = len(class_data[class_data[attr] == value])
                total = len(class_data)

                probabilities[attr][value][c] = (count + 1) / (total + len(data[attr].unique()))

    return priors, probabilities


def predict(row, priors, probabilities):
    scores = {}

    for c in priors:
        probability = priors[c]

        for attr in probabilities:
            value = row[attr]
            probability *= probabilities[attr][value][c]

        scores[c] = probability

    return max(scores, key=scores.get)


df = pd.read_csv("mushrooms.csv")

target = "class"

train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

priors, probabilities = train_naive_bayes(train_df, target)


print("Class Prior Probabilities:")

for c in priors:
    print("P(", c, ") =", priors[c])


print("\nConditional Probabilities:")

for attr in probabilities:
    for value in probabilities[attr]:
        for c in probabilities[attr][value]:
            print("P(", attr, "=", value, "| class =", c, ") =", probabilities[attr][value][c])


y_pred = [predict(row, priors, probabilities) for _, row in test_df.iterrows()]
y_actual = test_df[target].tolist()


print("\nConfusion Matrix")
print(confusion_matrix(y_actual, y_pred, labels=["p", "e"]))

accuracy = accuracy_score(y_actual, y_pred)
precision = precision_score(y_actual, y_pred, pos_label="p")
recall = recall_score(y_actual, y_pred, pos_label="p")
f1 = f1_score(y_actual, y_pred, pos_label="p")

print("Accuracy", accuracy)
print("Precision", precision)
print("Recall", recall)
print("F1 score", f1)
print("Error rate", 1 - accuracy)