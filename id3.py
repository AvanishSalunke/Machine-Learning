import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def entropy(target_col):
    values, counts = np.unique(target_col, return_counts=True)
    total = np.sum(counts)
    return sum(-(c / total) * math.log2(c / total) for c in counts)

def info_gain(data, attribute, target_name):
    total_entropy = entropy(data[target_name])
    values, counts = np.unique(data[attribute], return_counts=True)
    total = np.sum(counts)
    weighted_entropy = sum((c / total) * entropy(data[data[attribute] == v][target_name]) for v, c in zip(values, counts))
    return total_entropy - weighted_entropy

def build_tree(data, attributes, target_name, depth=0):
    labels = np.unique(data[target_name])
    if len(labels) == 1:
        return labels[0]
    if len(attributes) == 0:
        values, counts = np.unique(data[target_name], return_counts=True)
        return values[np.argmax(counts)]

    node_entropy = entropy(data[target_name])
    print("depth", depth, "samples", len(data), "entropy", node_entropy)

    gains = {attr: info_gain(data, attr, target_name) for attr in attributes}
    for attr, gain in gains.items():
        print("  gain", attr, gain)

    best_attr = max(gains, key=gains.get)
    print("  selected", best_attr)

    tree = {best_attr: {}}
    remaining_attrs = [a for a in attributes if a != best_attr]

    for value in np.unique(data[best_attr]):
        subset = data[data[best_attr] == value]
        if len(subset) == 0:
            values, counts = np.unique(data[target_name], return_counts=True)
            tree[best_attr][value] = values[np.argmax(counts)]
        else:
            tree[best_attr][value] = build_tree(subset, remaining_attrs, target_name, depth + 1)

    return tree

def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "class =", tree)
        return
    attr = list(tree.keys())[0]
    for value, subtree in tree[attr].items():
        print(indent + f"if {attr} == '{value}':")
        print_tree(subtree, indent + "    ")

def predict(tree, row, default="e"):
    if not isinstance(tree, dict):
        return tree
    attr = list(tree.keys())[0]
    value = row.get(attr)
    subtree = tree[attr].get(value, default)
    return predict(subtree, row, default)

df = pd.read_csv("mushrooms.csv")
target = "class"
attributes = [c for c in df.columns if c != target]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

tree = build_tree(train_df, attributes, target)

print("Decision Tree Rules")
print_tree(tree)

y_pred = [predict(tree, row.to_dict()) for _, row in test_df.iterrows()]
y_actual = test_df[target].tolist()

print("Confusion Matrix")
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