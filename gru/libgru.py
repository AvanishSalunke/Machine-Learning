import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, GRU, Dense
from tensorflow.keras.optimizers import SGD

data = pd.read_csv("../datasets/mushrooms.csv")
data = data.replace("?", np.nan)
data = data.fillna(data.mode().iloc[0])
X = data.drop("class", axis=1)
y = data["class"].map({"e": 0, "p": 1}).values
for col in X.columns:
    X[col] = X[col].astype("category").cat.codes
X = X.values.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rows, cols = X_train.shape
symbols = X.max() + 1
X_train = np.eye(symbols)[X_train]
X_test = np.eye(symbols)[X_test]

hidden = 8
epochs = 500
learning_rate = 0.1

model = Sequential()
model.add(Input(shape=(cols, symbols)))
model.add(GRU(hidden, reset_after=False))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=SGD(learning_rate=learning_rate), loss="binary_crossentropy")

history = model.fit(X_train, y_train, epochs=epochs, batch_size=rows, verbose=0)
errors = history.history["loss"]

prob = model.predict(X_test).ravel()
predicted = (prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, predicted)
precision = precision_score(y_test, predicted)
recall = recall_score(y_test, predicted)
f1 = f1_score(y_test, predicted)
matrix = confusion_matrix(y_test, predicted)

print("\nGRU - Library")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(matrix)

plt.figure(figsize=(8,5))
plt.plot(range(1, epochs + 1), errors)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("GRU - Library")
plt.grid()
plt.show()
