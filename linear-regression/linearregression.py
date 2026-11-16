import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("../datasets/electricity_cost_dataset.csv")
X = data.drop("electricity cost", axis=1)
y = data["electricity cost"].values
X = pd.get_dummies(X, drop_first=True)
X = X.values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rows, cols = X_train.shape

weights = np.zeros(cols)
bias = 0

learning_rate = 0.01
epochs = 1000
errors = []

for i in range(epochs):

    pred = np.dot(X_train, weights) + bias

    error = np.sum((pred - y_train) ** 2) / (2 * rows)

    dw = np.dot(X_train.T, pred - y_train) / rows
    db = np.sum(pred - y_train) / rows

    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    errors.append(error)

    if i % 100 == 0:
        print("epoch", i, "loss =", error)

predicted = np.dot(X_test, weights) + bias

mae = mean_absolute_error(y_test, predicted)
mse = mean_squared_error(y_test, predicted)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predicted)

print("\nLinear Regression - Scratch")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
print("Weights:", weights)
print("Bias:", bias)

plt.figure(figsize=(8,5))
plt.plot(errors)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Linear Regression - Scratch")
plt.grid()
plt.show()