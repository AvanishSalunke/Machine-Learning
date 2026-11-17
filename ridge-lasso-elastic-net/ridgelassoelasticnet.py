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
names = list(X.columns)
X = X.values.astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

rows, cols = X_train.shape

learning_rate = 0.05
epochs = 5000

# name, alpha (penalty strength), l1_ratio (0 = only L2, 1 = only L1)
models = [
    ("No Penalty", 0, 0),
    ("Ridge", 0.1, 0),
    ("Lasso", 30, 1),
    ("Elastic Net", 20, 0.99),
]

all_weights = []

for name, alpha, ratio in models:

    weights = np.zeros(cols)
    bias = 0
    errors = []

    for i in range(epochs):

        pred = np.dot(X_train, weights) + bias

        error = np.sum((pred - y_train) ** 2) / (2 * rows)
        error = error + alpha * ratio * np.sum(np.abs(weights)) + 0.5 * alpha * (1 - ratio) * np.sum(weights ** 2)

        dw = np.dot(X_train.T, pred - y_train) / rows + alpha * (1 - ratio) * weights
        db = np.sum(pred - y_train) / rows

        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        shrink = learning_rate * alpha * ratio
        weights = np.sign(weights) * np.maximum(np.abs(weights) - shrink, 0)

        errors.append(error)

    predicted = np.dot(X_test, weights) + bias

    mae = mean_absolute_error(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predicted)

    all_weights.append(weights)

    print("\n" + name + " - Scratch")
    print("alpha:", alpha, "l1 ratio:", ratio)
    print("Final Loss:", errors[-1])
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)
    print("Weights:", np.round(weights, 3))
    print("Bias:", bias)
    print("Zero Weights:", np.sum(weights == 0))

positions = np.arange(cols)
width = 0.2

plt.figure(figsize=(10,5))

for i in range(len(models)):
    plt.bar(positions + i * width, all_weights[i], width, label=models[i][0])

plt.xticks(positions + 1.5 * width, names, rotation=45, ha="right")
plt.ylabel("Weight")
plt.title("Weights - Scratch")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
