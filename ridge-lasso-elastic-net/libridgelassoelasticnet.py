import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
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

# alpha is the same penalty strength as in the scratch file
# ridge in sklearn does not divide the error by 2 * rows, so its alpha is alpha * rows
models = [
    ("No Penalty", LinearRegression(), 0, 0),
    ("Ridge", Ridge(alpha=0.1 * rows), 0.1, 0),
    ("Lasso", Lasso(alpha=30, max_iter=100000, tol=1e-10), 30, 1),
    ("Elastic Net", ElasticNet(alpha=20, l1_ratio=0.99, max_iter=100000, tol=1e-10), 20, 0.99),
]

all_weights = []

for name, model, penalty, ratio in models:

    model.fit(X_train, y_train)

    predicted = model.predict(X_test)

    mae = mean_absolute_error(y_test, predicted)
    mse = mean_squared_error(y_test, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predicted)

    weights = model.coef_
    bias = model.intercept_

    pred = model.predict(X_train)
    error = np.sum((pred - y_train) ** 2) / (2 * rows)
    error = error + penalty * ratio * np.sum(np.abs(weights)) + 0.5 * penalty * (1 - ratio) * np.sum(weights ** 2)

    all_weights.append(weights)

    print("\n" + name + " - Library")
    print("alpha:", penalty, "l1 ratio:", ratio)
    print("Final Loss:", error)
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
plt.title("Weights - Library")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
