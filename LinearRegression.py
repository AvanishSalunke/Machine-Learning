import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("electricity_cost_dataset.csv")

X = df.drop("electricity cost", axis=1)
y = df["electricity cost"].values

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
X_cat = encoder.fit_transform(X[["structure type"]])

X_num = X.drop("structure type", axis=1).values
X = np.hstack((X_num, X_cat))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

np.random.seed(42)

weights = np.zeros(X_train.shape[1])
bias = 0

learning_rate = 0.01
epochs = 1000
errors = []

for epoch in range(epochs):
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    total_error = 0

    for i in range(len(X_train)):
        prediction = np.dot(X_train[i], weights) + bias
        error = prediction - y_train[i]

        weights = weights - learning_rate * error * X_train[i]
        bias = bias - learning_rate * error

        total_error += error ** 2

    mse = total_error / len(X_train)
    errors.append(mse)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch:{epoch + 1}, MSE:{mse:.4f}")

y_pred = np.dot(X_test, weights) + bias

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Linear Regression")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.plot(range(1, epochs + 1), errors)
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Error")
plt.show()