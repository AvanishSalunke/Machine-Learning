import numpy as np

X = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1]])
y = np.array([[0, 1, 1, 1, 1, 1, 1, 1]])
m = 8

W1 = np.array([[0.2, -0.3, 0.4], [0.1, 0.5, -0.2], [-0.4, 0.3, 0.2]])
b1 = np.array([[0.1], [-0.1], [0.2]])
W2 = np.array([[0.3, -0.2, 0.4], [-0.1, 0.5, 0.2]])
b2 = np.array([[0.1], [-0.2]])
W3 = np.array([[0.4, -0.3]])
b3 = np.array([[0.1]])

lr = 0.1

for i in range(3):
    Z1 = np.dot(W1, X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))
    Z3 = np.dot(W3, A2) + b3
    A3 = 1 / (1 + np.exp(-Z3))

    loss = -np.mean(y * np.log(A3) + (1 - y) * np.log(1 - A3))

    dZ3 = A3 - y
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    dZ2 = np.dot(W3.T, dZ3) * A2 * (1 - A2)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    W3 = W3 - lr * dW3
    b3 = b3 - lr * db3
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1

    print("Iteration", i + 1, "loss =", loss)
    print("W1", W1)
    print("b1", b1)
    print("W2", W2)
    print("b2", b2)
    print("W3", W3)
    print("b3", b3)
    print()