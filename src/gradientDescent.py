import numpy as np
from computeCost import computeCost


# (вложенные циклы)
def gradientDescent_loops(X, y, theta, alpha, iterations):
    m = len(y)
    n = len(theta)
    J_history = np.zeros(iterations)

    for it in range(iterations):
        gradients = [0] * n
        for j in range(n):
            for i in range(m):
                prediction = sum(theta[k] * X[i][k] for k in range(n))
                error = prediction - y[i]
                gradients[j] += error * X[i][j]
            gradients[j] = gradients[j] / m

        for j in range(n):
            theta[j] = theta[j] - alpha * gradients[j]

        J_history[it] = computeCost(X, y, theta, mode=1)

    return np.array(theta), J_history


# sum() и list-comprehensions
def gradientDescent_pythonic(X, y, theta, alpha, iterations):
    m = len(y)
    n = len(theta)
    J_history = np.zeros(iterations)

    for it in range(iterations):
        predictions = [sum(theta[j] * X[i][j] for j in range(n)) for i in range(m)]
        gradients = [
            sum((predictions[i] - y[i]) * X[i][j] for i in range(m)) / m
            for j in range(n)
        ]
        theta = [theta[j] - alpha * gradients[j] for j in range(n)]
        J_history[it] = computeCost(X, y, theta, mode=2)

    return np.array(theta), J_history


# Векторный способ (NumPy)
def gradientDescent_vectorized(X, y, theta, alpha, iterations):
    m = len(y)
    J_history = np.zeros(iterations)

    for it in range(iterations):
        theta = theta - (alpha / m) * (X.T @ (X @ theta - y))
        J_history[it] = computeCost(X, y, theta, mode=3)

    return theta, J_history


def gradientDescent(X, y, theta, alpha, iterations, mode=3):
    if mode == 1:
        return gradientDescent_loops(X, y, theta, alpha, iterations)
    elif mode == 2:
        return gradientDescent_pythonic(X, y, theta, alpha, iterations)
    else:
        return gradientDescent_vectorized(X, y, theta, alpha, iterations)