import numpy as np

# Элементы программирования (loops)
def computeCost_loops(X, y, theta):
    m = len(y)
    cost = 0.0
    for i in range(m):
        prediction = 0
        for j in range(len(theta)):
            prediction += theta[j] * X[i][j]
        error = prediction - y[i]
        cost += error ** 2
    return cost / (2 * m)

# 2. Стандартные функции Python (sum + поэлементное перемножение)
def computeCost_pythonic(X, y, theta):
    m = len(y)
    predictions = [sum(theta[j] * X[i][j] for j in range(len(theta))) for i in range(m)]
    squared_errors = [(predictions[i] - y[i]) ** 2 for i in range(m)]
    return sum(squared_errors) / (2 * m)

# Векторный способ (NumPy)
def computeCost_vectorized(X, y, theta):
    m = len(y)
    errors = X @ theta - y
    return (errors.T @ errors) / (2 * m)


def computeCost(X, y, theta, mode=3):
    if mode == 1:
        return computeCost_loops(X, y, theta)
    elif mode == 2:
        return computeCost_pythonic(X, y, theta)
    else:
        return computeCost_vectorized(X, y, theta)