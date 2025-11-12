import numpy as np
from computeCost import computeCost
import os

def test_initial_cost():
    # ─ загрузка датасета
    data = np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "data", "ex1data1.txt"), delimiter=",")
    X_raw, y = data[:, 0], data[:, 1]
    m = len(y)

    # ─ формирование матрицы признаков [1  x]
    X = np.column_stack((np.ones(m), X_raw))
    theta = np.zeros(2)

    cost = computeCost(X, y, theta)
    expected = 32.073

    assert abs(cost - expected) < 1e-3, f"Ожидалось {expected}, получено {cost}"

    print(f"Тест пройден: cost = {cost:.6f} (~ {expected})")  # ИСПРАВЛЕНО

if __name__ == "__main__":
    test_initial_cost()