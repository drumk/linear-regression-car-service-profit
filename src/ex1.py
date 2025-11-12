import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent
from predict import predict
import pickle
import os

# 1. загрузка данных

DATA_FILE = os.path.join(os.path.dirname(__file__), "ex1data1.txt")

try:
    data = np.loadtxt(DATA_FILE, delimiter=",")
except FileNotFoundError:
    raise SystemExit(f"Не найден файл {DATA_FILE}")

X_raw, y = data[:, 0], data[:, 1]
m = len(y)

# 2. визуализация исходных точек

plotData(X_raw, y)

# 3. «нормализация»: добавляем столбец единиц
#    (для одного признака z‑score не обязателен, поэтому оставляем
#     X_raw как есть, а «нормируем» только добавлением 1)

X = np.column_stack((np.ones(m), X_raw))

# 4. инициализация θ и стартовая стоимость

theta = np.zeros(2)
print("Начальная cost =", computeCost(X, y, theta))


# 5. градиентный спуск

alpha = 0.01
iterations = 1500
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print("Найденные θ :", theta)


# 6. график линии регрессии

plt.figure()
plt.scatter(X_raw, y, marker="x", color="blue", label="Обучающие данные")
plt.plot(X_raw, X @ theta, color="red", label="Линия регрессии")
plt.xlabel("Количество автомобилей (×10 000)")
plt.ylabel("Прибыль (×10 000$)")
plt.title("Линейная регрессия")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.4)
plt.tight_layout()
plt.show()


# 7. примеры предсказаний

pop1, pop2 = 5, 7.5
profit1 = predict(np.array([1, pop1]), theta) * 10_000
profit2 = predict(np.array([1, pop2]), theta) * 10_000
print(f"Для 35 000 автомобилей: ${profit1:,.2f}")
print(f"Для 70 000 автомобилей: ${profit2:,.2f}")


# 8. сохранение весов

with open("theta.pkl", "wb") as f:
    pickle.dump(theta, f)
print("Весовые коэффициенты сохранены в theta.pkl")