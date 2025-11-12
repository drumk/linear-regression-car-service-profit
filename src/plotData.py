import matplotlib.pyplot as plt

def plotData(X, y):
    plt.figure()
    plt.scatter(X, y, marker='x', color='blue')
    plt.xlabel("Количество автомобилей (×10 000)")
    plt.ylabel("Прибыль (×10 000$)")
    plt.title("Обучающая выборка")
    plt.grid(True, linestyle="--", linewidth=0.4)
    plt.tight_layout()
    plt.show()