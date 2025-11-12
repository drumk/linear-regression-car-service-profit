import numpy as np
import pickle

# загрузка обученных параметров
with open("theta.pkl", "rb") as f:
    theta = pickle.load(f)

def predict_profit(num_cars):
    """
    num_cars — количество автомобилей (×10000).
    Возвращает прибыль в долларах.
    """
    return np.dot([1, num_cars], theta) * 10_000

if __name__ == "__main__":
    print("Прогноз прибыли СТО (введите 'q' для выхода)")
    while True:
        val = input("Количество автомобилей (в 10 000): ")  # ИСПРАВЛЕНО
        if val.lower() == "q":
            break
        try:
            num = float(val)
            print(f"Ожидаемая прибыль: ${predict_profit(num):,.2f}\n")
        except ValueError:
            print("Ошибка: нужно число или 'q'")