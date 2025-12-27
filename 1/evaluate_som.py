import json
import time
import numpy as np
from som import SOM

def load_som(name):
    """Завантаження моделі SOM з файлу"""
    with open(f"1/results/models/{name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    net = SOM(data["input_size"], grid_size=tuple(data["grid_size"]))
    net.weights = np.array(data["weights"])

    return net, data

def evaluate_som(name):
    net, data = load_som(name)

    X_train = np.array(data["train_input"])
    X_test  = np.array(data["test_input"])

    # Класифікація тестової вибірки
    start_test = time.perf_counter()
    y_pred_test = net.predict(X_test)
    end_test = time.perf_counter()
    test_eval_time = end_test - start_test

    return test_eval_time
