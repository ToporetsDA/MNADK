import json
import time
import numpy as np
from lvq import LVQ

def load_lvq(name):
    """Завантаження моделі LVQ з файлу"""
    with open(f"1/results/models/{name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    net = LVQ(data["input_size"], data["output_size"], n_prototypes=len(data["prototypes"]))
    net.prototypes = np.array(data["prototypes"])
    net.labels = np.array(data["labels"])

    return net, data

def evaluate_lvq(name):
    net, data = load_lvq(name)

    X_train = np.array(data["train_input"])
    y_train_true = np.array(data["train_output"])
    X_test  = np.array(data["test_input"])
    y_test_true = np.array(data["test_output"])

    # Класифікація тестової вибірки
    start_test = time.perf_counter()
    y_pred_test = net.predict(X_test)
    end_test = time.perf_counter()
    test_eval_time = end_test - start_test
    test_error = np.mean(y_pred_test != y_test_true)

    return test_eval_time, test_error
