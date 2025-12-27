import json
import time
import numpy as np
from hopfield import Hopfield

def load_hopfield_model(name):
    with open(f"1/results/models/{name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    net = Hopfield(data["neurons"])
    net.W = np.array(data["weights"])

    return net, data

def evaluate_hopfield(name):
    net, data = load_hopfield_model(name)

    X_test = np.array(data["test_input"])
    Y_true = np.array(data["test_output"])

    # ⏱ час класифікації (відновлення)
    start = time.time()
    recalled = net.recall(X_test)
    end = time.time()

    test_time = end - start

    # ❌ помилка = частка неправильно відновлених патернів
    errors = np.mean(np.any(recalled != Y_true, axis=1))

    return test_time, errors
