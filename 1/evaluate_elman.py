import json
import time
import numpy as np
from elman import Elman

def load_elman(name):
    with open(f"1/results/models/{name}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    net = Elman(
        data["input_size"],
        data["hidden_size"],
        data["output_size"]
    )

    net.Wxh = np.array(data["Wxh"])
    net.Whh = np.array(data["Whh"])
    net.Why = np.array(data["Why"])
    net.bh  = np.array(data["bh"])
    net.by  = np.array(data["by"])

    return net, data

def evaluate_elman(name):
    net, data = load_elman(name)

    X_test = np.array(data["test_input"])
    y_true = np.array(data["test_output"])

    start = time.perf_counter()
    y_pred, _ = net.forward(X_test)
    end = time.perf_counter()

    test_time = end - start

    preds = np.argmax(y_pred, axis=1)
    true  = np.argmax(y_true, axis=1)
    error = np.mean(preds != true)

    return test_time, error
