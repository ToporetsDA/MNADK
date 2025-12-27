import numpy as np
import time
from utils import save_model_data

class SOM:
    def __init__(self, input_size, grid_size=(5,5), lr=0.1):
        self.input_size = input_size
        self.grid_size = grid_size
        self.lr = lr
        self.weights = np.random.randn(grid_size[0], grid_size[1], input_size)

    def train(self, X, epochs=20):
        start = time.perf_counter()
        for _ in range(epochs):
            for x in X:
                # знайти найближчий нейрон
                dists = np.linalg.norm(self.weights - x, axis=2)
                i,j = np.unravel_index(np.argmin(dists), self.grid_size)
                # оновлення з урахуванням LR
                self.weights[i,j] += self.lr * (x - self.weights[i,j])
        end = time.perf_counter()
        return end - start

    def predict(self, X):
        preds = []
        for x in X:
            dists = np.linalg.norm(self.weights - x, axis=2)
            i,j = np.unravel_index(np.argmin(dists), self.grid_size)
            preds.append((i,j))
        return np.array(preds)

def run_som(X_train, X_test, name="som_model"):
    net = SOM(X_train.shape[1], grid_size=(5,5), lr=0.1)
    train_time = net.train(X_train)

    # класифікація навчальної вибірки
    start_train = time.perf_counter()
    y_pred_train = net.predict(X_train)
    end_train = time.perf_counter()
    train_eval_time = end_train - start_train

    # класифікація тестової вибірки
    start_test = time.perf_counter()
    y_pred_test = net.predict(X_test)
    end_test = time.perf_counter()
    test_eval_time = end_test - start_test

    model_data = {
        "type": "SOM",
        "input_size": net.input_size,
        "grid_size": net.grid_size,
        "weights": net.weights.tolist(),
        "train_input": X_train.tolist(),
        "test_input": X_test.tolist(),
        "train_output": y_pred_train.tolist(),
        "test_output": y_pred_test.tolist(),
        "train_time": train_time,
        "train_eval_time": train_eval_time,
        "test_eval_time": test_eval_time
    }

    save_model_data(name, model_data)
    return train_time, train_eval_time, train_eval_time, test_eval_time, model_data
