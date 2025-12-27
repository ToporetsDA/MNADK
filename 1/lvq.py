import numpy as np
import time
from utils import save_model_data

class LVQ:
    def __init__(self, input_size, output_size, n_prototypes=10):
        self.input_size = input_size
        self.output_size = output_size
        self.n_prototypes = n_prototypes
        self.prototypes = None
        self.labels = None

    def init_prototypes(self, X, y):
        """Ініціалізація прототипів з навчальної вибірки"""
        indices = np.random.choice(len(X), self.n_prototypes, replace=False)
        self.prototypes = X[indices].copy()
        self.labels = np.argmax(y[indices], axis=1)

    def train(self, X, y, lr=0.01, epochs=20):
        start = time.perf_counter()
        for _ in range(epochs):
            for i in range(len(X)):
                xi = X[i]
                yi = np.argmax(y[i])
                # знаходження найближчого прототипу
                distances = np.linalg.norm(self.prototypes - xi, axis=1)
                j = np.argmin(distances)
                # оновлення прототипу
                if self.labels[j] == yi:
                    self.prototypes[j] += lr * (xi - self.prototypes[j])
                else:
                    self.prototypes[j] -= lr * (xi - self.prototypes[j])
        end = time.perf_counter()
        return end - start

    def predict(self, X):
        preds = []
        for xi in X:
            distances = np.linalg.norm(self.prototypes - xi, axis=1)
            j = np.argmin(distances)
            preds.append(self.labels[j])
        return np.array(preds)

def run_lvq(X_train, y_train, X_test, y_test, name="lvq_model", n_prototypes=15, lr=0.01, epochs=50):
    n_classes = y_train.shape[1]
    net = LVQ(X_train.shape[1], n_classes, n_prototypes=n_prototypes)
    
    # Ініціалізуємо прототипи з навчальної вибірки
    net.init_prototypes(X_train, y_train)

    # Навчання
    train_time = net.train(X_train, y_train, lr=lr, epochs=epochs)

    # Класифікація навчальної вибірки
    start_train = time.perf_counter()
    y_pred_train = net.predict(X_train)
    end_train = time.perf_counter()
    train_eval_time = end_train - start_train
    train_error = np.mean(y_pred_train != np.argmax(y_train, axis=1))

    # Класифікація тестової вибірки
    start_test = time.perf_counter()
    y_pred_test = net.predict(X_test)
    end_test = time.perf_counter()
    test_eval_time = end_test - start_test
    test_error = np.mean(y_pred_test != np.argmax(y_test, axis=1))

    # Збереження моделі та результатів
    model_data = {
        "type": "LVQ",
        "input_size": net.input_size,
        "output_size": net.output_size,
        "prototypes": net.prototypes.tolist(),
        "labels": net.labels.tolist(),
        "train_input": X_train.tolist(),
        "test_input": X_test.tolist(),
        "train_output": np.argmax(y_train, axis=1).tolist(),
        "test_output": np.argmax(y_test, axis=1).tolist(),
        "train_time": train_time,
        "train_eval_time": train_eval_time,
        "train_error": train_error,
        "test_eval_time": test_eval_time,
        "test_error": test_error
    }

    save_model_data(name, model_data)

    return train_time, train_eval_time, train_error, model_data
