import numpy as np
import time
from utils import save_model_data

class Hopfield:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.W = np.zeros((n_neurons, n_neurons))

    def train(self, patterns):
        """patterns: array of shape (n_patterns, n_neurons) зі значеннями -1/+1"""
        n = self.n_neurons
        self.W = np.zeros((n, n))
        for p in patterns:
            self.W += np.outer(p, p)
        np.fill_diagonal(self.W, 0)

    def recall(self, patterns, steps=64):
        out = patterns.copy()
        for _ in range(steps):
            out = np.sign(out @ self.W)
        return out

def run_hopfield(patterns_train, patterns_test, name="hopfield_model"):
    """Навчання та оцінка Hopfield"""
    n_neurons = patterns_train.shape[1]
    net = Hopfield(n_neurons)

    start_train = time.perf_counter()
    net.train(patterns_train)
    end_train = time.perf_counter()
    train_time = end_train - start_train

    start_test = time.perf_counter()
    recalled = net.recall(patterns_test)
    end_test = time.perf_counter()
    test_time = end_test - start_test

    error = np.mean(recalled != patterns_test)
    
    # Початкові параметри нейромоделі                              | `n_neurons`, початкові ваги `W`                    |
    # Результати навчання (матриця ваг)                            | Матриця ваг `W` після навчання                     |
    # Структура з функціями активації і кількістю нейроелементів   | `n_neurons`, активація `sign`                      |
    # Вхідні дані навчальної вибірки                               | `patterns_train`                                   |
    # Вихідні дані навчальної вибірки                              | Відновлені патерни `recall(patterns_train)`        |
    # Вхідні дані тестової вибірки                                 | `patterns_test`                                    |
    # Вихідні дані тестової вибірки                                | Відновлені патерни `recall(patterns_test)`         |
    # Помилка класифікації / оцінювання                            | `error = mean(recalled != original)`               |
    # Час навчання                                                 | Виміряємо через `time.time()` до і після навчання  |
    # Час класифікації / відновлення                               | Виміряємо через `time.time()` під час `recall()`   |

    model_data = {
        "type": "Hopfield",
        "neurons": n_neurons,
        "weights": net.W.tolist(),
        "train_input": patterns_train.tolist(),
        "test_input": patterns_test.tolist(),
        "test_output": recalled.tolist(),
        "train_time": train_time,
        "test_time": test_time,
        "error": error
    }

    save_model_data(name, model_data)

    return train_time, test_time, error, model_data
