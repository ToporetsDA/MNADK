import numpy as np
import time
from utils import save_model_data

class Elman:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Випадкові ваги
        self.Wxh = np.random.randn(input_size, hidden_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Why = np.random.randn(hidden_size, output_size) * 0.1

        self.bh = np.zeros(hidden_size)
        self.by = np.zeros(output_size)

    def forward(self, X):
        """X shape: (seq_len, input_size)"""
        h = np.zeros(self.hidden_size)
        hs = []
        ys = []
        for x in X:
            h = np.tanh(x @ self.Wxh + h @ self.Whh + self.bh)
            y = h @ self.Why + self.by
            hs.append(h.copy())
            ys.append(y.copy())
        return np.array(ys), np.array(hs)
    
    def train(self, X_train, y_train, lr=0.01, epochs=200):
        start = time.time()

        for _ in range(epochs):
            h = np.zeros(self.hidden_size)

            for t in range(len(X_train)):
                x = X_train[t]
                y_true = y_train[t]

                # forward step
                h_prev = h.copy()
                h = np.tanh(x @ self.Wxh + h @ self.Whh + self.bh)
                y = h @ self.Why + self.by

                error = y - y_true

                # gradients (спрощено)
                dWhy = np.outer(h, error)
                dby = error

                dh = error @ self.Why.T * (1 - h ** 2)
                dWxh = np.outer(x, dh)
                dWhh = np.outer(h_prev, dh)
                dbh = dh

                # update
                self.Why -= lr * dWhy
                self.by -= lr * dby
                self.Wxh -= lr * dWxh
                self.Whh -= lr * dWhh
                self.bh -= lr * dbh

        return time.time() - start

def run_elman(X_train, y_train, X_test, y_test, name="elman_model"):
    net = Elman(X_train.shape[1], hidden_size=10, output_size=y_train.shape[1])

    train_time = net.train(X_train, y_train)
    start_test = time.perf_counter()
    y_pred, _ = net.forward(X_test)
    end_test = time.perf_counter()
    test_time = end_test - start_test

    preds = np.argmax(y_pred, axis=1)
    true = np.argmax(y_test, axis=1)
    error = np.mean(preds != true)

    # Завдання                                                     | Елмана                                                                                                                     |
    # Початкові параметри нейромоделі                              | `input_size`, `hidden_size`, `output_size`; початкові ваги `Wxh`, `Whh`, `Why`, `bh`, `by`                                 |
    # Результати навчання (матриця ваг)                            | Вагові матриці після навчання: `Wxh`, `Whh`, `Why` та зміщення `bh`, `by`                                                  |
    # Структура з функціями активації і кількістю нейроелементів   | Кількість шарів: 3 (in → hidden → out), N нейронів: `[input_size, hidden_size, output_size]`, активації: `[tanh, linear]`  |
    # Вхідні дані навчальної вибірки                               | `X_train` (наприклад, MNIST), `y_train` (one-hot)                                                                          |
    # Вихідні дані навчальної вибірки                              | Передбачені значення `y_pred_train` = `forward(X_train)`                                                                   |
    # Вхідні дані тестової вибірки                                 | `X_test`, `y_test`                                                                                                         |
    # Вихідні дані тестової вибірки                                | Передбачені значення `y_pred_test` = `forward(X_test)`                                                                     |
    # Помилка класифікації / оцінювання                            | `error = mean(y_pred_test != y_test)` або 1-accuracy                                                                       |
    # Час навчання                                                 | Виміряємо через `time.time()` до і після навчання                                                                          |
    # Час класифікації / відновлення                               | Виміряємо через `time.time()` під час `recall()`                                                                           |

    model_data = {
        "type": "Elman",
        "input_size": net.input_size,
        "hidden_size": net.hidden_size,
        "output_size": net.output_size,
        "Wxh": net.Wxh.tolist(),
        "Whh": net.Whh.tolist(),
        "Why": net.Why.tolist(),
        "bh": net.bh.tolist(),
        "by": net.by.tolist(),
        "train_input": X_train.tolist(),
        "test_input": X_test.tolist(),
        "test_output": y_pred.tolist(),
        "train_time": train_time,
        "test_time": test_time,
        "error": error
    }

    save_model_data(name, model_data)

    return train_time, test_time, error, model_data
