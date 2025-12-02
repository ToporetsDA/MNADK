import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

from utils import load_mnist, one_hot, measure_time

def run_mlp(lr=None):

    if lr is None:
        print("\n=== Багатошаровий персептрон (MLP, варіант 20) ===")
        optimizer = "adam"
    else:
        optimizer = SGD(learning_rate=lr)
    
    x_train, y_train, x_test, y_test = load_mnist()

    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)

    activate_layer = ["tanh", "tanh", "softmax"]

    model = Sequential()
    model.add(Dense(8, activation=activate_layer[0], input_shape=(784,)))
    model.add(Dense(8, activation=activate_layer[1]))
    model.add(Dense(10, activation=activate_layer[2]))  # вихід на 10 класів

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    (_, train_time) = measure_time(model.fit, x_train, y_train_oh, epochs=8, batch_size=32, verbose=1)

    (_, test_time) = measure_time(model.evaluate, x_test, y_test_oh, verbose=0)

    loss, acc = model.evaluate(x_test, y_test_oh, verbose=0)

    model_data = None

    if lr is not None: # experiment
        model_data = {
            "PARAM___learning_rates": lr,
            "weights": [w.tolist() for w in model.get_weights()],
            "activation": activate_layer,
            "layers": 3,
            "neurons": [8, 8, 10],
            # "train_output": y_train.tolist(),
            # "test_output": y_test.tolist(),
            # "train_input": x_train.tolist(),
            # "test_input": x_test.tolist()
        }
    else: # train
        name = input("\nОберіть назву мережі: ").strip()
        model.save(f"1/models/mlp_{name}.keras")

    return train_time, test_time, acc, model_data
