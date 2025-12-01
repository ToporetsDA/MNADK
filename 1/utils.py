import time
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Нормалiзацiя та приведення до вектора (28*28 = 784)
    x_train = x_train.reshape((60000, 784)).astype("float32") / 255.0
    x_test = x_test.reshape((10000, 784)).astype("float32") / 255.0

    return x_train, y_train, x_test, y_test

def one_hot(y):
    enc = OneHotEncoder(sparse_output=False)

    return enc.fit_transform(y.reshape(-1,1))

def measure_time(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()

    return result, end - start
