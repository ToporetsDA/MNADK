import json
import time
import numpy as np

from tensorflow.keras.models import load_model as keras_load_model

#  RBF utils

def rbf_transform(X, centers, sigma):
    diffs = X[:, None, :] - centers[None, :, :]
    sqd = np.sum(diffs * diffs, axis=2)
    return np.exp(-sqd / (2.0 * sigma**2))

def predict_rbf(X, centers, sigma, W):
    R = rbf_transform(X, centers, sigma)
    logits = R @ W
    return np.argmax(logits, axis=1)

#  SLP / MLP utils

def activation(x, act):
    if act == "tanh":
        return np.tanh(x)
    if act == "relu":
        return np.maximum(0, x)
    if act == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if act == "linear":
        return x
    if act == "softmax":
        # числова стабілізація
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    raise ValueError(f"Unknown activation: {act}")

def forward_mlp(X, weights, biases, activations):
    out = X
    for W, b, act in zip(weights, biases, activations):
        out = activation(out @ W + b, act)
    return out

def predict_mlp(X, weights, biases, activations):
    logits = forward_mlp(X, weights, biases, activations)
    return np.argmax(logits, axis=1)


#  Main loader (universal)

def load_model(model_name):

    filename = "1/models/" + model_name

    try: # It's RBF
        with open(filename + ".json", "r") as f:
            data = json.load(f)

            if "centers" in data and "sigma" in data and "W" in data:
                
                centers = np.array(data["centers"])
                sigma = data["sigma"]
                W = np.array(data["W"])

                return {
                    "type": "RBF",
                    "centers": centers,
                    "sigma": sigma,
                    "W": W
                }
    except FileNotFoundError:
        pass

    try: # It's MLP or SLP
        model = keras_load_model(filename + ".keras")

        weights = []
        biases = []
        activations = []

        for layer in model.layers:
            if hasattr(layer, "get_weights"):
                Wb = layer.get_weights()
                if len(Wb) == 2:
                    W, b = Wb
                    weights.append(W)
                    biases.append(b)
                    # Ось тут витягуємо *справжню* активацію
                    activations.append(layer.activation.__name__)

        return {
            "type": "MLP",
            "weights": weights,
            "biases": biases,
            "activations": activations
        }
    except FileNotFoundError:
        pass

    raise ValueError("Unknown model format")

#  Unified predictor (MAIN)
def predict(model_name, X):
    model = load_model(model_name)

    if model["type"] == "RBF":
        return predict_rbf(
            X,
            model["centers"],
            model["sigma"],
            model["W"]
        )

    if model["type"] in ("MLP", "SLP"):
        return predict_mlp(
            X,
            model["weights"],
            model["biases"],
            model["activations"]
        )

    raise ValueError("Internal error: unknown model type")

def evaluate(model_name, X, y_true):
    """
    Повертає:
    - час класифікації (секунди)
    - помилку класифікації
    - accuracy
    - preds (масив передбачених класів)
    """

    start = time.time()
    preds = predict(model_name, X)
    end = time.time()

    classification_time = end - start
    error_rate = np.mean(preds != y_true)
    accuracy = 1.0 - error_rate

    return classification_time, error_rate, accuracy, preds