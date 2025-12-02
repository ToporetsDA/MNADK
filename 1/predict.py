import json
import time
import numpy as np

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

    filename = "1/models/" + model_name + ".json"
    data = None

    with open(filename, "r") as f:
        data = json.load(f)
    
    if data is None:
        raise ValueError("No such model")

    # Detect model type

    if "centers" in data and "sigma" in data and "W" in data:
        # It's RBF
        centers = np.array(data["centers"])
        sigma = data["sigma"]
        W = np.array(data["W"])

        return {
            "type": "RBF",
            "centers": centers,
            "sigma": sigma,
            "W": W
        }

    if "weights" in data and "biases" in data and "activations" in data:
        # It's MLP or SLP
        weights = [np.array(w) for w in data["weights"]]
        biases  = [np.array(b) for b in data["biases"]]
        activations = data["activations"]

        return {
            "type": "MLP",
            "weights": weights,
            "biases": biases,
            "activations": activations
        }

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