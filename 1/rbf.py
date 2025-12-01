import os
# Обмежуємо кількість потоків BLAS/OMP щоб не навантажувати систему
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from utils import load_mnist, one_hot, measure_time

def rbf_transform(X, centers, sigma):
    # Гаусова RBF: повертає матрицю (n_samples, n_centers)
    # Використовує ефективні операції векторизації
    # X: (n_samples, d), centers: (n_centers, d)
    diffs = X[:, None, :] - centers[None, :, :]     # (n_samples, n_centers, d)
    sqd = np.sum(diffs * diffs, axis=2)            # (n_samples, n_centers)
    return np.exp(-sqd / (2.0 * sigma**2))


def run_rbf(param_n_centers=None):

    if param_n_centers is None:
        print("\n=== RBF-мережа ===")
        n_centers=50
    else:
        n_centers = param_n_centers
    
    x_train, y_train, x_test, y_test = load_mnist()
    y_train_oh = one_hot(y_train)
    y_test_oh = one_hot(y_test)

    # 1) Кластеризація
    subset = x_train[:2000]
    kmeans = MiniBatchKMeans(n_clusters=n_centers, batch_size=512, n_init=3)
    (_, t_cluster) = measure_time(kmeans.fit, subset)
    centers = kmeans.cluster_centers_

    # 2) sigma
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    sigma = np.mean(np.min(dists, axis=1))

    # 3) RBF-простір
    (_, t_rbf_train) = measure_time(rbf_transform, x_train, centers, sigma)
    R_train = rbf_transform(x_train, centers, sigma)

    (_, t_rbf_test) = measure_time(rbf_transform, x_test, centers, sigma)
    R_test = rbf_transform(x_test, centers, sigma)

    # 4) лінійні ваги
    y_train_oh = one_hot(y_train)
    RtR = R_train.T @ R_train
    RtY = R_train.T @ y_train_oh
    A = RtR + 1e-3 * np.eye(n_centers)
    (_, t_solve) = measure_time(np.linalg.solve, A, RtY)
    W = np.linalg.solve(A, RtY)

    # 5) оцінка
    preds = R_test @ W
    preds = np.argmax(preds, axis=1)
    acc = accuracy_score(y_test, preds)

    # загальний час
    total_time = t_cluster + t_rbf_train + t_solve

    model_data = None

    if param_n_centers is not None:
        model_data = {
            "PARAM___n_centers": n_centers,
            "weights": W.tolist(),
            "activation": "tanh",
            # "train_output": y_train.tolist(),
            # "test_output": y_test.tolist(),
            # "train_input": x_train.tolist(),
            # "test_input": x_test.tolist()
        }

    return total_time, _, acc, model_data
