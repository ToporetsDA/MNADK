import json
import numpy as np
import matplotlib.pyplot as plt

from slp import run_slp
from mlp import run_mlp
from rbf import run_rbf

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score

from utils import load_mnist, one_hot, measure_time

#  МЕТОДИ

ACT = "tanh"  # варіант 20 — тангенційна сигмоїда

def rbf_transform(X, centers, sigma):
    diffs = X[:, None, :] - centers[None, :, :]
    sqd = np.sum(diffs * diffs, axis=2)
    return np.exp(-sqd / (2.0 * sigma**2))

def run_learning_rates(run_func, arr, param_name):
    times = []
    accs = []
    more = []

    for i in arr:
        print(f"\n{param_name} = {i}")

        tr, _, acc, md = run_func(i)

        times.append(tr)
        accs.append(float(acc))
        more.append(md)

        print(f"Час навчання: {tr:.2f} сек, точність: {acc:.4f}")

    return times, accs, more

def plot_experiment_subplots(subplots_info, save_path, dpi=150):
    n = len(subplots_info)
    plt.figure(figsize=(6, 4*n))

    for i, info in enumerate(subplots_info, start=1):
        plt.subplot(n, 1, i)
        plt.plot(info['x'], info['y'], marker="o")
        if 'xscale' in info:
            plt.xscale(info['xscale'])
        plt.title(info['title'])
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close()

#  ЕКСПЕРИМЕНТ

def run_experiment():
    print("\n=== ДОСЛІД: SLP + MLP + RBF ===")

    # Вибір формату збереження
    print("\nОберiть формат збереження графіків:")
    print("A — один файл з трьома subplot-графіками")
    print("B — три окремі файли")
    mode = input("Ваш вибір (A/B): ").strip().upper()

    if mode not in ("A", "B"):
        print("Невідомий вибір, за замовчуванням — B")
        mode = "B"

    # параметри експериментiв
    learning_rates = [0.1, 0.03, 0.01, 0.003, 0.001]
    rbf_centers_list = [10, 20, 30, 40]

    print("\n=== Експеримент SLP ===")
    slp_times, slp_accs, slp_more = run_learning_rates(run_slp, learning_rates, "LR")

    print("\n=== Експеримент MLP ===")
    mlp_times, mlp_accs, mlp_more = run_learning_rates(run_mlp, learning_rates, "LR")

    print("\n=== Експеримент RBF ===")
    rbf_times, rbf_accs, rbf_more = run_learning_rates(run_rbf, rbf_centers_list, "n_centers")

    # РЕЗУЛЬТАТИ

    option = ""
    
    # Графіки

    if mode == "A":

        subplots_info = [
            # SLP subplot
            {"x": learning_rates,   "y": slp_times, "title": "SLP: Час навчання vs learning rate", "xscale": "log"},
            # MLP subplot
            {"x": learning_rates,   "y": mlp_times, "title": "MLP: Час навчання vs learning rate", "xscale": "log"},
            # RBF subplot
            {"x": rbf_centers_list, "y": rbf_times, "title": "RBF: Час навчання vs кількість центрів"}
        ]

        plot_experiment_subplots(subplots_info, "1/results/exp/experiment_all_in_one.png")

        option = "A"

    else:  # mode == B

        # окремо SLP
        subplots_info = [
            {"x": learning_rates,   "y": slp_times, "title": "SLP: Час навчання vs learning rate", "xscale": "log"}
        ]
        plot_experiment_subplots(subplots_info, "1/results/exp/slp_time_vs_lr.png")
        # окремо MLP
        subplots_info = [
            {"x": learning_rates,   "y": mlp_times, "title": "MLP: Час навчання vs learning rate", "xscale": "log"}
        ]
        plot_experiment_subplots(subplots_info, "1/results/exp/mlp_time_vs_lr.png")
        # окремо RBF
        subplots_info = [
            {"x": rbf_centers_list, "y": rbf_times, "title": "RBF: Час навчання vs кількість центрів"}
        ]
        plot_experiment_subplots(subplots_info, "1/results/exp/rbf_time_vs_centers.png")

        option = "B"

    # Збереження результатів

    results = {
        "models": {
            "SLP": {
                "learning_rates": learning_rates,
                "times": slp_times,
                "accuracies": slp_accs,
                "activation": ACT,
                "layers": 1,
                "neurons": [10],
                "more": slp_more
            },
            "MLP": {
                "learning_rates": learning_rates,
                "times": mlp_times,
                "accuracies": mlp_accs,
                "activation": ACT,
                "layers": 3,
                "neurons": [8, 8, 10],
                "more": mlp_more
            },
            "RBF": {
                "centers": rbf_centers_list,
                "times": rbf_times,
                "accuracies": rbf_accs,
                "activation": "RBF + linear layer",
                "sigma_note": "avg nearest-center distance",
                "more": rbf_more
            }
        }
    }

    with open(f"1/results/exp/experiment_results_{option}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print("\n=== Експеримент завершено! ===")
    print("Результати та графіки збережено в папку results/")
