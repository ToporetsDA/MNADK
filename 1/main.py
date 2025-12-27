# Lab 1
from slp import run_slp
from mlp import run_mlp
from rbf import run_rbf
from experiment import run_experiment
from predict import evaluate
import json
import numpy as np

from tensorflow.keras.datasets import mnist

# Lab 2
from hopfield import run_hopfield
from elman import run_elman
from utils import load_mnist, one_hot
from evaluate_hopfield import evaluate_hopfield
from evaluate_elman import evaluate_elman

# Lab 3
from lvq import run_lvq
from som import run_som
from evaluate_lvq import evaluate_lvq
from evaluate_som import evaluate_som

import matplotlib.pyplot as plt
import numpy as np

def main():
    # Lab 1 (1-6), Lab 2 (7-10), Lab 3 (11-12)
    print("Оберіть мережі для запуску:")
    print("1 — Одношаровий персептрон (SLP)")
    print("2 — Багатошаровий персептрон (MLP)")
    print("3 — RBF-мережа")
    print("4 — Усі моделі з Lab1 одразу")
    print("5 — Дослід: графік часу від learning rate")
    print("6 — Використати існуючу мережу")
    print("7 — Мережа Хопфілда")
    print("8 — Мережа Елмана")
    print("9 — Використати збережену мережу Хопфілда")
    print("10 — Використати збережену мережу Елмана")
    print("11 — Мережа LVQ")
    print("12 — Мережа SOM")
    print("13 — Використати збережену мережу LVQ")
    print("14 — Використати збережену мережу SOM")
    
    choice = input("\nВаш вибір: ").strip()
    
    if choice == "1":
        tr, te, acc, _ = run_slp()
        print(f"\n[SLP] Час навчання: {tr:.2f} сек | Час класифікації: {te:.2f} сек | Точність: {acc:.4f}")
    
    elif choice == "2":
        tr, te, acc, _ = run_mlp()
        print(f"\n[MLP] Час навчання: {tr:.2f} сек | Час класифікації: {te:.2f} сек | Точність: {acc:.4f}")
    
    elif choice == "3":
        tr, te, acc, _ = run_rbf()
        print(f"\n[MLP] Час навчання: {tr:.2f} сек | Час класифікації: {te:.2f} сек | Точність: {acc:.4f}")
    
    elif choice == "4":
        tr, te, acc, _ = run_slp()
        print(f"\n[SLP] Час навчання: {tr:.2f} сек | Час класифікації: {te:.2f} сек | Точність: {acc:.4f}")
        tr, te, acc, _ = run_mlp()
        print(f"\n[MLP] Час навчання: {tr:.2f} сек | Час класифікації: {te:.2f} сек | Точність: {acc:.4f}")
        tr, te, acc, _ = run_rbf()
        print(f"\n[MLP] Час навчання: {tr:.2f} сек | Час класифікації: {te:.2f} сек | Точність: {acc:.4f}")
    
    elif choice == "5":
        run_experiment()
    
    elif choice == "6":
        # назва мережі
        name = input("\nВведіть назву моделі: ").strip()

        (_, _), (x_test, y_test) = mnist.load_data()
        # нормалізація 0..1
        x_test = x_test.astype(np.float32) / 255.0
        # розгортання в 784-вимірний вектор
        x_test = x_test.reshape((x_test.shape[0], -1))
        
        t, err, acc, preds = evaluate(name, x_test, y_test)
        print(f"\n[MLP] Час класифікації: {t:.2f} сек | Помилка класифікації: {err:.2f} | Точність: {acc:.4f}")

        with open(f"1/results/models/{name}.json", "w", encoding="utf-8") as f:
            json.dump(preds.tolist(), f, separators=(",", ":"), ensure_ascii=False)
    
    elif choice == "7":
        # Для демонстрації генеруємо бінарні патерни
        patterns_train = np.random.choice([-1, 1], size=(5, 16))
        patterns_test = patterns_train.copy()

        tr, te, err, _ = run_hopfield(patterns_train, patterns_test)
        print(f"[Hopfield] Час навчання: {tr:.6f} сек | Час класифікації: {te:.6f} сек | Помилка: {err:.4f}")

    elif choice == "8":
        x_train, y_train, x_test, y_test = load_mnist()
        y_train_oh = one_hot(y_train)
        y_test_oh = one_hot(y_test)

        tr, te, err, _ = run_elman(x_train[:500], y_train_oh[:500], x_test[:100], y_test_oh[:100])
        print(f"[Elman] Час навчання: {tr:.6f} сек | Час класифікації: {te:.6f} сек | Помилка: {err:.4f}")
    
    elif choice == "9":
        name = input("Введіть назву моделі Хопфілда: ").strip()

        te, err = evaluate_hopfield(name)
        print(f"[Hopfield] Час класифікації: {te:.6f} сек | Помилка: {err:.4f}")

    elif choice == "10":
        name = input("Введіть назву моделі Елмана: ").strip()

        te, err = evaluate_elman(name)
        print(f"[Elman] Час класифікації: {te:.6f} сек | Помилка: {err:.4f}")

    elif choice == "11":
        x_train, y_train, x_test, y_test = load_mnist()
        y_train_oh = one_hot(y_train)
        y_test_oh = one_hot(y_test)

        tr, tr_eval, tr_err, _ = run_lvq(
            x_train[:500], y_train_oh[:500], x_test[:100], y_test_oh[:100]
        )

        print(f"[LVQ] Час навчання: {tr:.6f} сек | Час класифікації: {tr_eval:.6f} сек | Помилка : {tr_err:.4f}")

    elif choice == "12":
        x_train, _, x_test, _ = load_mnist()
        train_time, train_eval_time, test_eval_time, test_eval_time2, model_data = run_som(x_train[:500], x_test[:100])
        print(f"[SOM] Час навчання: {train_time:.6f} сек | Час класифікації навчальної вибірки: {train_eval_time:.6f} сек | "
              f"Час класифікації тестової вибірки: {test_eval_time:.6f} сек")
        
        weights = np.array(model_data["weights"])  # карта нейронів після навчання
        plt.figure(figsize=(6,6))

        # Якщо дані 2D, можна просто scatter
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                plt.scatter(weights[i,j,0], weights[i,j,1], c='red')
        plt.title("SOM weights map")
        plt.show()

    elif choice == "13":
        name = input("Введіть назву моделі LVQ: ").strip()

        ts_e, ts_err = evaluate_lvq(name)
        print(f"[LVQ] Час класифікації тестової вибірки: {ts_e:.6f} сек | "
            f"Помилка тестової вибірки: {ts_err:.4f}")

    elif choice == "14":
        name = input("Введіть назву моделі SOM: ").strip()

        ts_e = evaluate_som(name)

        print(f"[SOM] Час класифікації тестової вибірки: {ts_e:.6f} сек")


    else:
        print("Невідомий вибір.")

if __name__ == "__main__":
    main()
