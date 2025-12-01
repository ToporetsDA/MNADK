from slp import run_slp
from mlp import run_mlp
from rbf import run_rbf
from experiment import run_experiment

def main():
    print("Оберіть мережі для запуску:")
    print("1 — Одношаровий персептрон (SLP)")
    print("2 — Багатошаровий персептрон (MLP)")
    print("3 — RBF-мережа")
    print("4 — Усі одразу")
    print("5 — Дослід: графік часу від learning rate")

    choice = input("\nВаш вибір: ").strip()

    if choice == "1":
        tr, te, acc, _ = run_slp()
        print(f"\n[SLP] Час навчання: {tr:.2f} сек | Час класифікації: {te:.2f} сек | Точність: {acc:.4f}")
    elif choice == "2":
        tr, te, acc, _ = run_mlp()
        print(f"\n[MLP] Час навчання: {tr:.2f} сек | Час класифікації: {te:.2f} сек | Точність: {acc:.4f}")
    elif choice == "3":
        acc, _ = run_rbf()
        print(f"Точність RBF: {acc:.4f}")
    elif choice == "4":
        tr, te, acc, _ = run_slp()
        print(f"\n[SLP] Час навчання: {tr:.2f} сек | Час класифікації: {te:.2f} сек | Точність: {acc:.4f}")
        tr, te, acc, _ = run_mlp()
        print(f"\n[MLP] Час навчання: {tr:.2f} сек | Час класифікації: {te:.2f} сек | Точність: {acc:.4f}")
        tr, _, acc, _ = run_rbf()
        print(f"\n[RBF] Час навчання: {tr:.2f} сек | точність: {acc:.4f}")
    elif choice == "5":
        run_experiment()
    else:
        print("Невідомий вибір.")

if __name__ == "__main__":
    main()
