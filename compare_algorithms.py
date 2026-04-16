import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUT_DIR = "comparison_plots"

os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]

plt.style.use("default")


def load_data():
    data = {}

    for file in os.listdir(RESULTS_DIR):
        if file.endswith("_metrics.csv"):
            df = pd.read_csv(os.path.join(RESULTS_DIR, file))

            print(f"\nLoaded {file}")
            print(df.head())

            algo = file.replace("_metrics.csv", "")
            data[algo] = df

    return data


def plot(metric, data):
    plt.figure()

    plotted = False

    for algo, df in data.items():
        if metric not in df.columns:
            continue

        df = df.sort_values("round")

        plt.plot(
            df["round"],
            df[metric],
            marker="o",
            linewidth=2,
            label=algo.upper()
        )

        plotted = True

    if not plotted:
        print(f"[WARNING] No data for {metric}")
        plt.close()
        return

    plt.legend()
    plt.title(metric.upper())
    plt.xlabel("Rounds")
    plt.ylabel(metric)
    plt.grid(True)

    plt.savefig(f"{OUT_DIR}/{metric}.png")
    plt.close()


def main():
    data = load_data()

    for m in METRICS:
        plot(m, data)

    print("\nComparison plots saved ✅")


if __name__ == "__main__":
    main()