import os
import json
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUT_DIR = "comparison_plots"

os.makedirs(OUT_DIR, exist_ok=True)

METRICS = ["accuracy", "f1", "precision", "recall", "roc_auc"]


def load_data():
    data = {}
    for file in os.listdir(RESULTS_DIR):
        if file.endswith(".json"):
            algo = file.replace("_round_metrics.json", "")
            with open(os.path.join(RESULTS_DIR, file)) as f:
                df = pd.DataFrame(json.load(f))
                data[algo] = df
    return data


def plot(metric, data):
    plt.figure()

    for algo, df in data.items():
        subset = df[df["metric"] == metric]

        if subset.empty:
            continue

        subset = subset.sort_values("round")

        plt.plot(subset["round"], subset["value"], label=algo.upper())

    plt.legend()
    plt.title(metric.upper())
    plt.xlabel("Rounds")
    plt.ylabel(metric)

    plt.savefig(f"{OUT_DIR}/{metric}.png")
    plt.close()


def main():
    data = load_data()

    for m in METRICS:
        plot(m, data)

    print("Comparison plots saved ✅")


if __name__ == "__main__":
    main()