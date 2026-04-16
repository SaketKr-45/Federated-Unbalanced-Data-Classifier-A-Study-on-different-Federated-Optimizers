import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def save_round_metrics_json(algorithm: str, round_metrics: List[Dict], out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{algorithm}_round_metrics.json")
    with open(path, "w") as f:
        json.dump(round_metrics, f, indent=2)
    print(f"[Evaluation] Saved round metrics: {path}")


def plot_confusion_matrix(tp, tn, fp, fn, title="Confusion Matrix", out_path="confusion_matrix.png"):
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Evaluation] Saved confusion matrix: {out_path}")


def plot_roc_curve(y_true, y_prob, title="ROC Curve", out_path="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Evaluation] Saved ROC curve: {out_path}")