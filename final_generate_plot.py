import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="fedavg")
args = parser.parse_args()

OUT_DIR = f"final_plots/{args.algo}"
os.makedirs(OUT_DIR, exist_ok=True)
CLIENTS = ["1", "2", "3", "4"]

def plot_confusion(y_true, y_prob, name):
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm = [[tn, fp], [fn, tp]]

    plt.figure()
    plt.imshow(cm)
    plt.title(name)
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i][j], ha="center", va="center")
    plt.savefig(f"{OUT_DIR}/{name}_confusion.png")
    plt.close()

def plot_roc(y_true, y_prob, name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    plt.title(name)
    plt.savefig(f"{OUT_DIR}/{name}_roc.png")
    plt.close()

def run():
    global_y_true, global_y_prob = [], []

    for cid in CLIENTS:
        yt = f"predictions/clients/client_{cid}_y_true.npy"
        yp = f"predictions/clients/client_{cid}_y_prob.npy"

        if os.path.exists(yt) and os.path.exists(yp):
            y_true, y_prob = np.load(yt), np.load(yp)
            plot_confusion(y_true, y_prob, f"client_{cid}")
            plot_roc(y_true, y_prob, f"client_{cid}")
            global_y_true.extend(y_true.flatten().tolist())
            global_y_prob.extend(y_prob.flatten().tolist())

    if global_y_true:
        plot_confusion(np.array(global_y_true), np.array(global_y_prob), "global")
        plot_roc(np.array(global_y_true), np.array(global_y_prob), "global")

if __name__ == "__main__":
    run()