import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl

from utils import (
    set_seed,
    load_and_preprocess,
    make_torch_dataset,
    get_model_parameters,
    set_model_parameters,
    compute_metrics,
)


# Basic Neural Network for Fraud Detection
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, dataset_path: str):
        self.cid = cid
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_count, meta = load_and_preprocess(dataset_path)
        self.y_test = y_test

        train_dataset = make_torch_dataset(X_train, y_train)
        test_dataset = make_torch_dataset(X_test, y_test)

        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Initialize model and loss function
        self.model = SimpleModel(feature_count).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        epochs = config.get("local_epochs", 1)

        for _ in range(epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                optimizer.step()

        return get_model_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        self.model.eval()

        losses = []
        probs_all = []

        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                probs = torch.sigmoid(logits)

                losses.append(loss.item())
                probs_all.extend(probs.cpu().numpy().flatten().tolist())

        # =========================
        # SAVE PREDICTIONS (CLIENT ONLY)
        # =========================
        os.makedirs("predictions/clients", exist_ok=True)

        y_true = self.y_test
        y_prob = np.array(probs_all)

        np.save(f"predictions/clients/client_{self.cid}_y_true.npy", y_true)
        np.save(f"predictions/clients/client_{self.cid}_y_prob.npy", y_prob)

        print(f"[Client {self.cid}] Predictions saved ✅")

        metrics = compute_metrics(self.y_test, y_prob)

        return float(sum(losses) / len(losses)), len(self.test_loader.dataset), metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    set_seed(42)
    client = FlowerClient(cid="1", dataset_path="dataset_random_split1.csv")
    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()