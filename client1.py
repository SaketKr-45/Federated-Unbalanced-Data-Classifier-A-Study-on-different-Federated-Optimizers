import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import flwr as fl
from pathlib import Path

from utils import (
    set_seed, load_and_preprocess, make_torch_dataset,
    get_model_parameters, set_model_parameters, compute_metrics,
)

class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class UpgradedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, dataset_path: str):
        self.cid = cid
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train, X_test, y_train, y_test, feature_count, _ = load_and_preprocess(dataset_path)
        self.y_test = y_test

        self.train_loader = DataLoader(make_torch_dataset(X_train, y_train), batch_size=32, shuffle=True)
        self.test_loader = DataLoader(make_torch_dataset(X_test, y_test), batch_size=32, shuffle=False)

        self.model = SimpleModel(feature_count).to(self.device)
        pos_weight = torch.tensor([3.0]).to(self.device) 
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def get_parameters(self, config):
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        
        for _ in range(config.get("local_epochs", 2)):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()

        return get_model_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        self.model.eval()
        losses, probs_all = [], []

        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                losses.append(self.criterion(logits, yb).item())
                probs_all.extend(torch.sigmoid(logits).cpu().numpy().flatten().tolist())

        os.makedirs("predictions/clients", exist_ok=True)
        y_prob = np.array(probs_all)

        np.save(f"predictions/clients/client_{self.cid}_y_true.npy", self.y_test)
        np.save(f"predictions/clients/client_{self.cid}_y_prob.npy", y_prob)

        print(f"[Client {self.cid}] Predictions saved ✅")
        return float(sum(losses) / len(losses)), len(self.test_loader.dataset), compute_metrics(self.y_test, y_prob)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    set_seed(42)
    client = FlowerClient(cid="1", dataset_path="datasets\\random_split\\dataset_random_split1.csv")
    fl.client.start_numpy_client(server_address=args.server, client=client)

if __name__ == "__main__":
    main()