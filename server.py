import argparse
import os
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import FitRes, Metrics, Parameters, EvaluateRes
from flwr.server.client_proxy import ClientProxy

from evaluation import save_round_metrics_json


ALGORITHMS = {
    "fedavg": fl.server.strategy.FedAvg,
    "fedavgm": fl.server.strategy.FedAvgM,
    "fedprox": fl.server.strategy.FedProx,
    "fedadam": fl.server.strategy.FedAdam,
    "fedadagrad": fl.server.strategy.FedAdagrad,
    "fedyogi": fl.server.strategy.FedYogi,
}


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    num_examples = sum(n for n, _ in metrics)
    if num_examples == 0:
        return {}

    agg = {}
    keys = metrics[0][1].keys()
    for k in keys:
        agg[k] = sum(n * float(m[k]) for n, m in metrics) / num_examples
    return agg


def get_strategy(name: str, rounds: int) -> fl.server.strategy.Strategy:
    StrategyClass = ALGORITHMS[name]

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
    )

    if name == "fedprox":
        strategy = StrategyClass(proximal_mu=0.01, **common_kwargs)
    elif name in ("fedadam", "fedadagrad", "fedyogi"):
        strategy = StrategyClass(
            eta=1e-2,
            eta_l=1e-2,
            beta_1=0.9,
            beta_2=0.99,
            tau=1e-9,
            **common_kwargs,
        )
    elif name == "fedavgm":
        strategy = StrategyClass(server_learning_rate=1.0, server_momentum=0.9, **common_kwargs)
    else:
        strategy = StrategyClass(**common_kwargs)

    return strategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="fedavg", choices=list(ALGORITHMS.keys()))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    args = parser.parse_args()

    print(f"[Server] Starting with algorithm={args.algorithm}, rounds={args.rounds}, address={args.address}")

    strategy = get_strategy(args.algorithm, args.rounds)

    history = fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    round_metrics = []
    for rnd, vals in history.metrics_distributed.items():
        for k, v in vals:
            round_metrics.append({"round": rnd, "metric": k, "value": v})
    os.makedirs("results", exist_ok=True)
    save_round_metrics_json(args.algorithm, round_metrics, out_dir="results")
    print("[Server] Training completed.")


if __name__ == "__main__":
    main()