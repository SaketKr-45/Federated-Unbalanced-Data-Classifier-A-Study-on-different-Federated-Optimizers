import argparse
import os
from typing import List, Tuple

import flwr as fl
import pandas as pd
from flwr.common import Metrics, ndarrays_to_parameters

from client1 import SimpleModel
from utils import get_model_parameters, load_and_preprocess

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
    for n, m in metrics:
        for k, v in m.items():
            agg[k] = agg.get(k, 0) + n * float(v)

    for k in agg:
        agg[k] /= num_examples

    return agg


def get_strategy(name: str):
    StrategyClass = ALGORITHMS[name]

    _, _, _, _, feature_count, _ = load_and_preprocess(
        "datasets\\random_split\\dataset_random_split1.csv"
    )

    model = SimpleModel(feature_count)
    params = ndarrays_to_parameters(get_model_parameters(model))

    common_kwargs = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=params,
    )

    if name == "fedprox":
        return StrategyClass(proximal_mu=0.01, **common_kwargs)
    elif name in ("fedadam", "fedadagrad", "fedyogi"):
        return StrategyClass(
            eta=1e-2, eta_l=1e-2, beta_1=0.9, beta_2=0.99, tau=1e-9,
            **common_kwargs
        )
    elif name == "fedavgm":
        return StrategyClass(
            server_learning_rate=1.0,
            server_momentum=0.9,
            **common_kwargs
        )
    else:
        return StrategyClass(**common_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="fedavg",
                        choices=list(ALGORITHMS.keys()))
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--address", type=str, default="0.0.0.0:8080")
    args = parser.parse_args()

    print(f"[Server] Starting {args.algorithm.upper()}...")

    strategy = get_strategy(args.algorithm)

    history = fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    data = {}

    for metric_name, values in history.metrics_distributed.items():
        for rnd, val in values:
            if rnd not in data:
                data[rnd] = {}
            data[rnd][metric_name] = val

    df = pd.DataFrame.from_dict(data, orient="index").reset_index()
    df = df.rename(columns={"index": "round"})
    df = df.sort_values("round")

    print("\nDEBUG DATA:")
    print(df.head())

    os.makedirs("results", exist_ok=True)

    df.to_csv(f"results/{args.algorithm}_metrics.csv", index=False)

    prf_cols = ["round", "precision", "recall", "f1"]
    prf_df = df.loc[:, df.columns.intersection(prf_cols)]
    prf_df.to_csv(f"results/{args.algorithm}_prf_metrics.csv", index=False)

    print("\nSaved:")
    print(f"- results/{args.algorithm}_metrics.csv")
    print(f"- results/{args.algorithm}_prf_metrics.csv")

    print("\nLast rounds:")
    print(df.tail())


if __name__ == "__main__":
    main()