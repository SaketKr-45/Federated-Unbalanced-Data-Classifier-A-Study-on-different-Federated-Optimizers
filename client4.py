from client1 import FlowerClient, set_seed
import flwr as fl
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    set_seed(42)
    client = FlowerClient(cid="4", dataset_path="datasets\\random_split\\dataset_random_split4.csv")
    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()