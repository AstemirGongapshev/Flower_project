import flwr as fl


def server_fn():
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10),
    )


if __name__ == "__main__":
    server_fn()
