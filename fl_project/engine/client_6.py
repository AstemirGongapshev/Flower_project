import flwr as fl
import os
import torch
from flwr.client import NumPyClient
from engine.model import LogisticRegressionModel
from engine.tools import (
    get_data,
    get_model_parameters,
    set_initial_parameters,
    set_model_parameters,
    train,
    test,
    prepare_data,
)


class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, model, device):
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.device = device
        self.local_metrics = list()

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        train(self.model, self.trainloader, lr=0.001, num_epochs=1, device=self.device)
        return get_model_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        metrics = test(self.model, self.valloader, device=self.device)
        return metrics["logloss_test"], len(self.valloader.dataset), metrics


def client_fn(file_path_train, file_path_test):
    data_noniid = get_data(file_path_train)
    data_test = get_data(file_path_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, input_dim = prepare_data(
        data_noniid, data_test.drop(columns="Fraud"), data_test["Fraud"], batch_size=64
    )

    model = LogisticRegressionModel(input_dim=input_dim)
    print(f"!*10{type(model)}")

    set_initial_parameters(model)

    return fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(train_loader, test_loader, model, device).to_client(),
    )


if __name__ == "__main__":
    client_fn(
        os.path.join("engine", "data", "df6.csv"),
        os.path.join("engine", "data", "TEST_SAMPLE.csv"),
    )
