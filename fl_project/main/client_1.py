from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from model import LogisticRegressionModel
from tools import (
    get_data,
    get_model_parameters,
    set_initial_parameters,
    set_model_parameters,
    train,
    test,
    prepare_data,
)


class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, model):
        self.trainloader = trainloader
        self.valloader = valloader
        self.model = model
        self.local_metrics = list()

    def fit(self, parameters, config):
        set_model_parameters(self.model, parameters)
        train(self.model, self.trainloader, lr=0.001, num_epochs=1)
        return get_model_parameters(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_model_parameters(self.model, parameters)
        metrics = test(self.model, self.valloader)
        # self.local_metrics.append(metrics)
        return metrics["logloss_test"], len(self.valloader.dataset), metrics


def client_fn():
    return FlowerClient().to_client()


app = ClientApp(client_fn)
