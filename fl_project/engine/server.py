import flwr as fl
import os
import json
import wandb
from flwr.common import Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional, Union
from flwr.server import History
from datetime import datetime

roc_auc_history = []
MAX_ROUNDS_WITHOUT_IMPROVEMENT = 5


wandb.init(
    project="federated-learning_tests_2",
    name="FedAvg (Heterogeneous) with Adam learning_rate(0.001)",
    config={"num_rounds": 50},
)


def save_metrics(history, is_prox=False):
    path = "prox_metrics.json_sgd" if is_prox else "fed_avg_metrics_sgd.json"

    try:
        new_experiment = {
            "timestamp": datetime.now().isoformat(),
            "Loss": history.metrics_distributed.get("logloss_test"),
            "ROC_AUC": history.metrics_distributed.get("roc_auc_test"),
            "Accuracy": history.metrics_distributed.get("accuracy_test"),
            "F1_Score": history.metrics_distributed.get("f1_test"),
        }

        if os.path.exists(path):
            with open(path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        data.append(new_experiment)

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

    except Exception as e:
        print(f"Error saving metrics: {e}")


def aggregate_metrics(
    metrics_list: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:
    if not metrics_list:
        return {}

    aggregated_metrics = {}
    total_samples = sum(num_examples for num_examples, _ in metrics_list)
    print(f"ITS_METRICS_LIST:{metrics_list}")

    for _, metrics in metrics_list:
        for key, value in metrics.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0.0
            aggregated_metrics[key] += value * (_ / total_samples)

    return aggregated_metrics


class FedAvgCustom(FedAvg):

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        print(
            f"[Round {server_round}] Aggregated training metrics: {metrics_aggregated}"
        )
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        global roc_auc_history
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )
        print(
            f"[Round {server_round}] Aggregated evaluation metrics: {metrics_aggregated}"
        )

        
        wandb.log(
            {
                "server_round": server_round,
                "Loss": loss_aggregated,
                "ROC_AUC": metrics_aggregated.get("roc_auc_test", None), #TODO TEST\...
                "Accuracy": metrics_aggregated.get("accuracy_test", None),
                "F1_Score": metrics_aggregated.get("f1_test", None),
            }
        )

        
        if "roc_auc_test" in metrics_aggregated:
            roc_auc_history.append(metrics_aggregated["roc_auc_test"])

            if len(roc_auc_history) > MAX_ROUNDS_WITHOUT_IMPROVEMENT:
                last_five = roc_auc_history[-MAX_ROUNDS_WITHOUT_IMPROVEMENT:]
                if all(x >= y for x, y in zip(last_five, last_five[1:])):
                    print(
                        f"Early stopping warning: ROC AUC has been decreasing for {MAX_ROUNDS_WITHOUT_IMPROVEMENT} rounds!"
                    )

        return loss_aggregated, metrics_aggregated


def server_fn(num_rounds: int) -> None:

    config = fl.server.ServerConfig(num_rounds=num_rounds)
    strategy = FedAvgCustom(
        min_fit_clients=6,
        min_evaluate_clients=6,
        min_available_clients=6,
        fit_metrics_aggregation_fn=aggregate_metrics,
        evaluate_metrics_aggregation_fn=aggregate_metrics,
    )
    history = fl.server.start_server(
        server_address="0.0.0.0:8080", strategy=strategy, config=config
    )

    
    save_metrics(history=history)
    wandb.finish()


if __name__ == "__main__":
    server_fn(num_rounds=50)
