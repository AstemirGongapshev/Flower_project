import flwr as fl
from flwr.common import Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from typing import List, Tuple, Dict, Optional, Union


roc_auc_history = []
MAX_ROUNDS_WITHOUT_IMPROVEMENT = 5


def aggregate_metrics(
    metrics_list: List[Tuple[int, Dict[str, Scalar]]]
) -> Dict[str, Scalar]:

    if not metrics_list:
        return {}

    aggregated_metrics = {}
    total_samples = sum(num_examples for num_examples, _ in metrics_list)

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
            f"📊 [Round {server_round}] Aggregated training metrics: {metrics_aggregated}"
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
            f"📊 [Round {server_round}] Aggregated evaluation metrics: {metrics_aggregated}"
        )
        if "roc_auc_test" in metrics_aggregated:
            roc_auc_history.append(metrics_aggregated["roc_auc_test"])

            if len(roc_auc_history) > MAX_ROUNDS_WITHOUT_IMPROVEMENT:
                last_five = roc_auc_history[-MAX_ROUNDS_WITHOUT_IMPROVEMENT:]
                if all(x >= y for x, y in zip(last_five, last_five[1:])):
                    print(
                        f"⚠️ Early stopping warning: ROC AUC has been decreasing for {MAX_ROUNDS_WITHOUT_IMPROVEMENT} rounds!"
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
    fl.server.start_server(
        server_address="0.0.0.0:8080", strategy=strategy, config=config
    ),


if __name__ == "__main__":
    server_fn(num_rounds=50)
