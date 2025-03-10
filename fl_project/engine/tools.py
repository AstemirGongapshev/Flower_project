import os
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from imblearn.combine import SMOTEENN
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, f1_score
from imblearn.over_sampling import SMOTE
from tqdm import tqdm


def get_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    try:
        with open(file_path) as file:
            data = pd.read_csv(file)

    except pd.errors.ParserError:
        raise ValueError("The file could not be parsed.")

    return data


def get_model_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [param.data.cpu().numpy() for param in model.parameters()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=False)


# def set_initial_parameters(model: torch.nn.Module) -> None:
#     for param in model.parameters():
#         if param.dim() > 1:
#             nn.init.xavier_uniform_(param)
#         else:
#             nn.init.zeros_(param)
def set_initial_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.normal_(param, mean=0.0, std=1.0)
        else:
            torch.nn.init.zeros_(param)



def prepare_data(
    df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 321,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, int]:

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    X_test = X_test.drop(columns=["Unnamed: 0"], errors="ignore")

    try:
        X = df.drop(columns="Fraud")
        y = df["Fraud"]

        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ]
        )

        X_train_poly = pipeline.fit_transform(X)
        X_test_poly = pipeline.transform(X_test)

        smoteenn = SMOTEENN(random_state=random_state)
        X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train_poly, y)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_train_tensor = torch.tensor(
            X_train_resampled, dtype=torch.float32, device=device
        )
        y_train_tensor = torch.tensor(
            y_train_resampled.to_numpy(), dtype=torch.int64, device=device
        )

        X_test_tensor = torch.tensor(X_test_poly, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(
            y_test.to_numpy(), dtype=torch.int64, device=device
        )

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        input_dim = X_train_tensor.shape[1]

        return train_loader, test_loader, input_dim

    except Exception as e:
        logging.error(f"Ошибка при обработке данных: {e}")
        raise


def train(
    model: torch.nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    device: str,
    lr: float,
    is_proximal=False,
    proximal_mu=0.0,
    global_params: Optional[List[np.ndarray]] = None,
) -> None:
    try:
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for inputs, labels in tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} - Training",
                leave=False,
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if is_proximal:
                print(f"ITS FEDPROX with proxy:{proximal_mu}")
                proximal_term = 0.0
                # # print(F"global_params:{global_params}")
                # # print("="*100)
                # # print(f"local_params:{list(model.parameters())}")

                for local_param, global_param in zip(model.parameters(), global_params):
                    global_tensor = torch.tensor(global_param, device=device)
                    proximal_term += torch.norm(local_param - global_tensor, p=2) ** 2
                loss += (proximal_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


def eval(
    model: torch.nn.Module, test_loader: DataLoader, device: str
) -> Dict[str, float]:
    model.to(device)
    model.eval()
    test_labels = []
    test_predictions = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(probabilities.cpu().numpy())

    try:
        test_logloss = log_loss(test_labels, test_predictions)
        test_roc_auc = roc_auc_score(test_labels, test_predictions)
        test_pred_binary = np.array(test_predictions) > 0.5
        test_accuracy = accuracy_score(test_labels, test_pred_binary)
        test_f1 = f1_score(test_labels, test_pred_binary)

        # logging.info(
        #     f"Test Metrics | Log-loss: {test_logloss:.4f} | "
        #     f"ROC AUC: {test_roc_auc:.4f} | Accuracy: {test_accuracy:.4f} | F1 Score: {test_f1:.4f}"
        # )

        return {
            "logloss_test": test_logloss,
            "roc_auc_test": test_roc_auc,
            "accuracy_test": test_accuracy,
            "f1_test": test_f1,
        }

    except ValueError as e:
        logging.error(f"Error in metric computation: {e}")
        return {}
