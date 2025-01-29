import pytest
import pandas as pd
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from torch.utils.data import DataLoader, TensorDataset
from engine.tools import (
    get_data,
    get_model_parameters,
    set_initial_parameters,
    prepare_data,
    train,
)


def test_get_data_success(tmp_path):
    file_path = tmp_path / "test.csv"
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    df.to_csv(file_path, index=False)

    loaded_df = get_data(str(file_path))
    pd.testing.assert_frame_equal(df, loaded_df)


def test_get_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        get_data("non_existent_file.csv")


def test_get_model_parameters():
    model = torch.nn.Linear(2, 1)
    params = get_model_parameters(model)

    assert isinstance(params, list)
    assert all(isinstance(p, np.ndarray) for p in params)


def test_prepare_data():
    df = pd.DataFrame(
        {
            "Feature1": [1, 2, 3, 4, 5, 6],
            "Feature2": [4, 5, 6, 7, 8, 9],
            "Fraud": [0, 1, 0, 1, 0, 1],
        }
    )
    X_test = df.drop(columns="Fraud")
    y_test = df["Fraud"]

    train_loader, test_loader, input_dim = prepare_data(df, X_test, y_test)

    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert isinstance(input_dim, int) and input_dim > 0


@patch("torch.optim.SGD.step")
@patch("torch.optim.SGD.zero_grad")
@patch("torch.nn.CrossEntropyLoss")
def test_train(mock_loss, mock_zero_grad, mock_step):
    model = torch.nn.Linear(2, 2)
    train_loader = DataLoader(
        TensorDataset(torch.randn(10, 2), torch.randint(0, 2, (10,)))
    )
    train(model, train_loader, lr=0.01, num_epochs=1, device="cpu")

    mock_loss.assert_called()
    mock_zero_grad.assert_called()
    mock_step.assert_called()


if __name__ == "__main__":
    pytest.main(["-v"])
