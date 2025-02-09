import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.2):
        """
        :param input_dim: количество признаков (фичей) во входных данных
        :param hidden_dims: список с размерностями скрытых слоев
        :param dropout: коэффициент Dropout для регуляризации
        """
        super(MLPModel, self).__init__()

        layers = []
        prev_dim = input_dim

        # Добавляем скрытые слои с активацией ReLU и Dropout
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))  # Регуляризация
            prev_dim = hidden_dim

        # Выходной слой (2 выхода, так как бинарная классификация)
        layers.append(nn.Linear(prev_dim, 2))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
