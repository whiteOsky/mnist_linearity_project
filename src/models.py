import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, input_dim: int = 784, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim, num_classes, bias=True)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.fc(x)
        return logits

class MLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden: int = 256, num_classes: int = 10, 
                 use_batchnorm: bool = False, dropout_p: float = 0.0):
        super().__init__()
        layers = [nn.Flatten(), nn.Linear(input_dim, hidden)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(hidden, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
