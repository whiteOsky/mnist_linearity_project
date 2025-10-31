import torch
from torch import nn

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def select_loss(loss_name: str) -> nn.Module:
    loss_name = loss_name.lower()
    if loss_name in ["mse", "mse_loss"]:
        return nn.MSELoss()
    elif loss_name in ["ce", "crossentropy", "cross_entropy"]:
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def to_onehot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
