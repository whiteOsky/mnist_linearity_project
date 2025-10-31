import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_mnist_dataloaders(batch_size: int = 128, val_split: float = 0.1, num_workers: int = 2):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_full = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    val_size = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    train, val = random_split(train_full, [train_size, val_size])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
