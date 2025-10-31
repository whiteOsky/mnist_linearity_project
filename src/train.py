import argparse
import time
import torch
import torch.optim as optim
from tqdm import tqdm

from .data import get_mnist_dataloaders
from .models import LinearClassifier, MLP
from .utils import get_device, select_loss, accuracy_from_logits, to_onehot

def train_one_epoch(model, dataloader, optimizer, criterion, device, loss_name: str, num_classes: int = 10):
    model.train()
    total_loss, total_acc, total_count = 0.0, 0.0, 0
    for inputs, targets in tqdm(dataloader, leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        if loss_name == "mse":
            targets_oh = to_onehot(targets, num_classes).to(device)
            loss = criterion(torch.softmax(logits, dim=1), targets_oh)
        else:
            loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy_from_logits(logits, targets) * batch_size
        total_count += batch_size
    return total_loss / total_count, total_acc / total_count

@torch.no_grad()
def evaluate(model, dataloader, criterion, device, loss_name: str, num_classes: int = 10):
    model.eval()
    total_loss, total_acc, total_count = 0.0, 0.0, 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        if loss_name == "mse":
            targets_oh = to_onehot(targets, num_classes).to(device)
            loss = criterion(torch.softmax(logits, dim=1), targets_oh)
        else:
            loss = criterion(logits, targets)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy_from_logits(logits, targets) * batch_size
        total_count += batch_size
    return total_loss / total_count, total_acc / total_count

def main():
    parser = argparse.ArgumentParser(description="MNIST Linearity → Nonlinearity Study")
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--loss", type=str, default="crossentropy", help="mse or crossentropy")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batchnorm", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = get_device()
    train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size=args.batch_size)

    if args.model == "linear":
        model = LinearClassifier()
    else:
        model = MLP(hidden=args.hidden, use_batchnorm=args.batchnorm, dropout_p=args.dropout)

    model.to(device)

    loss_name = "mse" if args.loss.lower().startswith("mse") else "crossentropy"
    criterion = select_loss(loss_name)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Device: {device}")
    print(f"Model: {model}")
    print(f"Loss: {loss_name}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, LR: {args.lr}")

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, loss_name)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, loss_name)
        dt = time.time() - t0

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% | "
              f"{dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"mnist_{args.model}_{loss_name}_best.pt")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, loss_name)
    print(f"[Test] loss={test_loss:.4f} acc={test_acc*100:.2f}%")

if __name__ == "__main__":
    main()
