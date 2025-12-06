import time
import torch
import torch.nn as nn
from models import MLP

def train_one_activation(
    activation_name,
    train_loader,
    test_loader,
    input_dim,
    output_dim,
    hidden_dims=[256, 128],
    epochs=5,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(input_dim, hidden_dims, output_dim, activation_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {
        "activation": activation_name,
        "train_loss": [],
        "val_loss": [],
        "train_time_total": 0.0,
        "test_accuracy": None,
        "sparsity_first_hidden": None,
    }

    for epoch in range(epochs):
        model.train()
        start = time.time()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        epoch_time = time.time() - start
        history["train_time_total"] += epoch_time
        train_loss = running_loss / len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_loss /= len(test_loader.dataset)
        val_acc = correct / total

        history["val_loss"].append(val_loss)

        print(f"[{activation_name}] Epoch {epoch+1}/{epochs} "
              f"| Train {train_loss:.4f} | Val {val_loss:.4f} "
              f"| ValAcc {val_acc:.4f} | Time {epoch_time:.2f}s")

    history["test_accuracy"] = val_acc

    # sparsity measurement
    with torch.no_grad():
        x, _ = next(iter(train_loader))
        x = x.to(device)
        h = model.net[1](model.net[0](x))
        sparsity = (h.abs() < 1e-3).float().mean().item()
        history["sparsity_first_hidden"] = sparsity

    return history