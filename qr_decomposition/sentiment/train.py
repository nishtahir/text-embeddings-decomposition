import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from qr_decomposition.sentiment.dataset import ClassificationDataset
from qr_decomposition.sentiment.model import SentimentClassifier
from qr_decomposition.utils import resolve_path


def train_sentiment_model(
    batch_size: int,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    learning_rate: float,
    num_epochs: int,
    device: torch.device,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_dir: str,
):
    train_dataset = ClassificationDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ClassificationDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SentimentClassifier(input_dim, hidden_dim, output_dim, dropout)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        step(
            key="train",
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        step(
            key="validate",
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            grad_enabled=False,
        )

    checkpoint = {
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "dropout": dropout,
        },
        "training_config": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
    }

    model_path = resolve_path([target_dir, "sentiment_checkpoint.pt"])
    print(f"Saving checkpoint to {model_path}")
    torch.save(checkpoint, model_path)


def step(
    key: str,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    grad_enabled: bool = True,
):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, desc=f"{key.capitalize()}, Epoch: {epoch + 1}/{num_epochs}")
    for embeddings, labels in pbar:
        embeddings, labels = embeddings.to(device), labels.to(device)

        torch.set_grad_enabled(grad_enabled)
        outputs = model(embeddings)
        loss = criterion(outputs, labels)

        if grad_enabled:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        prediction = torch.argmax(outputs, dim=1)

        total_loss += loss.item()
        total_correct += (prediction == labels).sum().item()
        total_samples += labels.size(0)

        pbar.set_postfix_str(
            f"Loss: {total_loss / total_samples:.4f}, Accuracy: {total_correct / total_samples * 100:.2f}%"
        )
