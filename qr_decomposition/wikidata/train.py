import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from qr_decomposition.wikidata.dataset import ClassificationDataset
from qr_decomposition.wikidata.model import WikidataClassifier
from qr_decomposition.utils import resolve_path


def train_wikidata_classification_model(
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

    model = WikidataClassifier(input_dim, hidden_dim, output_dim, dropout)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
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

    model_path = resolve_path([target_dir, "wikidata_checkpoint.pt"])
    print(f"Saving checkpoint to {model_path}")
    torch.save(checkpoint, model_path)  # type: ignore


def step(
    key: str,
    model: torch.nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    grad_enabled: bool = True,
):
    threshold = 0.8
    losses: list[float] = []
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"{key.capitalize()}: {epoch + 1}/{num_epochs}")
    for embeddings, labels in pbar:
        embeddings, labels = embeddings.to(device), labels.to(device)

        torch.set_grad_enabled(grad_enabled)
        outputs = model(embeddings)
        loss = criterion(outputs, labels)

        if grad_enabled:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calculate accuracy
        predictions = (outputs > threshold).float()  # Convert logits to binary predictions
        correct += (predictions == labels).sum().item()
        total += labels.size(0) * labels.size(1)  # Batch size * num_classes

        losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        accuracy = (correct / total) * 100 if total > 0 else 0
        pbar.set_postfix_str(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
