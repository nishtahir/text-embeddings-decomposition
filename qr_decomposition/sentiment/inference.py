from typing import Literal

import torch
import numpy as np
import numpy.typing as npt
from qr_decomposition.device import select_device
from qr_decomposition.embedding import embed_text
from qr_decomposition.sentiment.model import SentimentClassifier
from qr_decomposition.utils import resolve_path


def load_sentiment_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)  # type: ignore

    # Recreate model with saved configuration
    model = SentimentClassifier(
        input_dim=checkpoint["model_config"]["input_dim"],
        hidden_dim=checkpoint["model_config"]["hidden_dim"],
        output_dim=checkpoint["model_config"]["output_dim"],
        dropout=checkpoint["model_config"]["dropout"],
    )

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Recreate optimizer if needed
    optimizer = torch.optim.Adam(
        model.parameters(), lr=checkpoint["training_config"]["learning_rate"]
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, checkpoint


def predict_embedding(
    embedding: npt.NDArray[np.float32],
    model_path: str = "target/sentiment_model/sentiment_checkpoint.pt",
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
) -> list[dict[str, float]]:
    resolved_device = select_device(device)
    model_path = resolve_path(model_path)
    model, _, _ = load_sentiment_model(model_path, resolved_device)
    model.eval()
    with torch.no_grad():
        input = torch.tensor(embedding, dtype=torch.float32, device=resolved_device)
        out = model(input)
        probs = torch.softmax(out, dim=1)
        probs = probs.cpu().numpy()  # type: ignore

    return [{"negative": float(prob[0]), "positive": float(prob[1])} for prob in probs]


def predict(
    text: list[str] | str,
    model_path: str = "target/sentiment_model/sentiment_checkpoint.pt",
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
) -> list[dict[str, float]]:
    if isinstance(text, str):
        text = [text]

    embeddings = embed_text(text)
    return predict_embedding(embeddings, model_path, device)
