from typing import Literal

import numpy as np
import numpy.typing as npt
import torch

from qr_decomposition.device import select_device
from qr_decomposition.embedding import embed_text
from qr_decomposition.wikidata.model import WikidataClassifier
from qr_decomposition.utils import resolve_path


def load_wikidata_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)  # type: ignore

    # Recreate model with saved configuration
    model = WikidataClassifier(
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
    model_path: str = "target/wikidata_model/wikidata_checkpoint.pt",
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
) -> list[dict[str, float]]:
    resolved_device = select_device(device)
    model_path = resolve_path(model_path)
    model, _, _ = load_wikidata_model(model_path, resolved_device)
    model.eval()
    with torch.no_grad():
        input = torch.tensor(embedding, dtype=torch.float32, device=resolved_device)
        out = model(input)
        probs = torch.sigmoid(out)
        probs = probs.cpu().numpy()  # type: ignore

    return [
        {
            "toxic": float(prob[0]),
            "severe_toxic": float(prob[1]),
            "obscene": float(prob[2]),
            "threat": float(prob[3]),
            "insult": float(prob[4]),
            "identity_hate": float(prob[5]),
        }
        for prob in probs
    ]


def predict(
    text: list[str] | str,
    model_path: str = "target/wikidata_model/wikidata_checkpoint.pt",
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
) -> list[dict[str, float]]:
    if isinstance(text, str):
        text = [text]

    embeddings = embed_text(text)
    return predict_embedding(embeddings, model_path, device)
