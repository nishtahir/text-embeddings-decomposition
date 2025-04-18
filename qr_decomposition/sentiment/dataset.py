import pandas as pd
import torch
from torch.utils.data import Dataset


class ClassificationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, items: pd.DataFrame):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.items.iloc[idx]  # type: ignore
        embedding = torch.tensor(row["embeddings"], dtype=torch.float32)
        label = torch.tensor(row["labels"], dtype=torch.long)
        return embedding, label
