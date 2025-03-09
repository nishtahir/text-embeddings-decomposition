from typing import Literal

import torch


def select_device(
    device: Literal["cpu", "cuda", "mps", "auto"] = "auto",
) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    else:
        return torch.device(device)
