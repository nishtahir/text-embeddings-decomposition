import numpy as np
import numpy.typing as npt


def cosine_distance(emb1: npt.NDArray[np.float32], emb2: npt.NDArray[np.float32]) -> float:
    return 1 - np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def euclidean_distance(emb1: npt.NDArray[np.float32], emb2: npt.NDArray[np.float32]) -> float:
    return float(np.linalg.norm(emb1 - emb2))
