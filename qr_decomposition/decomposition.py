import numpy as np
import numpy.typing as npt


def qr_decomposition(
    mat: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    Q, R = np.linalg.qr(mat.T)
    return Q, R


def projection_matrix(mat: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    P = mat @ mat.T
    return P
