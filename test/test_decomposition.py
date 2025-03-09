import numpy as np

from qr_decomposition.decomposition import qr_decomposition


def test_qr_decomposition():
    # using https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf
    # as a reference

    matrix = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]])
    q, r = qr_decomposition(matrix)

    expected_q = np.array(
        [
            [-0.70710678, 0.40824829, -0.57735027],
            [-0.70710678, -0.40824829, 0.57735027],
            [-0.0, 0.81649658, 0.57735027],
        ]
    )

    expected_r = np.array(
        [
            [-1.41421356, -0.70710678, -0.70710678],
            [0.0, 1.22474487, 0.40824829],
            [0.0, 0.0, 1.15470054],
        ]
    )

    assert np.allclose(q, expected_q)
    assert np.allclose(r, expected_r)

    assert np.allclose(q @ r, matrix)

    # # check that q is orthonormal
    assert np.allclose(q @ q.T, np.eye(3))

    # # check that r is upper triangular
    assert np.allclose(r, np.triu(r))
