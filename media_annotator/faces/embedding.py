from __future__ import annotations

import numpy as np


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vecs / norms


def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    query_norm = l2_normalize(query[np.newaxis, :])
    vectors_norm = l2_normalize(vectors)
    return vectors_norm @ query_norm.T
