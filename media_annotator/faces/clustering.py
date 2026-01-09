from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np

from media_annotator.faces.embedding import cosine_similarity, l2_normalize


@dataclass
class MatchResult:
    person_id: Optional[int]
    similarity: float


def _faiss_available() -> bool:
    return importlib.util.find_spec("faiss") is not None


def build_index(embeddings: np.ndarray):
    if not _faiss_available():
        return None
    import faiss  # noqa: PLC0415

    embeddings = l2_normalize(embeddings.astype("float32"))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def search_embeddings(
    query: np.ndarray,
    embeddings: np.ndarray,
    person_ids: Iterable[int],
    threshold: float,
    use_faiss: bool,
) -> MatchResult:
    if embeddings.size == 0:
        return MatchResult(person_id=None, similarity=0.0)
    if use_faiss and _faiss_available():
        import faiss  # noqa: PLC0415

        index = build_index(embeddings)
        query_norm = l2_normalize(query[np.newaxis, :].astype("float32"))
        scores, idx = index.search(query_norm, 1)
        similarity = float(scores[0][0])
        if similarity >= threshold:
            person_id = list(person_ids)[int(idx[0][0])]
            return MatchResult(person_id=person_id, similarity=similarity)
        return MatchResult(person_id=None, similarity=similarity)
    scores = cosine_similarity(query, embeddings).flatten()
    best_idx = int(np.argmax(scores))
    similarity = float(scores[best_idx])
    if similarity >= threshold:
        person_id = list(person_ids)[best_idx]
        return MatchResult(person_id=person_id, similarity=similarity)
    return MatchResult(person_id=None, similarity=similarity)
