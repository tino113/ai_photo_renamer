from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from insightface.app import FaceAnalysis


@dataclass
class DetectedFace:
    embedding: np.ndarray
    bbox: list[float]
    quality: float


class InsightFaceBackend:
    def __init__(self) -> None:
        self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        faces = self.app.get(image)
        results: List[DetectedFace] = []
        for face in faces:
            embedding = face.embedding.astype("float32")
            bbox = face.bbox.tolist() if hasattr(face, "bbox") else []
            quality = float(getattr(face, "det_score", 0.0))
            results.append(DetectedFace(embedding=embedding, bbox=bbox, quality=quality))
        return results


def load_image(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image
