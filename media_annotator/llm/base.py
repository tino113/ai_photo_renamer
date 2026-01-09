from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LLMResult:
    summary: str
    description: str
    tags: List[str]
    suggested_filename_base: str
    key_people: List[str]
    key_objects: List[str]
    key_actions: List[str]
    confidence: Optional[float] = None


class LLMBackend(ABC):
    @abstractmethod
    def describe(
        self,
        images: List[str],
        people: List[dict],
        location_text: str,
        capture_datetime: Optional[str],
        media_type: str,
        metadata: dict,
    ) -> LLMResult:
        raise NotImplementedError
