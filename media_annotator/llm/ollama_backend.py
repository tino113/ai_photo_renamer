from __future__ import annotations

import base64
import json
from typing import List, Optional

import httpx

from media_annotator.llm.base import LLMBackend, LLMResult
from media_annotator.llm.prompting import build_prompt, validate_output


class OllamaBackend(LLMBackend):
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        timeout_s: int = 120,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.base_url = base_url or "http://localhost:11434"
        self.timeout_s = timeout_s
        self.temperature = temperature

    def _encode_images(self, images: List[str]) -> List[str]:
        encoded = []
        for path in images:
            with open(path, "rb") as handle:
                encoded.append(base64.b64encode(handle.read()).decode("utf-8"))
        return encoded

    def _request(self, prompt: str, images: List[str]) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt, "images": self._encode_images(images)},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        with httpx.Client(timeout=self.timeout_s) as client:
            response = client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            return response.json()

    def describe(
        self,
        images: List[str],
        people: List[dict],
        location_text: str,
        capture_datetime: Optional[str],
        media_type: str,
        metadata: dict,
    ) -> LLMResult:
        prompt = build_prompt(media_type, capture_datetime, location_text, people, metadata, media_type == "video")
        last_error = None
        for _ in range(3):
            response = self._request(prompt, images)
            content = response.get("message", {}).get("content", "")
            try:
                payload = json.loads(content)
                validate_output(payload)
                return LLMResult(**payload)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                prompt = f"Repair JSON only. Error: {exc}. Original content: {content}"
        raise RuntimeError(f"LLM failed to return valid JSON: {last_error}")
