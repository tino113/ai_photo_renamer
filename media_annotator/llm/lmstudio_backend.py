from __future__ import annotations

import base64
import json
from typing import List, Optional

from openai import OpenAI

from media_annotator.llm.base import LLMBackend, LLMResult
from media_annotator.llm.prompting import build_prompt, validate_output


class LMStudioBackend(LLMBackend):
    def __init__(self, model: str, base_url: str, timeout_s: int = 120) -> None:
        self.client = OpenAI(base_url=base_url, api_key="lmstudio")
        self.model = model
        self.timeout_s = timeout_s

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
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            for path in images:
                with open(path, "rb") as handle:
                    content = base64.b64encode(handle.read()).decode("utf-8")
                messages[0]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{content}"}}
                )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                timeout=self.timeout_s,
            )
            content = response.choices[0].message.content or ""
            try:
                payload = json.loads(content)
                validate_output(payload)
                return LLMResult(**payload)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                prompt = f"Repair JSON only. Error: {exc}. Original content: {content}"
        raise RuntimeError(f"LLM failed to return valid JSON: {last_error}")
