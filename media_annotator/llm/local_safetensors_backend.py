from __future__ import annotations

import json
from typing import List, Optional

from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from media_annotator.llm.base import LLMBackend, LLMResult
from media_annotator.llm.prompting import build_prompt, validate_output


class LocalSafetensorsBackend(LLMBackend):
    def __init__(self, model: str) -> None:
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = AutoModelForVision2Seq.from_pretrained(model)

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
        pil_images = [Image.open(path).convert("RGB") for path in images]
        inputs = self.processor(images=pil_images, text=prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=800)
        content = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        last_error = None
        for _ in range(3):
            try:
                payload = json.loads(content)
                validate_output(payload)
                return LLMResult(**payload)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                content = content.replace("`", "")
        raise RuntimeError(f"LLM failed to return valid JSON: {last_error}")
