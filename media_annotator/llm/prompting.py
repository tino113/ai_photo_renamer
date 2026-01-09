from __future__ import annotations

import json
from typing import Any

PROMPT_TEMPLATE = """You are a media description assistant. You MUST output valid JSON only.

Context:
- media_type: {media_type}
- capture_datetime: {capture_datetime}
- location_text: {location_text}
- people: {people}
- metadata: {metadata}
{video_note}

Required JSON schema:
{{
  "summary": "<one sentence>",
  "description": "<2-6 paragraphs>",
  "tags": ["tag1", "tag2"],
  "suggested_filename_base": "<safe base name no extension>",
  "key_people": ["Name"],
  "key_objects": ["object"],
  "key_actions": ["action"]
}}

Rules:
- Use provided people names when known; unknowns must be labeled like unknown_001.
- Use approximate location text if present.
- Do not include markdown or extra text. Return JSON only.
"""


def build_prompt(
    media_type: str,
    capture_datetime: str | None,
    location_text: str,
    people: list[dict],
    metadata: dict,
    video: bool,
) -> str:
    video_note = "These are representative frames from one video." if video else ""
    return PROMPT_TEMPLATE.format(
        media_type=media_type,
        capture_datetime=capture_datetime or "unknown",
        location_text=location_text,
        people=json.dumps(people, ensure_ascii=False),
        metadata=json.dumps(metadata, ensure_ascii=False),
        video_note=video_note,
    )


def validate_output(payload: Any) -> None:
    required = [
        "summary",
        "description",
        "tags",
        "suggested_filename_base",
        "key_people",
        "key_objects",
        "key_actions",
    ]
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing key: {key}")
    if not isinstance(payload["tags"], list):
        raise ValueError("tags must be list")
