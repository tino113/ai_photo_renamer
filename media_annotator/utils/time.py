from __future__ import annotations

from datetime import datetime
from typing import Optional

from dateutil import parser


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return parser.parse(value)
    except (ValueError, TypeError):
        return None
