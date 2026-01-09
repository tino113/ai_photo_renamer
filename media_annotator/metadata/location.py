from __future__ import annotations

from typing import Optional

import httpx


def format_location(lat: Optional[float], lon: Optional[float], reverse_geocode: bool = False) -> str:
    if lat is None or lon is None:
        return "Location unknown"
    if not reverse_geocode:
        return f"Near {lat:.5f}, {lon:.5f}"
    try:
        response = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format": "jsonv2", "lat": lat, "lon": lon},
            headers={"User-Agent": "media-annotator"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        name = data.get("display_name")
        if name:
            return f"Near {name}"
    except httpx.HTTPError:
        return f"Near {lat:.5f}, {lon:.5f}"
    return f"Near {lat:.5f}, {lon:.5f}"
