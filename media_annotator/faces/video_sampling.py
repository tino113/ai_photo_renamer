from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SamplingPlan:
    times_ms: List[int]


def build_sampling_plan(
    duration_s: float,
    sample_rate: float,
    min_frames: int,
    max_frames: int,
) -> SamplingPlan:
    if duration_s <= 0:
        return SamplingPlan(times_ms=[])
    interval = max(1.0 / sample_rate, 0.1)
    times = []
    t = 0.0
    while t < duration_s and len(times) < max_frames:
        times.append(int(t * 1000))
        t += interval
    if len(times) < min_frames:
        step = max(duration_s / min_frames, 0.1)
        times = [int(i * step * 1000) for i in range(min_frames) if i * step < duration_s]
    return SamplingPlan(times_ms=times)
