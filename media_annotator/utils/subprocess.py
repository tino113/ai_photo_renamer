from __future__ import annotations

import json
import subprocess
from typing import Iterable

from loguru import logger


def run_command(command: Iterable[str]) -> subprocess.CompletedProcess:
    logger.debug("Running command: {}", " ".join(command))
    return subprocess.run(command, capture_output=True, text=True, check=False)


def run_json(command: Iterable[str]) -> dict:
    result = run_command(command)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{result.stderr}")
    return json.loads(result.stdout)
