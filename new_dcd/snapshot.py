from __future__ import annotations

from pathlib import Path

import torch

from .state import DCDState


def save_snapshot(state: DCDState, snapshot_dir: str | Path) -> None:
    state.save(snapshot_dir)


def load_snapshot(snapshot_dir: str | Path, *, map_location: str | torch.device | None = None) -> DCDState:
    return DCDState.load(snapshot_dir, map_location=map_location)
