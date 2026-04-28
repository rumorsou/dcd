from __future__ import annotations

import sys
from pathlib import Path

import torch


def _ensure_truss_path() -> None:
    root = Path(__file__).resolve().parents[1]
    truss_dir = root / "truss"
    for path in (str(root), str(truss_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)


def decompose_from_csr(rowptr: torch.Tensor, col: torch.Tensor) -> torch.Tensor:
    if col.numel() == 0:
        return torch.empty((0,), device=col.device, dtype=torch.int32)
    _ensure_truss_path()
    try:
        from truss.maintain_engine import decompose_from_csr as _decompose
        from truss.utils import device as truss_device
    except Exception:
        from maintain_engine import decompose_from_csr as _decompose
        from utils import device as truss_device

    result = _decompose(rowptr.to(device=truss_device, dtype=torch.long), col.to(device=truss_device, dtype=torch.long))
    return result.to(device=col.device, dtype=torch.int32)
