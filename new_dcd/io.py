from __future__ import annotations

import numpy as np
import torch

from .state import DCDState
from .updates import DeltaGraph


def load_graph_from_txt(path: str, *, device: str | torch.device | None = None, relabel: bool = True) -> DCDState:
    data = np.loadtxt(path, dtype=np.int64)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError("Graph file must contain at least two columns.")
    return DCDState.from_edge_pairs(data[:, :2], device=device, relabel=relabel)


def load_edge_pairs_from_txt(path: str, *, device: str | torch.device | None = None) -> torch.Tensor:
    data = np.loadtxt(path, dtype=np.int64)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError("Update file must contain at least two columns.")
    return torch.as_tensor(data[:, :2], device=torch.device(device) if device is not None else None, dtype=torch.long)


def load_delta_from_txt(
    *,
    ins_path: str | None = None,
    del_path: str | None = None,
    device: str | torch.device | None = None,
) -> DeltaGraph:
    return DeltaGraph(
        ins_edges=None if ins_path is None else load_edge_pairs_from_txt(ins_path, device=device),
        del_edges=None if del_path is None else load_edge_pairs_from_txt(del_path, device=device),
    )
