from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from .state import DCDState


@dataclass(frozen=True)
class EdgeChunk:
    edge_ids: torch.Tensor
    work: torch.Tensor


def edge_work(state: DCDState, edge_ids: torch.Tensor) -> torch.Tensor:
    edge_ids = edge_ids.to(device=state.device, dtype=torch.long)
    if edge_ids.numel() == 0:
        return torch.empty((0,), device=state.device, dtype=torch.long)
    deg = state.rowptr[1:] - state.rowptr[:-1]
    src = state.edge_src[edge_ids].to(torch.long)
    dst = state.edge_dst[edge_ids].to(torch.long)
    return deg[src] + deg[dst]


def chunk_by_budget(state: DCDState, edge_ids: torch.Tensor, *, edge_budget: int) -> Iterator[EdgeChunk]:
    edge_ids = torch.unique(edge_ids.to(device=state.device, dtype=torch.long), sorted=True)
    if edge_ids.numel() == 0:
        return
    work = edge_work(state, edge_ids)
    order = torch.argsort(work, descending=True, stable=True)
    edge_ids = edge_ids[order]
    work = work[order]
    budget = max(int(edge_budget), 1)
    for start in range(0, int(edge_ids.numel()), budget):
        yield EdgeChunk(edge_ids[start : start + budget], work[start : start + budget])
