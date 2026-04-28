from __future__ import annotations

from dataclasses import dataclass

import torch

from .state import DCDState
from .triangle_index import EdgeTriangleIndex


@dataclass
class BoundState:
    tau_base: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    delta_plus: torch.Tensor
    delta_minus: torch.Tensor
    cand: torch.Tensor
    active: torch.Tensor
    fixed: torch.Tensor
    last_mark: torch.Tensor
    cone_edges: torch.Tensor
    seed_edges: torch.Tensor


def _map_codes(source_codes: torch.Tensor, query_codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if query_codes.numel() == 0:
        empty = torch.empty((0,), device=source_codes.device, dtype=torch.long)
        return empty, torch.empty((0,), device=source_codes.device, dtype=torch.bool)
    pos = torch.searchsorted(source_codes, query_codes)
    valid = pos < source_codes.numel()
    if torch.any(valid):
        idx = torch.nonzero(valid, as_tuple=False).flatten()
        valid[idx] = source_codes[pos[idx]] == query_codes[idx]
    return pos[valid], valid


def _count_for_codes(
    source_state: DCDState,
    source_index: EdgeTriangleIndex,
    target_state: DCDState,
    target_edges: torch.Tensor,
    *,
    edge_budget: int | None = None,
) -> torch.Tensor:
    if target_edges.numel() == 0:
        return torch.empty((0,), device=target_state.device, dtype=torch.long)
    codes = target_state.edge_code[target_edges]
    source_ids, valid = _map_codes(source_state.edge_code, codes)
    counts = torch.zeros((target_edges.numel(),), device=target_state.device, dtype=torch.long)
    if torch.any(valid):
        counts[valid] = source_index.support_counts(source_ids, edge_budget=edge_budget)
    return counts


def initial_seed_edges(
    old_state: DCDState,
    new_state: DCDState,
    old_index: EdgeTriangleIndex,
    new_index: EdgeTriangleIndex,
    deleted_edges_old: torch.Tensor,
    inserted_edges_new: torch.Tensor,
    *,
    edge_budget: int | None = None,
) -> torch.Tensor:
    parts = []
    if inserted_edges_new.numel() > 0:
        parts.append(inserted_edges_new)
        neighbors = new_index.triangle_neighbors(inserted_edges_new, edge_budget=edge_budget)
        if neighbors.numel() > 0:
            parts.append(neighbors)
    if deleted_edges_old.numel() > 0:
        old_neighbors = old_index.triangle_neighbors(deleted_edges_old, edge_budget=edge_budget)
        if old_neighbors.numel() > 0:
            old_codes = old_state.edge_code[old_neighbors]
            mapped, valid = _map_codes(new_state.edge_code, old_codes)
            if torch.any(valid):
                parts.append(mapped)
    if not parts:
        return torch.empty((0,), device=new_state.device, dtype=torch.long)
    return torch.unique(torch.cat(parts), sorted=True)


def build_initial_bounds(
    old_state: DCDState,
    new_state: DCDState,
    old_index: EdgeTriangleIndex,
    new_index: EdgeTriangleIndex,
    cone_edges: torch.Tensor,
    seed_edges: torch.Tensor,
    *,
    inserted_edges_new: torch.Tensor,
    edge_budget: int | None = None,
) -> BoundState:
    device = new_state.device
    tau_base = new_state.tau.clone().to(torch.int32)
    lower = tau_base.clone()
    upper = tau_base.clone()
    delta_plus = torch.zeros((new_state.num_edges,), device=device, dtype=torch.int32)
    delta_minus = torch.zeros((new_state.num_edges,), device=device, dtype=torch.int32)
    cand = torch.zeros((new_state.num_edges,), device=device, dtype=torch.bool)
    fixed = torch.zeros_like(cand)
    last_mark = torch.full((new_state.num_edges,), -1, device=device, dtype=torch.int32)

    cone_edges = torch.unique(cone_edges.to(device=device, dtype=torch.long), sorted=True)
    if cone_edges.numel() > 0:
        cand[cone_edges] = True
        new_counts = new_index.support_counts(cone_edges, edge_budget=edge_budget)
        old_counts = _count_for_codes(old_state, old_index, new_state, cone_edges, edge_budget=edge_budget)
        plus = torch.clamp(new_counts - old_counts, min=0).to(torch.int32)
        minus = torch.clamp(old_counts - new_counts, min=0).to(torch.int32)
        delta_plus[cone_edges] = plus
        delta_minus[cone_edges] = torch.minimum(torch.clamp(tau_base[cone_edges] - 2, min=0), minus)
        lower[cone_edges] = torch.clamp(tau_base[cone_edges] - delta_minus[cone_edges], min=2)
        structural_upper = (new_counts + 2).to(torch.int32)
        upper[cone_edges] = torch.maximum(lower[cone_edges], structural_upper)

    if inserted_edges_new.numel() > 0:
        inserted_edges_new = inserted_edges_new.to(device=device, dtype=torch.long)
        cand[inserted_edges_new] = True
        lower[inserted_edges_new] = 2
        upper[inserted_edges_new] = torch.maximum(
            upper[inserted_edges_new],
            new_index.support_counts(inserted_edges_new, edge_budget=edge_budget).to(torch.int32) + 2,
        )
        tau_base[inserted_edges_new] = 2

    active = cand & (lower < upper)
    return BoundState(
        tau_base=tau_base,
        lower=lower,
        upper=upper,
        delta_plus=delta_plus,
        delta_minus=delta_minus,
        cand=cand,
        active=active,
        fixed=fixed,
        last_mark=last_mark,
        cone_edges=torch.nonzero(cand, as_tuple=False).flatten(),
        seed_edges=seed_edges,
    )
