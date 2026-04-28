from __future__ import annotations

import torch

from .csr import canonicalize_edge_pairs
from .state import DCDState


def _aligned_tau(
    old_state: DCDState,
    new_pairs: torch.Tensor,
    *,
    inserted_value: int = 2,
) -> torch.Tensor:
    if new_pairs.numel() == 0:
        return torch.empty((0,), device=old_state.device, dtype=torch.int32)
    new_code = new_pairs[:, 0].to(torch.long) * old_state.num_vertices + new_pairs[:, 1].to(torch.long)
    pos = torch.searchsorted(old_state.edge_code, new_code)
    valid = pos < old_state.edge_code.numel()
    if torch.any(valid):
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        valid[valid_idx] = old_state.edge_code[pos[valid_idx]] == new_code[valid_idx]
    tau = torch.full((new_pairs.size(0),), inserted_value, device=old_state.device, dtype=torch.int32)
    if torch.any(valid):
        tau[valid] = old_state.tau[pos[valid]].to(torch.int32)
    return tau


def apply_updates_to_graph(state: DCDState, del_edges: torch.Tensor, ins_edges: torch.Tensor) -> DCDState:
    current = state.canonical_edge_pairs()
    n = state.num_vertices
    if current.numel() == 0:
        current_code = torch.empty((0,), device=state.device, dtype=torch.long)
    else:
        current_code = current[:, 0] * n + current[:, 1]

    del_edges = canonicalize_edge_pairs(del_edges.to(state.device), num_vertices=n)
    ins_edges = canonicalize_edge_pairs(ins_edges.to(state.device), num_vertices=n)
    del_code = del_edges[:, 0] * n + del_edges[:, 1] if del_edges.numel() > 0 else torch.empty((0,), device=state.device, dtype=torch.long)
    keep = ~torch.isin(current_code, del_code) if del_code.numel() > 0 else torch.ones_like(current_code, dtype=torch.bool)
    kept = current[keep] if current.numel() > 0 else current

    if ins_edges.numel() > 0:
        merged = torch.cat((kept, ins_edges), dim=0)
    else:
        merged = kept
    merged = canonicalize_edge_pairs(merged, num_vertices=n)
    tau = _aligned_tau(state, merged)
    return DCDState.from_local_edge_pairs(merged, state.vertex_ids, tau=tau, device=state.device)
