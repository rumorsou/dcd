from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .csr import canonicalize_edge_pairs
from .state import DCDState


@dataclass(frozen=True)
class DeltaGraph:
    ins_edges: torch.Tensor | None = None
    del_edges: torch.Tensor | None = None
    add_vertices: torch.Tensor | None = None
    remove_vertices: torch.Tensor | None = None
    is_local: bool = False


@dataclass(frozen=True)
class NormalizedDelta:
    state: DCDState
    ins_edges: torch.Tensor
    del_edges: torch.Tensor
    seed_edge_ids_old: torch.Tensor
    seed_edge_ids_new: torch.Tensor


def _edge_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    tensor = torch.as_tensor(value, device=device, dtype=torch.long)
    if tensor.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    return tensor.reshape(-1, 2)


def _vertex_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.empty((0,), device=device, dtype=torch.long)
    tensor = torch.as_tensor(value, device=device, dtype=torch.long)
    if tensor.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    return tensor.flatten()


def _incident_edges(state: DCDState, vertices: torch.Tensor) -> torch.Tensor:
    vertices = torch.unique(vertices.to(device=state.device, dtype=torch.long))
    if vertices.numel() == 0 or state.num_edges == 0:
        return torch.empty((0, 2), device=state.device, dtype=torch.long)
    src = state.edge_src.to(torch.long)
    dst = state.edge_dst.to(torch.long)
    mask = torch.isin(src, vertices) | torch.isin(dst, vertices)
    if not torch.any(mask):
        return torch.empty((0, 2), device=state.device, dtype=torch.long)
    return torch.stack((src[mask], dst[mask]), dim=1)


def _map_raw_edges(state: DCDState, raw_edges: torch.Tensor, *, create_missing: bool) -> tuple[DCDState, torch.Tensor]:
    if raw_edges.numel() == 0:
        return state, torch.empty((0, 2), device=state.device, dtype=torch.long)
    work_state = state.with_vertices(raw_edges.reshape(-1).to(torch.int64)) if create_missing else state
    mapped, valid = work_state.map_vertex_ids(raw_edges.reshape(-1).to(torch.int64))
    if not torch.any(valid):
        return work_state, torch.empty((0, 2), device=state.device, dtype=torch.long)
    full = torch.full((raw_edges.numel(),), -1, device=state.device, dtype=torch.long)
    full[torch.nonzero(valid, as_tuple=False).flatten()] = mapped
    keep = valid.reshape(-1, 2).all(dim=1)
    if not torch.any(keep):
        return work_state, torch.empty((0, 2), device=state.device, dtype=torch.long)
    return work_state, full.reshape(-1, 2)[keep]


def normalize_updates(state: DCDState, delta: DeltaGraph | dict[str, Any]) -> NormalizedDelta:
    if isinstance(delta, dict):
        delta = DeltaGraph(**delta)

    work_state = state
    ins_edges = _edge_tensor(delta.ins_edges, state.device)
    del_edges = _edge_tensor(delta.del_edges, state.device)

    add_vertices = _vertex_tensor(delta.add_vertices, state.device)
    if add_vertices.numel() > 0:
        if delta.is_local:
            raw_labels = work_state.vertex_ids[add_vertices]
            work_state = work_state.with_vertices(raw_labels)
        else:
            work_state = work_state.with_vertices(add_vertices.to(torch.int64))

    remove_vertices = _vertex_tensor(delta.remove_vertices, state.device)
    if remove_vertices.numel() > 0:
        if delta.is_local:
            local_vertices = remove_vertices
        else:
            local_vertices, valid = work_state.map_vertex_ids(remove_vertices.to(torch.int64))
            local_vertices = local_vertices[valid]
        del_edges = torch.cat((del_edges, _incident_edges(work_state, local_vertices)), dim=0)

    if not delta.is_local:
        work_state, del_edges = _map_raw_edges(work_state, del_edges, create_missing=False)
        work_state, ins_edges = _map_raw_edges(work_state, ins_edges, create_missing=True)

    ins_edges = canonicalize_edge_pairs(ins_edges, num_vertices=work_state.num_vertices)
    del_edges = canonicalize_edge_pairs(del_edges, num_vertices=work_state.num_vertices)
    if ins_edges.numel() > 0 and del_edges.numel() > 0:
        # A mixed batch is interpreted as delete-then-insert. Identical pairs cancel
        # from deletion and remain as inserted final edges.
        ins_code = torch.unique(ins_edges[:, 0] * work_state.num_vertices + ins_edges[:, 1], sorted=True)
        del_code = del_edges[:, 0] * work_state.num_vertices + del_edges[:, 1]
        del_edges = del_edges[~torch.isin(del_code, ins_code)]

    old_seed, _ = work_state.edge_ids_from_pairs(del_edges)
    seed_new = torch.empty((0,), device=state.device, dtype=torch.long)
    return NormalizedDelta(work_state, ins_edges, del_edges, old_seed, seed_new)
