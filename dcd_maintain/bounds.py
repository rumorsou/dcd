from __future__ import annotations

from dataclasses import dataclass

import torch

from ._compat import _lookup_edge_codes, _lookup_edge_ids, device
from .graph_state import GraphState
from .support_estimator import estimate_support_bounds_for_edges
from .triangle_index import TriangleIndex


@dataclass
class BoundState:
    tau_base: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    delta_plus: torch.Tensor
    delta_minus: torch.Tensor
    candidate_mask: torch.Tensor
    seed_edges: torch.Tensor
    seed_neighbor_mask: torch.Tensor
    cone_edges: torch.Tensor


def _edge_ids_from_codes(state: GraphState, edge_codes: torch.Tensor) -> torch.Tensor:
    if edge_codes.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    edge_ids, valid = _lookup_edge_codes(state.edge_code, edge_codes.to(device=device, dtype=torch.long))
    if valid.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    return edge_ids


def _neighbor_seed_codes(old_state: GraphState, old_deleted_pairs: torch.Tensor, old_index: TriangleIndex) -> torch.Tensor:
    if old_deleted_pairs.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    deleted_ids, _ = _lookup_edge_ids(old_state.num_vertices, old_state.edge_code, old_deleted_pairs)
    if deleted_ids.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    neighbor_ids = old_index.triangle_neighbors(deleted_ids)
    if neighbor_ids.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    return old_state.edge_code[neighbor_ids]


def _triangle_neighbor_layer(
    old_state: GraphState,
    new_state: GraphState,
    old_index: TriangleIndex,
    new_index: TriangleIndex,
    frontier_edges: torch.Tensor,
) -> torch.Tensor:
    frontier = torch.unique(frontier_edges.to(device=device, dtype=torch.long))
    if frontier.numel() == 0:
        return frontier

    next_parts = []

    new_neighbors = new_index.triangle_neighbors(frontier)
    if new_neighbors.numel() > 0:
        next_parts.append(new_neighbors)

    frontier_codes = new_state.edge_code[frontier]
    old_frontier_ids, old_valid = _lookup_edge_codes(old_state.edge_code, frontier_codes)
    if old_valid.numel() > 0 and torch.any(old_valid):
        old_neighbors = old_index.triangle_neighbors(old_frontier_ids)
        if old_neighbors.numel() > 0:
            old_neighbor_codes = old_state.edge_code[old_neighbors]
            mapped_to_new = _edge_ids_from_codes(new_state, old_neighbor_codes)
            if mapped_to_new.numel() > 0:
                next_parts.append(mapped_to_new)

    if not next_parts:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.unique(torch.cat(next_parts))


def expand_triangle_candidates(
    old_state: GraphState,
    new_state: GraphState,
    old_index: TriangleIndex,
    new_index: TriangleIndex,
    seed_edges: torch.Tensor,
    *,
    max_hops: int | None,
) -> torch.Tensor:
    frontier = torch.unique(seed_edges.to(device=device, dtype=torch.long))
    if frontier.numel() == 0:
        return frontier

    visited = torch.zeros((new_state.num_edges,), device=device, dtype=torch.bool)
    visited[frontier] = True
    hops = 0

    while frontier.numel() > 0 and (max_hops is None or hops < max_hops):
        next_frontier = _triangle_neighbor_layer(
            old_state=old_state,
            new_state=new_state,
            old_index=old_index,
            new_index=new_index,
            frontier_edges=frontier,
        )
        if next_frontier.numel() == 0:
            break
        next_frontier = next_frontier[~visited[next_frontier]]
        if next_frontier.numel() == 0:
            break
        visited[next_frontier] = True
        frontier = next_frontier
        hops += 1

    return torch.nonzero(visited, as_tuple=False).flatten()


def _degree_upper_bound(state: GraphState, edge_ids: torch.Tensor) -> torch.Tensor:
    if edge_ids.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    bi_row_ptr, _, _ = state.bidirectional_view
    degrees = bi_row_ptr[1:] - bi_row_ptr[:-1]
    src = state.edge_src_index[edge_ids]
    dst = state.columns[edge_ids].to(torch.long)
    return torch.minimum(degrees[src], degrees[dst]).to(torch.long) + 1


def collect_closure_seed_edges(
    old_state: GraphState,
    new_state: GraphState,
    old_index: TriangleIndex,
    new_index: TriangleIndex,
    inserted_pairs: torch.Tensor,
    deleted_pairs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inserted_ids, _ = _lookup_edge_ids(new_state.num_vertices, new_state.edge_code, inserted_pairs)
    insert_neighbor_ids = new_index.triangle_neighbors(inserted_ids) if inserted_ids.numel() > 0 else torch.empty((0,), device=device, dtype=torch.long)
    delete_neighbor_codes = _neighbor_seed_codes(old_state, deleted_pairs, old_index)
    delete_neighbor_ids = _edge_ids_from_codes(new_state, delete_neighbor_codes)

    seed_edges = torch.unique(torch.cat((inserted_ids, insert_neighbor_ids, delete_neighbor_ids))) if (
        inserted_ids.numel() > 0 or insert_neighbor_ids.numel() > 0 or delete_neighbor_ids.numel() > 0
    ) else torch.empty((0,), device=device, dtype=torch.long)
    return inserted_ids, seed_edges


def collect_update_seed_edges(new_state: GraphState, inserted_pairs: torch.Tensor) -> torch.Tensor:
    inserted_ids, _ = _lookup_edge_ids(new_state.num_vertices, new_state.edge_code, inserted_pairs)
    return inserted_ids


def collect_seed_neighbor_edges(
    old_state: GraphState,
    new_state: GraphState,
    old_index: TriangleIndex,
    new_index: TriangleIndex,
    inserted_ids: torch.Tensor,
    deleted_pairs: torch.Tensor,
) -> torch.Tensor:
    insert_neighbor_ids = new_index.triangle_neighbors(inserted_ids) if inserted_ids.numel() > 0 else torch.empty((0,), device=device, dtype=torch.long)
    delete_neighbor_codes = _neighbor_seed_codes(old_state, deleted_pairs, old_index)
    delete_neighbor_ids = _edge_ids_from_codes(new_state, delete_neighbor_codes)
    if insert_neighbor_ids.numel() == 0 and delete_neighbor_ids.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    return torch.unique(torch.cat((insert_neighbor_ids, delete_neighbor_ids)))


def edge_window_metrics(
    old_state: GraphState,
    new_state: GraphState,
    old_index: TriangleIndex,
    new_index: TriangleIndex,
    edge_ids: torch.Tensor,
    inserted_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_ids = torch.unique(edge_ids.to(device=device, dtype=torch.long))
    if edge_ids.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty, empty, empty

    edge_codes = new_state.edge_code[edge_ids]
    old_edge_ids, valid_old = _lookup_edge_codes(old_state.edge_code, edge_codes)
    tau_base = new_state.tau[edge_ids].clone()
    new_triangle_count = new_index.incident_triangle_count(edge_ids)
    old_triangle_count = torch.zeros((edge_ids.numel(),), device=device, dtype=torch.long)

    if valid_old.numel() > 0:
        valid_pos = torch.nonzero(valid_old, as_tuple=False).flatten()
        tau_base[valid_pos] = old_state.tau[old_edge_ids]
        old_triangle_count[valid_pos] = old_index.incident_triangle_count(old_edge_ids)

    if inserted_ids.numel() > 0:
        inserted_mask = torch.isin(edge_ids, inserted_ids)
        tau_base[inserted_mask] = 2

    support_upper = new_triangle_count + 2
    structural_upper = torch.minimum(_degree_upper_bound(new_state, edge_ids), support_upper)
    delta_plus = torch.clamp(new_triangle_count - old_triangle_count, min=0)
    delta_minus = torch.clamp(old_triangle_count - new_triangle_count, min=0)
    return tau_base, structural_upper, delta_plus, delta_minus


def _narrow_window_bounds(
    tau_base: torch.Tensor,
    structural_upper: torch.Tensor,
    *,
    allow_decrease: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    lower = torch.clamp(tau_base - 1, min=2) if allow_decrease else torch.clamp(tau_base, min=2)
    # Batched edge insertions can raise an edge by more than one truss level,
    # so insert-side bounds must keep the full structural upper cap.
    upper = torch.minimum(structural_upper, tau_base + 1) if allow_decrease else structural_upper
    upper = torch.maximum(upper, lower)
    return lower, upper


def screen_seed_neighbors(
    old_state: GraphState,
    new_state: GraphState,
    old_index: TriangleIndex,
    new_index: TriangleIndex,
    inserted_ids: torch.Tensor,
    seed_neighbor_edges: torch.Tensor,
    *,
    allow_decrease: bool = True,
) -> torch.Tensor:
    seed_neighbor_edges = torch.unique(seed_neighbor_edges.to(device=device, dtype=torch.long))
    seed_neighbor_edges = seed_neighbor_edges[~torch.isin(seed_neighbor_edges, inserted_ids)]
    if seed_neighbor_edges.numel() == 0:
        return seed_neighbor_edges

    probe_edges = torch.unique(torch.cat((inserted_ids, seed_neighbor_edges))) if inserted_ids.numel() > 0 else seed_neighbor_edges
    probe_tau_base, probe_structural_upper, probe_plus, probe_minus = edge_window_metrics(
        old_state=old_state,
        new_state=new_state,
        old_index=old_index,
        new_index=new_index,
        edge_ids=probe_edges,
        inserted_ids=inserted_ids,
    )
    trial_lower = new_state.tau.clone()
    trial_upper = new_state.tau.clone()
    trial_lower[probe_edges] = probe_tau_base
    trial_upper[probe_edges] = probe_tau_base

    if inserted_ids.numel() > 0:
        seed_pos = torch.searchsorted(probe_edges, inserted_ids)
        seed_structural_upper = probe_structural_upper[seed_pos]
        trial_lower[inserted_ids] = 2
        trial_upper[inserted_ids] = torch.maximum(seed_structural_upper, torch.full_like(seed_structural_upper, 2))

    neighbor_pos = torch.searchsorted(probe_edges, seed_neighbor_edges)
    neighbor_tau_base = probe_tau_base[neighbor_pos]
    neighbor_structural_upper = probe_structural_upper[neighbor_pos]
    neighbor_plus = probe_plus[neighbor_pos]
    neighbor_minus = probe_minus[neighbor_pos]
    neighbor_lower, neighbor_upper = _narrow_window_bounds(
        neighbor_tau_base,
        neighbor_structural_upper,
        allow_decrease=allow_decrease,
    )
    trial_lower[seed_neighbor_edges] = neighbor_lower
    trial_upper[seed_neighbor_edges] = neighbor_upper

    critical_k = torch.div(neighbor_lower + neighbor_upper + 1, 2, rounding_mode="floor")
    smin, smax = estimate_support_bounds_for_edges(
        triangle_index=new_index,
        edge_ids=seed_neighbor_edges,
        k_values=critical_k,
        lower=trial_lower,
        upper=trial_upper,
    )
    window_open = neighbor_lower < neighbor_upper
    support_uncertain = (smin < (critical_k - 2)) & (smax >= (critical_k - 2))
    local_delta_active = (neighbor_plus > 0) | (neighbor_minus > 0)
    if allow_decrease:
        return seed_neighbor_edges[window_open & (support_uncertain | local_delta_active)]
    return seed_neighbor_edges[window_open & support_uncertain]


def build_bound_state_for_candidates(
    old_state: GraphState,
    new_state: GraphState,
    old_index: TriangleIndex,
    new_index: TriangleIndex,
    inserted_ids: torch.Tensor,
    seed_edges: torch.Tensor,
    seed_neighbor_mask: torch.Tensor,
    candidate_edges: torch.Tensor,
    *,
    allow_decrease: bool = True,
) -> BoundState:
    cone_edges = torch.unique(candidate_edges.to(device=device, dtype=torch.long))

    if cone_edges.numel() == 0:
        candidate_mask = torch.zeros((new_state.num_edges,), device=device, dtype=torch.bool)
        empty = torch.empty((0,), device=device, dtype=torch.long)
        tau_base = new_state.tau.clone()
        return BoundState(
            tau_base,
            tau_base.clone(),
            tau_base.clone(),
            torch.zeros_like(tau_base),
            torch.zeros_like(tau_base),
            candidate_mask,
            empty,
            torch.zeros((new_state.num_edges,), device=device, dtype=torch.bool),
            empty,
        )
    tau_base = new_state.tau.clone()
    delta_plus = torch.zeros((new_state.num_edges,), device=device, dtype=torch.long)
    delta_minus = torch.zeros((new_state.num_edges,), device=device, dtype=torch.long)
    base_subset, structural_upper, plus, minus = edge_window_metrics(
        old_state=old_state,
        new_state=new_state,
        old_index=old_index,
        new_index=new_index,
        edge_ids=cone_edges,
        inserted_ids=inserted_ids,
    )
    tau_base[cone_edges] = base_subset
    delta_plus[cone_edges] = plus
    delta_minus[cone_edges] = minus

    lower = tau_base.clone()
    upper = tau_base.clone()
    narrow_edges = cone_edges[~torch.isin(cone_edges, seed_edges)]
    if narrow_edges.numel() > 0:
        narrow_pos = torch.searchsorted(cone_edges, narrow_edges)
        narrow_lower, narrow_upper = _narrow_window_bounds(
            tau_base[narrow_edges],
            structural_upper[narrow_pos],
            allow_decrease=allow_decrease,
        )
        lower[narrow_edges] = narrow_lower
        upper[narrow_edges] = narrow_upper
    if seed_edges.numel() > 0:
        seed_pos = torch.searchsorted(cone_edges, seed_edges)
        lower[seed_edges] = 2
        if inserted_ids.numel() > 0:
            upper[seed_edges] = torch.maximum(structural_upper[seed_pos], torch.full_like(seed_pos, 2))
        else:
            upper[seed_edges] = torch.minimum(tau_base[seed_edges], structural_upper[seed_pos])

    candidate_mask = torch.zeros((new_state.num_edges,), device=device, dtype=torch.bool)
    candidate_mask[cone_edges] = True
    return BoundState(
        tau_base=tau_base,
        lower=lower,
        upper=upper,
        delta_plus=delta_plus,
        delta_minus=delta_minus,
        candidate_mask=candidate_mask,
        seed_edges=seed_edges,
        seed_neighbor_mask=seed_neighbor_mask,
        cone_edges=cone_edges,
    )


def extend_bound_state_for_candidates(
    base_bound_state: BoundState,
    old_state: GraphState,
    new_state: GraphState,
    old_index: TriangleIndex,
    new_index: TriangleIndex,
    inserted_ids: torch.Tensor,
    seed_edges: torch.Tensor,
    candidate_edges: torch.Tensor,
    *,
    allow_decrease: bool = True,
) -> BoundState:
    cone_edges = torch.unique(candidate_edges.to(device=device, dtype=torch.long))
    if cone_edges.numel() == 0:
        return build_bound_state_for_candidates(
            old_state=old_state,
            new_state=new_state,
            old_index=old_index,
            new_index=new_index,
            inserted_ids=inserted_ids,
            seed_edges=seed_edges,
            seed_neighbor_mask=base_bound_state.seed_neighbor_mask,
            candidate_edges=cone_edges,
            allow_decrease=allow_decrease,
        )

    new_edges = cone_edges[~base_bound_state.candidate_mask[cone_edges]]
    if new_edges.numel() == 0:
        return BoundState(
            tau_base=base_bound_state.tau_base.clone(),
            lower=base_bound_state.lower.clone(),
            upper=base_bound_state.upper.clone(),
            delta_plus=base_bound_state.delta_plus.clone(),
            delta_minus=base_bound_state.delta_minus.clone(),
            candidate_mask=base_bound_state.candidate_mask.clone(),
            seed_edges=base_bound_state.seed_edges.clone(),
            seed_neighbor_mask=base_bound_state.seed_neighbor_mask.clone(),
            cone_edges=cone_edges,
        )

    tau_base = base_bound_state.tau_base.clone()
    lower = base_bound_state.lower.clone()
    upper = base_bound_state.upper.clone()
    delta_plus = base_bound_state.delta_plus.clone()
    delta_minus = base_bound_state.delta_minus.clone()
    candidate_mask = base_bound_state.candidate_mask.clone()

    base_subset, structural_upper, plus, minus = edge_window_metrics(
        old_state=old_state,
        new_state=new_state,
        old_index=old_index,
        new_index=new_index,
        edge_ids=new_edges,
        inserted_ids=inserted_ids,
    )
    tau_base[new_edges] = base_subset
    delta_plus[new_edges] = plus
    delta_minus[new_edges] = minus
    lower[new_edges] = tau_base[new_edges]
    upper[new_edges] = tau_base[new_edges]

    narrow_edges = new_edges[~torch.isin(new_edges, seed_edges)]
    if narrow_edges.numel() > 0:
        narrow_pos = torch.searchsorted(new_edges, narrow_edges)
        narrow_lower, narrow_upper = _narrow_window_bounds(
            tau_base[narrow_edges],
            structural_upper[narrow_pos],
            allow_decrease=allow_decrease,
        )
        lower[narrow_edges] = narrow_lower
        upper[narrow_edges] = narrow_upper

    new_seed_edges = new_edges[torch.isin(new_edges, seed_edges)]
    if new_seed_edges.numel() > 0:
        seed_pos = torch.searchsorted(new_edges, new_seed_edges)
        lower[new_seed_edges] = 2
        if inserted_ids.numel() > 0:
            upper[new_seed_edges] = torch.maximum(
                structural_upper[seed_pos],
                torch.full_like(structural_upper[seed_pos], 2),
            )
        else:
            upper[new_seed_edges] = torch.minimum(tau_base[new_seed_edges], structural_upper[seed_pos])

    candidate_mask[new_edges] = True
    return BoundState(
        tau_base=tau_base,
        lower=lower,
        upper=upper,
        delta_plus=delta_plus,
        delta_minus=delta_minus,
        candidate_mask=candidate_mask,
        seed_edges=base_bound_state.seed_edges.clone(),
        seed_neighbor_mask=base_bound_state.seed_neighbor_mask.clone(),
        cone_edges=cone_edges,
    )


def build_initial_bounds(
    old_state: GraphState,
    new_state: GraphState,
    old_index: TriangleIndex,
    new_index: TriangleIndex,
    inserted_pairs: torch.Tensor,
    deleted_pairs: torch.Tensor,
    *,
    candidate_hops: int | None = 0,
) -> BoundState:
    allow_decrease = deleted_pairs.numel() > 0
    if candidate_hops is None:
        inserted_ids, closure_seed_edges = collect_closure_seed_edges(
            old_state=old_state,
            new_state=new_state,
            old_index=old_index,
            new_index=new_index,
            inserted_pairs=inserted_pairs,
            deleted_pairs=deleted_pairs,
        )
        if inserted_pairs.numel() > 0 and deleted_pairs.numel() == 0:
            candidate_edges = new_index.bfs_cone(closure_seed_edges)
        else:
            candidate_edges = expand_triangle_candidates(
                old_state=old_state,
                new_state=new_state,
                old_index=old_index,
                new_index=new_index,
                seed_edges=closure_seed_edges,
                max_hops=candidate_hops,
            )
        return build_bound_state_for_candidates(
            old_state=old_state,
            new_state=new_state,
            old_index=old_index,
            new_index=new_index,
            inserted_ids=inserted_ids,
            seed_edges=inserted_ids,
            seed_neighbor_mask=torch.zeros((new_state.num_edges,), device=device, dtype=torch.bool),
            candidate_edges=candidate_edges,
            allow_decrease=allow_decrease,
        )

    inserted_ids = collect_update_seed_edges(new_state, inserted_pairs)
    insert_seed_neighbor_edges = new_index.triangle_neighbors(inserted_ids) if inserted_ids.numel() > 0 else torch.empty((0,), device=device, dtype=torch.long)
    delete_neighbor_codes = _neighbor_seed_codes(old_state, deleted_pairs, old_index)
    delete_seed_neighbor_edges = _edge_ids_from_codes(new_state, delete_neighbor_codes)
    seed_neighbor_edges = (
        torch.unique(torch.cat((insert_seed_neighbor_edges, delete_seed_neighbor_edges)))
        if insert_seed_neighbor_edges.numel() > 0 or delete_seed_neighbor_edges.numel() > 0
        else torch.empty((0,), device=device, dtype=torch.long)
    )
    screened_insert_neighbors = screen_seed_neighbors(
        old_state=old_state,
        new_state=new_state,
        old_index=old_index,
        new_index=new_index,
        inserted_ids=inserted_ids,
        seed_neighbor_edges=insert_seed_neighbor_edges,
        allow_decrease=allow_decrease,
    )
    screened_neighbor_edges = (
        torch.unique(torch.cat((screened_insert_neighbors, delete_seed_neighbor_edges)))
        if screened_insert_neighbors.numel() > 0 or delete_seed_neighbor_edges.numel() > 0
        else torch.empty((0,), device=device, dtype=torch.long)
    )
    candidate_edges = (
        torch.unique(torch.cat((inserted_ids, screened_neighbor_edges)))
        if inserted_ids.numel() > 0 or screened_neighbor_edges.numel() > 0
        else torch.empty((0,), device=device, dtype=torch.long)
    )
    seed_neighbor_mask = torch.zeros((new_state.num_edges,), device=device, dtype=torch.bool)
    if seed_neighbor_edges.numel() > 0:
        seed_neighbor_mask[seed_neighbor_edges] = True
    return build_bound_state_for_candidates(
        old_state=old_state,
        new_state=new_state,
        old_index=old_index,
        new_index=new_index,
        inserted_ids=inserted_ids,
        seed_edges=inserted_ids,
        seed_neighbor_mask=seed_neighbor_mask,
        candidate_edges=candidate_edges,
        allow_decrease=allow_decrease,
    )
