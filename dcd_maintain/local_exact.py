from __future__ import annotations

import torch

from ._compat import device
from .triangle_index import ExplicitTriangleIndex, TriangleIndex


def _expand_ranges(starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    sizes = ends - starts
    if sizes.numel() == 0 or int(sizes.sum().item()) == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    ptr = torch.cat((torch.tensor([0], device=device, dtype=torch.long), sizes.cumsum(0)))
    return torch.arange(int(ptr[-1].item()), device=device, dtype=torch.long) - torch.repeat_interleave(ptr[:-1] - starts, sizes)


def _deactivate_triangles(
    tri_ids: torch.Tensor,
    active_triangles: torch.Tensor,
    local_support: torch.Tensor,
    alive_local: torch.Tensor,
    tri_local_edges: torch.Tensor,
) -> None:
    if tri_ids.numel() == 0:
        return
    tri_ids = torch.unique(tri_ids)
    tri_ids = tri_ids[active_triangles[tri_ids]]
    if tri_ids.numel() == 0:
        return
    active_triangles[tri_ids] = False
    local_edges = tri_local_edges[tri_ids]
    valid_mask = local_edges >= 0
    if not torch.any(valid_mask):
        return
    flat_local_edges = local_edges[valid_mask]
    alive_mask = alive_local[flat_local_edges]
    if not torch.any(alive_mask):
        return
    decrement = torch.bincount(flat_local_edges[alive_mask], minlength=alive_local.numel())
    local_support -= decrement


def solve_candidate_truss_exact(
    triangle_index: TriangleIndex,
    candidate_mask: torch.Tensor,
    fixed_tau: torch.Tensor,
    upper: torch.Tensor,
    explicit_index: ExplicitTriangleIndex | None = None,
) -> torch.Tensor:
    tau_new = fixed_tau.clone()
    candidate_edges = torch.nonzero(candidate_mask, as_tuple=False).flatten()
    if candidate_edges.numel() == 0:
        return tau_new

    if explicit_index is None:
        explicit_index = triangle_index.materialize(candidate_edges)
    tau_new[candidate_edges] = 2
    local_upper = upper[explicit_index.edge_ids]
    tri_boundary_limit = torch.full(
        (explicit_index.edges_of_tri.size(0),),
        int(torch.max(local_upper).item()) if local_upper.numel() > 0 else 2,
        device=device,
        dtype=torch.long,
    )
    for pos in range(3):
        boundary_mask = explicit_index.tri_local_edges[:, pos] < 0
        if torch.any(boundary_mask):
            tri_boundary_limit[boundary_mask] = torch.minimum(
                tri_boundary_limit[boundary_mask],
                fixed_tau[explicit_index.edges_of_tri[boundary_mask, pos]],
            )

    max_k = int(torch.max(upper[candidate_edges]).item())
    alive_local = local_upper >= 3
    active_triangles = tri_boundary_limit >= 3
    for pos in range(3):
        local_ids = explicit_index.tri_local_edges[:, pos]
        local_mask = local_ids >= 0
        if torch.any(local_mask):
            active_triangles[local_mask] &= alive_local[local_ids[local_mask]]
    local_support = torch.zeros((explicit_index.edge_ids.numel(),), device=device, dtype=torch.long)
    if torch.any(active_triangles):
        active_local_edges = explicit_index.tri_local_edges[active_triangles]
        valid_local = active_local_edges >= 0
        if torch.any(valid_local):
            local_support = torch.bincount(
                active_local_edges[valid_local],
                minlength=explicit_index.edge_ids.numel(),
            )

    for k in range(3, max_k + 1):
        if not torch.any(alive_local & (local_upper >= k)):
            continue

        if k > 3:
            expiring = torch.nonzero(active_triangles & (tri_boundary_limit < k), as_tuple=False).flatten()
            _deactivate_triangles(
                expiring,
                active_triangles,
                local_support,
                alive_local,
                explicit_index.tri_local_edges,
            )
            upper_expired = torch.nonzero(alive_local & (local_upper < k), as_tuple=False).flatten()
            if upper_expired.numel() > 0:
                alive_local[upper_expired] = False
                incidence_indices = _expand_ranges(
                    explicit_index.tri_ptr[upper_expired],
                    explicit_index.tri_ptr[upper_expired + 1],
                )
                if incidence_indices.numel() > 0:
                    _deactivate_triangles(
                        explicit_index.tri_ids_flat[incidence_indices],
                        active_triangles,
                        local_support,
                        alive_local,
                        explicit_index.tri_local_edges,
                    )

        if not torch.any(alive_local):
            break

        while True:
            evicted_mask = alive_local & (local_upper >= k) & (local_support < (k - 2))
            if not torch.any(evicted_mask):
                break
            evicted_local_ids = torch.nonzero(evicted_mask, as_tuple=False).flatten()
            alive_local[evicted_local_ids] = False
            incidence_indices = _expand_ranges(
                explicit_index.tri_ptr[evicted_local_ids],
                explicit_index.tri_ptr[evicted_local_ids + 1],
            )
            if incidence_indices.numel() > 0:
                _deactivate_triangles(
                    explicit_index.tri_ids_flat[incidence_indices],
                    active_triangles,
                    local_support,
                    alive_local,
                    explicit_index.tri_local_edges,
                )

        surviving_mask = alive_local & (local_upper >= k)
        if torch.any(surviving_mask):
            tau_new[explicit_index.edge_ids[surviving_mask]] = k
    return tau_new
