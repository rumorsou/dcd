from __future__ import annotations

from dataclasses import dataclass

import torch

from ._compat import device
from .support_estimator import estimate_support_bounds_for_edge
from .triangle_index import TriangleIndex


@dataclass
class PhaseStats:
    rounds: int = 0
    tightened_edges: int = 0
    backfilled_edges: int = 0


def pick_critical_k(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.div(lower + upper + 1, 2, rounding_mode="floor")


def _search_upper_bound(
    triangle_index: TriangleIndex,
    edge_id: int,
    lower_bound: int,
    upper_bound: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> int:
    left = int(lower_bound)
    right = int(upper_bound)
    best = left
    while left <= right:
        mid = (left + right) // 2
        support = estimate_support_bounds_for_edge(triangle_index, edge_id, mid, lower, upper)
        if support.smax >= mid - 2:
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best


def _search_lower_bound(
    triangle_index: TriangleIndex,
    edge_id: int,
    lower_bound: int,
    upper_bound: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> int:
    left = int(lower_bound)
    right = int(upper_bound)
    best = left
    while left <= right:
        mid = (left + right) // 2
        support = estimate_support_bounds_for_edge(triangle_index, edge_id, mid, lower, upper)
        if support.smin >= mid - 2:
            best = mid
            left = mid + 1
        else:
            right = mid - 1
    return best


def propagate_and_backfill(
    triangle_index: TriangleIndex,
    candidate_mask: torch.Tensor,
    tightened_edges: torch.Tensor,
) -> torch.Tensor:
    if tightened_edges.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    neighbors = triangle_index.triangle_neighbors(tightened_edges)
    if neighbors.numel() == 0:
        return neighbors
    new_edges = neighbors[~candidate_mask[neighbors]]
    if new_edges.numel() > 0:
        candidate_mask[new_edges] = True
    return new_edges


def refine_phase(
    triangle_index: TriangleIndex,
    lower: torch.Tensor,
    upper: torch.Tensor,
    candidate_mask: torch.Tensor,
    allow_raise_lower: bool,
    max_rounds: int = 1000,
) -> PhaseStats:
    stats = PhaseStats()
    while stats.rounds < max_rounds:
        active_edges = torch.nonzero(candidate_mask & (lower < upper), as_tuple=False).flatten()
        if active_edges.numel() == 0:
            break

        stats.rounds += 1
        critical_k = pick_critical_k(lower[active_edges], upper[active_edges])
        tightened = []
        for local_idx, edge_id in enumerate(active_edges.tolist()):
            current_lower = int(lower[edge_id].item())
            current_upper = int(upper[edge_id].item())
            current_k = int(critical_k[local_idx].item())

            next_upper = _search_upper_bound(
                triangle_index=triangle_index,
                edge_id=edge_id,
                lower_bound=current_lower,
                upper_bound=current_upper,
                lower=lower,
                upper=upper,
            )
            next_lower = current_lower
            if allow_raise_lower:
                next_lower = _search_lower_bound(
                    triangle_index=triangle_index,
                    edge_id=edge_id,
                    lower_bound=current_lower,
                    upper_bound=next_upper,
                    lower=lower,
                    upper=upper,
                )
                next_lower = max(next_lower, min(current_k, next_upper) if next_lower > current_lower else next_lower)

            if next_upper != current_upper or next_lower != current_lower:
                upper[edge_id] = next_upper
                lower[edge_id] = min(next_lower, next_upper)
                tightened.append(edge_id)

        if not tightened:
            break

        tightened_edges = torch.as_tensor(tightened, device=device, dtype=torch.long)
        stats.tightened_edges += int(tightened_edges.numel())
        new_edges = propagate_and_backfill(triangle_index, candidate_mask, tightened_edges)
        stats.backfilled_edges += int(new_edges.numel())

    return stats
