from __future__ import annotations

from dataclasses import dataclass

import torch

from ._compat import device
from .triangle_index import TriangleIndex


@dataclass
class SupportBounds:
    smin: int
    smax: int


def estimate_support_bounds_for_edge(
    triangle_index: TriangleIndex,
    edge_id: int,
    k: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> SupportBounds:
    smin, smax = estimate_support_bounds_for_edges(
        triangle_index=triangle_index,
        edge_ids=torch.tensor([edge_id], device=device, dtype=torch.long),
        k_values=torch.tensor([k], device=device, dtype=torch.long),
        lower=lower,
        upper=upper,
    )
    return SupportBounds(smin=int(smin[0].item()), smax=int(smax[0].item()))


def estimate_support_bounds_for_packed_edges(
    *,
    source_local: torch.Tensor,
    left_edge: torch.Tensor,
    right_edge: torch.Tensor,
    edge_count: int,
    k_values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    smin = torch.zeros((edge_count,), device=device, dtype=torch.long)
    smax = torch.zeros((edge_count,), device=device, dtype=torch.long)
    if source_local.numel() == 0 or edge_count == 0:
        return smin, smax

    source_k = k_values[source_local]
    smin_mask = (lower[left_edge] >= source_k) & (lower[right_edge] >= source_k)
    smax_mask = (upper[left_edge] >= source_k) & (upper[right_edge] >= source_k)
    if torch.any(smin_mask):
        smin = torch.bincount(source_local[smin_mask], minlength=edge_count)
    if torch.any(smax_mask):
        smax = torch.bincount(source_local[smax_mask], minlength=edge_count)
    return smin, smax


def estimate_support_bounds_for_edges(
    triangle_index: TriangleIndex,
    edge_ids: torch.Tensor,
    k_values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_ids = edge_ids.to(device=device, dtype=torch.long)
    k_values = k_values.to(device=device, dtype=torch.long)
    if edge_ids.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty

    order = torch.argsort(edge_ids, stable=True)
    sorted_edge_ids = edge_ids[order]
    sorted_k_values = k_values[order]

    if sorted_edge_ids.numel() > 1 and torch.any(sorted_edge_ids[1:] == sorted_edge_ids[:-1]):
        # Current call sites pass unique edge ids. Keep a small fallback for safety.
        smin = torch.zeros((edge_ids.numel(),), device=device, dtype=torch.long)
        smax = torch.zeros((edge_ids.numel(),), device=device, dtype=torch.long)
        for local_idx, edge_id in enumerate(edge_ids.tolist()):
            packed_edge = torch.tensor([edge_id], device=device, dtype=torch.long)
            _, left_edge, right_edge, _ = triangle_index.pack(packed_edge)
            if left_edge.numel() == 0:
                continue
            k = k_values[local_idx]
            smin[local_idx] = torch.count_nonzero((lower[left_edge] >= k) & (lower[right_edge] >= k))
            smax[local_idx] = torch.count_nonzero((upper[left_edge] >= k) & (upper[right_edge] >= k))
        return smin, smax

    _, left_edge, right_edge, source_local = triangle_index.pack(sorted_edge_ids)
    if source_local.numel() == 0:
        smin_sorted = torch.zeros((sorted_edge_ids.numel(),), device=device, dtype=torch.long)
        smax_sorted = torch.zeros((sorted_edge_ids.numel(),), device=device, dtype=torch.long)
        smin = torch.empty_like(smin_sorted)
        smax = torch.empty_like(smax_sorted)
        smin[order] = smin_sorted
        smax[order] = smax_sorted
        return smin, smax

    smin_sorted, smax_sorted = estimate_support_bounds_for_packed_edges(
        source_local=source_local,
        left_edge=left_edge,
        right_edge=right_edge,
        edge_count=sorted_edge_ids.numel(),
        k_values=sorted_k_values,
        lower=lower,
        upper=upper,
    )

    smin = torch.empty_like(smin_sorted)
    smax = torch.empty_like(smax_sorted)
    smin[order] = smin_sorted
    smax[order] = smax_sorted
    return smin, smax
