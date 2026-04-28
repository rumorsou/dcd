from __future__ import annotations

from dataclasses import dataclass

import torch

from .triangle_index import TrianglePack


@dataclass
class WitnessCache:
    edge_ids: torch.Tensor
    hard_ptr: torch.Tensor
    hard_witness: torch.Tensor
    soft_ptr: torch.Tensor
    soft_witness: torch.Tensor

    @classmethod
    def empty(cls, device: torch.device) -> "WitnessCache":
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return cls(empty, torch.zeros((1,), device=device, dtype=torch.long), torch.empty((0, 2), device=device, dtype=torch.int32), torch.zeros((1,), device=device, dtype=torch.long), torch.empty((0, 2), device=device, dtype=torch.int32))


def estimate_support_bounds(
    pack: TrianglePack,
    k_values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_count = int(pack.edge_ids.numel())
    smin = torch.zeros((edge_count,), device=pack.edge_ids.device, dtype=torch.long)
    smax = torch.zeros((edge_count,), device=pack.edge_ids.device, dtype=torch.long)
    if pack.other_edges.numel() == 0 or edge_count == 0:
        return smin, smax
    counts = pack.tri_ptr[1:] - pack.tri_ptr[:-1]
    owner = torch.repeat_interleave(torch.arange(edge_count, device=pack.edge_ids.device, dtype=torch.long), counts)
    other = pack.other_edges.to(torch.long)
    k = k_values.to(device=pack.edge_ids.device, dtype=torch.long)[owner]
    hard = (lower[other[:, 0]] >= k) & (lower[other[:, 1]] >= k)
    soft = (upper[other[:, 0]] >= k) & (upper[other[:, 1]] >= k)
    if torch.any(hard):
        smin = torch.bincount(owner[hard], minlength=edge_count)
    if torch.any(soft):
        smax = torch.bincount(owner[soft], minlength=edge_count)
    return smin, smax


def build_witness_cache(
    pack: TrianglePack,
    k_values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> WitnessCache:
    if pack.other_edges.numel() == 0:
        return WitnessCache.empty(pack.edge_ids.device)
    counts = pack.tri_ptr[1:] - pack.tri_ptr[:-1]
    owner = torch.repeat_interleave(torch.arange(pack.edge_ids.numel(), device=pack.edge_ids.device, dtype=torch.long), counts)
    other = pack.other_edges.to(torch.long)
    k = k_values.to(device=pack.edge_ids.device, dtype=torch.long)[owner]
    hard_mask = (lower[other[:, 0]] >= k) & (lower[other[:, 1]] >= k)
    soft_mask = (upper[other[:, 0]] >= k) & (upper[other[:, 1]] >= k) & ~hard_mask

    def _pack(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.any(mask):
            return torch.zeros((pack.edge_ids.numel() + 1,), device=pack.edge_ids.device, dtype=torch.long), torch.empty((0, 2), device=pack.edge_ids.device, dtype=torch.int32)
        local_owner = owner[mask]
        witness = pack.other_edges[mask]
        order = torch.argsort(local_owner, stable=True)
        local_owner = local_owner[order]
        witness = witness[order]
        local_counts = torch.bincount(local_owner, minlength=pack.edge_ids.numel())
        ptr = torch.cat((torch.zeros((1,), device=pack.edge_ids.device, dtype=torch.long), local_counts.cumsum(0)))
        return ptr, witness.to(torch.int32)

    hard_ptr, hard_witness = _pack(hard_mask)
    soft_ptr, soft_witness = _pack(soft_mask)
    return WitnessCache(pack.edge_ids, hard_ptr, hard_witness, soft_ptr, soft_witness)
