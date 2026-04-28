from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CanonicalCSR:
    rowptr: torch.Tensor
    col: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_code: torch.Tensor


@dataclass(frozen=True)
class FullCSR:
    rowptr: torch.Tensor
    col: torch.Tensor
    edge_id_of_col: torch.Tensor


def canonicalize_edge_pairs(edge_pairs: torch.Tensor, *, num_vertices: int | None = None) -> torch.Tensor:
    if edge_pairs.numel() == 0:
        return torch.empty((0, 2), device=edge_pairs.device, dtype=torch.long)
    edges = edge_pairs.reshape(-1, 2).to(dtype=torch.long)
    src = torch.minimum(edges[:, 0], edges[:, 1])
    dst = torch.maximum(edges[:, 0], edges[:, 1])
    keep = src != dst
    if num_vertices is not None:
        keep &= (src >= 0) & (dst >= 0) & (src < num_vertices) & (dst < num_vertices)
    if not torch.any(keep):
        return torch.empty((0, 2), device=edge_pairs.device, dtype=torch.long)
    src = src[keep]
    dst = dst[keep]
    n = int(num_vertices) if num_vertices is not None else int(torch.max(dst).item()) + 1
    code = src * n + dst
    code = torch.unique(code, sorted=True)
    return torch.stack((torch.div(code, n, rounding_mode="floor"), torch.remainder(code, n)), dim=1)


def edge_src_index(rowptr: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(rowptr.numel() - 1, device=rowptr.device, dtype=torch.long),
        rowptr[1:] - rowptr[:-1],
    )


def build_canonical_csr(edge_pairs: torch.Tensor, num_vertices: int) -> CanonicalCSR:
    device = edge_pairs.device
    edges = canonicalize_edge_pairs(edge_pairs, num_vertices=num_vertices)
    if edges.numel() == 0:
        rowptr = torch.zeros((num_vertices + 1,), device=device, dtype=torch.long)
        empty_i32 = torch.empty((0,), device=device, dtype=torch.int32)
        empty_long = torch.empty((0,), device=device, dtype=torch.long)
        return CanonicalCSR(rowptr, empty_i32, empty_long, empty_long, empty_long)

    src = edges[:, 0].to(torch.long)
    dst = edges[:, 1].to(torch.long)
    code = src * num_vertices + dst
    order = torch.argsort(code, stable=True)
    src = src[order]
    dst = dst[order]
    code = code[order]
    degree = torch.bincount(src, minlength=num_vertices)
    rowptr = torch.cat((torch.zeros((1,), device=device, dtype=torch.long), degree.cumsum(0)))
    return CanonicalCSR(rowptr, dst.to(torch.int32), src, dst, code)


def build_full_csr(edge_src: torch.Tensor, edge_dst: torch.Tensor, num_vertices: int) -> FullCSR:
    device = edge_src.device
    if edge_src.numel() == 0:
        rowptr = torch.zeros((num_vertices + 1,), device=device, dtype=torch.long)
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return FullCSR(rowptr, empty, empty)

    edge_ids = torch.arange(edge_src.numel(), device=device, dtype=torch.long)
    src = torch.cat((edge_src.to(torch.long), edge_dst.to(torch.long)))
    dst = torch.cat((edge_dst.to(torch.long), edge_src.to(torch.long)))
    ids = torch.cat((edge_ids, edge_ids))
    order = torch.argsort(src * num_vertices + dst, stable=True)
    src = src[order]
    dst = dst[order]
    ids = ids[order]
    degree = torch.bincount(src, minlength=num_vertices)
    rowptr = torch.cat((torch.zeros((1,), device=device, dtype=torch.long), degree.cumsum(0)))
    return FullCSR(rowptr, dst.to(torch.int32), ids.to(torch.int32))


def csr_edge_pairs(csr: CanonicalCSR) -> torch.Tensor:
    if csr.edge_src.numel() == 0:
        return torch.empty((0, 2), device=csr.rowptr.device, dtype=torch.long)
    return torch.stack((csr.edge_src.to(torch.long), csr.edge_dst.to(torch.long)), dim=1)
