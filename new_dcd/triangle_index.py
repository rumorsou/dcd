from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from .state import DCDState


def _expand_ranges(starts: torch.Tensor, ends: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = (ends - starts).to(torch.long)
    if sizes.numel() == 0 or int(sizes.sum().item()) == 0:
        empty = torch.empty((0,), device=starts.device, dtype=torch.long)
        return empty, sizes
    ptr = torch.cat((torch.zeros((1,), device=starts.device, dtype=torch.long), sizes.cumsum(0)))
    idx = torch.arange(int(ptr[-1].item()), device=starts.device, dtype=torch.long)
    idx = idx - torch.repeat_interleave(ptr[:-1] - starts.to(torch.long), sizes)
    return idx, sizes


@dataclass(frozen=True)
class TrianglePack:
    edge_ids: torch.Tensor
    tri_ptr: torch.Tensor
    other_edges: torch.Tensor
    third_vertex: torch.Tensor | None = None

    @property
    def record_count(self) -> int:
        return int(self.other_edges.size(0))


class EdgeTriangleIndex:
    def __init__(self, state: DCDState, *, triangle_budget: int | None = None):
        self.state = state
        self.triangle_budget = triangle_budget
        self._cache: dict[tuple[int, int, int, int] | bytes, TrianglePack] = {}

    def _cache_key(self, edge_ids: torch.Tensor) -> tuple[int, int, int, int] | bytes:
        if edge_ids.numel() == 0:
            return b""
        if edge_ids.is_contiguous() and edge_ids.numel() > 1:
            first = int(edge_ids[0].item())
            last = int(edge_ids[-1].item())
            if last >= first and edge_ids.numel() == last - first + 1:
                return ("range", first, last, int(edge_ids.numel()))
        return edge_ids.detach().to("cpu", dtype=torch.long).contiguous().numpy().tobytes()

    def collect(self, edge_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_ids = torch.unique(edge_ids.to(device=self.state.device, dtype=torch.long), sorted=True)
        if edge_ids.numel() == 0:
            empty = torch.empty((0,), device=self.state.device, dtype=torch.long)
            return empty, empty, empty, empty

        src = self.state.edge_src[edge_ids].to(torch.long)
        dst = self.state.edge_dst[edge_ids].to(torch.long)
        u_pos, u_sizes = _expand_ranges(self.state.rowptr[src], self.state.rowptr[src + 1])
        v_pos, v_sizes = _expand_ranges(self.state.rowptr[dst], self.state.rowptr[dst + 1])
        if u_pos.numel() == 0 or v_pos.numel() == 0:
            empty = torch.empty((0,), device=self.state.device, dtype=torch.long)
            return empty, empty, empty, empty

        local_u = torch.repeat_interleave(torch.arange(edge_ids.numel(), device=self.state.device, dtype=torch.long), u_sizes)
        local_v = torch.repeat_interleave(torch.arange(edge_ids.numel(), device=self.state.device, dtype=torch.long), v_sizes)
        u_codes = local_u * self.state.num_vertices + self.state.col[u_pos].to(torch.long)
        v_codes = local_v * self.state.num_vertices + self.state.col[v_pos].to(torch.long)

        sorted_u, u_order = torch.sort(u_codes, stable=True)
        v_codes = v_codes.contiguous()
        pos = torch.searchsorted(sorted_u, v_codes)
        matched = pos < sorted_u.numel()
        if torch.any(matched):
            idx = torch.nonzero(matched, as_tuple=False).flatten()
            matched[idx] = sorted_u[pos[idx]] == v_codes[idx]
        if not torch.any(matched):
            empty = torch.empty((0,), device=self.state.device, dtype=torch.long)
            return empty, empty, empty, empty

        matched_v = torch.nonzero(matched, as_tuple=False).flatten()
        matched_u_pos = u_pos[u_order[pos[matched_v]]].to(torch.long)
        matched_v_pos = v_pos[matched_v].to(torch.long)
        source_local = local_v[matched_v].to(torch.long)
        source_edge = edge_ids[source_local]
        left_edge = self.state.edge_id_of_col[matched_u_pos].to(torch.long)
        right_edge = self.state.edge_id_of_col[matched_v_pos].to(torch.long)
        third_vertex = self.state.col[matched_v_pos].to(torch.long)
        return source_edge, left_edge, right_edge, third_vertex

    def materialize(self, edge_ids: torch.Tensor) -> TrianglePack:
        edge_ids = torch.unique(edge_ids.to(device=self.state.device, dtype=torch.long), sorted=True)
        key = self._cache_key(edge_ids)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if edge_ids.numel() == 0:
            empty = torch.empty((0,), device=self.state.device, dtype=torch.long)
            pack = TrianglePack(edge_ids, torch.zeros((1,), device=self.state.device, dtype=torch.long), torch.empty((0, 2), device=self.state.device, dtype=torch.int32), empty)
            self._cache[key] = pack
            return pack

        source, left, right, third = self.collect(edge_ids)
        if source.numel() == 0:
            tri_ptr = torch.zeros((edge_ids.numel() + 1,), device=self.state.device, dtype=torch.long)
            pack = TrianglePack(edge_ids, tri_ptr, torch.empty((0, 2), device=self.state.device, dtype=torch.int32), torch.empty((0,), device=self.state.device, dtype=torch.int32))
            self._cache[key] = pack
            return pack

        source_local = torch.searchsorted(edge_ids, source.contiguous())
        order = torch.argsort(source_local, stable=True)
        source_local = source_local[order]
        other = torch.stack((left[order], right[order]), dim=1).to(torch.int32)
        third = third[order].to(torch.int32)
        counts = torch.bincount(source_local, minlength=edge_ids.numel())
        tri_ptr = torch.cat((torch.zeros((1,), device=self.state.device, dtype=torch.long), counts.cumsum(0)))
        pack = TrianglePack(edge_ids, tri_ptr, other, third)
        self._cache[key] = pack
        return pack

    def _edge_work(self, edge_ids: torch.Tensor) -> torch.Tensor:
        if edge_ids.numel() == 0:
            return torch.empty((0,), device=self.state.device, dtype=torch.long)
        degree = self.state.rowptr[1:] - self.state.rowptr[:-1]
        src = self.state.edge_src[edge_ids].to(torch.long)
        dst = self.state.edge_dst[edge_ids].to(torch.long)
        return (degree[src] + degree[dst]).to(torch.long)

    def stream(self, edge_ids: torch.Tensor, *, edge_budget: int) -> Iterator[TrianglePack]:
        edge_ids = torch.unique(edge_ids.to(device=self.state.device, dtype=torch.long), sorted=True)
        if edge_ids.numel() == 0:
            return
        budget = max(int(edge_budget), 1)
        work = self._edge_work(edge_ids)
        total_work = int(torch.sum(work).item()) if work.numel() > 0 else 0
        max_work = int(torch.max(work).item()) if work.numel() > 0 else 0
        if total_work <= budget * 1024 and max_work <= budget * 16:
            for start in range(0, int(edge_ids.numel()), budget):
                yield self.materialize(edge_ids[start : start + budget])
            return

        order = torch.argsort(work, descending=True, stable=True)
        edge_ids = edge_ids[order]
        work = work[order]

        start = 0
        total_edges = int(edge_ids.numel())
        while start < total_edges:
            running = 0
            end = start
            while end < total_edges:
                next_work = int(work[end].item())
                if end > start and running + next_work > budget:
                    break
                running += next_work
                end += 1
            if end == start:
                end = start + 1
            yield self.materialize(torch.sort(edge_ids[start:end]).values)
            start = end

    def support_counts(self, edge_ids: torch.Tensor, *, edge_budget: int | None = None) -> torch.Tensor:
        edge_ids = torch.unique(edge_ids.to(device=self.state.device, dtype=torch.long), sorted=True)
        if edge_ids.numel() == 0:
            return torch.empty((0,), device=self.state.device, dtype=torch.long)
        if edge_budget is None:
            pack = self.materialize(edge_ids)
            return pack.tri_ptr[1:] - pack.tri_ptr[:-1]

        counts = torch.zeros((edge_ids.numel(),), device=self.state.device, dtype=torch.long)
        for pack in self.stream(edge_ids, edge_budget=edge_budget):
            pack_counts = pack.tri_ptr[1:] - pack.tri_ptr[:-1]
            pos = torch.searchsorted(edge_ids, pack.edge_ids)
            counts[pos] = pack_counts
        return counts

    def triangle_neighbors(self, edge_ids: torch.Tensor, *, edge_budget: int | None = None) -> torch.Tensor:
        edge_ids = torch.unique(edge_ids.to(device=self.state.device, dtype=torch.long), sorted=True)
        if edge_ids.numel() == 0:
            return edge_ids
        if edge_budget is None:
            pack = self.materialize(edge_ids)
            flat = pack.other_edges.reshape(-1).to(torch.long)
            return torch.unique(flat) if flat.numel() > 0 else torch.empty((0,), device=self.state.device, dtype=torch.long)
        parts = []
        for pack in self.stream(edge_ids, edge_budget=edge_budget):
            flat = pack.other_edges.reshape(-1).to(torch.long)
            if flat.numel() > 0:
                parts.append(flat)
        if not parts:
            return torch.empty((0,), device=self.state.device, dtype=torch.long)
        return torch.unique(torch.cat(parts))


def materialize_state_triangles(state: DCDState, *, triangle_budget: int | None = None) -> DCDState:
    index = EdgeTriangleIndex(state, triangle_budget=triangle_budget)
    all_edges = torch.arange(state.num_edges, device=state.device, dtype=torch.long)
    pack = index.materialize(all_edges)
    next_state = state.clone()
    next_state.tri_ptr = pack.tri_ptr
    next_state.other_edges = pack.other_edges
    next_state.third_vertex = pack.third_vertex
    return next_state
