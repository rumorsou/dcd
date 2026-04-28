from __future__ import annotations

from dataclasses import dataclass

import torch

from ._compat import _collect_edge_triangles, device
from .graph_state import GraphState


@dataclass
class ExplicitTriangleIndex:
    edge_ids: torch.Tensor
    tri_ptr: torch.Tensor
    tri_ids_flat: torch.Tensor
    left_edge_flat: torch.Tensor
    right_edge_flat: torch.Tensor
    edges_of_tri: torch.Tensor
    tri_local_edges: torch.Tensor
    incidence_sizes: torch.Tensor
    local_owner: torch.Tensor
    left_local: torch.Tensor
    right_local: torch.Tensor
    left_is_local: torch.Tensor
    right_is_local: torch.Tensor


def _expand_ranges(starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    sizes = ends - starts
    if sizes.numel() == 0 or int(sizes.sum().item()) == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    ptr = torch.cat((torch.tensor([0], device=device, dtype=torch.long), sizes.cumsum(0)))
    return torch.arange(int(ptr[-1].item()), device=device, dtype=torch.long) - torch.repeat_interleave(ptr[:-1] - starts, sizes)


class TriangleIndex:
    def __init__(self, state: GraphState):
        self.state = state
        self._triangle_pack_cache: dict[bytes, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self._explicit_cache: dict[bytes, ExplicitTriangleIndex] = {}

    def evolved(self, state: GraphState) -> "TriangleIndex":
        return TriangleIndex(state)

    def _cache_key(self, edge_ids: torch.Tensor) -> bytes:
        if edge_ids.numel() == 0:
            return b""
        return edge_ids.to(device="cpu", dtype=torch.long).contiguous().numpy().tobytes()

    def pack(self, edge_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_ids = torch.unique(edge_ids.to(device=device, dtype=torch.long))
        if edge_ids.numel() == 0:
            empty = torch.empty((0,), device=device, dtype=torch.long)
            return empty, empty, empty, empty
        edge_ids = torch.sort(edge_ids).values
        key = self._cache_key(edge_ids)
        cached = self._triangle_pack_cache.get(key)
        if cached is not None:
            return cached

        source_edge, left_edge, right_edge = _collect_edge_triangles(
            self.state.row_ptr,
            self.state.columns,
            edge_ids,
            bidirectional_view=self.state.bidirectional_view,
            edge_src_index=self.state.edge_src_index,
        )
        if source_edge.numel() == 0:
            source_local = torch.empty((0,), device=device, dtype=torch.long)
        else:
            local_positions = torch.searchsorted(edge_ids, source_edge)
            source_local = local_positions.to(torch.long)
        packed = (source_edge, left_edge, right_edge, source_local)
        self._triangle_pack_cache[key] = packed
        return packed

    def incident_triangle_count(self, edge_ids: torch.Tensor) -> torch.Tensor:
        edge_ids = torch.unique(edge_ids.to(device=device, dtype=torch.long))
        if edge_ids.numel() == 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        _, _, _, source_local = self.pack(edge_ids)
        if source_local.numel() == 0:
            return torch.zeros((edge_ids.numel(),), device=device, dtype=torch.long)
        return torch.bincount(source_local, minlength=edge_ids.numel())

    def triangle_neighbors(self, edge_ids: torch.Tensor) -> torch.Tensor:
        edge_ids = torch.unique(edge_ids.to(device=device, dtype=torch.long))
        if edge_ids.numel() == 0:
            return edge_ids
        _, left_edge, right_edge, _ = self.pack(edge_ids)
        if left_edge.numel() == 0 and right_edge.numel() == 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        return torch.unique(torch.cat((left_edge, right_edge)))

    def support_counts(self, edge_ids: torch.Tensor, available_mask: torch.Tensor) -> torch.Tensor:
        edge_ids = torch.unique(edge_ids.to(device=device, dtype=torch.long))
        if edge_ids.numel() == 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        source_edge, left_edge, right_edge, source_local = self.pack(edge_ids)
        if source_edge.numel() == 0:
            return torch.zeros((edge_ids.numel(),), device=device, dtype=torch.long)
        valid = available_mask[left_edge] & available_mask[right_edge]
        if not torch.any(valid):
            return torch.zeros((edge_ids.numel(),), device=device, dtype=torch.long)
        return torch.bincount(source_local[valid], minlength=edge_ids.numel())

    def materialize(self, edge_ids: torch.Tensor) -> ExplicitTriangleIndex:
        edge_ids = torch.unique(edge_ids.to(device=device, dtype=torch.long))
        edge_ids = torch.sort(edge_ids).values
        key = self._cache_key(edge_ids)
        cached = self._explicit_cache.get(key)
        if cached is not None:
            return cached

        source_edge, left_edge, right_edge, _ = self.pack(edge_ids)
        triangle_edges = (
            torch.stack((source_edge, left_edge, right_edge), dim=1)
            if source_edge.numel() > 0
            else torch.empty((0, 3), device=device, dtype=torch.long)
        )
        return self._build_explicit_from_triangle_edges(edge_ids, triangle_edges)

    def extend_materialize(
        self,
        explicit_index: ExplicitTriangleIndex | None,
        edge_ids: torch.Tensor,
    ) -> ExplicitTriangleIndex:
        edge_ids = torch.unique(edge_ids.to(device=device, dtype=torch.long))
        edge_ids = torch.sort(edge_ids).values
        if explicit_index is None:
            return self.materialize(edge_ids)
        if torch.equal(explicit_index.edge_ids, edge_ids):
            return explicit_index
        if edge_ids.numel() == 0:
            return self.materialize(edge_ids)
        new_edges = edge_ids[~torch.isin(edge_ids, explicit_index.edge_ids)]
        if new_edges.numel() == 0:
            return self.materialize(edge_ids)
        added_explicit = self.materialize(new_edges)
        if explicit_index.edges_of_tri.numel() == 0:
            merged_triangles = added_explicit.edges_of_tri
        elif added_explicit.edges_of_tri.numel() == 0:
            merged_triangles = explicit_index.edges_of_tri
        else:
            merged_triangles = torch.cat((explicit_index.edges_of_tri, added_explicit.edges_of_tri), dim=0)
        return self._build_explicit_from_triangle_edges(edge_ids, merged_triangles)

    def _build_explicit_from_triangle_edges(
        self,
        edge_ids: torch.Tensor,
        triangle_edges: torch.Tensor,
    ) -> ExplicitTriangleIndex:
        edge_ids = torch.unique(edge_ids.to(device=device, dtype=torch.long))
        edge_ids = torch.sort(edge_ids).values
        key = self._cache_key(edge_ids)
        cached = self._explicit_cache.get(key)
        if cached is not None:
            return cached
        if triangle_edges.numel() == 0:
            empty = torch.empty((0,), device=device, dtype=torch.long)
            explicit = ExplicitTriangleIndex(
                edge_ids=edge_ids,
                tri_ptr=torch.zeros((edge_ids.numel() + 1,), device=device, dtype=torch.long),
                tri_ids_flat=empty,
                left_edge_flat=empty,
                right_edge_flat=empty,
                edges_of_tri=torch.empty((0, 3), device=device, dtype=torch.long),
                tri_local_edges=torch.empty((0, 3), device=device, dtype=torch.long),
                incidence_sizes=empty,
                local_owner=empty,
                left_local=empty,
                right_local=empty,
                left_is_local=torch.empty((0,), device=device, dtype=torch.bool),
                right_is_local=torch.empty((0,), device=device, dtype=torch.bool),
            )
            self._explicit_cache[key] = explicit
            return explicit
        triangle_edges = triangle_edges.to(device=device, dtype=torch.long)
        triangle_edges = torch.sort(triangle_edges, dim=1).values
        triangle_edges = torch.unique(triangle_edges, dim=0)
        flat_edges = triangle_edges.reshape(-1)
        tri_ids = torch.repeat_interleave(torch.arange(triangle_edges.size(0), device=device, dtype=torch.long), 3)
        tri_pos = torch.arange(3, device=device, dtype=torch.long).repeat(triangle_edges.size(0))
        sorted_edges, order = torch.sort(flat_edges, stable=True)
        tri_ids_flat = tri_ids[order]
        sorted_pos = tri_pos[order]

        incidence_mask = torch.isin(sorted_edges, edge_ids)
        local_edges = sorted_edges[incidence_mask]
        local_tri_ids = tri_ids_flat[incidence_mask]
        local_pos = sorted_pos[incidence_mask]
        local_indices = torch.searchsorted(edge_ids, local_edges)
        counts = torch.bincount(local_indices, minlength=edge_ids.numel())
        tri_ptr = torch.cat((torch.tensor([0], device=device, dtype=torch.long), counts.cumsum(0)))
        left_edge_flat = triangle_edges[local_tri_ids, (local_pos + 1) % 3]
        right_edge_flat = triangle_edges[local_tri_ids, (local_pos + 2) % 3]
        incidence_sizes = tri_ptr[1:] - tri_ptr[:-1]
        local_owner = torch.repeat_interleave(
            torch.arange(edge_ids.numel(), device=device, dtype=torch.long),
            incidence_sizes,
        )
        left_pos = torch.searchsorted(edge_ids, left_edge_flat)
        right_pos = torch.searchsorted(edge_ids, right_edge_flat)
        left_is_local = left_pos < edge_ids.numel()
        right_is_local = right_pos < edge_ids.numel()
        if edge_ids.numel() > 0:
            left_valid_idx = torch.nonzero(left_is_local, as_tuple=False).flatten()
            if left_valid_idx.numel() > 0:
                left_is_local[left_valid_idx] = edge_ids[left_pos[left_valid_idx]] == left_edge_flat[left_valid_idx]
            right_valid_idx = torch.nonzero(right_is_local, as_tuple=False).flatten()
            if right_valid_idx.numel() > 0:
                right_is_local[right_valid_idx] = edge_ids[right_pos[right_valid_idx]] == right_edge_flat[right_valid_idx]
        tri_flat_pos = torch.searchsorted(edge_ids, flat_edges)
        tri_flat_is_local = tri_flat_pos < edge_ids.numel()
        if edge_ids.numel() > 0:
            tri_valid_idx = torch.nonzero(tri_flat_is_local, as_tuple=False).flatten()
            if tri_valid_idx.numel() > 0:
                tri_flat_is_local[tri_valid_idx] = edge_ids[tri_flat_pos[tri_valid_idx]] == flat_edges[tri_valid_idx]
        tri_local_flat = torch.full_like(flat_edges, -1)
        tri_local_flat[tri_flat_is_local] = tri_flat_pos[tri_flat_is_local]
        explicit = ExplicitTriangleIndex(
            edge_ids=edge_ids,
            tri_ptr=tri_ptr,
            tri_ids_flat=local_tri_ids,
            left_edge_flat=left_edge_flat,
            right_edge_flat=right_edge_flat,
            edges_of_tri=triangle_edges,
            tri_local_edges=tri_local_flat.reshape(-1, 3),
            incidence_sizes=incidence_sizes,
            local_owner=local_owner,
            left_local=left_pos.clamp(max=max(int(edge_ids.numel()) - 1, 0)),
            right_local=right_pos.clamp(max=max(int(edge_ids.numel()) - 1, 0)),
            left_is_local=left_is_local,
            right_is_local=right_is_local,
        )
        self._explicit_cache[key] = explicit
        return explicit

    def bfs_cone(self, seed_edge_ids: torch.Tensor) -> torch.Tensor:
        frontier = torch.unique(seed_edge_ids.to(device=device, dtype=torch.long))
        if frontier.numel() == 0:
            return frontier
        visited = torch.zeros((self.state.num_edges,), device=device, dtype=torch.bool)
        visited[frontier] = True
        while frontier.numel() > 0:
            neighbors = self.triangle_neighbors(frontier)
            if neighbors.numel() == 0:
                break
            next_frontier = neighbors[~visited[neighbors]]
            if next_frontier.numel() == 0:
                break
            visited[next_frontier] = True
            frontier = torch.unique(next_frontier)
        return torch.nonzero(visited, as_tuple=False).flatten()
