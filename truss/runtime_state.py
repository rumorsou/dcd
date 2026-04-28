from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch

try:
    from .CSRGraph4 import edgelist_to_CSR, max_vertex
    from .updated_graph import build_edge_src_index, build_reverse_csr
except ImportError:
    from CSRGraph4 import edgelist_to_CSR, max_vertex
    from updated_graph import build_edge_src_index, build_reverse_csr


def _as_numpy_edges(filename: str) -> np.ndarray:
    array = np.loadtxt(filename, dtype=np.int64)
    array = np.atleast_2d(array)
    if array.shape[1] < 2:
        raise ValueError("Edge file must contain at least two columns.")
    return array[:, :2]


def _canonicalize_numpy_edges(edge_starts: np.ndarray, edge_ends: np.ndarray, dataset_type: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if dataset_type == 0:
        mask = edge_starts != edge_ends
    else:
        mask = edge_starts < edge_ends
    edge_starts = edge_starts[mask]
    edge_ends = edge_ends[mask]

    if edge_starts.size == 0:
        return edge_starts.astype(np.int64, copy=False), edge_ends.astype(np.int64, copy=False)

    swap_mask = edge_starts > edge_ends
    if np.any(swap_mask):
        temp = edge_starts[swap_mask].copy()
        edge_starts[swap_mask] = edge_ends[swap_mask]
        edge_ends[swap_mask] = temp

    edges = np.unique(np.stack((edge_starts, edge_ends), axis=1), axis=0)
    return edges[:, 0].astype(np.int64, copy=False), edges[:, 1].astype(np.int64, copy=False)


def read_base_graph_txt(filename: str, dataset_type: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = _as_numpy_edges(filename)
    edge_starts, edge_ends = _canonicalize_numpy_edges(array[:, 0], array[:, 1], dataset_type)

    if edge_starts.size == 0:
        max_vertex.value = -1
        return edge_starts.astype(np.int32), edge_ends.astype(np.int32), np.empty((0,), dtype=np.int64)

    vertices = np.unique(np.concatenate((edge_starts, edge_ends), axis=0))
    max_vertex.value = int(vertices.size - 1)
    mapped_src = np.searchsorted(vertices, edge_starts).astype(np.int32, copy=False)
    mapped_dst = np.searchsorted(vertices, edge_ends).astype(np.int32, copy=False)
    return mapped_src, mapped_dst, vertices.astype(np.int64, copy=False)


def read_update_edge_txt(
    filename: str,
    vertex_ids: Union[np.ndarray, torch.Tensor],
    dataset_type: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    array = _as_numpy_edges(filename)
    edge_starts, edge_ends = _canonicalize_numpy_edges(array[:, 0], array[:, 1], dataset_type)
    if edge_starts.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    if isinstance(vertex_ids, torch.Tensor):
        vertices = vertex_ids.detach().cpu().numpy()
    else:
        vertices = vertex_ids

    src_pos = np.searchsorted(vertices, edge_starts)
    dst_pos = np.searchsorted(vertices, edge_ends)
    src_valid = (src_pos < vertices.size) & (vertices[src_pos] == edge_starts)
    dst_valid = (dst_pos < vertices.size) & (vertices[dst_pos] == edge_ends)
    valid = src_valid & dst_valid
    if not np.all(valid):
        missing_edges = np.stack((edge_starts[~valid], edge_ends[~valid]), axis=1)
        raise ValueError(f"Update file contains unseen vertices, examples: {missing_edges[:3].tolist()}")

    mapped_src = src_pos.astype(np.int32, copy=False)
    mapped_dst = dst_pos.astype(np.int32, copy=False)
    return mapped_src, mapped_dst


@dataclass
class TensorTrussState:
    row_ptr: torch.Tensor
    columns: torch.Tensor
    rev_row_ptr: torch.Tensor
    rev_columns: torch.Tensor
    rev_edge_ids: torch.Tensor
    truss: torch.Tensor
    vertex_ids: torch.Tensor
    _edge_src_index: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _edge_codes: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _sorted_vertex_ids: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _sorted_vertex_local: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _bidirectional_view: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.row_ptr = self.row_ptr.to(dtype=torch.long)
        self.columns = self.columns.to(dtype=torch.int32)
        self.rev_row_ptr = self.rev_row_ptr.to(dtype=torch.long)
        self.rev_columns = self.rev_columns.to(dtype=torch.int32)
        self.rev_edge_ids = self.rev_edge_ids.to(dtype=torch.int32)
        self.truss = self.truss.to(dtype=torch.int32)
        self.vertex_ids = self.vertex_ids.to(dtype=torch.int64)
        self.invalidate_caches()

    @classmethod
    def from_csr(
        cls,
        row_ptr: torch.Tensor,
        columns: torch.Tensor,
        truss: torch.Tensor,
        vertex_ids: torch.Tensor,
    ) -> "TensorTrussState":
        rev_row_ptr, rev_columns, rev_edge_ids = build_reverse_csr(row_ptr, columns)
        return cls(row_ptr, columns, rev_row_ptr, rev_columns, rev_edge_ids, truss, vertex_ids)

    @classmethod
    def load(
        cls,
        snapshot_dir: Union[str, Path],
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> "TensorTrussState":
        path = Path(snapshot_dir)
        kwargs = {}
        if map_location is not None:
            kwargs["map_location"] = map_location
        row_ptr = torch.load(path / "row_ptr.pt", **kwargs)
        columns = torch.load(path / "columns.pt", **kwargs)
        rev_row_ptr = torch.load(path / "rev_row_ptr.pt", **kwargs)
        rev_columns = torch.load(path / "rev_columns.pt", **kwargs)
        rev_edge_ids = torch.load(path / "rev_edge_ids.pt", **kwargs)
        truss = torch.load(path / "truss.pt", **kwargs)
        vertex_ids = torch.load(path / "vertex_ids.pt", **kwargs)
        return cls(row_ptr, columns, rev_row_ptr, rev_columns, rev_edge_ids, truss, vertex_ids)

    @property
    def num_vertices(self) -> int:
        return int(self.row_ptr.size(0) - 1)

    @property
    def num_edges(self) -> int:
        return int(self.columns.numel())

    @property
    def device(self) -> torch.device:
        return self.row_ptr.device

    def clone(self) -> "TensorTrussState":
        return TensorTrussState(
            self.row_ptr.clone(),
            self.columns.clone(),
            self.rev_row_ptr.clone(),
            self.rev_columns.clone(),
            self.rev_edge_ids.clone(),
            self.truss.clone(),
            self.vertex_ids.clone(),
        )

    def to(self, device: Union[str, torch.device]) -> "TensorTrussState":
        target = torch.device(device)
        return TensorTrussState(
            self.row_ptr.to(target),
            self.columns.to(target),
            self.rev_row_ptr.to(target),
            self.rev_columns.to(target),
            self.rev_edge_ids.to(target),
            self.truss.to(target),
            self.vertex_ids.to(target),
        )

    def invalidate_caches(self) -> None:
        self._edge_src_index = None
        self._edge_codes = None
        self._sorted_vertex_ids = None
        self._sorted_vertex_local = None
        self._bidirectional_view = None

    def save(self, snapshot_dir: Union[str, Path]) -> None:
        path = Path(snapshot_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"version": 1}, path / "meta.pt")
        torch.save(self.row_ptr.cpu(), path / "row_ptr.pt")
        torch.save(self.columns.cpu(), path / "columns.pt")
        torch.save(self.rev_row_ptr.cpu(), path / "rev_row_ptr.pt")
        torch.save(self.rev_columns.cpu(), path / "rev_columns.pt")
        torch.save(self.rev_edge_ids.cpu(), path / "rev_edge_ids.pt")
        torch.save(self.truss.cpu(), path / "truss.pt")
        torch.save(self.vertex_ids.cpu(), path / "vertex_ids.pt")

    def refresh_graph(self, row_ptr: torch.Tensor, columns: torch.Tensor, truss: torch.Tensor) -> None:
        self.row_ptr = row_ptr.to(dtype=torch.long)
        self.columns = columns.to(dtype=torch.int32)
        self.truss = truss.to(dtype=torch.int32)
        self.rev_row_ptr, self.rev_columns, self.rev_edge_ids = build_reverse_csr(self.row_ptr, self.columns)
        self.invalidate_caches()

    def edge_src_index(self) -> torch.Tensor:
        if self._edge_src_index is None:
            self._edge_src_index = build_edge_src_index(self.row_ptr)
        return self._edge_src_index

    def edge_codes(self) -> torch.Tensor:
        if self._edge_codes is None:
            src = self.edge_src_index()
            self._edge_codes = src.to(torch.long) * self.num_vertices + self.columns.to(torch.long)
        return self._edge_codes

    def sorted_vertex_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._sorted_vertex_ids is None or self._sorted_vertex_local is None:
            sorted_ids, perm = torch.sort(self.vertex_ids)
            self._sorted_vertex_ids = sorted_ids
            self._sorted_vertex_local = perm.to(torch.long)
        return self._sorted_vertex_ids, self._sorted_vertex_local

    def map_vertex_ids(self, raw_vertex_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_vertex_ids = raw_vertex_ids.to(device=self.vertex_ids.device, dtype=torch.int64)
        if raw_vertex_ids.numel() == 0:
            empty = torch.empty((0,), device=self.vertex_ids.device, dtype=torch.long)
            return empty, torch.empty((0,), device=self.vertex_ids.device, dtype=torch.bool)

        sorted_ids, sorted_local = self.sorted_vertex_index()
        pos = torch.searchsorted(sorted_ids, raw_vertex_ids)
        valid = pos < sorted_ids.numel()
        if valid.any():
            valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
            valid[valid_idx] = sorted_ids[pos[valid_idx]] == raw_vertex_ids[valid_idx]
        mapped = torch.empty((0,), device=self.vertex_ids.device, dtype=torch.long)
        if valid.any():
            mapped = sorted_local[pos[valid]]
        return mapped, valid

    def ensure_vertices(self, raw_vertex_ids: torch.Tensor) -> torch.Tensor:
        raw_vertex_ids = raw_vertex_ids.to(device=self.vertex_ids.device, dtype=torch.int64)
        if raw_vertex_ids.numel() == 0:
            return torch.empty((0,), device=self.vertex_ids.device, dtype=torch.long)

        _, valid = self.map_vertex_ids(raw_vertex_ids)
        if not torch.all(valid):
            missing = torch.unique(raw_vertex_ids[~valid], sorted=True)
            self.vertex_ids = torch.cat((self.vertex_ids, missing))
            tail = self.row_ptr[-1]
            extra_ptr = torch.full((missing.numel(),), tail, device=self.row_ptr.device, dtype=torch.long)
            self.row_ptr = torch.cat((self.row_ptr, extra_ptr))
            self.rev_row_ptr = torch.cat((self.rev_row_ptr, extra_ptr))
            self.invalidate_caches()
        mapped, valid = self.map_vertex_ids(raw_vertex_ids)
        if not torch.all(valid):
            raise RuntimeError("Failed to map ensured vertices.")
        return mapped

    def get_bidirectional_view(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._bidirectional_view is None:
            fwd_deg = self.row_ptr[1:] - self.row_ptr[:-1]
            rev_deg = self.rev_row_ptr[1:] - self.rev_row_ptr[:-1]
            deg = fwd_deg + rev_deg
            bi_row_ptr = torch.cat(
                (
                    torch.tensor([0], device=self.device, dtype=torch.long),
                    deg.cumsum(0),
                )
            )

            bi_columns = torch.empty((self.num_edges * 2,), device=self.device, dtype=torch.int32)
            bi_edge_ids = torch.empty((self.num_edges * 2,), device=self.device, dtype=torch.int32)
            rev_sizes = rev_deg.to(torch.long)
            fwd_sizes = fwd_deg.to(torch.long)
            rev_ptr = torch.cat(
                (
                    torch.tensor([0], device=self.device, dtype=torch.long),
                    rev_sizes.cumsum(0),
                )
            )
            fwd_ptr = torch.cat(
                (
                    torch.tensor([0], device=self.device, dtype=torch.long),
                    fwd_sizes.cumsum(0),
                )
            )
            if self.rev_columns.numel() > 0:
                rev_vertices = torch.repeat_interleave(
                    torch.arange(self.num_vertices, device=self.device, dtype=torch.long),
                    rev_sizes,
                )
                rev_rank = torch.arange(self.rev_columns.numel(), device=self.device, dtype=torch.long) - \
                    torch.repeat_interleave(rev_ptr[:-1], rev_sizes)
                rev_pos = bi_row_ptr[:-1][rev_vertices] + rev_rank
                bi_columns[rev_pos] = self.rev_columns
                bi_edge_ids[rev_pos] = self.rev_edge_ids
            if self.columns.numel() > 0:
                fwd_vertices = torch.repeat_interleave(
                    torch.arange(self.num_vertices, device=self.device, dtype=torch.long),
                    fwd_sizes,
                )
                fwd_rank = torch.arange(self.columns.numel(), device=self.device, dtype=torch.long) - \
                    torch.repeat_interleave(fwd_ptr[:-1], fwd_sizes)
                fwd_pos = bi_row_ptr[:-1][fwd_vertices] + rev_deg[fwd_vertices].to(torch.long) + fwd_rank
                bi_columns[fwd_pos] = self.columns
                bi_edge_ids[fwd_pos] = torch.arange(self.num_edges, device=self.device, dtype=torch.int32)
            self._bidirectional_view = (bi_row_ptr, bi_columns, bi_edge_ids)
        return self._bidirectional_view
