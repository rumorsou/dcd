from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .csr import FullCSR, build_canonical_csr, build_full_csr, canonicalize_edge_pairs, csr_edge_pairs
from .static import decompose_from_csr


def _as_long_tensor(values: torch.Tensor | np.ndarray | Iterable[int], device: torch.device) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.long)
    if isinstance(values, np.ndarray):
        return torch.as_tensor(values, device=device, dtype=torch.long)
    return torch.as_tensor(list(values), device=device, dtype=torch.long)


@dataclass
class DCDState:
    rowptr: torch.Tensor
    col: torch.Tensor
    edge_id_of_col: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    edge_code: torch.Tensor
    tau: torch.Tensor
    vertex_ids: torch.Tensor
    tri_ptr: torch.Tensor | None = None
    other_edges: torch.Tensor | None = None
    third_vertex: torch.Tensor | None = None
    _sorted_vertex_ids: torch.Tensor | None = field(default=None, init=False, repr=False)
    _sorted_vertex_local: torch.Tensor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.rowptr = self.rowptr.to(dtype=torch.long)
        self.col = self.col.to(dtype=torch.int32)
        self.edge_id_of_col = self.edge_id_of_col.to(dtype=torch.int32)
        self.edge_src = self.edge_src.to(dtype=torch.int32)
        self.edge_dst = self.edge_dst.to(dtype=torch.int32)
        self.edge_code = self.edge_code.to(dtype=torch.long)
        self.tau = self.tau.to(dtype=torch.int32)
        self.vertex_ids = self.vertex_ids.to(dtype=torch.int64)
        if self.tri_ptr is not None:
            self.tri_ptr = self.tri_ptr.to(dtype=torch.long)
        if self.other_edges is not None:
            self.other_edges = self.other_edges.to(dtype=torch.int32)
        if self.third_vertex is not None:
            self.third_vertex = self.third_vertex.to(dtype=torch.int32)

    @classmethod
    def from_edge_pairs(
        cls,
        edge_pairs: torch.Tensor | np.ndarray,
        *,
        vertex_ids: torch.Tensor | np.ndarray | None = None,
        tau: torch.Tensor | None = None,
        device: str | torch.device | None = None,
        relabel: bool = True,
    ) -> "DCDState":
        target = torch.device(device) if device is not None else torch.device("cpu")
        edges = _as_long_tensor(edge_pairs, target).reshape(-1, 2)
        if vertex_ids is None:
            if edges.numel() == 0:
                labels = torch.empty((0,), device=target, dtype=torch.int64)
                local_edges = torch.empty((0, 2), device=target, dtype=torch.long)
            elif relabel:
                labels = torch.unique(edges.reshape(-1).to(torch.int64), sorted=True)
                local_edges = torch.searchsorted(labels, edges.to(torch.int64)).to(torch.long)
            else:
                n = int(edges.max().item()) + 1
                labels = torch.arange(n, device=target, dtype=torch.int64)
                local_edges = edges.to(torch.long)
        else:
            labels = _as_long_tensor(vertex_ids, target).to(torch.int64)
            local_edges = edges.to(torch.long)
        return cls.from_local_edge_pairs(local_edges, labels, tau=tau, device=target)

    @classmethod
    def from_local_edge_pairs(
        cls,
        edge_pairs: torch.Tensor,
        vertex_ids: torch.Tensor,
        *,
        tau: torch.Tensor | None = None,
        device: str | torch.device | None = None,
    ) -> "DCDState":
        target = torch.device(device) if device is not None else edge_pairs.device
        labels = vertex_ids.to(device=target, dtype=torch.int64)
        canonical = build_canonical_csr(edge_pairs.to(target), int(labels.numel()))
        full = build_full_csr(canonical.edge_src, canonical.edge_dst, int(labels.numel()))
        if tau is None:
            tau = decompose_from_csr(canonical.rowptr.cpu(), canonical.col.cpu()).to(target)
        else:
            tau = tau.to(device=target, dtype=torch.int32)
        return cls(
            rowptr=full.rowptr,
            col=full.col,
            edge_id_of_col=full.edge_id_of_col,
            edge_src=canonical.edge_src.to(torch.int32),
            edge_dst=canonical.edge_dst.to(torch.int32),
            edge_code=canonical.edge_code,
            tau=tau,
            vertex_ids=labels,
        )

    @property
    def device(self) -> torch.device:
        return self.rowptr.device

    @property
    def num_vertices(self) -> int:
        return int(self.vertex_ids.numel())

    @property
    def num_edges(self) -> int:
        return int(self.edge_src.numel())

    def clone(self) -> "DCDState":
        return DCDState(
            rowptr=self.rowptr.clone(),
            col=self.col.clone(),
            edge_id_of_col=self.edge_id_of_col.clone(),
            edge_src=self.edge_src.clone(),
            edge_dst=self.edge_dst.clone(),
            edge_code=self.edge_code.clone(),
            tau=self.tau.clone(),
            vertex_ids=self.vertex_ids.clone(),
            tri_ptr=None if self.tri_ptr is None else self.tri_ptr.clone(),
            other_edges=None if self.other_edges is None else self.other_edges.clone(),
            third_vertex=None if self.third_vertex is None else self.third_vertex.clone(),
        )

    def to(self, device: str | torch.device) -> "DCDState":
        target = torch.device(device)
        return DCDState(
            rowptr=self.rowptr.to(target),
            col=self.col.to(target),
            edge_id_of_col=self.edge_id_of_col.to(target),
            edge_src=self.edge_src.to(target),
            edge_dst=self.edge_dst.to(target),
            edge_code=self.edge_code.to(target),
            tau=self.tau.to(target),
            vertex_ids=self.vertex_ids.to(target),
            tri_ptr=None if self.tri_ptr is None else self.tri_ptr.to(target),
            other_edges=None if self.other_edges is None else self.other_edges.to(target),
            third_vertex=None if self.third_vertex is None else self.third_vertex.to(target),
        )

    def canonical_edge_pairs(self) -> torch.Tensor:
        if self.num_edges == 0:
            return torch.empty((0, 2), device=self.device, dtype=torch.long)
        return torch.stack((self.edge_src.to(torch.long), self.edge_dst.to(torch.long)), dim=1)

    def _sorted_vertex_index(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._sorted_vertex_ids is None or self._sorted_vertex_local is None:
            sorted_ids, perm = torch.sort(self.vertex_ids)
            self._sorted_vertex_ids = sorted_ids
            self._sorted_vertex_local = perm.to(torch.long)
        return self._sorted_vertex_ids, self._sorted_vertex_local

    def map_vertex_ids(self, raw_vertex_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = raw_vertex_ids.to(device=self.device, dtype=torch.int64)
        if raw.numel() == 0:
            empty = torch.empty((0,), device=self.device, dtype=torch.long)
            return empty, torch.empty((0,), device=self.device, dtype=torch.bool)
        sorted_ids, sorted_local = self._sorted_vertex_index()
        pos = torch.searchsorted(sorted_ids, raw)
        valid = pos < sorted_ids.numel()
        if torch.any(valid):
            valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
            valid[valid_idx] = sorted_ids[pos[valid_idx]] == raw[valid_idx]
        mapped = torch.empty((0,), device=self.device, dtype=torch.long)
        if torch.any(valid):
            mapped = sorted_local[pos[valid]]
        return mapped, valid

    def with_vertices(self, raw_vertex_ids: torch.Tensor) -> "DCDState":
        raw = raw_vertex_ids.to(device=self.device, dtype=torch.int64)
        if raw.numel() == 0:
            return self
        _, valid = self.map_vertex_ids(raw)
        if torch.all(valid):
            return self
        missing = torch.unique(raw[~valid], sorted=True)
        next_state = self.clone()
        next_state.vertex_ids = torch.cat((next_state.vertex_ids, missing))
        full = build_full_csr(next_state.edge_src.to(torch.long), next_state.edge_dst.to(torch.long), int(next_state.vertex_ids.numel()))
        next_state.rowptr = full.rowptr
        next_state.col = full.col
        next_state.edge_id_of_col = full.edge_id_of_col
        next_state.edge_code = next_state.edge_src.to(torch.long) * next_state.num_vertices + next_state.edge_dst.to(torch.long)
        next_state._sorted_vertex_ids = None
        next_state._sorted_vertex_local = None
        return next_state

    def edge_ids_from_pairs(self, local_pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pairs = canonicalize_edge_pairs(local_pairs.to(self.device), num_vertices=self.num_vertices)
        if pairs.numel() == 0:
            empty = torch.empty((0,), device=self.device, dtype=torch.long)
            return empty, torch.empty((0,), device=self.device, dtype=torch.bool)
        code = pairs[:, 0] * self.num_vertices + pairs[:, 1]
        pos = torch.searchsorted(self.edge_code, code)
        valid = pos < self.edge_code.numel()
        if torch.any(valid):
            valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
            valid[valid_idx] = self.edge_code[pos[valid_idx]] == code[valid_idx]
        return pos[valid], valid

    def save(self, snapshot_dir: str | Path) -> None:
        path = Path(snapshot_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"version": 1, "has_triangles": self.tri_ptr is not None}, path / "meta.pt")
        for name in ("rowptr", "col", "edge_id_of_col", "edge_src", "edge_dst", "edge_code", "tau", "vertex_ids"):
            torch.save(getattr(self, name).cpu(), path / f"{name}.pt")
        if self.tri_ptr is not None:
            torch.save(self.tri_ptr.cpu(), path / "tri_ptr.pt")
            torch.save(self.other_edges.cpu(), path / "other_edges.pt")
            if self.third_vertex is not None:
                torch.save(self.third_vertex.cpu(), path / "third_vertex.pt")

    @classmethod
    def load(cls, snapshot_dir: str | Path, *, map_location: str | torch.device | None = None) -> "DCDState":
        path = Path(snapshot_dir)
        kwargs = {"map_location": map_location} if map_location is not None else {}
        tensors = {
            name: torch.load(path / f"{name}.pt", **kwargs)
            for name in ("rowptr", "col", "edge_id_of_col", "edge_src", "edge_dst", "edge_code", "tau", "vertex_ids")
        }
        tri_ptr = torch.load(path / "tri_ptr.pt", **kwargs) if (path / "tri_ptr.pt").exists() else None
        other_edges = torch.load(path / "other_edges.pt", **kwargs) if (path / "other_edges.pt").exists() else None
        third_vertex = torch.load(path / "third_vertex.pt", **kwargs) if (path / "third_vertex.pt").exists() else None
        return cls(**tensors, tri_ptr=tri_ptr, other_edges=other_edges, third_vertex=third_vertex)


def rebuild_state_with_edges(base: DCDState, edge_pairs: torch.Tensor, tau: torch.Tensor | None = None) -> DCDState:
    return DCDState.from_local_edge_pairs(edge_pairs, base.vertex_ids, tau=tau, device=base.device)
