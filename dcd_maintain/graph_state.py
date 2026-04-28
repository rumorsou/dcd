from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from ._compat import (
    _prepare_graph_runtime,
    _lookup_edge_ids,
    decompose_from_csr,
    device,
    insert_edges_csr,
    remove_edges_csr,
    tetree_edgelist_to_csr,
    tetree_max_vertex,
)


def _as_long_tensor(values: torch.Tensor | np.ndarray | Iterable[int]) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.long)
    return torch.as_tensor(list(values) if not isinstance(values, np.ndarray) else values, device=device, dtype=torch.long)


def _normalize_edge_pairs(edge_pairs: torch.Tensor) -> torch.Tensor:
    if edge_pairs.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    edge_pairs = edge_pairs.to(device=device, dtype=torch.long)
    keep_mask = edge_pairs[:, 0] != edge_pairs[:, 1]
    edge_pairs = edge_pairs[keep_mask]
    if edge_pairs.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    left = torch.minimum(edge_pairs[:, 0], edge_pairs[:, 1])
    right = torch.maximum(edge_pairs[:, 0], edge_pairs[:, 1])
    edge_pairs = torch.stack((left, right), dim=1)
    return torch.unique(edge_pairs, dim=0)


@dataclass
class GraphState:
    row_ptr: torch.Tensor
    columns: torch.Tensor
    tau: torch.Tensor
    vertex_labels: np.ndarray
    edge_code: torch.Tensor
    bidirectional_view: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    edge_src_index: torch.Tensor

    @classmethod
    def from_edge_pairs(
        cls,
        edge_pairs: torch.Tensor | np.ndarray,
        vertex_labels: np.ndarray,
        tau: torch.Tensor | None = None,
    ) -> "GraphState":
        if isinstance(edge_pairs, torch.Tensor):
            edge_pairs_np = edge_pairs.detach().cpu().numpy()
        else:
            edge_pairs_np = np.asarray(edge_pairs, dtype=np.int64)
        edge_pairs_np = np.atleast_2d(edge_pairs_np)

        if edge_pairs_np.size == 0:
            n = int(len(vertex_labels))
            row_ptr = torch.zeros((n + 1,), device=device, dtype=torch.long)
            columns = torch.empty((0,), device=device, dtype=torch.long)
        else:
            starts = edge_pairs_np[:, 0].astype(np.int32, copy=True)
            ends = edge_pairs_np[:, 1].astype(np.int32, copy=True)
            tetree_max_vertex.value = len(vertex_labels) - 1
            row_ptr_np, columns_np, _ = tetree_edgelist_to_csr(starts, ends, direct=True)
            row_ptr = torch.as_tensor(row_ptr_np, device=device, dtype=torch.long)
            columns = torch.as_tensor(columns_np, device=device, dtype=torch.long)
            expected_n = int(edge_pairs_np.max()) + 1
            if expected_n < len(vertex_labels):
                degrees = row_ptr[1:] - row_ptr[:-1]
                if row_ptr.numel() - 1 < len(vertex_labels):
                    pad = torch.zeros((len(vertex_labels) - (row_ptr.numel() - 1),), device=device, dtype=torch.long)
                    row_ptr = torch.cat((torch.tensor([0], device=device, dtype=torch.long), torch.cumsum(torch.cat((degrees, pad)), dim=0)))

        if tau is None:
            tau = decompose_from_csr(row_ptr, columns)
        else:
            tau = tau.to(device=device, dtype=torch.long)

        edge_code, bidirectional_view, edge_src_index = _prepare_graph_runtime(row_ptr, columns)
        return cls(
            row_ptr=row_ptr,
            columns=columns,
            tau=tau,
            vertex_labels=np.asarray(vertex_labels, dtype=np.int64),
            edge_code=edge_code,
            bidirectional_view=bidirectional_view,
            edge_src_index=edge_src_index,
        )

    @property
    def num_vertices(self) -> int:
        return int(self.row_ptr.numel() - 1)

    @property
    def num_edges(self) -> int:
        return int(self.columns.numel())

    def clone(self) -> "GraphState":
        return GraphState(
            row_ptr=self.row_ptr.clone(),
            columns=self.columns.clone(),
            tau=self.tau.clone(),
            vertex_labels=self.vertex_labels.copy(),
            edge_code=self.edge_code.clone(),
            bidirectional_view=tuple(part.clone() for part in self.bidirectional_view),
            edge_src_index=self.edge_src_index.clone(),
        )

    def edge_ids_from_pairs(self, edge_pairs: torch.Tensor) -> torch.Tensor:
        edge_pairs = _normalize_edge_pairs(edge_pairs)
        edge_ids, _ = _lookup_edge_ids(self.num_vertices, self.edge_code, edge_pairs)
        return edge_ids

    def edge_codes_from_pairs(self, edge_pairs: torch.Tensor) -> torch.Tensor:
        edge_pairs = _normalize_edge_pairs(edge_pairs)
        if edge_pairs.numel() == 0:
            return torch.empty((0,), device=device, dtype=torch.long)
        return edge_pairs[:, 0].to(torch.long) * self.num_vertices + edge_pairs[:, 1].to(torch.long)

    def edge_pairs(self) -> torch.Tensor:
        src = torch.repeat_interleave(
            torch.arange(self.num_vertices, device=device, dtype=torch.long),
            self.row_ptr[1:] - self.row_ptr[:-1],
        )
        return torch.stack((src, self.columns.to(torch.long)), dim=1)

    def incident_edge_pairs(self, vertex_ids: torch.Tensor) -> torch.Tensor:
        vertex_ids = torch.unique(vertex_ids.to(device=device, dtype=torch.long))
        if vertex_ids.numel() == 0:
            return torch.empty((0, 2), device=device, dtype=torch.long)
        starts = self.row_ptr[vertex_ids]
        ends = self.row_ptr[vertex_ids + 1]
        pieces = []
        for local_idx, vertex_id in enumerate(vertex_ids.tolist()):
            start = int(starts[local_idx].item())
            end = int(ends[local_idx].item())
            if start == end:
                continue
            neighbors = self.columns[start:end].to(torch.long)
            pairs = torch.stack(
                (
                    torch.full_like(neighbors, vertex_id, dtype=torch.long, device=device),
                    neighbors,
                ),
                dim=1,
            )
            pieces.append(pairs)
        if not pieces:
            return torch.empty((0, 2), device=device, dtype=torch.long)
        return _normalize_edge_pairs(torch.cat(pieces, dim=0))

    def with_inserted_edges(self, edge_pairs: torch.Tensor) -> "GraphState":
        edge_pairs = _normalize_edge_pairs(edge_pairs)
        if edge_pairs.numel() == 0:
            return self.clone()
        row_ptr, columns, tau, edge_code, bidirectional_view = insert_edges_csr(
            self.row_ptr,
            self.columns,
            edge_pairs[:, 0],
            edge_pairs[:, 1],
            self.tau,
            return_bidirectional_view=True,
        )
        edge_src_index = torch.repeat_interleave(
            torch.arange(row_ptr.size(0) - 1, device=device, dtype=torch.long),
            row_ptr[1:] - row_ptr[:-1],
        )
        return GraphState(
            row_ptr=row_ptr,
            columns=columns,
            tau=tau,
            vertex_labels=self.vertex_labels.copy(),
            edge_code=edge_code,
            bidirectional_view=bidirectional_view,
            edge_src_index=edge_src_index,
        )

    def with_removed_edges(self, edge_pairs: torch.Tensor) -> "GraphState":
        edge_pairs = _normalize_edge_pairs(edge_pairs)
        if edge_pairs.numel() == 0:
            return self.clone()
        row_ptr, columns, tau, edge_code, bidirectional_view = remove_edges_csr(
            self.row_ptr,
            self.columns,
            edge_pairs[:, 0],
            edge_pairs[:, 1],
            self.tau,
            return_bidirectional_view=True,
            bidirectional_view=self.bidirectional_view,
        )
        edge_src_index = torch.repeat_interleave(
            torch.arange(row_ptr.size(0) - 1, device=device, dtype=torch.long),
            row_ptr[1:] - row_ptr[:-1],
        )
        return GraphState(
            row_ptr=row_ptr,
            columns=columns,
            tau=tau,
            vertex_labels=self.vertex_labels.copy(),
            edge_code=edge_code,
            bidirectional_view=bidirectional_view,
            edge_src_index=edge_src_index,
        )
