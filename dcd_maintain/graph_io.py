from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

from ._compat import device
from .graph_state import GraphState


def _load_txt_array(path: str) -> np.ndarray:
    array = np.loadtxt(path, dtype=np.int64)
    return np.atleast_2d(array)


def _normalize_pairs_np(edge_pairs: np.ndarray) -> np.ndarray:
    edge_pairs = np.atleast_2d(edge_pairs).astype(np.int64, copy=False)
    if edge_pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    keep_mask = edge_pairs[:, 0] != edge_pairs[:, 1]
    edge_pairs = edge_pairs[keep_mask]
    if edge_pairs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    left = np.minimum(edge_pairs[:, 0], edge_pairs[:, 1])
    right = np.maximum(edge_pairs[:, 0], edge_pairs[:, 1])
    edge_pairs = np.stack((left, right), axis=1)
    return np.unique(edge_pairs, axis=0)


def _to_long_pairs(pairs: Iterable[tuple[int, int]] | np.ndarray | torch.Tensor | None) -> torch.Tensor:
    if pairs is None:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    if isinstance(pairs, torch.Tensor):
        tensor = pairs.to(device=device, dtype=torch.long)
    else:
        tensor = torch.as_tensor(np.asarray(list(pairs) if not isinstance(pairs, np.ndarray) else pairs), device=device, dtype=torch.long)
    if tensor.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    if tensor.ndim == 1:
        tensor = tensor.reshape(-1, 2)
    keep_mask = tensor[:, 0] != tensor[:, 1]
    tensor = tensor[keep_mask]
    if tensor.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    left = torch.minimum(tensor[:, 0], tensor[:, 1])
    right = torch.maximum(tensor[:, 0], tensor[:, 1])
    return torch.unique(torch.stack((left, right), dim=1), dim=0)


@dataclass
class GraphUpdate:
    edge_inserts: torch.Tensor = field(default_factory=lambda: torch.empty((0, 2), device=device, dtype=torch.long))
    edge_deletes: torch.Tensor = field(default_factory=lambda: torch.empty((0, 2), device=device, dtype=torch.long))
    vertex_inserts: dict[int, list[int]] = field(default_factory=dict)
    vertex_deletes: torch.Tensor = field(default_factory=lambda: torch.empty((0,), device=device, dtype=torch.long))

    @classmethod
    def from_raw(
        cls,
        edge_inserts: Iterable[tuple[int, int]] | np.ndarray | torch.Tensor | None = None,
        edge_deletes: Iterable[tuple[int, int]] | np.ndarray | torch.Tensor | None = None,
        vertex_inserts: dict[int, Iterable[int]] | None = None,
        vertex_deletes: Iterable[int] | np.ndarray | torch.Tensor | None = None,
    ) -> "GraphUpdate":
        normalized_vertex_inserts = {
            int(vertex): [int(neighbor) for neighbor in neighbors]
            for vertex, neighbors in (vertex_inserts or {}).items()
        }
        if vertex_deletes is None:
            vertex_delete_tensor = torch.empty((0,), device=device, dtype=torch.long)
        elif isinstance(vertex_deletes, torch.Tensor):
            vertex_delete_tensor = torch.unique(vertex_deletes.to(device=device, dtype=torch.long))
        else:
            vertex_delete_tensor = torch.unique(torch.as_tensor(list(vertex_deletes), device=device, dtype=torch.long))
        return cls(
            edge_inserts=_to_long_pairs(edge_inserts),
            edge_deletes=_to_long_pairs(edge_deletes),
            vertex_inserts=normalized_vertex_inserts,
            vertex_deletes=vertex_delete_tensor,
        )

    def is_empty(self) -> bool:
        return (
            self.edge_inserts.numel() == 0
            and self.edge_deletes.numel() == 0
            and not self.vertex_inserts
            and self.vertex_deletes.numel() == 0
        )


def load_graph_from_txt(path: str) -> GraphState:
    array = _load_txt_array(path)
    edge_pairs = _normalize_pairs_np(array[:, :2])
    if edge_pairs.size == 0:
        vertex_labels = np.empty((0,), dtype=np.int64)
        mapped_edges = np.empty((0, 2), dtype=np.int64)
    else:
        vertices = np.unique(edge_pairs.reshape(-1))
        label_to_id = {int(label): idx for idx, label in enumerate(vertices.tolist())}
        mapped_edges = np.array(
            [[label_to_id[int(u)], label_to_id[int(v)]] for u, v in edge_pairs],
            dtype=np.int64,
        )
        vertex_labels = vertices.astype(np.int64, copy=False)
    return GraphState.from_edge_pairs(mapped_edges, vertex_labels)


def load_edge_pairs_from_txt(path: str) -> torch.Tensor:
    array = _load_txt_array(path)
    edge_pairs = _normalize_pairs_np(array[:, :2])
    return _to_long_pairs(edge_pairs)


def expand_update(base_state: GraphState, update: GraphUpdate) -> tuple[GraphState, dict[str, torch.Tensor]]:
    label_to_id = {int(label): idx for idx, label in enumerate(base_state.vertex_labels.tolist())}
    vertex_labels = list(base_state.vertex_labels.tolist())

    def ensure_id(label: int) -> int:
        label = int(label)
        if label not in label_to_id:
            label_to_id[label] = len(vertex_labels)
            vertex_labels.append(label)
        return label_to_id[label]

    insert_pairs = []
    delete_pairs = []

    if update.edge_inserts.numel() > 0:
        for pair in update.edge_inserts.detach().cpu().tolist():
            insert_pairs.append((ensure_id(pair[0]), ensure_id(pair[1])))

    if update.edge_deletes.numel() > 0:
        for pair in update.edge_deletes.detach().cpu().tolist():
            u = label_to_id.get(int(pair[0]))
            v = label_to_id.get(int(pair[1]))
            if u is not None and v is not None:
                delete_pairs.append((u, v))

    for vertex_label, neighbors in update.vertex_inserts.items():
        vertex_id = ensure_id(vertex_label)
        for neighbor in neighbors:
            insert_pairs.append((vertex_id, ensure_id(neighbor)))

    delete_vertex_ids = []
    if update.vertex_deletes.numel() > 0:
        for label in update.vertex_deletes.detach().cpu().tolist():
            vertex_id = label_to_id.get(int(label))
            if vertex_id is not None:
                delete_vertex_ids.append(vertex_id)

    expanded_labels = np.asarray(vertex_labels, dtype=np.int64)
    if expanded_labels.size == base_state.vertex_labels.size:
        expanded_state = base_state
    else:
        expanded_state = GraphState.from_edge_pairs(base_state.edge_pairs(), expanded_labels, tau=base_state.tau)

    delete_pairs_tensor = _to_long_pairs(delete_pairs)
    if delete_vertex_ids:
        delete_vertex_tensor = torch.as_tensor(delete_vertex_ids, device=device, dtype=torch.long)
        delete_pairs_tensor = torch.cat((delete_pairs_tensor, expanded_state.incident_edge_pairs(delete_vertex_tensor)), dim=0)
        delete_pairs_tensor = _to_long_pairs(delete_pairs_tensor)

    return expanded_state, {
        "edge_inserts": _to_long_pairs(insert_pairs),
        "edge_deletes": delete_pairs_tensor,
        "vertex_deletes": torch.as_tensor(delete_vertex_ids, device=device, dtype=torch.long),
    }
