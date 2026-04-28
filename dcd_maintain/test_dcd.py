from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from .engine import tensorized_dcd_maintain
from .graph_io import GraphUpdate, load_graph_from_txt
from .oracle import compare_against_recompute


BASE_EDGES = np.array(
    [
        [0, 1],
        [0, 2],
        [1, 2],
        [1, 3],
        [2, 3],
        [1, 4],
        [2, 4],
        [3, 4],
    ],
    dtype=np.int64,
)


def _build_base_state():
    with tempfile.TemporaryDirectory() as tmp_dir:
        graph_path = Path(tmp_dir) / "base_graph.txt"
        np.savetxt(graph_path, BASE_EDGES, fmt="%d")
        return load_graph_from_txt(str(graph_path))


def _run_case(name: str, update: GraphUpdate):
    base_state = _build_base_state()
    result = tensorized_dcd_maintain(base_state, update, allow_full_recompute_fallback=False)
    ok, recomputed = compare_against_recompute(result.state, result.tau_new)
    print(
        f"{name}: ok={ok}, fallback={result.used_fallback}, "
        f"cone={int(result.cone_edges.numel())}, delete_rounds={result.delete_stats.rounds}, "
        f"insert_rounds={result.insert_stats.rounds}"
    )
    if result.used_fallback:
        raise AssertionError(f"{name} unexpectedly used fallback")
    if not ok:
        diff_idx = torch.nonzero(result.tau_new != recomputed, as_tuple=False).flatten()
        raise AssertionError(f"{name} mismatch on edges {diff_idx.tolist()}")


def main():
    _run_case(
        "edge_delete_only",
        GraphUpdate.from_raw(edge_deletes=[(1, 4)]),
    )
    _run_case(
        "edge_insert_only",
        GraphUpdate.from_raw(edge_inserts=[(0, 3)]),
    )
    _run_case(
        "mixed_delete_insert",
        GraphUpdate.from_raw(edge_deletes=[(1, 4)], edge_inserts=[(0, 3)]),
    )
    _run_case(
        "vertex_delete",
        GraphUpdate.from_raw(vertex_deletes=[4]),
    )
    _run_case(
        "vertex_insert",
        GraphUpdate.from_raw(vertex_inserts={5: [1, 2, 3]}),
    )
    print("all dcd tests passed")


if __name__ == "__main__":
    main()
