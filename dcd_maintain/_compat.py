from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRUSS_DIR = ROOT / "truss"
TETREE_DIR = ROOT / "TETree"

# `truss/` must take precedence because its modules use bare imports like
# `from utils import ...`, which would otherwise resolve to `TETree/utils.py`.
for path in (str(ROOT), str(TETREE_DIR), str(TRUSS_DIR)):
    if path in sys.path:
        sys.path.remove(path)
for path in (str(ROOT), str(TETREE_DIR), str(TRUSS_DIR)):
    sys.path.insert(0, path)

from CSRGraph import edgelist_to_CSR as tetree_edgelist_to_csr, max_vertex as tetree_max_vertex  # noqa: E402
from delete_maintenance import _lookup_edge_codes  # noqa: E402
from delete_maintenance import _superior_remove_prepared, superior_remove  # noqa: E402
from insert_maintenace import (  # noqa: E402
    _build_edge_src_index,
    _collect_edge_triangles,
    _lookup_edge_ids,
    _map_ids_to_local,
    _prepare_graph_runtime,
    _superior_insert_prepared,
    decompose_from_csr,
    device,
    superior_insert,
)
from truss_save6_2 import calculate_support3, truss_decomposition  # noqa: E402
from updated_graph import insert_edges_csr, remove_edges_csr  # noqa: E402

__all__ = [
    "_build_edge_src_index",
    "_collect_edge_triangles",
    "_lookup_edge_codes",
    "_lookup_edge_ids",
    "_map_ids_to_local",
    "_prepare_graph_runtime",
    "_superior_insert_prepared",
    "_superior_remove_prepared",
    "calculate_support3",
    "decompose_from_csr",
    "device",
    "insert_edges_csr",
    "remove_edges_csr",
    "superior_insert",
    "superior_remove",
    "tetree_edgelist_to_csr",
    "tetree_max_vertex",
    "truss_decomposition",
]
