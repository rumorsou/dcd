from __future__ import annotations

import torch

from ._compat import decompose_from_csr
from .graph_state import GraphState


def recompute_truss(state: GraphState) -> torch.Tensor:
    return decompose_from_csr(state.row_ptr, state.columns)


def compare_against_recompute(state: GraphState, tau: torch.Tensor) -> tuple[bool, torch.Tensor]:
    recomputed = recompute_truss(state)
    return torch.equal(tau, recomputed), recomputed
