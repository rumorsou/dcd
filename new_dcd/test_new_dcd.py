import tempfile
import unittest

import torch

from new_dcd import DCDState, DeltaGraph, EdgeTriangleIndex, load_snapshot, maintain_dcd, save_snapshot


def _reference_after(state: DCDState, ins=None, delete=None) -> DCDState:
    edges = state.canonical_edge_pairs()
    n = state.num_vertices
    if delete is not None:
        delete = torch.as_tensor(delete, dtype=torch.long, device=state.device).reshape(-1, 2)
        del_code = torch.unique(torch.minimum(delete[:, 0], delete[:, 1]) * n + torch.maximum(delete[:, 0], delete[:, 1]))
        code = edges[:, 0] * n + edges[:, 1]
        edges = edges[~torch.isin(code, del_code)]
    if ins is not None:
        ins = torch.as_tensor(ins, dtype=torch.long, device=state.device).reshape(-1, 2)
        edges = torch.cat((edges, ins), dim=0)
    return DCDState.from_local_edge_pairs(edges, state.vertex_ids, device=state.device)


class NewDCDTests(unittest.TestCase):
    def test_full_csr_edge_ids_and_triangle_pack(self):
        state = DCDState.from_edge_pairs(torch.tensor([[0, 1], [1, 2], [0, 2], [2, 3]]), relabel=False)
        self.assertEqual(int(state.col.numel()), 2 * state.num_edges)
        self.assertEqual(int(state.edge_id_of_col.numel()), int(state.col.numel()))
        index = EdgeTriangleIndex(state)
        tri_edge = state.edge_ids_from_pairs(torch.tensor([[0, 1]]))[0]
        pack = index.materialize(tri_edge)
        self.assertEqual(pack.record_count, 1)
        self.assertEqual(set(pack.other_edges.flatten().tolist()), set(state.edge_ids_from_pairs(torch.tensor([[0, 2], [1, 2]]))[0].tolist()))

    def test_insert_matches_recompute(self):
        base = DCDState.from_edge_pairs(torch.tensor([[0, 1], [1, 2], [0, 2], [2, 3], [3, 4]]), relabel=False)
        result = maintain_dcd(base, DeltaGraph(ins_edges=torch.tensor([[1, 3], [0, 3]]), is_local=True), edge_budget=2)
        reference = _reference_after(base, ins=[[1, 3], [0, 3]])
        self.assertTrue(torch.equal(result.tau_new.cpu(), reference.tau.cpu()))

    def test_delete_matches_recompute(self):
        base = DCDState.from_edge_pairs(torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]), relabel=False)
        result = maintain_dcd(base, DeltaGraph(del_edges=torch.tensor([[0, 1], [2, 3]]), is_local=True), edge_budget=2)
        reference = _reference_after(base, delete=[[0, 1], [2, 3]])
        self.assertTrue(torch.equal(result.tau_new.cpu(), reference.tau.cpu()))

    def test_vertex_remove_expands_incident_edges(self):
        base = DCDState.from_edge_pairs(torch.tensor([[0, 1], [0, 2], [1, 2], [2, 3]]), relabel=False)
        result = maintain_dcd(base, DeltaGraph(remove_vertices=torch.tensor([2]), is_local=True), edge_budget=2)
        reference = _reference_after(base, delete=[[0, 2], [1, 2], [2, 3]])
        self.assertTrue(torch.equal(result.tau_new.cpu(), reference.tau.cpu()))

    def test_snapshot_roundtrip(self):
        state = DCDState.from_edge_pairs(torch.tensor([[10, 20], [20, 30], [10, 30]]))
        with tempfile.TemporaryDirectory() as tmp:
            save_snapshot(state, tmp)
            loaded = load_snapshot(tmp)
        self.assertTrue(torch.equal(state.rowptr.cpu(), loaded.rowptr.cpu()))
        self.assertTrue(torch.equal(state.edge_id_of_col.cpu(), loaded.edge_id_of_col.cpu()))
        self.assertTrue(torch.equal(state.tau.cpu(), loaded.tau.cpu()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_cuda_consistency(self):
        edges = torch.tensor([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
        delta = DeltaGraph(ins_edges=torch.tensor([[0, 3]]), is_local=True)
        cpu_state = DCDState.from_edge_pairs(edges, relabel=False, device="cpu")
        cuda_state = cpu_state.to("cuda")
        cpu_result = maintain_dcd(cpu_state, delta, device="cpu")
        cuda_result = maintain_dcd(cuda_state, delta, device="cuda")
        self.assertTrue(torch.equal(cpu_result.tau_new.cpu(), cuda_result.tau_new.cpu()))


if __name__ == "__main__":
    unittest.main()
