import torch

from insert_maintenace import (
    compare_with_recompute,
    decompose_from_csr,
    device,
    read_update_edge_txt,
    _build_edge_index,
    _build_edge_src_index,
    _build_bidirectional_view,
    _collect_edge_triangles,
    _compute_ts_local_packed,
    _compute_ts_local_with_support_mask,
    _expand_ranges,
    load_graph_as_csr,
    _lookup_edge_ids,
    _merge_active_candidate_edges,
    _map_ids_to_local,
    _prepare_graph_runtime,
    _prepare_traverse_runtime,
    build_affected_edge_groups,
    _select_est,
)
from updated_graph import remove_edges_csr
try:
    from .maintain_engine import maintain_truss, state_from_csr
except ImportError:
    from maintain_engine import maintain_truss, state_from_csr


def _lookup_edge_codes(state, edge_codes: torch.Tensor):
    if edge_codes.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long), torch.empty((0,), device=device, dtype=torch.bool)

    pos = torch.searchsorted(state, edge_codes)
    valid = pos < state.numel()
    if valid.any():
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        valid[valid_idx] = state[pos[valid_idx]] == edge_codes[valid_idx]
    return pos[valid], valid


def _prepare_delete_edges(row_ptr, columns, del_src, del_dst):
    if del_src.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)

    del_src = del_src.clone()
    del_dst = del_dst.clone()

    swap_mask = del_src > del_dst
    temp = del_src[swap_mask].clone()
    del_src[swap_mask] = del_dst[swap_mask]
    del_dst[swap_mask] = temp

    mask = del_src != del_dst
    del_src = del_src[mask]
    del_dst = del_dst[mask]

    if del_src.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)

    edges = torch.unique(torch.stack([del_src, del_dst], dim=1), dim=0)
    n, edge_code = _build_edge_index(row_ptr, columns)
    _, valid = _lookup_edge_ids(n, edge_code, edges)
    keep_mask = torch.zeros((edges.size(0),), device=device, dtype=torch.bool)
    keep_mask[torch.nonzero(valid, as_tuple=False).flatten()] = True
    return edges[keep_mask]


def _build_source_code_groups(affected_flat, affected_ptr, selected_mask, edge_code, truss_result):
    selected_local = torch.nonzero(selected_mask, as_tuple=False).flatten()
    if selected_local.numel() == 0:
        return {}

    selected_indices = _expand_ranges(affected_ptr[selected_local], affected_ptr[selected_local + 1])
    if selected_indices.numel() == 0 or affected_flat.numel() == 0:
        return {}

    affected_ids = torch.unique(affected_flat[selected_indices])
    affected_truss = truss_result[affected_ids]
    affected_codes = edge_code[affected_ids]

    groups = {}
    for k in torch.unique(affected_truss).detach().cpu().tolist():
        k = int(k)
        groups[k] = affected_codes[affected_truss == k]
    return groups

def _removal_traverse_prepared(row_ptr, columns, truss_result, source_edges, k, bidirectional_view, edge_src_index, traverse_stats=None):
    """
    论文 Algorithm 4: RemovalTraverse 的张量版实现。
    """
    if source_edges.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    num_edges = truss_result.size(0)
    k_mask = truss_result == k
    source_edges = torch.unique(source_edges[k_mask[source_edges]])
    if source_edges.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    evicted_mask = torch.zeros((num_edges,), device=device, dtype=torch.bool)
    valid_support_mask = k_mask.clone()
    valid_support_mask |= truss_result > k
    touched_mask = torch.zeros((num_edges,), device=device, dtype=torch.bool)
    processed_edge_visits = 0
    processed_triangle_visits = 0

    source_edge, left_edge, right_edge = _collect_edge_triangles(
        row_ptr,
        columns,
        source_edges,
        bidirectional_view=bidirectional_view,
        edge_src_index=edge_src_index,
    )
    processed_triangle_visits += int(source_edge.numel())
    if traverse_stats is not None and source_edge.numel() > 0:
        triangle_pack = torch.sort(
            torch.stack((source_edge, left_edge, right_edge), dim=1),
            dim=1,
        ).values
        traverse_stats.setdefault("triangle_batches", []).append(triangle_pack)
    source_local = _map_ids_to_local(source_edge, source_edges) if source_edge.numel() > 0 else \
        torch.empty((0,), device=device, dtype=torch.long)
    ts = _compute_ts_local_with_support_mask(
        source_edge, left_edge, right_edge, source_local, source_edges.size(0), valid_support_mask, evicted_mask
    )
    current_local_mask = k_mask[source_edges] & (ts < k - 2)
    current = torch.unique(source_edges[current_local_mask])
    touched_mask[source_edges] = True
    processed_edge_visits += int(source_edges.numel())
    current_triangle_mask = current_local_mask[source_local] if source_local.numel() > 0 else \
        torch.empty((0,), device=device, dtype=torch.bool)
    current_source_edge = source_edge[current_triangle_mask]
    current_left_edge = left_edge[current_triangle_mask]
    current_right_edge = right_edge[current_triangle_mask]

    while current.numel() > 0:
        evicted_mask[current] = True
        valid_support_mask[current] = False

        candidate_edges = _merge_active_candidate_edges(
            current_left_edge,
            current_right_edge,
            k_mask,
            evicted_mask,
        )
        if candidate_edges.numel() == 0:
            break

        candidate_source_edge, candidate_left_edge, candidate_right_edge = _collect_edge_triangles(
            row_ptr,
            columns,
            candidate_edges,
            bidirectional_view=bidirectional_view,
            edge_src_index=edge_src_index,
        )
        processed_triangle_visits += int(candidate_source_edge.numel())
        if traverse_stats is not None and candidate_source_edge.numel() > 0:
            triangle_pack = torch.sort(
                torch.stack((candidate_source_edge, candidate_left_edge, candidate_right_edge), dim=1),
                dim=1,
            ).values
            traverse_stats.setdefault("triangle_batches", []).append(triangle_pack)
        candidate_source_local = _map_ids_to_local(candidate_source_edge, candidate_edges) if \
            candidate_source_edge.numel() > 0 else torch.empty((0,), device=device, dtype=torch.long)
        touched_mask[candidate_edges] = True
        processed_edge_visits += int(candidate_edges.numel())
        ts = _compute_ts_local_with_support_mask(
            candidate_source_edge,
            candidate_left_edge,
            candidate_right_edge,
            candidate_source_local,
            candidate_edges.size(0),
            valid_support_mask,
            evicted_mask,
        )
        current_local_mask = ts < k - 2
        current = torch.unique(candidate_edges[current_local_mask])
        if current.numel() > 0:
            valid_support_mask[current] = False
        current_triangle_mask = current_local_mask[candidate_source_local] if candidate_source_local.numel() > 0 else \
            torch.empty((0,), device=device, dtype=torch.bool)
        current_source_edge = candidate_source_edge[current_triangle_mask]
        current_left_edge = candidate_left_edge[current_triangle_mask]
        current_right_edge = candidate_right_edge[current_triangle_mask]

    if traverse_stats is not None:
        traverse_stats["unique_edges"] = traverse_stats.get("unique_edges", 0) + int(torch.count_nonzero(touched_mask).item())
        traverse_stats["edge_visits"] = traverse_stats.get("edge_visits", 0) + processed_edge_visits
        traverse_stats["triangle_visits"] = traverse_stats.get("triangle_visits", 0) + processed_triangle_visits
    return torch.nonzero(evicted_mask & k_mask, as_tuple=False).flatten()


def removal_traverse(row_ptr, columns, truss_result, source_edges, k, bidirectional_view=None, edge_src_index=None, traverse_stats=None):
    bidirectional_view, edge_src_index = _prepare_traverse_runtime(
        row_ptr,
        columns,
        bidirectional_view=bidirectional_view,
        edge_src_index=edge_src_index,
    )
    return _removal_traverse_prepared(
        row_ptr,
        columns,
        truss_result,
        source_edges,
        k,
        bidirectional_view,
        edge_src_index,
        traverse_stats=traverse_stats,
    )


def _superior_remove_prepared(row_ptr, columns, del_src, del_dst, truss_result, edge_code, bidirectional_view, edge_src_index, return_stats: bool = False):
    """
    论文 Algorithm 3 的顺序外层 + 张量化遍历实现：
    1. 从待删除边中识别一个 EST；
    2. 在删除前构造 ES；
    3. 删除 EST；
    4. 调用 Algorithm 4 找到需要 -1 的边。
    """
    remaining_edges = _prepare_delete_edges(row_ptr, columns, del_src, del_dst)
    n = row_ptr.size(0) - 1
    stats = {
        "unique_edges": 0,
        "edge_visits": 0,
        "triangle_visits": 0,
        "triangle_batches": [],
        "affected_edge_code_batches": [],
        "affected_triangle_code_batches": [],
    } if return_stats else None

    while remaining_edges.size(0) > 0:
        delete_ids, _ = _lookup_edge_ids(n, edge_code, remaining_edges)
        source_edge, left_edge, right_edge = _collect_edge_triangles(
            row_ptr, columns, delete_ids, bidirectional_view=bidirectional_view, edge_src_index=edge_src_index
        )

        affected_flat, affected_ptr = build_affected_edge_groups(
            delete_ids, truss_result, source_edge, left_edge, right_edge
        )
        selected_edges, selected_mask = _select_est(
            remaining_edges, affected_flat, affected_ptr, truss_result.size(0)
        )

        source_code_groups = _build_source_code_groups(
            affected_flat, affected_ptr, selected_mask, edge_code, truss_result
        )

        row_ptr, columns, truss_result, edge_code, new_bidirectional_view = remove_edges_csr(
            row_ptr,
            columns,
            selected_edges[:, 0],
            selected_edges[:, 1],
            truss_result,
            return_bidirectional_view=True,
            bidirectional_view=bidirectional_view,
        )
        edge_src_index = _build_edge_src_index(row_ptr)

        for k in sorted(source_code_groups):
            source_edges, valid = _lookup_edge_codes(edge_code, source_code_groups[k])
            if valid.numel() == 0 or source_edges.numel() == 0:
                continue
            if stats is not None:
                source_codes = source_code_groups[k][valid]
                stats.setdefault("affected_edge_code_batches", []).append(source_codes)
                tri_source, tri_left, tri_right = _collect_edge_triangles(
                    row_ptr,
                    columns,
                    source_edges,
                    bidirectional_view=new_bidirectional_view,
                    edge_src_index=edge_src_index,
                )
                if tri_source.numel() > 0:
                    triangle_codes = torch.sort(
                        torch.stack(
                            (
                                edge_code[tri_source],
                                edge_code[tri_left],
                                edge_code[tri_right],
                            ),
                            dim=1,
                        ),
                        dim=1,
                    ).values
                    stats.setdefault("affected_triangle_code_batches", []).append(triangle_codes)
            decreased_edges = _removal_traverse_prepared(
                row_ptr,
                columns,
                truss_result,
                source_edges,
                k,
                new_bidirectional_view,
                edge_src_index,
                traverse_stats=stats,
            )
            if decreased_edges.numel() > 0:
                truss_result[decreased_edges] -= 1
        bidirectional_view = new_bidirectional_view

        remain_mask = ~torch.isin(
            remaining_edges[:, 0].to(torch.long) * n + remaining_edges[:, 1].to(torch.long),
            selected_edges[:, 0].to(torch.long) * n + selected_edges[:, 1].to(torch.long)
        )
        remaining_edges = remaining_edges[remain_mask]

    if stats is not None:
        return row_ptr, columns, truss_result, stats
    return row_ptr, columns, truss_result


def superior_remove(row_ptr, columns, del_src, del_dst, truss_result, edge_code=None, bidirectional_view=None, return_stats: bool = False):
    del edge_code, bidirectional_view
    state = state_from_csr(row_ptr.clone(), columns.clone(), truss_result.clone())
    delta = {
        "del_edges": torch.stack((del_src.to(torch.long), del_dst.to(torch.long)), dim=1)
        if del_src.numel() > 0 else torch.empty((0, 2), device=row_ptr.device, dtype=torch.long),
        "is_local": True,
    }
    new_state, stats = maintain_truss(state, delta, device=row_ptr.device)
    if return_stats:
        return new_state.row_ptr, new_state.columns, new_state.truss, stats["delete"]
    return new_state.row_ptr, new_state.columns, new_state.truss


def run(filename: str, updated_filename: str):
    _, _, old_vertices_hash, row_ptr, columns, bidirectional_view = load_graph_as_csr(filename, 0)
    truss_result = decompose_from_csr(row_ptr, columns)

    print("original truss_result", truss_result)
    print("original row_ptr, columns", row_ptr, columns)

    del_src, del_dst = read_update_edge_txt(updated_filename, old_vertices_hash, 0)
    del_src = torch.tensor(del_src, device=device, dtype=torch.long)
    del_dst = torch.tensor(del_dst, device=device, dtype=torch.long)

    row_ptr, columns, truss_result = superior_remove(
        row_ptr, columns, del_src, del_dst, truss_result, bidirectional_view=bidirectional_view
    )
    print("updated truss_result", truss_result)
    print("updated row_ptr, columns", row_ptr, columns)

    recomputed_ok, recomputed_truss_result = compare_with_recompute(row_ptr, columns, truss_result, "delete")
    if recomputed_ok:
        print("recomputed truss_result", recomputed_truss_result)

    return row_ptr, columns, truss_result


if __name__ == '__main__':
    run(r"C:\Users\Administrator\Desktop\update_g\g1.txt", r"C:\Users\Administrator\Desktop\update_g\g1.txt")
