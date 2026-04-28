from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

try:
    from .CSRGraph4 import edgelist_to_CSR
    from .runtime_state import TensorTrussState, read_base_graph_txt
    from .updated_graph import insert_edges_csr, remove_edges_csr
except ImportError:
    from CSRGraph4 import edgelist_to_CSR
    from runtime_state import TensorTrussState, read_base_graph_txt
    from updated_graph import insert_edges_csr, remove_edges_csr


@dataclass
class NormalizedDelta:
    ins_edges: torch.Tensor
    del_edges: torch.Tensor


def _ensure_edge_tensor(edge_pairs: Any, device: torch.device) -> torch.Tensor:
    if edge_pairs is None:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    edge_pairs = torch.as_tensor(edge_pairs, device=device, dtype=torch.long)
    if edge_pairs.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)
    edge_pairs = edge_pairs.reshape(-1, 2)
    return edge_pairs


def _canonicalize_edge_pairs(edge_pairs: torch.Tensor) -> torch.Tensor:
    if edge_pairs.numel() == 0:
        return torch.empty((0, 2), device=edge_pairs.device, dtype=torch.long)
    edge_pairs = edge_pairs.to(torch.long)
    src = edge_pairs[:, 0].clone()
    dst = edge_pairs[:, 1].clone()
    swap_mask = src > dst
    if swap_mask.any():
        temp = src[swap_mask].clone()
        src[swap_mask] = dst[swap_mask]
        dst[swap_mask] = temp
    keep_mask = src != dst
    src = src[keep_mask]
    dst = dst[keep_mask]
    if src.numel() == 0:
        return torch.empty((0, 2), device=edge_pairs.device, dtype=torch.long)
    return torch.unique(torch.stack((src, dst), dim=1), dim=0)


def _expand_ranges(starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    sizes = (ends - starts).to(torch.long)
    if sizes.numel() == 0:
        return torch.empty((0,), device=starts.device, dtype=torch.long)
    total = int(sizes.sum().item())
    if total == 0:
        return torch.empty((0,), device=starts.device, dtype=torch.long)
    ptr = torch.cat((torch.tensor([0], device=starts.device, dtype=torch.long), sizes.cumsum(0)))
    return torch.arange(total, device=starts.device, dtype=torch.long) - \
        torch.repeat_interleave(ptr[:-1] - starts.to(torch.long), sizes)


def _segment_indices(starts: torch.Tensor, ends: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = (ends - starts).to(torch.long)
    if sizes.numel() == 0:
        empty = torch.empty((0,), device=starts.device, dtype=torch.long)
        return empty, empty
    total = int(sizes.sum().item())
    if total == 0:
        empty = torch.empty((0,), device=starts.device, dtype=torch.long)
        return empty, sizes
    ptr = torch.cat((torch.tensor([0], device=starts.device, dtype=torch.long), sizes.cumsum(0)))
    idx = torch.arange(total, device=starts.device, dtype=torch.long) - \
        torch.repeat_interleave(ptr[:-1] - starts.to(torch.long), sizes)
    return idx, sizes


def _map_ids_to_local(ids: torch.Tensor, reference_ids: torch.Tensor) -> torch.Tensor:
    if ids.numel() == 0:
        return torch.empty((0,), device=reference_ids.device, dtype=torch.long)
    pos = torch.searchsorted(reference_ids, ids)
    return pos.to(torch.long)


def _map_ids_to_local_unsorted(ids: torch.Tensor, reference_ids: torch.Tensor) -> torch.Tensor:
    if ids.numel() == 0:
        return torch.empty((0,), device=reference_ids.device, dtype=torch.long)
    sorted_reference, perm = torch.sort(reference_ids)
    pos = torch.searchsorted(sorted_reference, ids)
    return perm[pos].to(torch.long)


def _isin_sorted(sorted_reference: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if sorted_reference.numel() == 0 or values.numel() == 0:
        return torch.zeros((values.numel(),), device=values.device, dtype=torch.bool)
    pos = torch.searchsorted(sorted_reference, values)
    valid = pos < sorted_reference.numel()
    if valid.any():
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        valid[valid_idx] = sorted_reference[pos[valid_idx]] == values[valid_idx]
    return valid


def _lookup_edge_ids(edge_code: torch.Tensor, edge_pairs: torch.Tensor, num_vertices: int) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_pairs.numel() == 0:
        empty = torch.empty((0,), device=edge_code.device, dtype=torch.long)
        return empty, torch.empty((0,), device=edge_code.device, dtype=torch.bool)
    query_code = edge_pairs[:, 0].to(torch.long) * num_vertices + edge_pairs[:, 1].to(torch.long)
    pos = torch.searchsorted(edge_code, query_code)
    valid = pos < edge_code.numel()
    if valid.any():
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        valid[valid_idx] = edge_code[pos[valid_idx]] == query_code[valid_idx]
    return pos[valid], valid


def _remove_selected_pairs(remaining_edges: torch.Tensor, selected_edges: torch.Tensor, num_vertices: int) -> torch.Tensor:
    if remaining_edges.numel() == 0 or selected_edges.numel() == 0:
        return remaining_edges
    remaining_code = remaining_edges[:, 0].to(torch.long) * num_vertices + remaining_edges[:, 1].to(torch.long)
    selected_code = torch.sort(selected_edges[:, 0].to(torch.long) * num_vertices + selected_edges[:, 1].to(torch.long)).values
    keep_mask = ~_isin_sorted(selected_code, remaining_code)
    return remaining_edges[keep_mask]


def _collect_edge_triangles(
    state: TensorTrussState,
    edge_ids: torch.Tensor,
    batch_limit: int = 4_000_000,
    bidirectional_view: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    edge_src_index: Optional[torch.Tensor] = None,
    query_src: Optional[torch.Tensor] = None,
    query_dst: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if edge_ids.numel() == 0:
        empty = torch.empty((0,), device=state.device, dtype=torch.long)
        return empty, empty, empty

    if bidirectional_view is None:
        bi_row_ptr, bi_columns, bi_edge_ids = state.get_bidirectional_view()
    else:
        bi_row_ptr, bi_columns, bi_edge_ids = bidirectional_view

    if query_src is None or query_dst is None:
        if edge_src_index is None:
            edge_src_index = state.edge_src_index()
        src = edge_src_index[edge_ids.to(torch.long)].to(torch.long)
        dst = state.columns[edge_ids.to(torch.long)].to(torch.long)
    else:
        src = query_src.to(torch.long)
        dst = query_dst.to(torch.long)

    num_vertices = state.num_vertices
    work = (bi_row_ptr[src + 1] - bi_row_ptr[src] + bi_row_ptr[dst + 1] - bi_row_ptr[dst]).to(torch.long)
    work_cumsum = torch.cumsum(work, dim=0)

    source_parts = []
    left_parts = []
    right_parts = []
    head = 0
    while head < edge_ids.numel():
        base = work_cumsum[head - 1] if head > 0 else torch.tensor(0, device=state.device, dtype=torch.long)
        limit = base + batch_limit
        tail = int(torch.searchsorted(work_cumsum, limit, right=True).item())
        if tail <= head:
            tail = head + 1

        chunk_edge_ids = edge_ids[head:tail].to(torch.long)
        chunk_src = src[head:tail]
        chunk_dst = dst[head:tail]
        local = torch.arange(chunk_edge_ids.size(0), device=state.device, dtype=torch.long)

        u_nbr_indices, u_nbr_sizes = _segment_indices(bi_row_ptr[chunk_src], bi_row_ptr[chunk_src + 1])
        v_nbr_indices, v_nbr_sizes = _segment_indices(bi_row_ptr[chunk_dst], bi_row_ptr[chunk_dst + 1])

        if u_nbr_indices.numel() == 0 or v_nbr_indices.numel() == 0:
            head = tail
            continue

        u_repeat = torch.repeat_interleave(local, u_nbr_sizes)
        v_repeat = torch.repeat_interleave(local, v_nbr_sizes)
        u_codes = u_repeat * num_vertices + bi_columns[u_nbr_indices].to(torch.long)
        v_codes = v_repeat * num_vertices + bi_columns[v_nbr_indices].to(torch.long)

        sorted_u_codes, sorted_u_order = torch.sort(u_codes, stable=True)
        sorted_u_indices = u_nbr_indices[sorted_u_order].to(torch.long)
        matched_u_pos = torch.searchsorted(sorted_u_codes, v_codes)
        match_mask = matched_u_pos < sorted_u_codes.numel()
        if match_mask.any():
            match_idx = torch.nonzero(match_mask, as_tuple=False).flatten()
            match_mask[match_idx] = sorted_u_codes[matched_u_pos[match_idx]] == v_codes[match_idx]
        if not match_mask.any():
            head = tail
            continue

        matched_edge_local = v_repeat[match_mask]
        matched_v_indices = bi_edge_ids[v_nbr_indices[match_mask].to(torch.long)].to(torch.long)
        matched_u_indices = bi_edge_ids[sorted_u_indices[matched_u_pos[match_mask]]].to(torch.long)

        source_parts.append(chunk_edge_ids[matched_edge_local])
        left_parts.append(matched_v_indices)
        right_parts.append(matched_u_indices)
        head = tail

    if not source_parts:
        empty = torch.empty((0,), device=state.device, dtype=torch.long)
        return empty, empty, empty
    return torch.cat(source_parts), torch.cat(left_parts), torch.cat(right_parts)


def _augment_bidirectional_view(
    bidirectional_view: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    extra_src: torch.Tensor,
    extra_dst: torch.Tensor,
    extra_edge_ids: torch.Tensor,
    num_vertices: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if extra_src.numel() == 0:
        return bidirectional_view

    device = extra_src.device
    bi_row_ptr, bi_columns, bi_edge_ids = bidirectional_view
    base_deg = bi_row_ptr[1:] - bi_row_ptr[:-1]

    extra_bi_src = torch.cat((extra_src.to(torch.long), extra_dst.to(torch.long)))
    extra_bi_dst = torch.cat((extra_dst.to(torch.long), extra_src.to(torch.long)))
    extra_bi_edge_ids = torch.cat((extra_edge_ids.to(torch.long), extra_edge_ids.to(torch.long)))
    extra_order = torch.argsort(extra_bi_src, stable=True)
    extra_bi_src = extra_bi_src[extra_order]
    extra_bi_dst = extra_bi_dst[extra_order]
    extra_bi_edge_ids = extra_bi_edge_ids[extra_order]
    extra_deg = torch.bincount(extra_bi_src, minlength=num_vertices)

    aug_row_ptr = torch.cat(
        (
            torch.tensor([0], device=device, dtype=torch.long),
            (base_deg + extra_deg).cumsum(0),
        )
    )

    aug_columns = torch.empty((bi_columns.size(0) + extra_bi_dst.size(0),), device=device, dtype=torch.int32)
    aug_edge_ids = torch.empty((bi_edge_ids.size(0) + extra_bi_edge_ids.size(0),), device=device, dtype=torch.int32)

    if bi_columns.numel() > 0:
        base_src = torch.repeat_interleave(torch.arange(num_vertices, device=device, dtype=torch.long), base_deg)
        base_ptr = torch.cat((torch.tensor([0], device=device, dtype=torch.long), base_deg.cumsum(0)))
        base_rank = torch.arange(bi_columns.size(0), device=device, dtype=torch.long) - \
            torch.repeat_interleave(base_ptr[:-1], base_deg)
        base_pos = aug_row_ptr[:-1][base_src] + base_rank
        aug_columns[base_pos] = bi_columns
        aug_edge_ids[base_pos] = bi_edge_ids

    extra_ptr = torch.cat((torch.tensor([0], device=device, dtype=torch.long), extra_deg.cumsum(0)))
    extra_rank = torch.arange(extra_bi_dst.size(0), device=device, dtype=torch.long) - \
        torch.repeat_interleave(extra_ptr[:-1], extra_deg)
    extra_pos = aug_row_ptr[:-1][extra_bi_src] + base_deg[extra_bi_src].to(torch.long) + extra_rank
    aug_columns[extra_pos] = extra_bi_dst.to(torch.int32)
    aug_edge_ids[extra_pos] = extra_bi_edge_ids.to(torch.int32)
    return aug_row_ptr, aug_columns, aug_edge_ids


def _bootstrap_truss_from_csr(row_ptr: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    try:
        from .truss_save6_2 import calculate_support3, truss_decomposition
    except ImportError:
        from truss_save6_2 import calculate_support3, truss_decomposition

    edge_src = torch.repeat_interleave(
        torch.arange(row_ptr.size(0) - 1, device=row_ptr.device, dtype=torch.long),
        row_ptr[1:] - row_ptr[:-1],
    )
    edge_num = columns.size(0)
    triangle_source, triangle_id, _, _, _ = calculate_support3(edge_src, columns.to(torch.long), row_ptr, columns.to(torch.long))
    return truss_decomposition(triangle_source, triangle_id, edge_num).to(torch.int32)


def build_state_from_text(
    graph_file: str,
    snapshot_dir: Optional[Union[str, Path]] = None,
    dataset_type: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> TensorTrussState:
    edge_starts, edge_ends, vertex_ids = read_base_graph_txt(graph_file, dataset_type)
    row_ptr, columns = edgelist_to_CSR(edge_starts.copy(), edge_ends.copy(), direct=True)
    row_ptr_t = torch.as_tensor(row_ptr, dtype=torch.long)
    columns_t = torch.as_tensor(columns, dtype=torch.int32)
    truss = _bootstrap_truss_from_csr(row_ptr_t.clone(), columns_t.clone())
    state = TensorTrussState.from_csr(
        row_ptr_t.cpu(),
        columns_t.cpu(),
        truss.cpu(),
        torch.as_tensor(vertex_ids, dtype=torch.int64),
    )
    if device is not None:
        state = state.to(device)
    if snapshot_dir is not None:
        state.save(snapshot_dir)
    return state


def state_from_csr(
    row_ptr: torch.Tensor,
    columns: torch.Tensor,
    truss_result: torch.Tensor,
    vertex_ids: Optional[torch.Tensor] = None,
) -> TensorTrussState:
    if vertex_ids is None:
        vertex_ids = torch.arange(row_ptr.size(0) - 1, device=row_ptr.device, dtype=torch.int64)
    return TensorTrussState.from_csr(row_ptr, columns, truss_result, vertex_ids)


def decompose_from_csr(row_ptr: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    return _bootstrap_truss_from_csr(row_ptr, columns)


def compare_with_recompute(
    row_ptr: torch.Tensor,
    columns: torch.Tensor,
    truss_result: torch.Tensor,
    title: str,
) -> tuple[bool, torch.Tensor]:
    recomputed = decompose_from_csr(row_ptr, columns)
    is_same = torch.equal(truss_result.to(recomputed.dtype), recomputed)
    print(f"{title} maintenance == recomputed", is_same)
    if not is_same:
        diff_idx = torch.nonzero(truss_result.to(recomputed.dtype) != recomputed, as_tuple=False).flatten()
        print(f"{title} diff edge ids", diff_idx)
        print(f"{title} maintenance values", truss_result[diff_idx])
        print(f"{title} recomputed values", recomputed[diff_idx])
    return is_same, recomputed


def _incident_edges_for_vertices(state: TensorTrussState, local_vertices: torch.Tensor) -> torch.Tensor:
    local_vertices = torch.unique(local_vertices.to(torch.long))
    if local_vertices.numel() == 0:
        return torch.empty((0, 2), device=state.device, dtype=torch.long)

    out_idx = _expand_ranges(state.row_ptr[local_vertices], state.row_ptr[local_vertices + 1])
    out_src = torch.repeat_interleave(local_vertices, state.row_ptr[local_vertices + 1] - state.row_ptr[local_vertices])
    out_dst = state.columns[out_idx].to(torch.long)

    in_idx = _expand_ranges(state.rev_row_ptr[local_vertices], state.rev_row_ptr[local_vertices + 1])
    in_src = state.rev_columns[in_idx].to(torch.long)
    in_dst = torch.repeat_interleave(local_vertices, state.rev_row_ptr[local_vertices + 1] - state.rev_row_ptr[local_vertices])

    edge_pairs = torch.cat(
        (
            torch.stack((out_src, out_dst), dim=1) if out_idx.numel() > 0 else torch.empty((0, 2), device=state.device, dtype=torch.long),
            torch.stack((in_src, in_dst), dim=1) if in_idx.numel() > 0 else torch.empty((0, 2), device=state.device, dtype=torch.long),
        ),
        dim=0,
    )
    return _canonicalize_edge_pairs(edge_pairs)


def normalize_updates(state: TensorTrussState, delta: Dict[str, Any]) -> NormalizedDelta:
    is_local = bool(delta.get("is_local", False))
    ins_edges = _ensure_edge_tensor(delta.get("ins_edges"), state.device)
    del_edges = _ensure_edge_tensor(delta.get("del_edges"), state.device)

    add_vertices = delta.get("add_vertices")
    if add_vertices is not None:
        add_vertices = torch.as_tensor(add_vertices, device=state.device, dtype=torch.long)
        if add_vertices.ndim == 1:
            state.ensure_vertices(add_vertices)
        else:
            ins_edges = torch.cat((ins_edges, add_vertices.reshape(-1, 2)), dim=0)

    remove_vertices = delta.get("remove_vertices")
    if remove_vertices is not None:
        remove_vertices = torch.as_tensor(remove_vertices, device=state.device, dtype=torch.long)
        if remove_vertices.ndim == 1:
            if is_local:
                local_vertices = remove_vertices
            else:
                local_vertices, valid = state.map_vertex_ids(remove_vertices.to(torch.int64))
                local_vertices = local_vertices[valid]
            del_edges = torch.cat((del_edges, _incident_edges_for_vertices(state, local_vertices)), dim=0)
        else:
            del_edges = torch.cat((del_edges, remove_vertices.reshape(-1, 2)), dim=0)

    if not is_local:
        if del_edges.numel() > 0:
            flat_del = del_edges.reshape(-1).to(torch.int64)
            mapped_del, valid_del = state.map_vertex_ids(flat_del)
            if valid_del.any():
                mapped_full = torch.empty_like(flat_del, dtype=torch.long)
                valid_idx = torch.nonzero(valid_del, as_tuple=False).flatten()
                mapped_full[valid_idx] = mapped_del
                keep_del = valid_del.reshape(-1, 2).all(dim=1)
                del_edges = mapped_full.reshape(-1, 2)[keep_del]
            else:
                del_edges = torch.empty((0, 2), device=state.device, dtype=torch.long)
        if ins_edges.numel() > 0:
            flat_ins = ins_edges.reshape(-1).to(torch.int64)
            mapped_ins = state.ensure_vertices(flat_ins).reshape(-1, 2)
            ins_edges = mapped_ins

    del_edges = _canonicalize_edge_pairs(del_edges)
    ins_edges = _canonicalize_edge_pairs(ins_edges)
    return NormalizedDelta(ins_edges=ins_edges, del_edges=del_edges)


def _prepare_insert_edges(state: TensorTrussState, ins_edges: torch.Tensor) -> torch.Tensor:
    if ins_edges.numel() == 0:
        return torch.empty((0, 2), device=state.device, dtype=torch.long)
    ins_edges = _canonicalize_edge_pairs(ins_edges)
    if ins_edges.numel() == 0:
        return ins_edges
    edge_ids, valid = _lookup_edge_ids(state.edge_codes(), ins_edges, state.num_vertices)
    if valid.numel() == 0:
        return ins_edges
    exist_mask = torch.zeros((ins_edges.size(0),), device=state.device, dtype=torch.bool)
    exist_mask[torch.nonzero(valid, as_tuple=False).flatten()] = True
    return ins_edges[~exist_mask]


def _prepare_delete_edges(state: TensorTrussState, del_edges: torch.Tensor) -> torch.Tensor:
    if del_edges.numel() == 0:
        return torch.empty((0, 2), device=state.device, dtype=torch.long)
    del_edges = _canonicalize_edge_pairs(del_edges)
    if del_edges.numel() == 0:
        return del_edges
    _, valid = _lookup_edge_ids(state.edge_codes(), del_edges, state.num_vertices)
    if valid.numel() == 0:
        return torch.empty((0, 2), device=state.device, dtype=torch.long)
    keep_mask = torch.zeros((del_edges.size(0),), device=state.device, dtype=torch.bool)
    keep_mask[torch.nonzero(valid, as_tuple=False).flatten()] = True
    return del_edges[keep_mask]


def calculate_low_b(
    inserted_edge_ids: torch.Tensor,
    truss_result: torch.Tensor,
    source_edge: torch.Tensor,
    left_edge: torch.Tensor,
    right_edge: torch.Tensor,
) -> torch.Tensor:
    pre_values = torch.full((inserted_edge_ids.size(0),), 2, dtype=truss_result.dtype, device=truss_result.device)
    if inserted_edge_ids.numel() == 0 or source_edge.numel() == 0:
        return pre_values

    triangle_levels = torch.minimum(truss_result[left_edge], truss_result[right_edge]).to(torch.long)
    source_mask = _isin_sorted(torch.sort(inserted_edge_ids).values, source_edge)
    if not source_mask.any():
        return pre_values

    matched_source = source_edge[source_mask]
    matched_levels = triangle_levels[source_mask]
    local_ids = _map_ids_to_local_unsorted(matched_source, inserted_edge_ids)

    repeat_cnt = torch.clamp(matched_levels - 1, min=0)
    valid_repeat = repeat_cnt > 0
    if not valid_repeat.any():
        return pre_values

    local_ids = local_ids[valid_repeat]
    matched_levels = matched_levels[valid_repeat]
    repeat_cnt = repeat_cnt[valid_repeat]

    expanded_local = torch.repeat_interleave(local_ids, repeat_cnt)
    k_values = _expand_ranges(
        torch.full_like(matched_levels, 2, dtype=torch.long),
        matched_levels + 1,
    )
    max_k = int(matched_levels.max().item())
    bucket_code = expanded_local * (max_k + 1) + k_values
    counts = torch.bincount(
        bucket_code,
        minlength=inserted_edge_ids.size(0) * (max_k + 1),
    ).reshape(inserted_edge_ids.size(0), max_k + 1)
    k_range = torch.arange(2, max_k + 1, device=truss_result.device, dtype=truss_result.dtype)
    valid_k = counts[:, 2:] >= (k_range - 2).unsqueeze(0)
    pre_values = torch.where(
        valid_k,
        k_range.unsqueeze(0).expand_as(valid_k),
        torch.full_like(valid_k, 2, dtype=truss_result.dtype),
    ).amax(dim=1)
    return pre_values


def build_affected_edge_groups(
    inserted_edge_ids: torch.Tensor,
    truss_result: torch.Tensor,
    source_edge: torch.Tensor,
    left_edge: torch.Tensor,
    right_edge: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if inserted_edge_ids.numel() == 0 or source_edge.numel() == 0:
        ptr = torch.zeros((inserted_edge_ids.size(0) + 1,), device=truss_result.device, dtype=torch.long)
        return torch.empty((0,), device=truss_result.device, dtype=torch.long), ptr

    tri_truss = torch.minimum(truss_result[left_edge], truss_result[right_edge])
    source_mask = _isin_sorted(torch.sort(inserted_edge_ids).values, source_edge)
    if not source_mask.any():
        ptr = torch.zeros((inserted_edge_ids.size(0) + 1,), device=truss_result.device, dtype=torch.long)
        return torch.empty((0,), device=truss_result.device, dtype=torch.long), ptr

    valid_mask = source_mask & (tri_truss <= truss_result[source_edge])
    left_mask = valid_mask & (truss_result[left_edge] == tri_truss)
    right_mask = valid_mask & (truss_result[right_edge] == tri_truss)

    affected_source = torch.cat((source_edge[left_mask], source_edge[right_mask]))
    affected_edge = torch.cat((left_edge[left_mask], right_edge[right_mask]))
    if affected_edge.numel() == 0:
        ptr = torch.zeros((inserted_edge_ids.size(0) + 1,), device=truss_result.device, dtype=torch.long)
        return torch.empty((0,), device=truss_result.device, dtype=torch.long), ptr

    pair_rows = torch.unique(torch.stack((affected_source, affected_edge), dim=1), dim=0)
    local_ids = _map_ids_to_local_unsorted(pair_rows[:, 0], inserted_edge_ids)
    order = torch.argsort(local_ids, stable=True)
    local_ids = local_ids[order]
    affected_edge = pair_rows[:, 1][order]
    counts = torch.bincount(local_ids, minlength=inserted_edge_ids.size(0))
    affected_ptr = torch.cat((torch.tensor([0], device=truss_result.device, dtype=torch.long), counts.cumsum(0)))
    return affected_edge, affected_ptr


def _select_est(inserted_pairs: torch.Tensor, affected_flat: torch.Tensor, affected_ptr: torch.Tensor, edge_num: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_inserted = inserted_pairs.size(0)
    if num_inserted == 0:
        return inserted_pairs, torch.empty((0,), device=inserted_pairs.device, dtype=torch.bool)

    membership_counts = affected_ptr[1:] - affected_ptr[:-1]
    selected_mask = torch.zeros((num_inserted,), device=inserted_pairs.device, dtype=torch.bool)
    zero_membership = membership_counts == 0
    selected_mask |= zero_membership

    if affected_flat.numel() == 0:
        if not selected_mask.any():
            selected_mask[0] = True
        return inserted_pairs[selected_mask], selected_mask

    membership_owner = torch.repeat_interleave(
        torch.arange(num_inserted, device=inserted_pairs.device, dtype=torch.long),
        membership_counts,
    )
    active_mask = ~zero_membership
    large_rank = torch.tensor(num_inserted, device=inserted_pairs.device, dtype=torch.long)

    while active_mask.any():
        active_membership = active_mask[membership_owner]
        if not active_membership.any():
            break

        membership_rank = membership_owner.clone()
        membership_rank[~active_membership] = large_rank
        min_rank_per_affected = torch.full((edge_num,), large_rank, device=inserted_pairs.device, dtype=torch.long)
        min_rank_per_affected.scatter_reduce_(
            0,
            affected_flat,
            membership_rank,
            reduce="amin",
            include_self=True,
        )
        winning_membership = active_membership & (membership_owner == min_rank_per_affected[affected_flat])
        losing_counts = torch.bincount(
            membership_owner[active_membership & (~winning_membership)],
            minlength=num_inserted,
        )
        round_selected = active_mask & (losing_counts == 0)
        if not round_selected.any():
            first_active = torch.nonzero(active_mask, as_tuple=False).flatten()[0]
            round_selected = torch.zeros_like(active_mask)
            round_selected[first_active] = True

        selected_mask |= round_selected
        selected_membership = round_selected[membership_owner]
        occupied_by_selected = torch.zeros((edge_num,), device=inserted_pairs.device, dtype=torch.bool)
        if selected_membership.any():
            occupied_by_selected[affected_flat[selected_membership]] = True
        conflicted = torch.bincount(
            membership_owner[active_membership & occupied_by_selected[affected_flat]],
            minlength=num_inserted,
        ) > 0
        active_mask &= ~conflicted

    if not selected_mask.any():
        selected_mask[0] = True
    return inserted_pairs[selected_mask], selected_mask


def _build_source_edge_groups(affected_flat: torch.Tensor, affected_ptr: torch.Tensor, truss_result: torch.Tensor) -> Dict[int, torch.Tensor]:
    if affected_ptr.numel() <= 1:
        return {}
    selected_indices = _expand_ranges(affected_ptr[:-1], affected_ptr[1:])
    if selected_indices.numel() == 0 or affected_flat.numel() == 0:
        return {}
    affected_edges = torch.unique(affected_flat[selected_indices])
    edge_truss = truss_result[affected_edges]
    groups: Dict[int, torch.Tensor] = {}
    for k in torch.unique(edge_truss).detach().cpu().tolist():
        k = int(k)
        groups[k] = affected_edges[edge_truss == k]
    return groups


def _build_source_code_groups(
    affected_flat: torch.Tensor,
    affected_ptr: torch.Tensor,
    selected_mask: torch.Tensor,
    edge_code: torch.Tensor,
    truss_result: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    selected_local = torch.nonzero(selected_mask, as_tuple=False).flatten()
    if selected_local.numel() == 0:
        return {}
    selected_indices = _expand_ranges(affected_ptr[selected_local], affected_ptr[selected_local + 1])
    if selected_indices.numel() == 0 or affected_flat.numel() == 0:
        return {}
    affected_ids = torch.unique(affected_flat[selected_indices])
    affected_truss = truss_result[affected_ids]
    affected_codes = edge_code[affected_ids]
    groups: Dict[int, torch.Tensor] = {}
    for k in torch.unique(affected_truss).detach().cpu().tolist():
        k = int(k)
        groups[k] = affected_codes[affected_truss == k]
    return groups


class ConeTriangleCache:
    def __init__(
        self,
        state: TensorTrussState,
        k: int,
        triangle_budget: int = 12_000_000,
        batch_limit: int = 4_000_000,
    ) -> None:
        self.state = state
        self.k = k
        self.triangle_budget = triangle_budget
        self.batch_limit = batch_limit
        self.collected_sources = torch.empty((0,), device=state.device, dtype=torch.long)
        self.global_edges = torch.empty((0,), device=state.device, dtype=torch.long)
        self.tri_edges_global = torch.empty((0, 3), device=state.device, dtype=torch.long)
        self.tri_edges_local = torch.empty((0, 3), device=state.device, dtype=torch.long)
        self.local_truss = torch.empty((0,), device=state.device, dtype=state.truss.dtype)
        self.inc_ptr = torch.zeros((1,), device=state.device, dtype=torch.long)
        self.ordered_source = torch.empty((0,), device=state.device, dtype=torch.long)
        self.ordered_left = torch.empty((0,), device=state.device, dtype=torch.long)
        self.ordered_right = torch.empty((0,), device=state.device, dtype=torch.long)

    def _rebuild(self) -> None:
        if self.tri_edges_global.numel() == 0:
            self.global_edges = torch.unique(self.global_edges)
            self.local_truss = self.state.truss[self.global_edges] if self.global_edges.numel() > 0 else \
                torch.empty((0,), device=self.state.device, dtype=self.state.truss.dtype)
            self.tri_edges_local = torch.empty((0, 3), device=self.state.device, dtype=torch.long)
            self.inc_ptr = torch.zeros((self.global_edges.numel() + 1,), device=self.state.device, dtype=torch.long)
            self.ordered_source = torch.empty((0,), device=self.state.device, dtype=torch.long)
            self.ordered_left = torch.empty((0,), device=self.state.device, dtype=torch.long)
            self.ordered_right = torch.empty((0,), device=self.state.device, dtype=torch.long)
            return

        self.global_edges = torch.unique(
            torch.cat((self.global_edges, self.tri_edges_global.reshape(-1))),
            sorted=True,
        )
        tri_flat_local = _map_ids_to_local(self.tri_edges_global.reshape(-1), self.global_edges)
        self.tri_edges_local = tri_flat_local.reshape(-1, 3)
        self.local_truss = self.state.truss[self.global_edges]

        a = self.tri_edges_local[:, 0]
        b = self.tri_edges_local[:, 1]
        c = self.tri_edges_local[:, 2]
        oriented_source = torch.cat((a, b, c))
        oriented_left = torch.cat((b, c, a))
        oriented_right = torch.cat((c, a, b))
        order = torch.argsort(oriented_source, stable=True)
        self.ordered_source = oriented_source[order]
        self.ordered_left = oriented_left[order]
        self.ordered_right = oriented_right[order]
        counts = torch.bincount(self.ordered_source, minlength=self.global_edges.numel())
        self.inc_ptr = torch.cat((torch.tensor([0], device=self.state.device, dtype=torch.long), counts.cumsum(0)))

    def add_edges(self, edge_ids: torch.Tensor, stats: Optional[Dict[str, Any]] = None) -> None:
        edge_ids = torch.unique(edge_ids.to(torch.long))
        if edge_ids.numel() == 0:
            return
        if self.collected_sources.numel() > 0:
            new_mask = ~_isin_sorted(self.collected_sources, edge_ids)
            edge_ids = edge_ids[new_mask]
        if edge_ids.numel() == 0:
            return

        source_edge, left_edge, right_edge = _collect_edge_triangles(
            self.state,
            edge_ids,
            batch_limit=self.batch_limit,
            bidirectional_view=self.state.get_bidirectional_view(),
            edge_src_index=self.state.edge_src_index(),
        )
        if stats is not None:
            stats["triangle_visits"] += int(source_edge.numel())
            stats["unique_edges"] += int(edge_ids.numel())
        self.collected_sources = torch.unique(torch.cat((self.collected_sources, edge_ids)), sorted=True)
        if source_edge.numel() == 0:
            self.global_edges = torch.unique(torch.cat((self.global_edges, edge_ids)), sorted=True)
            self._rebuild()
            return

        tri_edges = torch.sort(torch.stack((source_edge, left_edge, right_edge), dim=1), dim=1).values
        if self.tri_edges_global.numel() == 0:
            self.tri_edges_global = torch.unique(tri_edges, dim=0)
        else:
            self.tri_edges_global = torch.unique(torch.cat((self.tri_edges_global, tri_edges), dim=0), dim=0)
        self.global_edges = torch.unique(torch.cat((self.global_edges, edge_ids, tri_edges.reshape(-1))), sorted=True)
        self._rebuild()

    def _evicted_local_mask(self, evicted_global: torch.Tensor) -> torch.Tensor:
        if self.global_edges.numel() == 0 or evicted_global.numel() == 0:
            return torch.zeros((self.global_edges.numel(),), device=self.state.device, dtype=torch.bool)
        return _isin_sorted(torch.sort(evicted_global).values, self.global_edges)

    def map_global_edges(self, edge_ids: torch.Tensor) -> torch.Tensor:
        if edge_ids.numel() == 0:
            return torch.empty((0,), device=self.state.device, dtype=torch.long)
        return _map_ids_to_local(edge_ids.to(torch.long), self.global_edges)

    def support_counts(self, edge_ids: torch.Tensor, evicted_global: torch.Tensor) -> torch.Tensor:
        edge_ids = torch.unique(edge_ids.to(torch.long))
        if edge_ids.numel() == 0:
            return torch.empty((0,), device=self.state.device, dtype=torch.long)
        local_ids = self.map_global_edges(edge_ids)
        starts = self.inc_ptr[local_ids]
        ends = self.inc_ptr[local_ids + 1]
        inc_indices = _expand_ranges(starts, ends)
        if inc_indices.numel() == 0:
            return torch.zeros((edge_ids.numel(),), device=self.state.device, dtype=torch.long)
        sizes = (ends - starts).to(torch.long)
        local_source = torch.repeat_interleave(torch.arange(local_ids.numel(), device=self.state.device, dtype=torch.long), sizes)
        evicted_local = self._evicted_local_mask(evicted_global)
        valid_support = (self.local_truss > self.k) | ((self.local_truss == self.k) & (~evicted_local))
        left = self.ordered_left[inc_indices]
        right = self.ordered_right[inc_indices]
        mask = valid_support[left] & valid_support[right]
        if not mask.any():
            return torch.zeros((edge_ids.numel(),), device=self.state.device, dtype=torch.long)
        return torch.bincount(local_source[mask], minlength=edge_ids.numel())

    def neighbor_candidates(
        self,
        edge_ids: torch.Tensor,
        evicted_global: torch.Tensor,
        visited_global: Optional[torch.Tensor] = None,
        require_valid_triangles: bool = False,
    ) -> torch.Tensor:
        edge_ids = torch.unique(edge_ids.to(torch.long))
        if edge_ids.numel() == 0:
            return torch.empty((0,), device=self.state.device, dtype=torch.long)
        local_ids = self.map_global_edges(edge_ids)
        starts = self.inc_ptr[local_ids]
        ends = self.inc_ptr[local_ids + 1]
        inc_indices = _expand_ranges(starts, ends)
        if inc_indices.numel() == 0:
            return torch.empty((0,), device=self.state.device, dtype=torch.long)
        left = self.ordered_left[inc_indices]
        right = self.ordered_right[inc_indices]

        if require_valid_triangles:
            evicted_local = self._evicted_local_mask(evicted_global)
            valid_support = (self.local_truss > self.k) | ((self.local_truss == self.k) & (~evicted_local))
            tri_mask = valid_support[left] & valid_support[right]
            left = left[tri_mask]
            right = right[tri_mask]

        if left.numel() == 0 and right.numel() == 0:
            return torch.empty((0,), device=self.state.device, dtype=torch.long)

        candidates = torch.unique(torch.cat((self.global_edges[left], self.global_edges[right])), sorted=True)
        k_mask = self.state.truss[candidates] == self.k
        candidates = candidates[k_mask]
        if candidates.numel() == 0:
            return candidates
        if evicted_global.numel() > 0:
            candidates = candidates[~_isin_sorted(torch.sort(evicted_global).values, candidates)]
        if visited_global is not None and visited_global.numel() > 0:
            candidates = candidates[~_isin_sorted(torch.sort(visited_global).values, candidates)]
        return candidates


def _removal_traverse_cached(
    state: TensorTrussState,
    source_edges: torch.Tensor,
    k: int,
    triangle_budget: int = 12_000_000,
    stats: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    k_mask = state.truss == k
    source_edges = torch.unique(source_edges[k_mask[source_edges]], sorted=True)
    if source_edges.numel() == 0:
        return torch.empty((0,), device=state.device, dtype=torch.long)

    cache = ConeTriangleCache(state, k, triangle_budget=triangle_budget)
    frontier = source_edges
    evicted = torch.empty((0,), device=state.device, dtype=torch.long)
    touched = torch.empty((0,), device=state.device, dtype=torch.long)

    while frontier.numel() > 0:
        cache.add_edges(frontier, stats=stats)
        touched = torch.unique(torch.cat((touched, frontier)), sorted=True)
        if stats is not None:
            stats["edge_visits"] += int(frontier.numel())
        ts = cache.support_counts(frontier, evicted)
        current = frontier[ts < k - 2]
        if current.numel() == 0:
            break
        evicted = torch.unique(torch.cat((evicted, current)), sorted=True)
        frontier = cache.neighbor_candidates(current, evicted, require_valid_triangles=False)

    if stats is not None:
        stats["cone_edges"] = max(stats.get("cone_edges", 0), cache.global_edges.numel())
        stats["cone_triangles"] = max(stats.get("cone_triangles", 0), cache.tri_edges_global.size(0))
    return evicted


def _insertion_traverse_cached(
    state: TensorTrussState,
    source_edges: torch.Tensor,
    k: int,
    triangle_budget: int = 12_000_000,
    stats: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    k_mask = state.truss == k
    source_edges = torch.unique(source_edges[k_mask[source_edges]], sorted=True)
    if source_edges.numel() == 0:
        return torch.empty((0,), device=state.device, dtype=torch.long)

    cache = ConeTriangleCache(state, k, triangle_budget=triangle_budget)
    frontier = source_edges
    visited = source_edges
    tracked = source_edges
    evicted = torch.empty((0,), device=state.device, dtype=torch.long)

    cache.add_edges(frontier, stats=stats)
    while frontier.numel() > 0:
        next_frontier = cache.neighbor_candidates(
            frontier,
            evicted,
            visited_global=visited,
            require_valid_triangles=True,
        )
        if next_frontier.numel() == 0:
            break

        cache.add_edges(next_frontier, stats=stats)
        candidate_active = torch.unique(torch.cat((tracked, next_frontier)), sorted=True)
        if stats is not None:
            stats["edge_visits"] += int(candidate_active.numel())
        while True:
            active_mask = torch.ones((candidate_active.numel(),), device=state.device, dtype=torch.bool)
            if evicted.numel() > 0:
                active_mask &= ~_isin_sorted(torch.sort(evicted).values, candidate_active)
            ts = cache.support_counts(candidate_active, evicted)
            new_evict = candidate_active[active_mask & (ts <= k - 2)]
            if new_evict.numel() == 0:
                break
            evicted = torch.unique(torch.cat((evicted, new_evict)), sorted=True)

        tracked = candidate_active
        if evicted.numel() > 0:
            next_frontier = next_frontier[~_isin_sorted(torch.sort(evicted).values, next_frontier)]
        if next_frontier.numel() == 0:
            break
        visited = torch.unique(torch.cat((visited, next_frontier)), sorted=True)
        frontier = next_frontier

    if stats is not None:
        stats["cone_edges"] = max(stats.get("cone_edges", 0), cache.global_edges.numel())
        stats["cone_triangles"] = max(stats.get("cone_triangles", 0), cache.tri_edges_global.size(0))
    if evicted.numel() == 0:
        return tracked
    keep_mask = ~_isin_sorted(torch.sort(evicted).values, tracked)
    return tracked[keep_mask]


def _new_stats() -> Dict[str, Any]:
    return {
        "unique_edges": 0,
        "edge_visits": 0,
        "triangle_visits": 0,
        "cone_edges": 0,
        "cone_triangles": 0,
        "delta_plus_total": 0,
        "delta_minus_total": 0,
    }


def _apply_delete_phase(
    state: TensorTrussState,
    delete_edges: torch.Tensor,
    stats: Dict[str, Any],
    triangle_budget: int,
) -> None:
    remaining_edges = _prepare_delete_edges(state, delete_edges)
    while remaining_edges.size(0) > 0:
        edge_code = state.edge_codes()
        delete_ids, _ = _lookup_edge_ids(edge_code, remaining_edges, state.num_vertices)
        source_edge, left_edge, right_edge = _collect_edge_triangles(
            state,
            delete_ids,
            bidirectional_view=state.get_bidirectional_view(),
            edge_src_index=state.edge_src_index(),
        )
        stats["delta_minus_total"] += int(source_edge.numel())
        affected_flat, affected_ptr = build_affected_edge_groups(
            delete_ids,
            state.truss,
            source_edge,
            left_edge,
            right_edge,
        )
        selected_edges, selected_mask = _select_est(
            remaining_edges,
            affected_flat,
            affected_ptr,
            state.num_edges,
        )
        source_code_groups = _build_source_code_groups(
            affected_flat,
            affected_ptr,
            selected_mask,
            edge_code,
            state.truss,
        )

        row_ptr, columns, truss_result = remove_edges_csr(
            state.row_ptr,
            state.columns,
            selected_edges[:, 0],
            selected_edges[:, 1],
            state.truss,
            return_bidirectional_view=False,
        )
        state.refresh_graph(row_ptr, columns, truss_result)

        for k in sorted(source_code_groups):
            source_edges_k, valid = _lookup_edge_ids(state.edge_codes(), torch.stack((
                torch.div(source_code_groups[k], state.num_vertices, rounding_mode="floor"),
                torch.remainder(source_code_groups[k], state.num_vertices),
            ), dim=1), state.num_vertices)
            if valid.numel() == 0 or source_edges_k.numel() == 0:
                continue
            decreased_edges = _removal_traverse_cached(
                state,
                source_edges_k,
                k,
                triangle_budget=triangle_budget,
                stats=stats,
            )
            if decreased_edges.numel() > 0:
                state.truss[decreased_edges] -= 1

        remaining_edges = _remove_selected_pairs(remaining_edges, selected_edges, state.num_vertices)


def _apply_insert_phase(
    state: TensorTrussState,
    insert_edges: torch.Tensor,
    stats: Dict[str, Any],
    triangle_budget: int,
) -> None:
    remaining_edges = _prepare_insert_edges(state, insert_edges)
    while remaining_edges.size(0) > 0:
        rem_src = remaining_edges[:, 0]
        rem_dst = remaining_edges[:, 1]
        temp_inserted_ids = state.num_edges + torch.arange(remaining_edges.size(0), device=state.device, dtype=torch.long)
        temp_truss = torch.cat((
            state.truss,
            torch.full((remaining_edges.size(0),), 2, dtype=state.truss.dtype, device=state.device),
        ))
        temp_bidirectional_view = _augment_bidirectional_view(
            state.get_bidirectional_view(),
            rem_src,
            rem_dst,
            temp_inserted_ids,
            state.num_vertices,
        )
        temp_source_edge, temp_left_edge, temp_right_edge = _collect_edge_triangles(
            state,
            temp_inserted_ids,
            bidirectional_view=temp_bidirectional_view,
            query_src=rem_src,
            query_dst=rem_dst,
        )
        stats["delta_plus_total"] += int(temp_source_edge.numel())
        pre_truss = calculate_low_b(
            temp_inserted_ids,
            temp_truss,
            temp_source_edge,
            temp_left_edge,
            temp_right_edge,
        )
        temp_truss[temp_inserted_ids] = pre_truss

        affected_flat, affected_ptr = build_affected_edge_groups(
            temp_inserted_ids,
            temp_truss,
            temp_source_edge,
            temp_left_edge,
            temp_right_edge,
        )
        selected_edges, selected_mask = _select_est(
            remaining_edges,
            affected_flat,
            affected_ptr,
            temp_truss.size(0),
        )

        row_ptr, columns, truss_result = insert_edges_csr(
            state.row_ptr,
            state.columns,
            selected_edges[:, 0],
            selected_edges[:, 1],
            state.truss,
            return_bidirectional_view=False,
        )
        state.refresh_graph(row_ptr, columns, truss_result)

        selected_ids, _ = _lookup_edge_ids(state.edge_codes(), selected_edges, state.num_vertices)
        state.truss[selected_ids] = pre_truss[selected_mask]

        selected_source_edge, selected_left_edge, selected_right_edge = _collect_edge_triangles(
            state,
            selected_ids,
            bidirectional_view=state.get_bidirectional_view(),
            edge_src_index=state.edge_src_index(),
        )
        selected_affected_flat, selected_affected_ptr = build_affected_edge_groups(
            selected_ids,
            state.truss,
            selected_source_edge,
            selected_left_edge,
            selected_right_edge,
        )
        source_edge_groups = _build_source_edge_groups(selected_affected_flat, selected_affected_ptr, state.truss)
        for k in sorted(source_edge_groups):
            increased_edges = _insertion_traverse_cached(
                state,
                source_edge_groups[k],
                k,
                triangle_budget=triangle_budget,
                stats=stats,
            )
            if increased_edges.numel() > 0:
                state.truss[increased_edges] += 1

        remaining_edges = _remove_selected_pairs(remaining_edges, selected_edges, state.num_vertices)


def maintain_truss(
    state: TensorTrussState,
    delta: Dict[str, Any],
    device: Optional[Union[str, torch.device]] = None,
    edge_budget: int = 2_000_000,
    triangle_budget: int = 12_000_000,
) -> tuple[TensorTrussState, Dict[str, Any]]:
    working = state.clone()
    if device is not None:
        working = working.to(device)
    normalized = normalize_updates(working, delta)

    stats = {
        "config": {
            "edge_budget": edge_budget,
            "triangle_budget": triangle_budget,
            "device": str(working.device),
        },
        "delete": _new_stats(),
        "insert": _new_stats(),
    }
    with torch.no_grad():
        if normalized.del_edges.numel() > 0:
            _apply_delete_phase(working, normalized.del_edges, stats["delete"], triangle_budget)
        if normalized.ins_edges.numel() > 0:
            _apply_insert_phase(working, normalized.ins_edges, stats["insert"], triangle_budget)
    return working, stats
