import numpy as np
import torch

from CSRGraph4 import edgelist_to_CSR, max_vertex
from truss_save6_2 import calculate_support3, truss_decomposition
from updated_graph import insert_edges_csr
from utils import get_all_nbr_size
try:
    from .maintain_engine import maintain_truss, state_from_csr
except ImportError:
    from maintain_engine import maintain_truss, state_from_csr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_base_graph_txt(filename: str, dataset_type: int = 0):
    """
    为动态图维护读取基础图：
    1. 不删除 leaf 边；
    2. 保留原始图中所有出现过的顶点；
    3. 返回与原始顶点集合一致的连续编号。
    """
    array = np.loadtxt(filename, dtype=np.int32)
    array = np.atleast_2d(array)

    edge_starts = array[:, 0]
    edge_ends = array[:, 1]

    if dataset_type == 0:
        mask = edge_starts != edge_ends
    else:
        mask = edge_starts < edge_ends

    edge_starts = edge_starts[mask]
    edge_ends = edge_ends[mask]

    swap_mask = edge_starts > edge_ends
    temp = edge_starts[swap_mask].copy()
    edge_starts[swap_mask] = edge_ends[swap_mask]
    edge_ends[swap_mask] = temp

    if edge_starts.size == 0:
        max_vertex.value = -1
        return edge_starts, edge_ends, np.empty((0,), dtype=np.int32)

    edges = np.unique(np.stack([edge_starts, edge_ends], axis=1), axis=0)
    vertices = np.unique(edges.reshape(-1))
    max_vertex.value = len(vertices) - 1

    mapped_src = np.searchsorted(vertices, edges[:, 0]).astype(np.int32, copy=False)
    mapped_dst = np.searchsorted(vertices, edges[:, 1]).astype(np.int32, copy=False)
    return mapped_src, mapped_dst, vertices


def read_update_edge_txt(filename: str, vertices: np.ndarray, dataset_type: int = 0):
    """
    读取待插入边文件，只做自环过滤和编号映射，不删除 leaf 边。
    """
    array = np.loadtxt(filename, dtype=np.int32)
    array = np.atleast_2d(array)

    edge_starts = array[:, 0]
    edge_ends = array[:, 1]

    if dataset_type == 0:
        mask = edge_starts != edge_ends
    else:
        mask = edge_starts < edge_ends

    edge_starts = edge_starts[mask]
    edge_ends = edge_ends[mask]

    if edge_starts.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    src_pos = np.searchsorted(vertices, edge_starts)
    dst_pos = np.searchsorted(vertices, edge_ends)

    src_valid = (src_pos < vertices.size) & (vertices[src_pos] == edge_starts)
    dst_valid = (dst_pos < vertices.size) & (vertices[dst_pos] == edge_ends)
    valid = src_valid & dst_valid

    if not np.all(valid):
        missing_edges = np.stack([edge_starts[~valid], edge_ends[~valid]], axis=1)
        raise ValueError(
            f"更新边中存在基础图文件里从未出现过的顶点，例如 {missing_edges[:3]}。"
        )

    mapped_src = src_pos.astype(np.int32, copy=False)
    mapped_dst = dst_pos.astype(np.int32, copy=False)

    swap_mask = mapped_src > mapped_dst
    temp = mapped_src[swap_mask].copy()
    mapped_src[swap_mask] = mapped_dst[swap_mask]
    mapped_dst[swap_mask] = temp

    keep_mask = mapped_src != mapped_dst
    mapped_src = mapped_src[keep_mask]
    mapped_dst = mapped_dst[keep_mask]

    if mapped_src.size == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    edges = np.unique(np.stack([mapped_src, mapped_dst], axis=1), axis=0)
    return edges[:, 0], edges[:, 1]


def _csr_to_coo(row_ptr: torch.Tensor, columns: torch.Tensor):
    src = torch.repeat_interleave(
        torch.arange(row_ptr.size(0) - 1, device=row_ptr.device, dtype=columns.dtype),
        row_ptr[1:] - row_ptr[:-1]
    )
    return src, columns


def load_graph_as_csr(filename: str, dataset_type: int = 0):
    edge_starts, edge_ends, vertex_to_index = read_base_graph_txt(filename, dataset_type)
    row_ptr, columns = edgelist_to_CSR(edge_starts, edge_ends, direct=True)
    row_ptr = torch.tensor(row_ptr, device=device, dtype=torch.long)
    columns = torch.tensor(columns, device=device, dtype=torch.long)
    bidirectional_view = _build_bidirectional_view(row_ptr, columns)
    return edge_starts, edge_ends, vertex_to_index, row_ptr, columns, bidirectional_view


def _build_edge_index(row_ptr: torch.Tensor, columns: torch.Tensor):
    src, dst = _csr_to_coo(row_ptr, columns)
    n = row_ptr.size(0) - 1
    edge_code = src.to(torch.long) * n + dst.to(torch.long)
    return n, edge_code


def _build_bidirectional_view(row_ptr: torch.Tensor, columns: torch.Tensor):
    if columns.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return row_ptr.clone(), empty, empty

    src, dst = _csr_to_coo(row_ptr, columns)
    edge_ids = torch.arange(columns.size(0), device=device, dtype=torch.long)
    n = row_ptr.size(0) - 1
    out_counts = row_ptr[1:] - row_ptr[:-1]
    in_counts = torch.bincount(dst.to(torch.long), minlength=n)
    deg = in_counts + out_counts
    bi_row_ptr = torch.cat((
        torch.tensor([0], device=device, dtype=torch.long),
        deg.cumsum(0)
    ))

    bi_columns = torch.empty((columns.size(0) * 2,), device=device, dtype=torch.long)
    bi_edge_ids = torch.empty((columns.size(0) * 2,), device=device, dtype=torch.long)

    in_row_ptr = torch.cat((
        torch.tensor([0], device=device, dtype=torch.long),
        in_counts.cumsum(0)
    ))
    in_offset = bi_row_ptr[:-1]
    out_offset = in_offset + in_counts

    order_in = torch.argsort(dst, stable=True)
    dst_sorted = dst[order_in].to(torch.long)
    src_sorted = src[order_in].to(torch.long)
    edge_ids_sorted = edge_ids[order_in]
    in_group_start = torch.repeat_interleave(in_row_ptr[:-1], in_counts)
    in_local_rank = torch.arange(dst_sorted.size(0), device=device, dtype=torch.long) - in_group_start
    incoming_positions = in_offset[dst_sorted] + in_local_rank

    out_group_start = torch.repeat_interleave(row_ptr[:-1], out_counts)
    out_local_rank = torch.arange(src.size(0), device=device, dtype=torch.long) - out_group_start
    outgoing_positions = out_offset[src.to(torch.long)] + out_local_rank

    bi_columns[incoming_positions] = src_sorted
    bi_edge_ids[incoming_positions] = edge_ids_sorted
    bi_columns[outgoing_positions] = dst.to(torch.long)
    bi_edge_ids[outgoing_positions] = edge_ids

    return bi_row_ptr, bi_columns, bi_edge_ids


def _augment_bidirectional_view(
    bidirectional_view,
    extra_src: torch.Tensor,
    extra_dst: torch.Tensor,
    extra_edge_ids: torch.Tensor,
    n: int,
):
    if extra_src.numel() == 0:
        return bidirectional_view

    bi_row_ptr, bi_columns, bi_edge_ids = bidirectional_view
    base_deg = bi_row_ptr[1:] - bi_row_ptr[:-1]

    extra_bi_src = torch.cat((extra_src.to(torch.long), extra_dst.to(torch.long)))
    extra_bi_dst = torch.cat((extra_dst.to(torch.long), extra_src.to(torch.long)))
    extra_bi_edge_ids = torch.cat((extra_edge_ids, extra_edge_ids))
    extra_order = torch.argsort(extra_bi_src, stable=True)
    extra_bi_src = extra_bi_src[extra_order]
    extra_bi_dst = extra_bi_dst[extra_order]
    extra_bi_edge_ids = extra_bi_edge_ids[extra_order]
    extra_deg = torch.bincount(extra_bi_src, minlength=n)

    aug_row_ptr = torch.cat((
        torch.tensor([0], device=device, dtype=torch.long),
        (base_deg + extra_deg).cumsum(0)
    ))

    aug_columns = torch.empty(
        (bi_columns.size(0) + extra_bi_dst.size(0),),
        device=device,
        dtype=torch.long,
    )
    aug_edge_ids = torch.empty(
        (bi_edge_ids.size(0) + extra_bi_edge_ids.size(0),),
        device=device,
        dtype=torch.long,
    )

    if bi_columns.numel() > 0:
        base_src = torch.repeat_interleave(
            torch.arange(n, device=device, dtype=torch.long),
            base_deg
        )
        base_local_ptr = torch.cat((
            torch.tensor([0], device=device, dtype=torch.long),
            base_deg.cumsum(0)
        ))
        base_local_rank = torch.arange(bi_columns.size(0), device=device, dtype=torch.long) - \
            torch.repeat_interleave(base_local_ptr[:-1], base_deg)
        base_positions = aug_row_ptr[:-1][base_src] + base_local_rank
        aug_columns[base_positions] = bi_columns
        aug_edge_ids[base_positions] = bi_edge_ids

    extra_local_ptr = torch.cat((
        torch.tensor([0], device=device, dtype=torch.long),
        extra_deg.cumsum(0)
    ))
    extra_local_rank = torch.arange(extra_bi_dst.size(0), device=device, dtype=torch.long) - \
        torch.repeat_interleave(extra_local_ptr[:-1], extra_deg)
    extra_positions = aug_row_ptr[:-1][extra_bi_src] + base_deg[extra_bi_src] + extra_local_rank
    aug_columns[extra_positions] = extra_bi_dst
    aug_edge_ids[extra_positions] = extra_bi_edge_ids
    return aug_row_ptr, aug_columns, aug_edge_ids


def _collect_edge_triangles(
    row_ptr: torch.Tensor,
    columns: torch.Tensor,
    edge_ids: torch.Tensor,
    batch_limit: int = 4_000_000,
    bidirectional_view=None,
    edge_src_index: torch.Tensor = None,
    query_src: torch.Tensor = None,
    query_dst: torch.Tensor = None,
):
    if edge_ids.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty, empty

    if bidirectional_view is None:
        bi_row_ptr, bi_columns, bi_edge_ids = _build_bidirectional_view(row_ptr, columns)
    else:
        bi_row_ptr, bi_columns, bi_edge_ids = bidirectional_view

    n = row_ptr.size(0) - 1
    if query_src is None or query_dst is None:
        if edge_src_index is None:
            src = torch.searchsorted(row_ptr[1:], edge_ids, right=True)
        else:
            src = edge_src_index[edge_ids]
        dst = columns[edge_ids]
    else:
        src = query_src.to(torch.long)
        dst = query_dst.to(torch.long)
    work = (bi_row_ptr[src + 1] - bi_row_ptr[src] + bi_row_ptr[dst + 1] - bi_row_ptr[dst]).to(torch.int64)
    work_cumsum = torch.cumsum(work, dim=0)

    source_parts = []
    left_parts = []
    right_parts = []
    head = 0
    while head < edge_ids.numel():
        base = work_cumsum[head - 1] if head > 0 else torch.tensor(0, device=device, dtype=torch.int64)
        limit = base + batch_limit
        tail = int(torch.searchsorted(work_cumsum, limit, right=True).item())
        if tail <= head:
            tail = head + 1

        chunk_edge_ids = edge_ids[head:tail]
        chunk_src = src[head:tail]
        chunk_dst = dst[head:tail]
        local = torch.arange(chunk_edge_ids.size(0), device=device, dtype=torch.long)

        u_nbr_indices, u_nbr_sizes = get_all_nbr_size(bi_row_ptr[chunk_src], bi_row_ptr[chunk_src + 1])
        v_nbr_indices, v_nbr_sizes = get_all_nbr_size(bi_row_ptr[chunk_dst], bi_row_ptr[chunk_dst + 1])

        if u_nbr_indices.numel() == 0 or v_nbr_indices.numel() == 0:
            head = tail
            continue

        u_repeat = torch.repeat_interleave(local, u_nbr_sizes.to(torch.long))
        v_repeat = torch.repeat_interleave(local, v_nbr_sizes.to(torch.long))
        u_codes = u_repeat * n + bi_columns[u_nbr_indices].to(torch.long)
        v_codes = v_repeat * n + bi_columns[v_nbr_indices].to(torch.long)

        sorted_u_codes, sorted_u_order = torch.sort(u_codes, stable=True)
        sorted_u_indices = u_nbr_indices[sorted_u_order].to(torch.long)
        matched_u_pos = torch.searchsorted(sorted_u_codes, v_codes)
        match_mask = matched_u_pos < sorted_u_codes.numel()
        if torch.any(match_mask):
            match_local = torch.nonzero(match_mask, as_tuple=False).flatten()
            match_mask[match_local] = sorted_u_codes[matched_u_pos[match_local]] == v_codes[match_local]
        if not torch.any(match_mask):
            head = tail
            continue

        matched_edge_local = v_repeat[match_mask]
        matched_v_indices = bi_edge_ids[v_nbr_indices[match_mask].to(torch.long)]
        matched_u_indices = bi_edge_ids[sorted_u_indices[matched_u_pos[match_mask]]]

        source_parts.append(chunk_edge_ids[matched_edge_local].to(torch.long))
        left_parts.append(matched_v_indices)
        right_parts.append(matched_u_indices)
        head = tail

    if not source_parts:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty, empty

    return torch.cat(source_parts), torch.cat(left_parts), torch.cat(right_parts)


def _edge_set_cache_key(edge_ids: torch.Tensor):
    if edge_ids.numel() == 0:
        return (0, b"")
    edge_ids_cpu = edge_ids.detach().to("cpu", dtype=torch.int64).contiguous()
    return edge_ids_cpu.numel(), edge_ids_cpu.numpy().tobytes()


def _build_edge_src_index(row_ptr: torch.Tensor):
    degrees = row_ptr[1:] - row_ptr[:-1]
    return torch.repeat_interleave(
        torch.arange(row_ptr.size(0) - 1, device=row_ptr.device, dtype=torch.long),
        degrees.to(torch.long),
    )


def _prepare_graph_runtime(row_ptr, columns, edge_code=None, bidirectional_view=None, edge_src_index=None):
    if edge_code is None:
        _, edge_code = _build_edge_index(row_ptr, columns)
    if bidirectional_view is None:
        bidirectional_view = _build_bidirectional_view(row_ptr, columns)
    if edge_src_index is None:
        edge_src_index = _build_edge_src_index(row_ptr)
    return edge_code, bidirectional_view, edge_src_index


def _prepare_traverse_runtime(row_ptr, columns, bidirectional_view=None, edge_src_index=None):
    if bidirectional_view is None:
        bidirectional_view = _build_bidirectional_view(row_ptr, columns)
    if edge_src_index is None:
        edge_src_index = _build_edge_src_index(row_ptr)
    return bidirectional_view, edge_src_index


def _get_cached_local_triangles(
    row_ptr,
    columns,
    edge_ids,
    bidirectional_view=None,
    edge_src_index=None,
    triangle_cache=None,
):
    if triangle_cache is None:
        return _collect_edge_triangles(
            row_ptr, columns, edge_ids, bidirectional_view=bidirectional_view, edge_src_index=edge_src_index
        )

    key = _edge_set_cache_key(edge_ids)
    triangles = triangle_cache.get(key)
    if triangles is None:
        triangles = _collect_edge_triangles(
            row_ptr, columns, edge_ids, bidirectional_view=bidirectional_view, edge_src_index=edge_src_index
        )
        triangle_cache[key] = triangles
    return triangles


def _get_cached_triangle_pack(
    row_ptr,
    columns,
    edge_ids,
    bidirectional_view=None,
    edge_src_index=None,
    triangle_cache=None,
):
    if triangle_cache is None:
        source_edge, left_edge, right_edge = _collect_edge_triangles(
            row_ptr, columns, edge_ids, bidirectional_view=bidirectional_view, edge_src_index=edge_src_index
        )
        source_local = _map_ids_to_local(source_edge, edge_ids) if source_edge.numel() > 0 else \
            torch.empty((0,), device=device, dtype=torch.long)
        return source_edge, left_edge, right_edge, source_local

    key = ("pack",) + _edge_set_cache_key(edge_ids)
    triangle_pack = triangle_cache.get(key)
    if triangle_pack is None:
        source_edge, left_edge, right_edge = _collect_edge_triangles(
            row_ptr, columns, edge_ids, bidirectional_view=bidirectional_view, edge_src_index=edge_src_index
        )
        source_local = _map_ids_to_local(source_edge, edge_ids) if source_edge.numel() > 0 else \
            torch.empty((0,), device=device, dtype=torch.long)
        triangle_pack = (source_edge, left_edge, right_edge, source_local)
        triangle_cache[key] = triangle_pack
    return triangle_pack


def _expand_ranges(starts: torch.Tensor, ends: torch.Tensor):
    sizes = ends - starts
    if sizes.numel() == 0 or int(sizes.sum().item()) == 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    ptr = torch.cat((torch.tensor([0], device=device), sizes.cumsum(0)))
    return torch.arange(ptr[-1], device=device) - torch.repeat_interleave(ptr[:-1] - starts, sizes)


def _map_ids_to_local(ids: torch.Tensor, reference_ids: torch.Tensor):
    if ids.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    sorted_reference, perm = torch.sort(reference_ids)
    pos = torch.searchsorted(sorted_reference, ids)
    return perm[pos]


def _isin_sorted(sorted_reference: torch.Tensor, values: torch.Tensor):
    if sorted_reference.numel() == 0 or values.numel() == 0:
        return torch.zeros((values.numel(),), device=device, dtype=torch.bool)

    pos = torch.searchsorted(sorted_reference, values)
    valid = pos < sorted_reference.numel()
    if valid.any():
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        valid[valid_idx] = sorted_reference[pos[valid_idx]] == values[valid_idx]
    return valid


def _filter_triangle_pack_by_source(source_edge, left_edge, right_edge, source_ids: torch.Tensor):
    if source_ids.numel() == 0 or source_edge.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.long)
        return empty, empty, empty

    keep_mask = _isin_sorted(torch.sort(source_ids).values, source_edge)
    return source_edge[keep_mask], left_edge[keep_mask], right_edge[keep_mask]


def _merge_active_candidate_edges(left_edges: torch.Tensor, right_edges: torch.Tensor, k_mask: torch.Tensor, evicted_mask: torch.Tensor):
    if left_edges.numel() == 0 and right_edges.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    if left_edges.numel() > 0:
        left_keep = k_mask[left_edges] & (~evicted_mask[left_edges])
        left_edges = left_edges[left_keep]
    if right_edges.numel() > 0:
        right_keep = k_mask[right_edges] & (~evicted_mask[right_edges])
        right_edges = right_edges[right_keep]
    if left_edges.numel() == 0:
        return torch.unique(right_edges)
    if right_edges.numel() == 0:
        return torch.unique(left_edges)
    return torch.unique(torch.cat((left_edges, right_edges)))


def _merge_new_frontier_edges(
    left_edges: torch.Tensor,
    right_edges: torch.Tensor,
    k_mask: torch.Tensor,
    evicted_mask: torch.Tensor,
    visited_mask: torch.Tensor,
):
    if left_edges.numel() == 0 and right_edges.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    if left_edges.numel() > 0:
        left_keep = k_mask[left_edges] & (~evicted_mask[left_edges]) & (~visited_mask[left_edges])
        left_edges = left_edges[left_keep]
    if right_edges.numel() > 0:
        right_keep = k_mask[right_edges] & (~evicted_mask[right_edges]) & (~visited_mask[right_edges])
        right_edges = right_edges[right_keep]
    if left_edges.numel() == 0:
        return torch.unique(right_edges)
    if right_edges.numel() == 0:
        return torch.unique(left_edges)
    return torch.unique(torch.cat((left_edges, right_edges)))


def _lookup_edge_ids(n: int, edge_code: torch.Tensor, edge_pairs: torch.Tensor):
    if edge_pairs.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long), torch.empty((0,), device=device, dtype=torch.bool)

    query_code = edge_pairs[:, 0].to(torch.long) * n + edge_pairs[:, 1].to(torch.long)
    pos = torch.searchsorted(edge_code, query_code)
    valid = pos < edge_code.numel()
    if valid.any():
        valid_idx = torch.nonzero(valid, as_tuple=False).flatten()
        valid[valid_idx] = edge_code[pos[valid_idx]] == query_code[valid_idx]
    return pos[valid], valid


def decompose_from_csr(row_ptr: torch.Tensor, columns: torch.Tensor):
    edge_starts, edge_ends = _csr_to_coo(row_ptr, columns)
    edge_num = columns.size(0)
    triangle_source, triangle_id, _, _, _ = calculate_support3(edge_starts, edge_ends, row_ptr, columns)
    return truss_decomposition(triangle_source, triangle_id, edge_num)


def compare_with_recompute(row_ptr: torch.Tensor, columns: torch.Tensor, truss_result: torch.Tensor, title: str):
    recomputed = decompose_from_csr(row_ptr, columns)
    is_same = torch.equal(truss_result, recomputed)

    print(f"{title} maintenance == recomputed", is_same)
    if not is_same:
        diff_idx = torch.nonzero(truss_result != recomputed, as_tuple=False).flatten()
        print(f"{title} diff edge ids", diff_idx)
        print(f"{title} maintenance values", truss_result[diff_idx])
        print(f"{title} recomputed values", recomputed[diff_idx])

    return is_same, recomputed


def calculate_low_b(inserted_edge_ids, truss_result, source_edge, left_edge, right_edge):
    """
    论文 Definition 4 / Algorithm 1 中的 pre-truss number 计算。
    """
    pre_values = torch.full((inserted_edge_ids.size(0),), 2, dtype=truss_result.dtype, device=device)
    if inserted_edge_ids.numel() == 0 or source_edge.numel() == 0:
        return pre_values

    triangle_levels = torch.minimum(truss_result[left_edge], truss_result[right_edge])
    match_mask = torch.isin(source_edge, inserted_edge_ids)
    if not torch.any(match_mask):
        return pre_values

    matched_source = source_edge[match_mask]
    matched_levels = triangle_levels[match_mask]
    local_ids = _map_ids_to_local(matched_source, inserted_edge_ids)

    repeat_cnt = torch.clamp(matched_levels.to(torch.long) - 1, min=0)
    valid_repeat = repeat_cnt > 0
    if not torch.any(valid_repeat):
        return pre_values

    local_ids = local_ids[valid_repeat]
    matched_levels = matched_levels[valid_repeat].to(torch.long)
    repeat_cnt = repeat_cnt[valid_repeat]

    expanded_local = torch.repeat_interleave(local_ids, repeat_cnt)
    k_values = _expand_ranges(
        torch.full_like(matched_levels, 2, dtype=torch.long),
        matched_levels + 1
    )

    max_k = int(matched_levels.max().item())
    bucket_code = expanded_local * (max_k + 1) + k_values
    counts = torch.bincount(
        bucket_code,
        minlength=inserted_edge_ids.size(0) * (max_k + 1)
    ).reshape(inserted_edge_ids.size(0), max_k + 1)

    k_range = torch.arange(2, max_k + 1, device=device, dtype=truss_result.dtype)
    valid_k = counts[:, 2:] >= (k_range - 2).unsqueeze(0)
    pre_values = torch.where(
        valid_k,
        k_range.unsqueeze(0).expand_as(valid_k),
        torch.full_like(valid_k, 2, dtype=truss_result.dtype)
    ).amax(dim=1)

    return pre_values


def build_affected_edge_groups(inserted_edge_ids, truss_result, source_edge, left_edge, right_edge):
    """
    论文 Definition 9 中各插入边对应的 source edge set 候选 Ae。
    返回：
        affected_flat: 所有 affected edge 的扁平拼接
        affected_ptr : 与 inserted_edge_ids 对齐的 CSR 指针
    """
    if inserted_edge_ids.numel() == 0 or source_edge.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long), torch.zeros(
            inserted_edge_ids.size(0) + 1, device=device, dtype=torch.long
        )

    tri_truss = torch.minimum(truss_result[left_edge], truss_result[right_edge])
    source_mask = torch.isin(source_edge, inserted_edge_ids)
    if not torch.any(source_mask):
        return torch.empty((0,), device=device, dtype=torch.long), torch.zeros(
            inserted_edge_ids.size(0) + 1, device=device, dtype=torch.long
        )

    valid_mask = source_mask & (tri_truss <= truss_result[source_edge])
    left_mask = valid_mask & (truss_result[left_edge] == tri_truss)
    right_mask = valid_mask & (truss_result[right_edge] == tri_truss)

    affected_source = torch.cat((source_edge[left_mask], source_edge[right_mask]))
    affected_edge = torch.cat((left_edge[left_mask], right_edge[right_mask]))

    if affected_edge.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long), torch.zeros(
            inserted_edge_ids.size(0) + 1, device=device, dtype=torch.long
        )

    pair_code = affected_source.to(torch.long) * truss_result.size(0) + affected_edge.to(torch.long)
    pair_code = torch.unique(pair_code, sorted=True)
    affected_source = torch.div(pair_code, truss_result.size(0), rounding_mode='floor')
    affected_edge = torch.remainder(pair_code, truss_result.size(0))

    local_ids = _map_ids_to_local(affected_source, inserted_edge_ids)
    order = torch.argsort(local_ids, stable=True)
    local_ids = local_ids[order]
    affected_edge = affected_edge[order]

    counts = torch.bincount(local_ids, minlength=inserted_edge_ids.size(0))
    affected_ptr = torch.cat((torch.tensor([0], device=device), counts.cumsum(0)))
    return affected_edge, affected_ptr


def _valid_support_mask(truss_result, k, evicted_mask):
    return (truss_result > k) | ((truss_result == k) & (~evicted_mask))


def _compute_ts(source_edge, left_edge, right_edge, truss_result, k, evicted_mask):
    if source_edge.numel() == 0:
        return torch.zeros_like(truss_result)

    valid_support = _valid_support_mask(truss_result, k, evicted_mask)
    source_mask = (truss_result[source_edge] == k) & (~evicted_mask[source_edge])
    triangle_mask = source_mask & valid_support[left_edge] & valid_support[right_edge]
    return torch.bincount(source_edge[triangle_mask], minlength=truss_result.size(0))


def _compute_ts_local(source_edge, left_edge, right_edge, edge_ids, truss_result, k, evicted_mask):
    if edge_ids.numel() == 0 or source_edge.numel() == 0:
        return torch.zeros((edge_ids.numel(),), device=device, dtype=torch.long)

    valid_support = _valid_support_mask(truss_result, k, evicted_mask)
    source_mask = (truss_result[source_edge] == k) & (~evicted_mask[source_edge])
    triangle_mask = source_mask & valid_support[left_edge] & valid_support[right_edge]
    if not torch.any(triangle_mask):
        return torch.zeros((edge_ids.numel(),), device=device, dtype=torch.long)

    local_ids = _map_ids_to_local(source_edge[triangle_mask], edge_ids)
    return torch.bincount(local_ids, minlength=edge_ids.size(0))


def _compute_ts_local_packed(
    source_edge,
    left_edge,
    right_edge,
    source_local_ids,
    edge_count,
    truss_result,
    k,
    evicted_mask,
):
    if edge_count == 0 or source_edge.numel() == 0:
        return torch.zeros((edge_count,), device=device, dtype=torch.long)

    valid_support = _valid_support_mask(truss_result, k, evicted_mask)
    source_mask = (truss_result[source_edge] == k) & (~evicted_mask[source_edge])
    triangle_mask = source_mask & valid_support[left_edge] & valid_support[right_edge]
    if not torch.any(triangle_mask):
        return torch.zeros((edge_count,), device=device, dtype=torch.long)
    return torch.bincount(source_local_ids[triangle_mask], minlength=edge_count)


def _compute_ts_local_with_support_mask(
    source_edge,
    left_edge,
    right_edge,
    source_local_ids,
    edge_count,
    valid_support_mask,
    evicted_mask,
):
    if edge_count == 0 or source_edge.numel() == 0:
        return torch.zeros((edge_count,), device=device, dtype=torch.long)

    triangle_mask = (~evicted_mask[source_edge]) & valid_support_mask[left_edge] & valid_support_mask[right_edge]
    if not torch.any(triangle_mask):
        return torch.zeros((edge_count,), device=device, dtype=torch.long)
    return torch.bincount(source_local_ids[triangle_mask], minlength=edge_count)


def _compute_ts_for_edges(
    row_ptr,
    columns,
    edge_ids,
    truss_result,
    k,
    evicted_mask,
    bidirectional_view=None,
    triangle_cache=None,
):
    if edge_ids.numel() == 0:
        return torch.zeros((0,), device=device, dtype=torch.long)

    source_edge, left_edge, right_edge, source_local = _get_cached_triangle_pack(
        row_ptr,
        columns,
        edge_ids,
        bidirectional_view=bidirectional_view,
        triangle_cache=triangle_cache,
    )
    return _compute_ts_local_packed(
        source_edge,
        left_edge,
        right_edge,
        source_local,
        edge_ids.size(0),
        truss_result,
        k,
        evicted_mask,
    )


def _insertion_traverse_prepared(row_ptr, columns, truss_result, source_edges, k, bidirectional_view, edge_src_index, traverse_stats=None):
    """
    使用 PyTorch Tensor 复现论文 Algorithm 2 的核心并行逻辑：
    用 frontier + mask + bincount 维护访问集合与淘汰集合。
    """
    if source_edges.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)

    num_edges = truss_result.size(0)
    k_mask = truss_result == k
    visited_mask = torch.zeros((num_edges,), device=device, dtype=torch.bool)
    evicted_mask = torch.zeros((num_edges,), device=device, dtype=torch.bool)
    valid_support_mask = k_mask.clone()
    valid_support_mask |= truss_result > k
    frontier = torch.unique(source_edges[k_mask[source_edges]])
    if frontier.numel() == 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    visited_mask[frontier] = True
    processed_edge_visits = 0
    processed_triangle_visits = 0
    tracked_edges = frontier
    frontier_source_edge, frontier_left_edge, frontier_right_edge = _collect_edge_triangles(
        row_ptr,
        columns,
        frontier,
        bidirectional_view=bidirectional_view,
        edge_src_index=edge_src_index,
    )
    processed_triangle_visits += int(frontier_source_edge.numel())
    if traverse_stats is not None and frontier_source_edge.numel() > 0:
        triangle_pack = torch.sort(
            torch.stack((frontier_source_edge, frontier_left_edge, frontier_right_edge), dim=1),
            dim=1,
        ).values
        traverse_stats.setdefault("triangle_batches", []).append(triangle_pack)
    tracked_source_edge = frontier_source_edge
    tracked_left_edge = frontier_left_edge
    tracked_right_edge = frontier_right_edge
    tracked_source_local = _map_ids_to_local(tracked_source_edge, tracked_edges) if tracked_source_edge.numel() > 0 else \
        torch.empty((0,), device=device, dtype=torch.long)

    while frontier.numel() > 0:
        tri_mask = valid_support_mask[frontier_left_edge] & valid_support_mask[frontier_right_edge]
        next_frontier = _merge_new_frontier_edges(
            frontier_left_edge[tri_mask],
            frontier_right_edge[tri_mask],
            k_mask,
            evicted_mask,
            visited_mask,
        )
        if next_frontier.numel() == 0:
            break

        new_source_edge, new_left_edge, new_right_edge = _collect_edge_triangles(
            row_ptr,
            columns,
            next_frontier,
            bidirectional_view=bidirectional_view,
            edge_src_index=edge_src_index,
        )
        processed_triangle_visits += int(new_source_edge.numel())
        if traverse_stats is not None and new_source_edge.numel() > 0:
            triangle_pack = torch.sort(
                torch.stack((new_source_edge, new_left_edge, new_right_edge), dim=1),
                dim=1,
            ).values
            traverse_stats.setdefault("triangle_batches", []).append(triangle_pack)
        new_source_local = _map_ids_to_local(new_source_edge, next_frontier) + tracked_edges.size(0) if \
            new_source_edge.numel() > 0 else torch.empty((0,), device=device, dtype=torch.long)
        candidate_active_edges = torch.cat((tracked_edges, next_frontier))
        candidate_source_edge = torch.cat((tracked_source_edge, new_source_edge))
        candidate_left_edge = torch.cat((tracked_left_edge, new_left_edge))
        candidate_right_edge = torch.cat((tracked_right_edge, new_right_edge))
        candidate_source_local = torch.cat((tracked_source_local, new_source_local))
        processed_edge_visits += int(candidate_active_edges.numel())
        while True:
            ts = _compute_ts_local_with_support_mask(
                candidate_source_edge,
                candidate_left_edge,
                candidate_right_edge,
                candidate_source_local,
                candidate_active_edges.size(0),
                valid_support_mask,
                evicted_mask,
            )
            active_mask = ~evicted_mask[candidate_active_edges]
            new_evict_edges = candidate_active_edges[active_mask & (ts <= k - 2)]
            if new_evict_edges.numel() == 0:
                break
            evicted_mask[new_evict_edges] = True
            valid_support_mask[new_evict_edges] = False

        tracked_edges = candidate_active_edges
        tracked_source_edge = candidate_source_edge
        tracked_left_edge = candidate_left_edge
        tracked_right_edge = candidate_right_edge
        tracked_source_local = candidate_source_local
        next_frontier = next_frontier[~evicted_mask[next_frontier]]
        if next_frontier.numel() == 0:
            break

        keep_mask = ~evicted_mask[new_source_edge]
        frontier_source_edge = new_source_edge[keep_mask]
        frontier_left_edge = new_left_edge[keep_mask]
        frontier_right_edge = new_right_edge[keep_mask]
        visited_mask[next_frontier] = True
        frontier = next_frontier

    result_edges = tracked_edges[~evicted_mask[tracked_edges]]
    if traverse_stats is not None:
        traverse_stats["unique_edges"] = traverse_stats.get("unique_edges", 0) + int(torch.count_nonzero(visited_mask).item())
        traverse_stats["edge_visits"] = traverse_stats.get("edge_visits", 0) + processed_edge_visits
        traverse_stats["triangle_visits"] = traverse_stats.get("triangle_visits", 0) + processed_triangle_visits
    return result_edges


def insertion_traverse(row_ptr, columns, truss_result, source_edges, k, bidirectional_view=None, edge_src_index=None, traverse_stats=None):
    bidirectional_view, edge_src_index = _prepare_traverse_runtime(
        row_ptr,
        columns,
        bidirectional_view=bidirectional_view,
        edge_src_index=edge_src_index,
    )
    return _insertion_traverse_prepared(
        row_ptr,
        columns,
        truss_result,
        source_edges,
        k,
        bidirectional_view,
        edge_src_index,
        traverse_stats=traverse_stats,
    )


def _prepare_insert_edges(row_ptr, columns, ins_src, ins_dst):
    if ins_src.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)

    ins_src = ins_src.clone()
    ins_dst = ins_dst.clone()

    swap_mask = ins_src > ins_dst
    temp = ins_src[swap_mask].clone()
    ins_src[swap_mask] = ins_dst[swap_mask]
    ins_dst[swap_mask] = temp

    mask = ins_src != ins_dst
    ins_src = ins_src[mask]
    ins_dst = ins_dst[mask]

    if ins_src.numel() == 0:
        return torch.empty((0, 2), device=device, dtype=torch.long)

    edges = torch.unique(torch.stack([ins_src, ins_dst], dim=1), dim=0)
    n, edge_code = _build_edge_index(row_ptr, columns)
    _, valid = _lookup_edge_ids(n, edge_code, edges)
    exist_mask = torch.zeros((edges.size(0),), device=device, dtype=torch.bool)
    exist_mask[torch.nonzero(valid, as_tuple=False).flatten()] = True
    return edges[~exist_mask]


def _select_est(inserted_pairs: torch.Tensor, affected_flat: torch.Tensor, affected_ptr: torch.Tensor, edge_num: int):
    num_inserted = inserted_pairs.size(0)
    if num_inserted == 0:
        return inserted_pairs, torch.empty((0,), device=device, dtype=torch.bool)

    membership_counts = affected_ptr[1:] - affected_ptr[:-1]
    selected_mask = torch.zeros((num_inserted,), device=device, dtype=torch.bool)

    # 没有 Ae 的插入边天然满足 EST 条件，可直接并行加入。
    zero_membership = membership_counts == 0
    selected_mask |= zero_membership

    if affected_flat.numel() == 0:
        if not torch.any(selected_mask):
            selected_mask[0] = True
        return inserted_pairs[selected_mask], selected_mask

    membership_owner = torch.repeat_interleave(
        torch.arange(num_inserted, device=device, dtype=torch.long),
        membership_counts
    )

    active_mask = ~zero_membership
    large_rank = torch.tensor(num_inserted, device=device, dtype=torch.long)

    while torch.any(active_mask):
        active_membership = active_mask[membership_owner]
        if not torch.any(active_membership):
            break

        membership_rank = membership_owner.clone()
        membership_rank[~active_membership] = large_rank

        min_rank_per_affected = torch.full(
            (edge_num,), large_rank, device=device, dtype=torch.long
        )
        min_rank_per_affected.scatter_reduce_(
            0, affected_flat, membership_rank, reduce="amin", include_self=True
        )

        winning_membership = active_membership & (
            membership_owner == min_rank_per_affected[affected_flat]
        )

        losing_counts = torch.bincount(
            membership_owner[active_membership & (~winning_membership)],
            minlength=num_inserted
        )
        round_selected = active_mask & (losing_counts == 0)

        if not torch.any(round_selected):
            first_active = torch.nonzero(active_mask, as_tuple=False).flatten()[0]
            round_selected = torch.zeros_like(active_mask)
            round_selected[first_active] = True

        selected_mask |= round_selected

        selected_membership = round_selected[membership_owner]
        occupied_by_selected = torch.zeros((edge_num,), device=device, dtype=torch.bool)
        if torch.any(selected_membership):
            occupied_by_selected[affected_flat[selected_membership]] = True

        conflicted = torch.bincount(
            membership_owner[active_membership & occupied_by_selected[affected_flat]],
            minlength=num_inserted
        ) > 0
        active_mask &= ~conflicted

    if not torch.any(selected_mask):
        selected_mask[0] = True
    return inserted_pairs[selected_mask], selected_mask


def _build_source_edge_groups(affected_flat, affected_ptr, truss_result):
    if affected_ptr.numel() <= 1:
        return {}

    selected_indices = _expand_ranges(affected_ptr[:-1], affected_ptr[1:])
    if selected_indices.numel() == 0 or affected_flat.numel() == 0:
        return {}

    affected_edges = torch.unique(affected_flat[selected_indices])
    edge_truss = truss_result[affected_edges]
    groups = {}
    for k in torch.unique(edge_truss).detach().cpu().tolist():
        k = int(k)
        groups[k] = affected_edges[edge_truss == k]
    return groups


def _superior_insert_prepared(row_ptr, columns, ins_src, ins_dst, truss_result, edge_code, bidirectional_view, edge_src_index, return_stats: bool = False):
    remaining_edges = _prepare_insert_edges(row_ptr, columns, ins_src, ins_dst)
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
        rem_src = remaining_edges[:, 0]
        rem_dst = remaining_edges[:, 1]
        temp_inserted_ids = columns.size(0) + torch.arange(
            remaining_edges.size(0), device=device, dtype=torch.long
        )
        temp_truss = torch.cat((
            truss_result,
            torch.full((remaining_edges.size(0),), 2, dtype=truss_result.dtype, device=device)
        ))
        temp_bidirectional_view = _augment_bidirectional_view(
            bidirectional_view, rem_src, rem_dst, temp_inserted_ids, n
        )
        temp_source_edge, temp_left_edge, temp_right_edge = _collect_edge_triangles(
            row_ptr,
            columns,
            temp_inserted_ids,
            bidirectional_view=temp_bidirectional_view,
            query_src=rem_src,
            query_dst=rem_dst,
        )
        pre_truss = calculate_low_b(temp_inserted_ids, temp_truss, temp_source_edge, temp_left_edge, temp_right_edge)
        temp_truss[temp_inserted_ids] = pre_truss

        affected_flat, affected_ptr = build_affected_edge_groups(
            temp_inserted_ids, temp_truss, temp_source_edge, temp_left_edge, temp_right_edge
        )
        selected_edges, selected_mask = _select_est(
            remaining_edges, affected_flat, affected_ptr, temp_truss.size(0)
        )

        row_ptr, columns, truss_result, edge_code, bidirectional_view = insert_edges_csr(
            row_ptr,
            columns,
            selected_edges[:, 0],
            selected_edges[:, 1],
            truss_result,
            return_bidirectional_view=True,
        )
        edge_src_index = _build_edge_src_index(row_ptr)
        selected_ids, _ = _lookup_edge_ids(n, edge_code, selected_edges)
        truss_result[selected_ids] = pre_truss[selected_mask]
        selected_source_edge, selected_left_edge, selected_right_edge = _collect_edge_triangles(
            row_ptr, columns, selected_ids, bidirectional_view=bidirectional_view, edge_src_index=edge_src_index
        )

        selected_affected_flat, selected_affected_ptr = build_affected_edge_groups(
            selected_ids, truss_result, selected_source_edge, selected_left_edge, selected_right_edge
        )
        source_edge_groups = _build_source_edge_groups(selected_affected_flat, selected_affected_ptr, truss_result)

        for k in sorted(source_edge_groups):
            if stats is not None and source_edge_groups[k].numel() > 0:
                source_edges = source_edge_groups[k]
                stats.setdefault("affected_edge_code_batches", []).append(edge_code[source_edges])
                tri_source, tri_left, tri_right = _collect_edge_triangles(
                    row_ptr,
                    columns,
                    source_edges,
                    bidirectional_view=bidirectional_view,
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
            increased_edges = _insertion_traverse_prepared(
                row_ptr,
                columns,
                truss_result,
                source_edge_groups[k],
                k,
                bidirectional_view,
                edge_src_index,
                traverse_stats=stats,
            )
            if increased_edges.numel() > 0:
                truss_result[increased_edges] += 1

        remain_mask = ~torch.isin(
            remaining_edges[:, 0].to(torch.long) * n + remaining_edges[:, 1].to(torch.long),
            selected_edges[:, 0].to(torch.long) * n + selected_edges[:, 1].to(torch.long)
        )
        remaining_edges = remaining_edges[remain_mask]

    if stats is not None:
        return row_ptr, columns, truss_result, stats
    return row_ptr, columns, truss_result


def superior_insert(row_ptr, columns, ins_src, ins_dst, truss_result, edge_code=None, bidirectional_view=None, return_stats: bool = False):
    del edge_code, bidirectional_view
    state = state_from_csr(row_ptr.clone(), columns.clone(), truss_result.clone())
    delta = {
        "ins_edges": torch.stack((ins_src.to(torch.long), ins_dst.to(torch.long)), dim=1)
        if ins_src.numel() > 0 else torch.empty((0, 2), device=row_ptr.device, dtype=torch.long),
        "is_local": True,
    }
    new_state, stats = maintain_truss(state, delta, device=row_ptr.device)
    if return_stats:
        return new_state.row_ptr, new_state.columns, new_state.truss, stats["insert"]
    return new_state.row_ptr, new_state.columns, new_state.truss


def run(filename: str, updated_filename: str):
    edge_starts, edge_ends, old_vertices_hash, row_ptr, columns, _ = load_graph_as_csr(filename, 0)
    truss_result = decompose_from_csr(row_ptr, columns)

    print("original truss_result", truss_result)
    print("original row_ptr, columns", row_ptr, columns)

    ins_src, ins_dst = read_update_edge_txt(updated_filename, old_vertices_hash, 0)
    ins_src = torch.tensor(ins_src, device=device, dtype=torch.long)
    ins_dst = torch.tensor(ins_dst, device=device, dtype=torch.long)

    row_ptr, columns, truss_result = superior_insert(row_ptr, columns, ins_src, ins_dst, truss_result)
    print("updated truss_result", truss_result)
    print("updated row_ptr, columns", row_ptr, columns)

    recomputed_ok, recomputed_truss_result = compare_with_recompute(row_ptr, columns, truss_result, "insert")
    if recomputed_ok:
        print("recomputed truss_result", recomputed_truss_result)

    return row_ptr, columns, truss_result


if __name__ == '__main__':
    run(r"C:\Users\Administrator\Desktop\update_g\g1.txt", r"C:\Users\Administrator\Desktop\update_g\g1_del.txt")
