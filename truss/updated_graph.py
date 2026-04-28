from typing import Optional, Tuple

import torch


def _csr_to_coo(row_ptr: torch.Tensor, columns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = row_ptr.size(0) - 1
    src = torch.repeat_interleave(
        torch.arange(n, device=row_ptr.device, dtype=torch.long),
        row_ptr[1:] - row_ptr[:-1],
    )
    return src, columns.to(torch.long)


def _build_csr_from_coo(
    src: torch.Tensor,
    dst: torch.Tensor,
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if src.numel() == 0:
        row_ptr = torch.zeros((n + 1,), device=device, dtype=torch.long)
        columns = torch.empty((0,), device=device, dtype=torch.int32)
        return row_ptr, columns

    deg = torch.bincount(src.to(torch.long), minlength=n)
    row_ptr = torch.cat(
        (
            torch.tensor([0], device=device, dtype=torch.long),
            deg.cumsum(0),
        )
    )
    return row_ptr, dst.to(torch.int32)


def build_edge_src_index(row_ptr: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(row_ptr.size(0) - 1, device=row_ptr.device, dtype=torch.long),
        row_ptr[1:] - row_ptr[:-1],
    )


def build_reverse_csr(
    row_ptr: torch.Tensor,
    columns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = row_ptr.device
    n = row_ptr.size(0) - 1
    if columns.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return torch.zeros((n + 1,), device=device, dtype=torch.long), empty, empty

    src, dst = _csr_to_coo(row_ptr, columns)
    edge_ids = torch.arange(columns.numel(), device=device, dtype=torch.long)
    order = torch.argsort(dst, stable=True)
    rev_dst = src[order]
    rev_edge_ids = edge_ids[order]
    in_counts = torch.bincount(dst.to(torch.long), minlength=n)
    rev_row_ptr = torch.cat(
        (
            torch.tensor([0], device=device, dtype=torch.long),
            in_counts.cumsum(0),
        )
    )
    return rev_row_ptr, rev_dst.to(torch.int32), rev_edge_ids.to(torch.int32)


def _build_bidirectional_view_from_sorted_edges(
    src: torch.Tensor,
    dst: torch.Tensor,
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dst.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        edge_ids = torch.empty((0,), device=device, dtype=torch.int32)
        row_ptr = torch.zeros((n + 1,), device=device, dtype=torch.long)
        return row_ptr, empty, edge_ids

    edge_ids = torch.arange(dst.size(0), device=device, dtype=torch.long)
    out_counts = torch.bincount(src.to(torch.long), minlength=n)
    in_counts = torch.bincount(dst.to(torch.long), minlength=n)
    deg = in_counts + out_counts
    bi_row_ptr = torch.cat(
        (
            torch.tensor([0], device=device, dtype=torch.long),
            deg.cumsum(0),
        )
    )

    bi_columns = torch.empty((dst.size(0) * 2,), device=device, dtype=torch.int32)
    bi_edge_ids = torch.empty((dst.size(0) * 2,), device=device, dtype=torch.int32)

    in_row_ptr = torch.cat(
        (
            torch.tensor([0], device=device, dtype=torch.long),
            in_counts.cumsum(0),
        )
    )
    in_offset = bi_row_ptr[:-1]
    out_offset = in_offset + in_counts

    order_in = torch.argsort(dst, stable=True)
    dst_sorted = dst[order_in].to(torch.long)
    src_sorted = src[order_in].to(torch.long)
    edge_ids_sorted = edge_ids[order_in]
    in_group_start = torch.repeat_interleave(in_row_ptr[:-1], in_counts)
    in_local_rank = torch.arange(dst_sorted.size(0), device=device, dtype=torch.long) - in_group_start
    incoming_positions = in_offset[dst_sorted] + in_local_rank

    src_row_ptr = torch.cat(
        (
            torch.tensor([0], device=device, dtype=torch.long),
            out_counts.cumsum(0),
        )
    )
    out_group_start = torch.repeat_interleave(src_row_ptr[:-1], out_counts)
    out_local_rank = torch.arange(src.size(0), device=device, dtype=torch.long) - out_group_start
    outgoing_positions = out_offset[src.to(torch.long)] + out_local_rank

    bi_columns[incoming_positions] = src_sorted.to(torch.int32)
    bi_edge_ids[incoming_positions] = edge_ids_sorted.to(torch.int32)
    bi_columns[outgoing_positions] = dst.to(torch.int32)
    bi_edge_ids[outgoing_positions] = edge_ids.to(torch.int32)
    return bi_row_ptr, bi_columns, bi_edge_ids


def _filter_bidirectional_view(
    bidirectional_view: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    keep_mask: torch.Tensor,
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bi_row_ptr, bi_columns, bi_edge_ids = bidirectional_view
    if bi_columns.numel() == 0:
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        return torch.zeros((n + 1,), device=device, dtype=torch.long), empty, empty

    bi_deg = bi_row_ptr[1:] - bi_row_ptr[:-1]
    bi_src = torch.repeat_interleave(
        torch.arange(n, device=device, dtype=torch.long),
        bi_deg,
    )
    bi_keep = keep_mask[bi_edge_ids.to(torch.long)]
    new_deg = torch.bincount(bi_src[bi_keep], minlength=n)
    new_row_ptr = torch.cat(
        (
            torch.tensor([0], device=device, dtype=torch.long),
            new_deg.cumsum(0),
        )
    )

    new_edge_ids = torch.cumsum(keep_mask.to(torch.long), dim=0) - 1
    new_columns = bi_columns[bi_keep]
    new_bi_edge_ids = new_edge_ids[bi_edge_ids[bi_keep].to(torch.long)].to(torch.int32)
    return new_row_ptr, new_columns, new_bi_edge_ids


def insert_edges_csr(
    row_ptr: torch.Tensor,
    columns: torch.Tensor,
    new_src: torch.Tensor,
    new_dst: torch.Tensor,
    truss_result: torch.Tensor,
    return_bidirectional_view: bool = False,
):
    device = row_ptr.device
    n = row_ptr.size(0) - 1
    src_orig, dst_orig = _csr_to_coo(row_ptr, columns)

    if truss_result.size(0) != columns.size(0):
        raise ValueError("truss_result length must match columns length.")

    orig_code = src_orig * n + dst_orig
    new_code = new_src.to(torch.long) * n + new_dst.to(torch.long)
    insert_mask = ~torch.isin(new_code, orig_code)
    new_code = torch.unique(new_code[insert_mask], sorted=True)

    if new_code.numel() == 0:
        if return_bidirectional_view:
            bidirectional_view = _build_bidirectional_view_from_sorted_edges(src_orig, dst_orig, n, device)
            return row_ptr, columns, truss_result, orig_code, bidirectional_view
        return row_ptr, columns, truss_result

    ins_src = torch.div(new_code, n, rounding_mode="floor")
    ins_dst = torch.remainder(new_code, n)
    ins_truss = torch.full((new_code.size(0),), 2, dtype=truss_result.dtype, device=device)

    all_src = torch.cat((src_orig, ins_src))
    all_dst = torch.cat((dst_orig, ins_dst))
    all_truss = torch.cat((truss_result, ins_truss))

    sort_code = all_src * n + all_dst
    order = torch.argsort(sort_code, stable=True)
    all_src = all_src[order]
    all_dst = all_dst[order]
    all_truss = all_truss[order]
    all_code = sort_code[order]

    new_row_ptr, new_columns = _build_csr_from_coo(all_src, all_dst, n, device)
    if return_bidirectional_view:
        bidirectional_view = _build_bidirectional_view_from_sorted_edges(all_src, all_dst, n, device)
        return new_row_ptr, new_columns, all_truss, all_code, bidirectional_view
    return new_row_ptr, new_columns, all_truss


def remove_edges_csr(
    row_ptr: torch.Tensor,
    columns: torch.Tensor,
    del_src: torch.Tensor,
    del_dst: torch.Tensor,
    truss_result: torch.Tensor,
    return_bidirectional_view: bool = False,
    bidirectional_view: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    device = row_ptr.device
    n = row_ptr.size(0) - 1
    src_orig, dst_orig = _csr_to_coo(row_ptr, columns)

    if truss_result.size(0) != columns.size(0):
        raise ValueError("truss_result length must match columns length.")

    orig_code = src_orig * n + dst_orig
    del_code = torch.unique(del_src.to(torch.long) * n + del_dst.to(torch.long), sorted=True)
    keep_mask = ~torch.isin(orig_code, del_code)

    src_keep = src_orig[keep_mask]
    dst_keep = dst_orig[keep_mask]
    truss_keep = truss_result[keep_mask]

    new_row_ptr, new_columns = _build_csr_from_coo(src_keep, dst_keep, n, device)
    keep_code = orig_code[keep_mask]
    if return_bidirectional_view:
        if bidirectional_view is not None:
            new_bidirectional_view = _filter_bidirectional_view(bidirectional_view, keep_mask, n, device)
        else:
            new_bidirectional_view = _build_bidirectional_view_from_sorted_edges(src_keep, dst_keep, n, device)
        return new_row_ptr, new_columns, truss_keep, keep_code, new_bidirectional_view
    return new_row_ptr, new_columns, truss_keep
