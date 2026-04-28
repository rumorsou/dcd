import argparse
import statistics
import time

import torch

from delete_maintenance import _superior_remove_prepared, superior_remove
from insert_maintenace import (
    _prepare_graph_runtime,
    _superior_insert_prepared,
    compare_with_recompute,
    decompose_from_csr,
    device,
    load_graph_as_csr,
    read_update_edge_txt,
    superior_insert,
)


def _sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_runs(fn, warmup_runs: int, repeat_runs: int):
    for _ in range(warmup_runs):
        fn()
        _sync_if_needed()

    timings = []
    result = None
    for _ in range(repeat_runs):
        _sync_if_needed()
        start = time.perf_counter()
        result = fn()
        _sync_if_needed()
        timings.append(time.perf_counter() - start)
    return result, timings


def _median(timings):
    return statistics.median(timings)


def _print_timing_stats(label: str, timings):
    print(f"{label} runs:", ", ".join(f"{t:.6f}s" for t in timings))
    print(f"{label} mean: {statistics.mean(timings):.6f}s")
    print(f"{label} median: {_median(timings):.6f}s")


def _print_speedup(label: str, baseline_timings, new_timings):
    baseline_median = _median(baseline_timings)
    new_median = _median(new_timings)
    speedup = baseline_median / new_median if new_median > 0 else float("inf")
    print(f"{label} speedup vs baseline: {speedup:.3f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark legacy baseline vs new cone-local truss maintenance.")
    parser.add_argument("graph_file", help="Path to the base graph file")
    parser.add_argument("sampled_edge_file", help="Path to the sampled update edge file")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup run count per stage")
    parser.add_argument("--repeat", type=int, default=5, help="Timed run count per stage")
    args = parser.parse_args()

    _, _, vertex_to_index, row_ptr, columns, bidirectional_view = load_graph_as_csr(args.graph_file, 0)
    sampled_src, sampled_dst = read_update_edge_txt(args.sampled_edge_file, vertex_to_index, 0)
    sampled_src = torch.tensor(sampled_src, device=device, dtype=torch.long)
    sampled_dst = torch.tensor(sampled_dst, device=device, dtype=torch.long)

    def run_decomposition():
        return decompose_from_csr(row_ptr, columns)

    original_truss, decomposition_timings = _measure_runs(run_decomposition, args.warmup, args.repeat)
    print("original truss decomposition completed")
    print("original truss edge count:", int(original_truss.numel()))

    delete_edge_code, delete_bidirectional_view, delete_edge_src_index = _prepare_graph_runtime(
        row_ptr,
        columns,
        bidirectional_view=bidirectional_view,
    )

    def run_delete_baseline():
        return _superior_remove_prepared(
            row_ptr.clone(),
            columns.clone(),
            sampled_src,
            sampled_dst,
            original_truss.clone(),
            delete_edge_code,
            delete_bidirectional_view,
            delete_edge_src_index,
            return_stats=False,
        )

    def run_delete_new():
        return superior_remove(
            row_ptr.clone(),
            columns.clone(),
            sampled_src,
            sampled_dst,
            original_truss.clone(),
            return_stats=False,
        )

    _, delete_baseline_timings = _measure_runs(run_delete_baseline, args.warmup, args.repeat)
    (deleted_row_ptr, deleted_columns, deleted_truss), delete_new_timings = _measure_runs(
        run_delete_new, args.warmup, args.repeat
    )
    deleted_row_ptr, deleted_columns, deleted_truss, delete_stats = superior_remove(
        row_ptr.clone(),
        columns.clone(),
        sampled_src,
        sampled_dst,
        original_truss.clone(),
        return_stats=True,
    )

    delete_ok, _ = compare_with_recompute(deleted_row_ptr, deleted_columns, deleted_truss, "delete")
    print("delete maintenance completed")
    print("delete graph edge count:", int(deleted_columns.numel()))
    print("delete cone unique edge visits:", delete_stats["unique_edges"])
    print("delete cone total edge visits:", delete_stats["edge_visits"])
    print("delete cone max edge count:", delete_stats["cone_edges"])
    print("delete cone max triangle count:", delete_stats["cone_triangles"])

    insert_edge_code, insert_bidirectional_view, insert_edge_src_index = _prepare_graph_runtime(
        deleted_row_ptr,
        deleted_columns,
    )

    def run_insert_baseline():
        return _superior_insert_prepared(
            deleted_row_ptr.clone(),
            deleted_columns.clone(),
            sampled_src,
            sampled_dst,
            deleted_truss.clone(),
            insert_edge_code,
            insert_bidirectional_view,
            insert_edge_src_index,
            return_stats=False,
        )

    def run_insert_new():
        return superior_insert(
            deleted_row_ptr.clone(),
            deleted_columns.clone(),
            sampled_src,
            sampled_dst,
            deleted_truss.clone(),
            return_stats=False,
        )

    _, insert_baseline_timings = _measure_runs(run_insert_baseline, args.warmup, args.repeat)
    (inserted_row_ptr, inserted_columns, inserted_truss), insert_new_timings = _measure_runs(
        run_insert_new, args.warmup, args.repeat
    )
    inserted_row_ptr, inserted_columns, inserted_truss, insert_stats = superior_insert(
        deleted_row_ptr.clone(),
        deleted_columns.clone(),
        sampled_src,
        sampled_dst,
        deleted_truss.clone(),
        return_stats=True,
    )

    insert_ok, _ = compare_with_recompute(inserted_row_ptr, inserted_columns, inserted_truss, "insert")
    print("insert maintenance completed")
    print("insert graph edge count:", int(inserted_columns.numel()))
    print("insert cone unique edge visits:", insert_stats["unique_edges"])
    print("insert cone total edge visits:", insert_stats["edge_visits"])
    print("insert cone max edge count:", insert_stats["cone_edges"])
    print("insert cone max triangle count:", insert_stats["cone_triangles"])

    print("final graph == original graph:", torch.equal(inserted_row_ptr, row_ptr) and torch.equal(inserted_columns, columns))
    print("final truss == original truss:", torch.equal(inserted_truss, original_truss))

    _print_timing_stats("initial decomposition", decomposition_timings)
    _print_timing_stats("delete baseline", delete_baseline_timings)
    _print_timing_stats("delete new", delete_new_timings)
    _print_speedup("delete", delete_baseline_timings, delete_new_timings)

    _print_timing_stats("insert baseline", insert_baseline_timings)
    _print_timing_stats("insert new", insert_new_timings)
    _print_speedup("insert", insert_baseline_timings, insert_new_timings)

    if not delete_ok:
        print("warning: delete maintenance result differs from full recomputation")
    if not insert_ok:
        print("warning: insert maintenance result differs from full recomputation")


if __name__ == "__main__":
    main()
