import argparse
import math
import statistics
import time
from pathlib import Path

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


def _geomean(values):
    safe = [value for value in values if value > 0]
    if not safe:
        return 0.0
    return math.exp(sum(math.log(value) for value in safe) / len(safe))


def _default_workloads(root: Path):
    datasets = ("facebook", "amazon", "dblp", "youtube")
    sizes = ("10", "100", "1000")
    workloads = []
    for dataset in datasets:
        graph = root / f"{dataset}.txt"
        for size in sizes:
            update = root / f"{dataset}_{size}.txt"
            workloads.append((graph, update))
    return workloads


def _benchmark_workload(graph_file: Path, update_file: Path, warmup: int, repeat: int):
    _, _, vertex_to_index, row_ptr, columns, bidirectional_view = load_graph_as_csr(str(graph_file), 0)
    sampled_src, sampled_dst = read_update_edge_txt(str(update_file), vertex_to_index, 0)
    sampled_src = torch.tensor(sampled_src, device=device, dtype=torch.long)
    sampled_dst = torch.tensor(sampled_dst, device=device, dtype=torch.long)
    original_truss = decompose_from_csr(row_ptr, columns)

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

    (_, _, _), delete_baseline_timings = _measure_runs(run_delete_baseline, warmup, repeat)
    (deleted_row_ptr, deleted_columns, deleted_truss), delete_new_timings = _measure_runs(run_delete_new, warmup, repeat)
    delete_ok, _ = compare_with_recompute(deleted_row_ptr, deleted_columns, deleted_truss, f"delete:{graph_file.name}:{update_file.name}")

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

    (_, _, _), insert_baseline_timings = _measure_runs(run_insert_baseline, warmup, repeat)
    (inserted_row_ptr, inserted_columns, inserted_truss), insert_new_timings = _measure_runs(run_insert_new, warmup, repeat)
    insert_ok, _ = compare_with_recompute(inserted_row_ptr, inserted_columns, inserted_truss, f"insert:{graph_file.name}:{update_file.name}")

    delete_speedup = _median(delete_baseline_timings) / _median(delete_new_timings)
    insert_speedup = _median(insert_baseline_timings) / _median(insert_new_timings)
    print(
        graph_file.name,
        update_file.name,
        f"delete={delete_speedup:.3f}x",
        f"insert={insert_speedup:.3f}x",
        f"delete_ok={delete_ok}",
        f"insert_ok={insert_ok}",
    )
    return {
        "delete_speedup": delete_speedup,
        "insert_speedup": insert_speedup,
        "delete_ok": delete_ok,
        "insert_ok": insert_ok,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark suite for baseline vs new truss maintenance.")
    parser.add_argument("--suite-root", default="update_g", help="Directory containing default graph/update workloads")
    parser.add_argument("--workload", action="append", nargs=2, metavar=("GRAPH", "UPDATE"), help="Explicit graph/update pair")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per workload")
    parser.add_argument("--repeat", type=int, default=5, help="Timed runs per workload")
    args = parser.parse_args()

    if args.workload:
        workloads = [(Path(graph), Path(update)) for graph, update in args.workload]
    else:
        workloads = _default_workloads(Path(args.suite_root))

    delete_speedups = []
    insert_speedups = []
    all_ok = True
    for graph_file, update_file in workloads:
        result = _benchmark_workload(graph_file, update_file, args.warmup, args.repeat)
        delete_speedups.append(result["delete_speedup"])
        insert_speedups.append(result["insert_speedup"])
        all_ok = all_ok and result["delete_ok"] and result["insert_ok"]

    print(f"delete geomean speedup: {_geomean(delete_speedups):.3f}x")
    print(f"insert geomean speedup: {_geomean(insert_speedups):.3f}x")
    print("all workloads correct:", all_ok)


if __name__ == "__main__":
    main()
