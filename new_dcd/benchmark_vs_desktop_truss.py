from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import torch

from new_dcd import DeltaGraph, load_edge_pairs_from_txt, load_graph_from_txt, maintain_dcd
import new_dcd.triangle_index as new_tri


ROOT = Path(__file__).resolve().parents[1]
DESKTOP_TRUSS = Path(r"C:\Users\Administrator\Desktop\truss")


@dataclass
class TraversalCounter:
    edge_visits: int = 0
    triangle_records: int = 0
    triangle_calls: int = 0


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _geomean(values: list[float]) -> float:
    safe = [value for value in values if value > 0]
    if not safe:
        return 0.0
    return math.exp(sum(math.log(value) for value in safe) / len(safe))


@contextmanager
def _count_new_dcd_traversal(counter: TraversalCounter) -> Iterator[None]:
    original = new_tri.EdgeTriangleIndex.collect

    def wrapped(self, edge_ids):
        counter.edge_visits += int(edge_ids.numel())
        counter.triangle_calls += 1
        result = original(self, edge_ids)
        counter.triangle_records += int(result[0].numel())
        return result

    new_tri.EdgeTriangleIndex.collect = wrapped
    try:
        yield
    finally:
        new_tri.EdgeTriangleIndex.collect = original


def _load_desktop_modules():
    if str(DESKTOP_TRUSS) not in sys.path:
        sys.path.insert(0, str(DESKTOP_TRUSS))
    from dcd_maintenance import dcd_maintain
    import dcd_maintenance as desktop_dcd
    from insert_maintenace import decompose_from_csr, device, load_graph_as_csr, read_update_edge_txt

    return desktop_dcd, dcd_maintain, decompose_from_csr, device, load_graph_as_csr, read_update_edge_txt


@contextmanager
def _count_desktop_truss_traversal(desktop_dcd, counter: TraversalCounter) -> Iterator[None]:
    original = desktop_dcd._get_triangle_pack

    def wrapped(snapshot, edge_ids, triangle_cache):
        counter.edge_visits += int(edge_ids.numel())
        counter.triangle_calls += 1
        result = original(snapshot, edge_ids, triangle_cache)
        counter.triangle_records += int(result[0].numel())
        return result

    desktop_dcd._get_triangle_pack = wrapped
    try:
        yield
    finally:
        desktop_dcd._get_triangle_pack = original


def _measure(fn: Callable, counter_context: Callable[[], Iterator[None]], *, repeat: int, warmup: int):
    for _ in range(warmup):
        with counter_context():
            fn()
        _sync()

    timings: list[float] = []
    counters: list[TraversalCounter] = []
    result = None
    for _ in range(repeat):
        counter = TraversalCounter()
        _sync()
        started = time.perf_counter()
        with counter_context(counter):
            result = fn()
        _sync()
        timings.append(time.perf_counter() - started)
        counters.append(counter)
    median_idx = sorted(range(len(timings)), key=lambda idx: timings[idx])[len(timings) // 2]
    return result, timings, counters[median_idx]


def _edge_codes(row_ptr: torch.Tensor, columns: torch.Tensor) -> torch.Tensor:
    src = torch.repeat_interleave(
        torch.arange(row_ptr.numel() - 1, device=row_ptr.device, dtype=torch.long),
        row_ptr[1:] - row_ptr[:-1],
    )
    return src * (row_ptr.numel() - 1) + columns.to(torch.long)


def _states_equal(new_state, row_ptr: torch.Tensor, columns: torch.Tensor, tau: torch.Tensor) -> bool:
    new_codes = new_state.edge_code.detach().cpu()
    desktop_codes = _edge_codes(row_ptr, columns).detach().cpu()
    if new_codes.numel() != desktop_codes.numel():
        return False
    new_order = torch.argsort(new_codes)
    desktop_order = torch.argsort(desktop_codes)
    if not torch.equal(new_codes[new_order], desktop_codes[desktop_order]):
        return False
    return torch.equal(
        new_state.tau.detach().cpu().to(torch.long)[new_order],
        tau.detach().cpu().to(torch.long)[desktop_order],
    )


def _profile_columns(result) -> dict[str, int | float]:
    keys = [
        "num_cone_edges",
        "num_active_rounds",
        "num_active_edges_total",
        "num_active_edges_max",
        "num_active_edges_avg",
        "num_enqueue_total",
        "num_enqueue_without_tighten",
        "ratio_enqueue_without_tighten",
        "num_tighten_L",
        "num_tighten_U",
        "num_triangle_records_scanned",
        "num_unique_triangle_records_scanned",
        "ratio_triangle_records_repeated",
        "num_repeated_scan_same_edge",
        "num_repeated_scan_same_edge_level",
        "num_2hop_backfill_edges",
        "ratio_2hop_backfill_edges_to_cone",
        "num_exact_peeling_edges",
    ]
    return {key: result.stats.profile.get(key, 0) for key in keys}


def _benchmark_workload(
    graph_file: Path,
    update_file: Path,
    *,
    repeat: int,
    warmup: int,
    edge_budget: int,
    enable_refinement: bool,
):
    desktop_dcd, dcd_maintain, decompose_from_csr, device, load_graph_as_csr, read_update_edge_txt = _load_desktop_modules()

    new_state = load_graph_from_txt(str(graph_file), device=device)
    update_edges = load_edge_pairs_from_txt(str(update_file), device=device)

    _, _, vertices, row_ptr, columns, _ = load_graph_as_csr(str(graph_file), 0)
    tau = decompose_from_csr(row_ptr, columns)
    upd_src_np, upd_dst_np = read_update_edge_txt(str(update_file), vertices, 0)
    upd_src = torch.as_tensor(upd_src_np, device=device, dtype=torch.long)
    upd_dst = torch.as_tensor(upd_dst_np, device=device, dtype=torch.long)

    def new_counter_context(counter: TraversalCounter | None = None):
        return _count_new_dcd_traversal(counter if counter is not None else TraversalCounter())

    def desktop_counter_context(counter: TraversalCounter | None = None):
        return _count_desktop_truss_traversal(desktop_dcd, counter if counter is not None else TraversalCounter())

    new_delete, new_delete_times, new_delete_counter = _measure(
        lambda: maintain_dcd(
            new_state,
            DeltaGraph(del_edges=update_edges),
            device=device,
            edge_budget=edge_budget,
            enable_refinement=enable_refinement,
        ),
        new_counter_context,
        repeat=repeat,
        warmup=warmup,
    )
    desktop_delete, desktop_delete_times, desktop_delete_counter = _measure(
        lambda: dcd_maintain(row_ptr.clone(), columns.clone(), tau.clone(), del_src=upd_src, del_dst=upd_dst),
        desktop_counter_context,
        repeat=repeat,
        warmup=warmup,
    )
    delete_ok = _states_equal(new_delete.state, desktop_delete[0], desktop_delete[1], desktop_delete[2])

    rows = [
        {
            "workload": update_file.stem,
            "operation": "delete",
            "new_dcd_median": statistics.median(new_delete_times),
            "desktop_truss_median": statistics.median(desktop_delete_times),
            "desktop_over_new": statistics.median(desktop_delete_times) / statistics.median(new_delete_times),
            "ok": delete_ok,
            "new_candidate_edges": new_delete.stats.candidate_edges,
            "new_candidate_triangles": new_delete.stats.candidate_triangles,
            "new_edge_visits": new_delete_counter.edge_visits,
            "new_triangle_records": new_delete_counter.triangle_records,
            "new_triangle_calls": new_delete_counter.triangle_calls,
            **_profile_columns(new_delete),
            "desktop_candidate_edges": int(torch.count_nonzero(desktop_delete[3].cand).item()),
            "desktop_unresolved_edges": int(desktop_delete[3].unresolved_edges.numel()),
            "desktop_fallback": bool(desktop_delete[3].fallback_used),
            "desktop_edge_visits": desktop_delete_counter.edge_visits,
            "desktop_triangle_records": desktop_delete_counter.triangle_records,
            "desktop_triangle_calls": desktop_delete_counter.triangle_calls,
        }
    ]

    new_insert, new_insert_times, new_insert_counter = _measure(
        lambda: maintain_dcd(
            new_delete.state,
            DeltaGraph(ins_edges=update_edges),
            device=device,
            edge_budget=edge_budget,
            enable_refinement=enable_refinement,
        ),
        new_counter_context,
        repeat=repeat,
        warmup=warmup,
    )
    desktop_insert, desktop_insert_times, desktop_insert_counter = _measure(
        lambda: dcd_maintain(
            desktop_delete[0].clone(),
            desktop_delete[1].clone(),
            desktop_delete[2].clone(),
            ins_src=upd_src,
            ins_dst=upd_dst,
        ),
        desktop_counter_context,
        repeat=repeat,
        warmup=warmup,
    )
    insert_ok = _states_equal(new_insert.state, desktop_insert[0], desktop_insert[1], desktop_insert[2])
    rows.append(
        {
            "workload": update_file.stem,
            "operation": "insert",
            "new_dcd_median": statistics.median(new_insert_times),
            "desktop_truss_median": statistics.median(desktop_insert_times),
            "desktop_over_new": statistics.median(desktop_insert_times) / statistics.median(new_insert_times),
            "ok": insert_ok,
            "new_candidate_edges": new_insert.stats.candidate_edges,
            "new_candidate_triangles": new_insert.stats.candidate_triangles,
            "new_edge_visits": new_insert_counter.edge_visits,
            "new_triangle_records": new_insert_counter.triangle_records,
            "new_triangle_calls": new_insert_counter.triangle_calls,
            **_profile_columns(new_insert),
            "desktop_candidate_edges": int(torch.count_nonzero(desktop_insert[3].cand).item()),
            "desktop_unresolved_edges": int(desktop_insert[3].unresolved_edges.numel()),
            "desktop_fallback": bool(desktop_insert[3].fallback_used),
            "desktop_edge_visits": desktop_insert_counter.edge_visits,
            "desktop_triangle_records": desktop_insert_counter.triangle_records,
            "desktop_triangle_calls": desktop_insert_counter.triangle_calls,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare new_dcd against C:/Users/Administrator/Desktop/truss.")
    parser.add_argument("--suite-root", default="update_g")
    parser.add_argument("--datasets", nargs="+", default=["facebook", "amazon", "dblp", "youtube"])
    parser.add_argument("--sizes", nargs="+", default=["10", "100", "1000"])
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--edge-budget", type=int, default=200_000)
    parser.add_argument("--enable-refinement", action="store_true")
    parser.add_argument("--csv", default="new_dcd_vs_desktop_truss.csv")
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    all_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for size in args.sizes:
            rows = _benchmark_workload(
                suite_root / f"{dataset}.txt",
                suite_root / f"{dataset}_{size}.txt",
                repeat=args.repeat,
                warmup=args.warmup,
                edge_budget=args.edge_budget,
                enable_refinement=args.enable_refinement,
            )
            all_rows.extend(rows)
            for row in rows:
                print(
                    row["workload"],
                    row["operation"],
                    f"new_dcd={row['new_dcd_median']:.6f}s",
                    f"desktop_truss={row['desktop_truss_median']:.6f}s",
                    f"desktop/new={row['desktop_over_new']:.3f}",
                    f"ok={row['ok']}",
                    f"new_edge_visits={row['new_edge_visits']}",
                    f"new_tri_records={row['new_triangle_records']}",
                    f"cone={row['num_cone_edges']}",
                    f"enqueue/no_tighten={row['ratio_enqueue_without_tighten']:.3f}",
                    f"tri_repeat={row['ratio_triangle_records_repeated']:.3f}",
                    f"backfill/cone={row['ratio_2hop_backfill_edges_to_cone']:.3f}",
                    f"desktop_edge_visits={row['desktop_edge_visits']}",
                    f"desktop_tri_records={row['desktop_triangle_records']}",
                    f"desktop_fallback={row['desktop_fallback']}",
                    flush=True,
                )

    delete_ratios = [float(row["desktop_over_new"]) for row in all_rows if row["operation"] == "delete"]
    insert_ratios = [float(row["desktop_over_new"]) for row in all_rows if row["operation"] == "insert"]
    all_ratios = [float(row["desktop_over_new"]) for row in all_rows]
    print(f"delete geomean desktop/new: {_geomean(delete_ratios):.3f}")
    print(f"insert geomean desktop/new: {_geomean(insert_ratios):.3f}")
    print(f"overall geomean desktop/new: {_geomean(all_ratios):.3f}")
    print("all correct:", all(bool(row["ok"]) for row in all_rows))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()
