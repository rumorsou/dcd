from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path

import torch

from new_dcd import DeltaGraph, load_edge_pairs_from_txt, load_graph_from_txt, maintain_dcd


ROOT = Path(__file__).resolve().parents[1]
TRUSS_DIR = ROOT / "truss"
if str(TRUSS_DIR) not in sys.path:
    sys.path.insert(0, str(TRUSS_DIR))

from insert_maintenace import decompose_from_csr, device, load_graph_as_csr  # noqa: E402
from truss.maintain_engine import maintain_truss, state_from_csr  # noqa: E402


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure(fn, *, warmup: int, repeat: int):
    for _ in range(warmup):
        fn()
        _sync()
    timings = []
    result = None
    for _ in range(repeat):
        _sync()
        started = time.perf_counter()
        result = fn()
        _sync()
        timings.append(time.perf_counter() - started)
    return result, timings


def _build_truss_state(graph_file: Path):
    _, _, vertices, row_ptr, columns, _ = load_graph_as_csr(str(graph_file), 0)
    tau = decompose_from_csr(row_ptr, columns)
    vertex_ids = torch.as_tensor(vertices, device=row_ptr.device, dtype=torch.int64)
    return state_from_csr(row_ptr, columns, tau, vertex_ids)


def _states_equal(new_state, truss_state) -> bool:
    new_codes = new_state.edge_code.detach().cpu()
    truss_codes = truss_state.edge_codes().detach().cpu()
    if new_codes.numel() != truss_codes.numel():
        return False
    new_order = torch.argsort(new_codes)
    truss_order = torch.argsort(truss_codes)
    if not torch.equal(new_codes[new_order], truss_codes[truss_order]):
        return False
    return torch.equal(
        new_state.tau.detach().cpu().to(torch.int64)[new_order],
        truss_state.truss.detach().cpu().to(torch.int64)[truss_order],
    )


def _geomean(values: list[float]) -> float:
    safe = [value for value in values if value > 0]
    if not safe:
        return 0.0
    return math.exp(sum(math.log(value) for value in safe) / len(safe))


def _benchmark_workload(graph_file: Path, update_file: Path, *, warmup: int, repeat: int, edge_budget: int) -> list[dict[str, object]]:
    new_state = load_graph_from_txt(str(graph_file), device=device)
    truss_state = _build_truss_state(graph_file)
    update_edges = load_edge_pairs_from_txt(str(update_file), device=device)
    rows: list[dict[str, object]] = []

    new_delete, new_delete_times = _measure(
        lambda: maintain_dcd(new_state, DeltaGraph(del_edges=update_edges), device=device, edge_budget=edge_budget),
        warmup=warmup,
        repeat=repeat,
    )
    truss_delete, truss_delete_times = _measure(
        lambda: maintain_truss(truss_state, {"del_edges": update_edges}, device=device, edge_budget=edge_budget),
        warmup=warmup,
        repeat=repeat,
    )
    delete_ok = _states_equal(new_delete.state, truss_delete[0])
    rows.append(
        {
            "workload": update_file.stem,
            "operation": "delete",
            "new_dcd_median": statistics.median(new_delete_times),
            "truss_median": statistics.median(truss_delete_times),
            "truss_over_new": statistics.median(truss_delete_times) / statistics.median(new_delete_times),
            "ok": delete_ok,
            "candidate_edges": new_delete.stats.candidate_edges,
            "candidate_triangles": new_delete.stats.candidate_triangles,
        }
    )

    new_insert, new_insert_times = _measure(
        lambda: maintain_dcd(new_delete.state, DeltaGraph(ins_edges=update_edges), device=device, edge_budget=edge_budget),
        warmup=warmup,
        repeat=repeat,
    )
    truss_insert, truss_insert_times = _measure(
        lambda: maintain_truss(truss_delete[0], {"ins_edges": update_edges}, device=device, edge_budget=edge_budget),
        warmup=warmup,
        repeat=repeat,
    )
    insert_ok = _states_equal(new_insert.state, truss_insert[0])
    rows.append(
        {
            "workload": update_file.stem,
            "operation": "insert",
            "new_dcd_median": statistics.median(new_insert_times),
            "truss_median": statistics.median(truss_insert_times),
            "truss_over_new": statistics.median(truss_insert_times) / statistics.median(new_insert_times),
            "ok": insert_ok,
            "candidate_edges": new_insert.stats.candidate_edges,
            "candidate_triangles": new_insert.stats.candidate_triangles,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare new_dcd maintenance against truss maintenance.")
    parser.add_argument("--suite-root", default="update_g")
    parser.add_argument("--datasets", nargs="+", default=["facebook", "amazon", "dblp", "youtube"])
    parser.add_argument("--sizes", nargs="+", default=["10", "100", "1000"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--edge-budget", type=int, default=200_000)
    parser.add_argument("--csv", default="")
    args = parser.parse_args()

    suite_root = Path(args.suite_root)
    all_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        graph_file = suite_root / f"{dataset}.txt"
        for size in args.sizes:
            update_file = suite_root / f"{dataset}_{size}.txt"
            rows = _benchmark_workload(graph_file, update_file, warmup=args.warmup, repeat=args.repeat, edge_budget=args.edge_budget)
            all_rows.extend(rows)
            for row in rows:
                print(
                    row["workload"],
                    row["operation"],
                    f"new_dcd={row['new_dcd_median']:.6f}s",
                    f"truss={row['truss_median']:.6f}s",
                    f"truss/new={row['truss_over_new']:.3f}",
                    f"ok={row['ok']}",
                    f"candidate_edges={row['candidate_edges']}",
                    f"candidate_triangles={row['candidate_triangles']}",
                    flush=True,
                )

    delete_ratios = [float(row["truss_over_new"]) for row in all_rows if row["operation"] == "delete"]
    insert_ratios = [float(row["truss_over_new"]) for row in all_rows if row["operation"] == "insert"]
    all_ratios = [float(row["truss_over_new"]) for row in all_rows]
    print(f"delete geomean truss/new: {_geomean(delete_ratios):.3f}")
    print(f"insert geomean truss/new: {_geomean(insert_ratios):.3f}")
    print(f"overall geomean truss/new: {_geomean(all_ratios):.3f}")
    print("all correct:", all(bool(row["ok"]) for row in all_rows))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)


if __name__ == "__main__":
    main()
