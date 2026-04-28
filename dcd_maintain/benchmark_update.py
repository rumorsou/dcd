from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch

from ._compat import (
    _prepare_graph_runtime,
    _superior_insert_prepared,
    _superior_remove_prepared,
)
from .engine import DCDPreparedRuntime, DCDResult, execute_prepared_dcd, prepare_dcd_runtime
from .graph_io import GraphUpdate, expand_update, load_edge_pairs_from_txt, load_graph_from_txt
from .graph_state import GraphState
from .oracle import recompute_truss


@dataclass
class AlgorithmPhaseSummary:
    name: str
    all_correct: bool
    total_timings: list[float]
    exec_timings: list[float]
    affected_edges: int
    affected_triangles: int
    traversed_edges: int
    traversed_triangles: int


def _sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_once(fn):
    _sync_if_needed()
    start = time.perf_counter()
    result = fn()
    _sync_if_needed()
    return result, time.perf_counter() - start


def _measure_repeat(fn, repeat: int) -> tuple[list[Any], list[float]]:
    results = []
    timings = []
    for _ in range(repeat):
        result, elapsed = _measure_once(fn)
        results.append(result)
        timings.append(elapsed)
    return results, timings


def _clone_state_with_tau(state: GraphState, tau: torch.Tensor) -> GraphState:
    return GraphState(
        row_ptr=state.row_ptr.clone(),
        columns=state.columns.clone(),
        tau=tau.clone(),
        vertex_labels=state.vertex_labels.copy(),
        edge_code=state.edge_code.clone(),
        bidirectional_view=tuple(part.clone() for part in state.bidirectional_view),
        edge_src_index=state.edge_src_index.clone(),
    )


def _map_delete_edges(base_state: GraphState, update_edges: torch.Tensor) -> torch.Tensor:
    _, expanded = expand_update(base_state, GraphUpdate.from_raw(edge_deletes=update_edges))
    return expanded["edge_deletes"]


def _map_insert_edges(base_state: GraphState, update_edges: torch.Tensor) -> torch.Tensor:
    _, expanded = expand_update(base_state, GraphUpdate.from_raw(edge_inserts=update_edges))
    return expanded["edge_inserts"]


def _mean_median_str(timings: list[float]) -> str:
    return f"均值 {statistics.mean(timings):.6f}s，中位数 {statistics.median(timings):.6f}s"


def _count_unique_triangles(stats: dict[str, Any] | None) -> int:
    if stats is None:
        return 0
    batches = stats.get("triangle_batches", [])
    if not batches:
        return 0
    triangle_tensor = torch.cat([batch.to(device="cpu", dtype=torch.long) for batch in batches], dim=0)
    return int(torch.unique(triangle_tensor, dim=0).size(0))


def _count_unique_codes(stats: dict[str, Any] | None, key: str) -> int:
    if stats is None:
        return 0
    batches = stats.get(key, [])
    if not batches:
        return 0
    code_tensor = torch.cat([batch.to(device="cpu", dtype=torch.long).reshape(-1) for batch in batches], dim=0)
    return int(torch.unique(code_tensor).numel())


def _count_unique_code_triangles(stats: dict[str, Any] | None, key: str) -> int:
    if stats is None:
        return 0
    batches = stats.get(key, [])
    if not batches:
        return 0
    triangle_tensor = torch.cat([batch.to(device="cpu", dtype=torch.long) for batch in batches], dim=0)
    return int(torch.unique(triangle_tensor, dim=0).size(0))


def _extract_dcd_tau(result: DCDResult) -> torch.Tensor:
    return result.tau_new


def _extract_superior_tau(result: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return result[2]


def _run_dcd_total(base_state: GraphState, update: GraphUpdate, repeat: int) -> tuple[list[DCDResult], list[float]]:
    def run_once():
        runtime = prepare_dcd_runtime(base_state, update)
        return execute_prepared_dcd(runtime)

    return _measure_repeat(run_once, repeat)


def _run_dcd_exec(runtime: DCDPreparedRuntime, repeat: int) -> tuple[list[DCDResult], list[float]]:
    return _measure_repeat(lambda: execute_prepared_dcd(runtime), repeat)


def _run_superior_delete_total(base_state: GraphState, raw_update_edges: torch.Tensor, repeat: int):
    def run_once():
        delete_edges = _map_delete_edges(base_state, raw_update_edges)
        edge_code, bidirectional_view, edge_src_index = _prepare_graph_runtime(
            base_state.row_ptr,
            base_state.columns,
            edge_code=base_state.edge_code,
            bidirectional_view=base_state.bidirectional_view,
            edge_src_index=base_state.edge_src_index,
        )
        return _superior_remove_prepared(
            base_state.row_ptr.clone(),
            base_state.columns.clone(),
            delete_edges[:, 0],
            delete_edges[:, 1],
            base_state.tau.clone(),
            edge_code,
            bidirectional_view,
            edge_src_index,
            return_stats=False,
        )

    return _measure_repeat(run_once, repeat)


def _run_superior_insert_total(base_state: GraphState, raw_update_edges: torch.Tensor, repeat: int):
    def run_once():
        insert_edges = _map_insert_edges(base_state, raw_update_edges)
        edge_code, bidirectional_view, edge_src_index = _prepare_graph_runtime(
            base_state.row_ptr,
            base_state.columns,
            edge_code=base_state.edge_code,
            bidirectional_view=base_state.bidirectional_view,
            edge_src_index=base_state.edge_src_index,
        )
        return _superior_insert_prepared(
            base_state.row_ptr.clone(),
            base_state.columns.clone(),
            insert_edges[:, 0],
            insert_edges[:, 1],
            base_state.tau.clone(),
            edge_code,
            bidirectional_view,
            edge_src_index,
            return_stats=False,
        )

    return _measure_repeat(run_once, repeat)


def _run_superior_delete_exec(base_state: GraphState, delete_edges: torch.Tensor, repeat: int):
    edge_code, bidirectional_view, edge_src_index = _prepare_graph_runtime(
        base_state.row_ptr,
        base_state.columns,
        edge_code=base_state.edge_code,
        bidirectional_view=base_state.bidirectional_view,
        edge_src_index=base_state.edge_src_index,
    )
    return _measure_repeat(
        lambda: _superior_remove_prepared(
            base_state.row_ptr.clone(),
            base_state.columns.clone(),
            delete_edges[:, 0],
            delete_edges[:, 1],
            base_state.tau.clone(),
            edge_code,
            bidirectional_view,
            edge_src_index,
            return_stats=False,
        ),
        repeat,
    )


def _run_superior_insert_exec(base_state: GraphState, insert_edges: torch.Tensor, repeat: int):
    edge_code, bidirectional_view, edge_src_index = _prepare_graph_runtime(
        base_state.row_ptr,
        base_state.columns,
        edge_code=base_state.edge_code,
        bidirectional_view=base_state.bidirectional_view,
        edge_src_index=base_state.edge_src_index,
    )
    return _measure_repeat(
        lambda: _superior_insert_prepared(
            base_state.row_ptr.clone(),
            base_state.columns.clone(),
            insert_edges[:, 0],
            insert_edges[:, 1],
            base_state.tau.clone(),
            edge_code,
            bidirectional_view,
            edge_src_index,
            return_stats=False,
        ),
        repeat,
    )


def _collect_superior_delete_stats(base_state: GraphState, delete_edges: torch.Tensor) -> tuple[int, int, int, int]:
    edge_code, bidirectional_view, edge_src_index = _prepare_graph_runtime(
        base_state.row_ptr,
        base_state.columns,
        edge_code=base_state.edge_code,
        bidirectional_view=base_state.bidirectional_view,
        edge_src_index=base_state.edge_src_index,
    )
    _, _, _, stats = _superior_remove_prepared(
        base_state.row_ptr.clone(),
        base_state.columns.clone(),
        delete_edges[:, 0],
        delete_edges[:, 1],
        base_state.tau.clone(),
        edge_code,
        bidirectional_view,
        edge_src_index,
        return_stats=True,
    )
    return (
        _count_unique_codes(stats, "affected_edge_code_batches"),
        _count_unique_code_triangles(stats, "affected_triangle_code_batches"),
        int(stats.get("unique_edges", 0)),
        _count_unique_triangles(stats),
    )


def _collect_superior_insert_stats(base_state: GraphState, insert_edges: torch.Tensor) -> tuple[int, int, int, int]:
    edge_code, bidirectional_view, edge_src_index = _prepare_graph_runtime(
        base_state.row_ptr,
        base_state.columns,
        edge_code=base_state.edge_code,
        bidirectional_view=base_state.bidirectional_view,
        edge_src_index=base_state.edge_src_index,
    )
    _, _, _, stats = _superior_insert_prepared(
        base_state.row_ptr.clone(),
        base_state.columns.clone(),
        insert_edges[:, 0],
        insert_edges[:, 1],
        base_state.tau.clone(),
        edge_code,
        bidirectional_view,
        edge_src_index,
        return_stats=True,
    )
    return (
        _count_unique_codes(stats, "affected_edge_code_batches"),
        _count_unique_code_triangles(stats, "affected_triangle_code_batches"),
        int(stats.get("unique_edges", 0)),
        _count_unique_triangles(stats),
    )


def _build_dcd_summary(
    name: str,
    total_results: list[DCDResult],
    total_timings: list[float],
    exec_timings: list[float],
    reference_tau: torch.Tensor,
) -> AlgorithmPhaseSummary:
    reference_result = total_results[0]
    return AlgorithmPhaseSummary(
        name=name,
        all_correct=all(torch.equal(_extract_dcd_tau(result), reference_tau) for result in total_results),
        total_timings=total_timings,
        exec_timings=exec_timings,
        affected_edges=int(reference_result.cone_edges.numel()),
        affected_triangles=reference_result.candidate_triangle_count,
        traversed_edges=int(reference_result.cone_edges.numel()),
        traversed_triangles=reference_result.candidate_triangle_count,
    )


def _build_superior_summary(
    name: str,
    total_results: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    total_timings: list[float],
    exec_timings: list[float],
    reference_tau: torch.Tensor,
    affected_edges: int,
    affected_triangles: int,
    traversed_edges: int,
    traversed_triangles: int,
) -> AlgorithmPhaseSummary:
    return AlgorithmPhaseSummary(
        name=name,
        all_correct=all(torch.equal(_extract_superior_tau(result), reference_tau) for result in total_results),
        total_timings=total_timings,
        exec_timings=exec_timings,
        affected_edges=affected_edges,
        affected_triangles=affected_triangles,
        traversed_edges=traversed_edges,
        traversed_triangles=traversed_triangles,
    )


def _print_phase_comparison(title: str, decompose_time: float, summaries: list[AlgorithmPhaseSummary]) -> None:
    print(f"{title}：")
    print(f"验证用单次 truss 分解时间：{decompose_time:.6f}s")
    for summary in summaries:
        print(
            f"{summary.name}：正确={'是' if summary.all_correct else '否'}，"
            f"总维护时间 {_mean_median_str(summary.total_timings)}，"
            f"纯执行时间 {_mean_median_str(summary.exec_timings)}，"
            f"可能受影响边数 {summary.affected_edges}，"
            f"可能受影响三角形数 {summary.affected_triangles}，"
            f"遍历边数 {summary.traversed_edges}，"
            f"遍历三角形数 {summary.traversed_triangles}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="统一统计 DCD 与 truss/superior 的维护总时间、纯执行时间与遍历规模。")
    parser.add_argument("graph_file", help="原图文件路径")
    parser.add_argument("update_file", help="待更新边文件路径")
    parser.add_argument("--repeat", type=int, default=5, help="维护重复执行次数，默认 5 次")
    parser.add_argument(
        "--backend",
        choices=("dcd", "superior", "both"),
        default="both",
        help="输出 DCD、superior 或两者对比，默认 both。",
    )
    args = parser.parse_args()

    if args.repeat != 5:
        raise ValueError("按照当前需求，维护次数必须为 5 次。")

    base_state = load_graph_from_txt(args.graph_file)
    update_edges = load_edge_pairs_from_txt(args.update_file)

    delete_edges = _map_delete_edges(base_state, update_edges)
    delete_reference_graph = base_state.with_removed_edges(delete_edges)
    delete_reference_tau, delete_decompose_time = _measure_once(lambda: recompute_truss(delete_reference_graph))
    deleted_base_state = _clone_state_with_tau(delete_reference_graph, delete_reference_tau)

    delete_summaries: list[AlgorithmPhaseSummary] = []
    if args.backend in ("dcd", "both"):
        delete_update = GraphUpdate.from_raw(edge_deletes=update_edges)
        delete_runtime = prepare_dcd_runtime(base_state, delete_update)
        delete_dcd_total_results, delete_dcd_total_timings = _run_dcd_total(base_state, delete_update, args.repeat)
        _, delete_dcd_exec_timings = _run_dcd_exec(delete_runtime, args.repeat)
        delete_summaries.append(
            _build_dcd_summary(
                name="DCD",
                total_results=delete_dcd_total_results,
                total_timings=delete_dcd_total_timings,
                exec_timings=delete_dcd_exec_timings,
                reference_tau=delete_reference_tau,
            )
        )

    if args.backend in ("superior", "both"):
        delete_superior_total_results, delete_superior_total_timings = _run_superior_delete_total(
            base_state,
            update_edges,
            args.repeat,
        )
        _, delete_superior_exec_timings = _run_superior_delete_exec(base_state, delete_edges, args.repeat)
        (
            delete_affected_edges,
            delete_affected_triangles,
            delete_traversed_edges,
            delete_traversed_triangles,
        ) = _collect_superior_delete_stats(base_state, delete_edges)
        delete_summaries.append(
            _build_superior_summary(
                name="Superior",
                total_results=delete_superior_total_results,
                total_timings=delete_superior_total_timings,
                exec_timings=delete_superior_exec_timings,
                reference_tau=delete_reference_tau,
                affected_edges=delete_affected_edges,
                affected_triangles=delete_affected_triangles,
                traversed_edges=delete_traversed_edges,
                traversed_triangles=delete_traversed_triangles,
            )
        )

    _print_phase_comparison("删除维护", delete_decompose_time, delete_summaries)

    insert_edges = _map_insert_edges(deleted_base_state, update_edges)
    insert_reference_graph = deleted_base_state.with_inserted_edges(insert_edges)
    insert_reference_tau, insert_decompose_time = _measure_once(lambda: recompute_truss(insert_reference_graph))

    insert_summaries: list[AlgorithmPhaseSummary] = []
    if args.backend in ("dcd", "both"):
        insert_update = GraphUpdate.from_raw(edge_inserts=update_edges)
        insert_runtime = prepare_dcd_runtime(deleted_base_state, insert_update)
        insert_dcd_total_results, insert_dcd_total_timings = _run_dcd_total(deleted_base_state, insert_update, args.repeat)
        _, insert_dcd_exec_timings = _run_dcd_exec(insert_runtime, args.repeat)
        insert_summaries.append(
            _build_dcd_summary(
                name="DCD",
                total_results=insert_dcd_total_results,
                total_timings=insert_dcd_total_timings,
                exec_timings=insert_dcd_exec_timings,
                reference_tau=insert_reference_tau,
            )
        )

    if args.backend in ("superior", "both"):
        insert_superior_total_results, insert_superior_total_timings = _run_superior_insert_total(
            deleted_base_state,
            update_edges,
            args.repeat,
        )
        _, insert_superior_exec_timings = _run_superior_insert_exec(deleted_base_state, insert_edges, args.repeat)
        (
            insert_affected_edges,
            insert_affected_triangles,
            insert_traversed_edges,
            insert_traversed_triangles,
        ) = _collect_superior_insert_stats(deleted_base_state, insert_edges)
        insert_summaries.append(
            _build_superior_summary(
                name="Superior",
                total_results=insert_superior_total_results,
                total_timings=insert_superior_total_timings,
                exec_timings=insert_superior_exec_timings,
                reference_tau=insert_reference_tau,
                affected_edges=insert_affected_edges,
                affected_triangles=insert_affected_triangles,
                traversed_edges=insert_traversed_edges,
                traversed_triangles=insert_traversed_triangles,
            )
        )

    _print_phase_comparison("插入维护", insert_decompose_time, insert_summaries)


if __name__ == "__main__":
    main()
