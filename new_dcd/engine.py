from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .bounds import BoundState, build_initial_bounds, initial_seed_edges
from .csr import build_canonical_csr
from .graph_ops import apply_updates_to_graph
from .state import DCDState
from .static import decompose_from_csr
from .support import WitnessCache, build_witness_cache, estimate_support_bounds
from .triangle_index import EdgeTriangleIndex, TrianglePack
from .updates import DeltaGraph, normalize_updates


@dataclass
class PhaseStats:
    rounds: int = 0
    tightened_edges: int = 0
    backfilled_edges: int = 0


@dataclass
class DCDProfileCounters:
    num_cone_edges: int = 0
    num_active_edges_per_round: list[int] = field(default_factory=list)
    num_enqueue_total: int = 0
    num_enqueue_without_tighten: int = 0
    num_tighten_L: int = 0
    num_tighten_U: int = 0
    num_triangle_records_scanned: int = 0
    num_unique_triangle_records_scanned: int = 0
    num_repeated_scan_same_edge: int = 0
    num_repeated_scan_same_edge_level: int = 0
    num_2hop_backfill_edges: int = 0
    num_exact_peeling_edges: int = 0
    _seen_edges: set[int] = field(default_factory=set, repr=False)
    _seen_edge_levels: set[tuple[int, int]] = field(default_factory=set, repr=False)
    _seen_triangle_records: set[tuple[int, int, int]] = field(default_factory=set, repr=False)

    def record_pack_scan(self, pack: TrianglePack, *, level: int | None = None) -> None:
        edge_ids = pack.edge_ids.detach().to("cpu", dtype=torch.long).tolist()
        repeated_edges = 0
        repeated_edge_levels = 0
        for edge_id in edge_ids:
            edge_id = int(edge_id)
            if edge_id in self._seen_edges:
                repeated_edges += 1
            else:
                self._seen_edges.add(edge_id)
            if level is not None:
                edge_level = (edge_id, int(level))
                if edge_level in self._seen_edge_levels:
                    repeated_edge_levels += 1
                else:
                    self._seen_edge_levels.add(edge_level)
        self.num_repeated_scan_same_edge += repeated_edges
        self.num_repeated_scan_same_edge_level += repeated_edge_levels
        self.num_triangle_records_scanned += pack.record_count

        if pack.record_count == 0:
            self.num_unique_triangle_records_scanned = len(self._seen_triangle_records)
            return

        counts = pack.tri_ptr[1:] - pack.tri_ptr[:-1]
        owner = torch.repeat_interleave(pack.edge_ids.to(torch.long), counts)
        records = torch.cat((owner.reshape(-1, 1), pack.other_edges.to(torch.long)), dim=1)
        records = torch.sort(records, dim=1).values.detach().to("cpu", dtype=torch.long)
        for a, b, c in records.tolist():
            self._seen_triangle_records.add((int(a), int(b), int(c)))
        self.num_unique_triangle_records_scanned = len(self._seen_triangle_records)

    def as_profile(self) -> dict[str, int | float]:
        max_active = max(self.num_active_edges_per_round) if self.num_active_edges_per_round else 0
        avg_active = (
            sum(self.num_active_edges_per_round) / len(self.num_active_edges_per_round)
            if self.num_active_edges_per_round
            else 0.0
        )
        enqueue_without_tighten_ratio = (
            self.num_enqueue_without_tighten / self.num_enqueue_total
            if self.num_enqueue_total
            else 0.0
        )
        repeated_triangle_scan_ratio = (
            self.num_triangle_records_scanned / self.num_unique_triangle_records_scanned
            if self.num_unique_triangle_records_scanned
            else 0.0
        )
        backfill_to_cone_ratio = (
            self.num_2hop_backfill_edges / self.num_cone_edges
            if self.num_cone_edges
            else 0.0
        )
        return {
            "num_cone_edges": self.num_cone_edges,
            "num_active_rounds": len(self.num_active_edges_per_round),
            "num_active_edges_total": sum(self.num_active_edges_per_round),
            "num_active_edges_max": max_active,
            "num_active_edges_avg": avg_active,
            "num_enqueue_total": self.num_enqueue_total,
            "num_enqueue_without_tighten": self.num_enqueue_without_tighten,
            "ratio_enqueue_without_tighten": enqueue_without_tighten_ratio,
            "num_tighten_L": self.num_tighten_L,
            "num_tighten_U": self.num_tighten_U,
            "num_triangle_records_scanned": self.num_triangle_records_scanned,
            "num_unique_triangle_records_scanned": self.num_unique_triangle_records_scanned,
            "ratio_triangle_records_repeated": repeated_triangle_scan_ratio,
            "num_repeated_scan_same_edge": self.num_repeated_scan_same_edge,
            "num_repeated_scan_same_edge_level": self.num_repeated_scan_same_edge_level,
            "num_2hop_backfill_edges": self.num_2hop_backfill_edges,
            "ratio_2hop_backfill_edges_to_cone": backfill_to_cone_ratio,
            "num_exact_peeling_edges": self.num_exact_peeling_edges,
        }


@dataclass
class DCDStats:
    candidate_edges: int
    candidate_triangles: int
    delete_phase: PhaseStats
    insert_phase: PhaseStats
    used_exact_cone: bool
    witness_cache: WitnessCache | None = None
    profile: dict[str, int | float] = field(default_factory=dict)


@dataclass
class DCDResult:
    state: DCDState
    tau_new: torch.Tensor
    bounds: BoundState
    stats: DCDStats


def pick_critical_k(lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return torch.div(lower.to(torch.long) + upper.to(torch.long) + 1, 2, rounding_mode="floor").to(torch.int32)


def _expand_cone(
    index: EdgeTriangleIndex,
    seed_edges: torch.Tensor,
    *,
    edge_budget: int,
    max_hops: int | None,
) -> torch.Tensor:
    frontier = torch.unique(seed_edges.to(device=index.state.device, dtype=torch.long), sorted=True)
    if frontier.numel() == 0:
        return frontier
    visited = torch.zeros((index.state.num_edges,), device=index.state.device, dtype=torch.bool)
    visited[frontier] = True
    hops = 0
    while frontier.numel() > 0 and (max_hops is None or hops < max_hops):
        neighbors = index.triangle_neighbors(frontier, edge_budget=edge_budget)
        if neighbors.numel() == 0:
            break
        next_frontier = neighbors[~visited[neighbors]]
        if next_frontier.numel() == 0:
            break
        visited[next_frontier] = True
        frontier = torch.unique(next_frontier, sorted=True)
        hops += 1
    return torch.nonzero(visited, as_tuple=False).flatten()


def _propagate_and_backfill(
    bound: BoundState,
    index: EdgeTriangleIndex,
    tightened_edges: torch.Tensor,
    *,
    edge_budget: int,
    mark: int,
    profile: DCDProfileCounters | None = None,
) -> torch.Tensor:
    if tightened_edges.numel() == 0:
        return torch.empty((0,), device=index.state.device, dtype=torch.long)
    neighbors = index.triangle_neighbors(tightened_edges, edge_budget=edge_budget)
    if neighbors.numel() == 0:
        return neighbors
    if profile is not None:
        profile.num_enqueue_total += int(neighbors.numel())
    unseen = neighbors[~bound.cand[neighbors]]
    if profile is not None:
        profile.num_enqueue_without_tighten += int(neighbors.numel() - unseen.numel())
        profile.num_2hop_backfill_edges += int(unseen.numel())
    if unseen.numel() > 0:
        bound.cand[unseen] = True
        bound.last_mark[unseen] = int(mark)
        bound.cone_edges = torch.nonzero(bound.cand, as_tuple=False).flatten()
    return unseen


def _run_phase(
    bound: BoundState,
    index: EdgeTriangleIndex,
    *,
    allow_raise_lower: bool,
    edge_budget: int,
    max_rounds: int,
    profile: DCDProfileCounters | None = None,
) -> PhaseStats:
    stats = PhaseStats()
    for round_id in range(max_rounds):
        active_edges = torch.nonzero(bound.cand & (bound.lower < bound.upper), as_tuple=False).flatten()
        if active_edges.numel() == 0:
            break
        if profile is not None:
            profile.num_active_edges_per_round.append(int(active_edges.numel()))
        stats.rounds += 1
        pack = index.materialize(active_edges)
        if profile is not None:
            profile.record_pack_scan(pack, level=round_id)
        k_values = pick_critical_k(bound.lower[pack.edge_ids], bound.upper[pack.edge_ids])
        smin, smax = estimate_support_bounds(pack, k_values, bound.lower, bound.upper)
        old_lower = bound.lower[pack.edge_ids].clone()
        old_upper = bound.upper[pack.edge_ids].clone()

        down = smax < (k_values.to(torch.long) - 2)
        if torch.any(down):
            bound.upper[pack.edge_ids[down]] = torch.minimum(bound.upper[pack.edge_ids[down]], (k_values[down] - 1).to(torch.int32))
        if allow_raise_lower:
            up = smin >= (k_values.to(torch.long) - 2)
            if torch.any(up):
                bound.lower[pack.edge_ids[up]] = torch.maximum(bound.lower[pack.edge_ids[up]], k_values[up])
        bound.lower[pack.edge_ids] = torch.minimum(bound.lower[pack.edge_ids], bound.upper[pack.edge_ids])

        if profile is not None:
            profile.num_tighten_L += int(torch.count_nonzero(bound.lower[pack.edge_ids] > old_lower).item())
            profile.num_tighten_U += int(torch.count_nonzero(bound.upper[pack.edge_ids] < old_upper).item())
        tightened_mask = (bound.lower[pack.edge_ids] > old_lower) | (bound.upper[pack.edge_ids] < old_upper)
        tightened = pack.edge_ids[tightened_mask]
        if tightened.numel() == 0:
            break
        stats.tightened_edges += int(tightened.numel())
        backfilled = _propagate_and_backfill(
            bound,
            index,
            tightened,
            edge_budget=edge_budget,
            mark=round_id,
            profile=profile,
        )
        stats.backfilled_edges += int(backfilled.numel())
    bound.fixed = bound.cand & (bound.lower == bound.upper)
    bound.active = bound.cand & (bound.lower < bound.upper)
    return stats


def _full_recompute_tau(state: DCDState) -> torch.Tensor:
    pairs = state.canonical_edge_pairs()
    canonical = build_canonical_csr(pairs, state.num_vertices)
    return decompose_from_csr(canonical.rowptr, canonical.col).to(device=state.device, dtype=torch.int32)


def _solve_cone_exact(
    index: EdgeTriangleIndex,
    bound: BoundState,
    *,
    edge_budget: int | None = None,
    profile: DCDProfileCounters | None = None,
    active_only: bool = False,
) -> tuple[torch.Tensor, TrianglePack, bool]:
    tau_new = bound.tau_base.clone().to(torch.int32)
    if active_only:
        fixed_edges = torch.nonzero(bound.cand & (bound.lower == bound.upper), as_tuple=False).flatten()
        if fixed_edges.numel() > 0:
            tau_new[fixed_edges] = bound.lower[fixed_edges].to(torch.int32)
        cone_edges = torch.nonzero(bound.cand & (bound.lower < bound.upper), as_tuple=False).flatten()
    else:
        cone_edges = torch.nonzero(bound.cand, as_tuple=False).flatten()
    if profile is not None:
        profile.num_exact_peeling_edges = int(cone_edges.numel())
    if cone_edges.numel() == 0:
        empty_pack = TrianglePack(
            cone_edges,
            torch.zeros((1,), device=index.state.device, dtype=torch.long),
            torch.empty((0, 2), device=index.state.device, dtype=torch.int32),
        )
        return tau_new, empty_pack, False

    if edge_budget is not None:
        degree = index.state.rowptr[1:] - index.state.rowptr[:-1]
        src = index.state.edge_src[cone_edges].to(torch.long)
        dst = index.state.edge_dst[cone_edges].to(torch.long)
        cone_work = int(torch.sum(degree[src] + degree[dst]).item())
        if cone_edges.numel() > 1_000_000 or cone_work > int(edge_budget) * 1024:
            tau_new = _full_recompute_tau(index.state)
            empty_pack = TrianglePack(
                cone_edges,
                torch.zeros((cone_edges.numel() + 1,), device=index.state.device, dtype=torch.long),
                torch.empty((0, 2), device=index.state.device, dtype=torch.int32),
            )
            return tau_new, empty_pack, True

    pack = index.materialize(cone_edges)
    if profile is not None:
        profile.record_pack_scan(pack)
    local_edges = pack.edge_ids
    if local_edges.numel() == 0:
        return tau_new, pack, False

    local_upper = bound.upper[local_edges].to(torch.int32)
    max_k = int(torch.max(local_upper).item()) if local_upper.numel() > 0 else 2
    tau_new[local_edges] = 2
    if pack.other_edges.numel() == 0 or max_k < 3:
        return tau_new, pack, False

    other = pack.other_edges.to(torch.long)
    left_other = other[:, 0].contiguous()
    right_other = other[:, 1].contiguous()
    left_pos = torch.searchsorted(local_edges, left_other)
    right_pos = torch.searchsorted(local_edges, right_other)
    max_local = max(local_edges.numel() - 1, 0)
    left_pos_clamped = left_pos.clamp(max=max_local)
    right_pos_clamped = right_pos.clamp(max=max_local)
    left_local = (left_pos < local_edges.numel()) & (local_edges[left_pos_clamped] == left_other)
    right_local = (right_pos < local_edges.numel()) & (local_edges[right_pos_clamped] == right_other)
    counts = pack.tri_ptr[1:] - pack.tri_ptr[:-1]
    owner = torch.repeat_interleave(torch.arange(local_edges.numel(), device=index.state.device, dtype=torch.long), counts)

    alive = local_upper >= 3
    for k in range(3, max_k + 1):
        alive &= local_upper >= k
        while True:
            left_ok = torch.where(left_local, alive[left_pos_clamped], tau_new[left_other] >= k)
            right_ok = torch.where(right_local, alive[right_pos_clamped], tau_new[right_other] >= k)
            valid = left_ok & right_ok & alive[owner]
            support = torch.bincount(owner[valid], minlength=local_edges.numel()) if torch.any(valid) else torch.zeros((local_edges.numel(),), device=index.state.device, dtype=torch.long)
            evict = alive & (support < (k - 2))
            if not torch.any(evict):
                break
            alive[torch.nonzero(evict, as_tuple=False).flatten()] = False
        survivors = alive & (local_upper >= k)
        if torch.any(survivors):
            tau_new[local_edges[survivors]] = k
    return tau_new, pack, False


def maintain_dcd(
    state: DCDState,
    delta: DeltaGraph | dict,
    *,
    device: str | torch.device | None = None,
    edge_budget: int = 200_000,
    triangle_budget: int | None = None,
    max_rounds: int = 64,
    candidate_hops: int | None = None,
    index_backend: str = "streaming",
    enable_refinement: bool = False,
    collect_witness: bool = False,
) -> DCDResult:
    target = torch.device(device) if device is not None else state.device
    work_state = state.to(target) if state.device != target else state
    with torch.no_grad():
        normalized = normalize_updates(work_state, delta)
        base_state = normalized.state
        deleted_edges_old, _ = base_state.edge_ids_from_pairs(normalized.del_edges)
        new_state = apply_updates_to_graph(base_state, normalized.del_edges, normalized.ins_edges)
        inserted_edges_new, _ = new_state.edge_ids_from_pairs(normalized.ins_edges)

        old_index = EdgeTriangleIndex(base_state, triangle_budget=triangle_budget)
        new_index = EdgeTriangleIndex(new_state, triangle_budget=triangle_budget)
        seeds = initial_seed_edges(
            base_state,
            new_state,
            old_index,
            new_index,
            deleted_edges_old,
            inserted_edges_new,
            edge_budget=edge_budget,
        )
        cone_edges = _expand_cone(new_index, seeds, edge_budget=edge_budget, max_hops=candidate_hops)
        profile_counters = DCDProfileCounters(num_cone_edges=int(cone_edges.numel()))
        bound = build_initial_bounds(
            base_state,
            new_state,
            old_index,
            new_index,
            cone_edges,
            seeds,
            inserted_edges_new=inserted_edges_new,
            edge_budget=edge_budget,
        )

        if enable_refinement:
            delete_stats = _run_phase(
                bound,
                new_index,
                allow_raise_lower=False,
                edge_budget=edge_budget,
                max_rounds=max_rounds,
                profile=profile_counters,
            )
            insert_stats = _run_phase(
                bound,
                new_index,
                allow_raise_lower=True,
                edge_budget=edge_budget,
                max_rounds=max_rounds,
                profile=profile_counters,
            )
        else:
            delete_stats = PhaseStats()
            insert_stats = PhaseStats()
        tau_new, cone_pack, used_fallback = _solve_cone_exact(
            new_index,
            bound,
            edge_budget=edge_budget,
            profile=profile_counters,
            active_only=normalized.del_edges.numel() == 0 and normalized.ins_edges.numel() > 0,
        )
        new_state.tau = tau_new
        profile_counters.num_cone_edges = int(bound.cone_edges.numel())

        witness = (
            build_witness_cache(
                cone_pack,
                bound.upper[cone_pack.edge_ids] if cone_pack.edge_ids.numel() > 0 else torch.empty((0,), device=new_state.device, dtype=torch.int32),
                bound.lower,
                bound.upper,
            )
            if collect_witness
            else None
        )
        stats = DCDStats(
            candidate_edges=int(bound.cone_edges.numel()),
            candidate_triangles=cone_pack.record_count,
            delete_phase=delete_stats,
            insert_phase=insert_stats,
            used_exact_cone=not used_fallback,
            witness_cache=witness,
            profile={
                "index_backend": 0 if index_backend == "materialized" else 1,
                "used_full_recompute_fallback": int(used_fallback),
                **profile_counters.as_profile(),
            },
        )
        return DCDResult(new_state, tau_new, bound, stats)
