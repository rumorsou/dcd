from __future__ import annotations

from dataclasses import dataclass, field
import time

import torch

from ._compat import _lookup_edge_ids, device
from .bounds import (
    BoundState,
    _narrow_window_bounds,
    build_initial_bounds,
    edge_window_metrics,
    extend_bound_state_for_candidates,
)
from .graph_io import GraphUpdate, expand_update
from .graph_state import GraphState
from .local_exact import solve_candidate_truss_exact
from .propagation import PhaseStats, refine_phase
from .support_estimator import estimate_support_bounds_for_packed_edges
from .triangle_index import ExplicitTriangleIndex, TriangleIndex


@dataclass
class DCDResult:
    state: GraphState
    tau_new: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    candidate_mask: torch.Tensor
    seed_edges: torch.Tensor
    cone_edges: torch.Tensor
    candidate_triangle_count: int
    candidate_rounds: int
    delete_stats: PhaseStats
    insert_stats: PhaseStats
    used_fallback: bool = False
    profile: DCDProfile | None = None


@dataclass
class DCDRoundProfile:
    round_id: int
    cone_edges: int
    candidate_triangle_count: int
    delta_cone: int
    refill_edges: int = 0
    rescue_edges: int = 0
    t_expand: float = 0.0
    t_materialize: float = 0.0
    t_exact: float = 0.0
    t_refill: float = 0.0
    t_rescue: float = 0.0
    t_support: float = 0.0


@dataclass
class DCDProfile:
    materialize_calls: int = 0
    t_expand: float = 0.0
    t_materialize: float = 0.0
    t_exact: float = 0.0
    t_refill: float = 0.0
    t_rescue: float = 0.0
    t_support: float = 0.0
    rounds: list[DCDRoundProfile] = field(default_factory=list)


@dataclass
class DCDPreparedRuntime:
    base_state: GraphState
    new_state: GraphState
    old_index: TriangleIndex
    new_index: TriangleIndex
    expanded_update: dict[str, torch.Tensor]
    allow_candidate_decrease: bool
    uses_full_insert_closure: bool
    uses_triangle_closed_insert_cone: bool
    bound_state: BoundState
    explicit_index: ExplicitTriangleIndex | None = None


def _apply_graph_updates(base_state: GraphState, expanded_update: dict[str, torch.Tensor]) -> GraphState:
    state = base_state.with_removed_edges(expanded_update["edge_deletes"])
    state = state.with_inserted_edges(expanded_update["edge_inserts"])
    return state


def _inject_inserted_bounds(bound_state: BoundState, new_state: GraphState, inserted_pairs: torch.Tensor) -> None:
    if inserted_pairs.numel() == 0:
        return
    inserted_ids, _ = _lookup_edge_ids(new_state.num_vertices, new_state.edge_code, inserted_pairs)
    if inserted_ids.numel() == 0:
        return
    bound_state.candidate_mask[inserted_ids] = True
    bound_state.lower[inserted_ids] = 2
    bound_state.upper[inserted_ids] = torch.maximum(bound_state.upper[inserted_ids], torch.full_like(inserted_ids, 2))


def tensorized_dcd_maintain(
    base_state: GraphState,
    update: GraphUpdate,
    *,
    max_rounds: int = 1000,
    allow_full_recompute_fallback: bool = True,
    enable_refinement: bool = False,
    initial_candidate_hops: int | None = 0,
    max_candidate_rounds: int = 64,
) -> DCDResult:
    _ = allow_full_recompute_fallback
    expanded_base_state, expanded_update = expand_update(base_state, update)
    runtime = prepare_dcd_runtime(
        expanded_base_state=expanded_base_state,
        expanded_update=expanded_update,
        candidate_hops=initial_candidate_hops,
    )
    return execute_prepared_dcd(
        runtime,
        max_rounds=max_rounds,
        enable_refinement=enable_refinement,
        max_candidate_rounds=max_candidate_rounds,
    )


def prepare_dcd_runtime(
    base_state: GraphState | None = None,
    update: GraphUpdate | None = None,
    *,
    expanded_base_state: GraphState | None = None,
    expanded_update: dict[str, torch.Tensor] | None = None,
    candidate_hops: int | None = 0,
) -> DCDPreparedRuntime:
    if expanded_base_state is None or expanded_update is None:
        if base_state is None or update is None:
            raise ValueError("Either provide base_state/update or expanded_base_state/expanded_update.")
        expanded_base_state, expanded_update = expand_update(base_state, update)

    new_state = _apply_graph_updates(expanded_base_state, expanded_update)
    old_index = TriangleIndex(expanded_base_state)
    new_index = TriangleIndex(new_state)
    uses_full_insert_closure = False
    uses_triangle_closed_insert_cone = False
    if candidate_hops == 0 and expanded_update["edge_inserts"].numel() > 0 and expanded_update["edge_deletes"].numel() == 0:
        tentative_bound_state = build_initial_bounds(
            old_state=expanded_base_state,
            new_state=new_state,
            old_index=old_index,
            new_index=new_index,
            inserted_pairs=expanded_update["edge_inserts"],
            deleted_pairs=expanded_update["edge_deletes"],
            candidate_hops=1,
        )
        if _is_triangle_closed_candidate(tentative_bound_state, new_index):
            bound_state = tentative_bound_state
            uses_triangle_closed_insert_cone = True
        else:
            bound_state = build_initial_bounds(
                old_state=expanded_base_state,
                new_state=new_state,
                old_index=old_index,
                new_index=new_index,
                inserted_pairs=expanded_update["edge_inserts"],
                deleted_pairs=expanded_update["edge_deletes"],
                candidate_hops=None,
            )
            uses_full_insert_closure = True
    else:
        effective_candidate_hops = candidate_hops
        if candidate_hops == 0 and expanded_update["edge_inserts"].numel() > 0:
            effective_candidate_hops = None
        uses_full_insert_closure = expanded_update["edge_inserts"].numel() > 0 and effective_candidate_hops is None
        bound_state = build_initial_bounds(
            old_state=expanded_base_state,
            new_state=new_state,
            old_index=old_index,
            new_index=new_index,
            inserted_pairs=expanded_update["edge_inserts"],
            deleted_pairs=expanded_update["edge_deletes"],
            candidate_hops=effective_candidate_hops,
        )
    _inject_inserted_bounds(bound_state, new_state, expanded_update["edge_inserts"])
    exact_edges = _exact_candidate_edges(bound_state)
    explicit_index = new_index.materialize(exact_edges) if exact_edges.numel() > 0 else None
    return DCDPreparedRuntime(
        base_state=expanded_base_state,
        new_state=new_state,
        old_index=old_index,
        new_index=new_index,
        expanded_update=expanded_update,
        allow_candidate_decrease=expanded_update["edge_deletes"].numel() > 0,
        uses_full_insert_closure=uses_full_insert_closure,
        uses_triangle_closed_insert_cone=uses_triangle_closed_insert_cone,
        bound_state=bound_state,
        explicit_index=explicit_index,
    )


def _clone_bound_state(bound_state: BoundState) -> BoundState:
    return BoundState(
        tau_base=bound_state.tau_base.clone(),
        lower=bound_state.lower.clone(),
        upper=bound_state.upper.clone(),
        delta_plus=bound_state.delta_plus.clone(),
        delta_minus=bound_state.delta_minus.clone(),
        candidate_mask=bound_state.candidate_mask.clone(),
        seed_edges=bound_state.seed_edges.clone(),
        seed_neighbor_mask=bound_state.seed_neighbor_mask.clone(),
        cone_edges=bound_state.cone_edges.clone(),
    )


def _exact_candidate_mask(bound_state: BoundState) -> torch.Tensor:
    return bound_state.candidate_mask & (bound_state.lower < bound_state.upper)


def _exact_candidate_edges(bound_state: BoundState) -> torch.Tensor:
    exact_mask = _exact_candidate_mask(bound_state)
    return torch.nonzero(exact_mask, as_tuple=False).flatten()


def _refresh_explicit_index(
    runtime: DCDPreparedRuntime,
    explicit_index: ExplicitTriangleIndex | None,
    exact_edges: torch.Tensor,
    profile: DCDProfile,
    round_profile: DCDRoundProfile,
    *,
    patch_growth_threshold: float = 0.12,
    patch_edge_limit: int = 4096,
) -> ExplicitTriangleIndex | None:
    if exact_edges.numel() == 0:
        return None
    if explicit_index is not None and torch.equal(explicit_index.edge_ids, exact_edges):
        return explicit_index

    use_patch = False
    if explicit_index is not None and explicit_index.edge_ids.numel() > 0:
        added_edges = exact_edges[~torch.isin(exact_edges, explicit_index.edge_ids)]
        growth_ratio = float(added_edges.numel()) / float(max(int(exact_edges.numel()), 1))
        use_patch = (
            added_edges.numel() > 0
            and exact_edges.numel() >= explicit_index.edge_ids.numel()
            and added_edges.numel() <= patch_edge_limit
            and growth_ratio <= patch_growth_threshold
        )

    started_at = time.perf_counter()
    refreshed = (
        runtime.new_index.extend_materialize(explicit_index, exact_edges)
        if use_patch
        else runtime.new_index.materialize(exact_edges)
    )
    elapsed = time.perf_counter() - started_at
    profile.materialize_calls += 1
    profile.t_materialize += elapsed
    round_profile.t_materialize += elapsed
    return refreshed


def _is_triangle_closed_candidate(bound_state: BoundState, triangle_index: TriangleIndex) -> bool:
    cone_edges = bound_state.cone_edges
    if cone_edges.numel() == 0:
        return True
    neighbor_edges = triangle_index.triangle_neighbors(cone_edges)
    if neighbor_edges.numel() == 0:
        return True
    return bool(torch.all(bound_state.candidate_mask[neighbor_edges]).item())


def _tighten_event_edges(bound_state: BoundState, tau_new: torch.Tensor) -> torch.Tensor:
    candidate_edges = bound_state.cone_edges
    if candidate_edges.numel() == 0:
        return candidate_edges
    tightened_mask = (tau_new[candidate_edges] > bound_state.lower[candidate_edges]) | (
        tau_new[candidate_edges] < bound_state.upper[candidate_edges]
    )
    return candidate_edges[tightened_mask]


def _sorted_pack(
    triangle_index: TriangleIndex,
    edge_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_ids = edge_ids.to(device=device, dtype=torch.long)
    order = torch.argsort(edge_ids, stable=True)
    sorted_edge_ids = edge_ids[order]
    _, left_edge, right_edge, source_local = triangle_index.pack(sorted_edge_ids)
    return order, left_edge, right_edge, source_local


def _saturated_candidate_edges(bound_state: BoundState, tau_new: torch.Tensor) -> torch.Tensor:
    candidate_edges = bound_state.cone_edges
    if candidate_edges.numel() == 0:
        return candidate_edges
    saturated_mask = tau_new[candidate_edges] == bound_state.upper[candidate_edges]
    return candidate_edges[saturated_mask]


def _rescue_candidate_edges(runtime: DCDPreparedRuntime, bound_state: BoundState, tau_new: torch.Tensor) -> torch.Tensor:
    saturated_edges = _saturated_candidate_edges(bound_state, tau_new)
    if saturated_edges.numel() == 0:
        return saturated_edges
    rescue_edges = runtime.new_index.triangle_neighbors(saturated_edges)
    if rescue_edges.numel() == 0:
        return rescue_edges
    return rescue_edges[~bound_state.candidate_mask[rescue_edges]]


def _rebuild_bound_state(
    runtime: DCDPreparedRuntime,
    current_bound_state: BoundState,
    candidate_edges: torch.Tensor,
) -> BoundState:
    inserted_ids, _ = _lookup_edge_ids(
        runtime.new_state.num_vertices,
        runtime.new_state.edge_code,
        runtime.expanded_update["edge_inserts"],
    )
    bound_state = extend_bound_state_for_candidates(
        base_bound_state=current_bound_state,
        old_state=runtime.base_state,
        new_state=runtime.new_state,
        old_index=runtime.old_index,
        new_index=runtime.new_index,
        inserted_ids=inserted_ids,
        seed_edges=current_bound_state.seed_edges,
        candidate_edges=candidate_edges,
        allow_decrease=runtime.allow_candidate_decrease,
    )
    _inject_inserted_bounds(bound_state, runtime.new_state, runtime.expanded_update["edge_inserts"])
    return bound_state


def _refill_candidate_edges(
    runtime: DCDPreparedRuntime,
    bound_state: BoundState,
    tau_new: torch.Tensor,
    refilled_mask: torch.Tensor,
    round_profile: DCDRoundProfile | None = None,
) -> torch.Tensor:
    tightened_edges = _tighten_event_edges(bound_state, tau_new)
    if tightened_edges.numel() == 0:
        return tightened_edges

    refill_pool = runtime.new_index.triangle_neighbors(tightened_edges)
    if refill_pool.numel() == 0:
        return refill_pool
    refill_pool = refill_pool[~bound_state.candidate_mask[refill_pool]]
    refill_pool = refill_pool[~refilled_mask[refill_pool]]
    if refill_pool.numel() == 0:
        return refill_pool

    before_lower = bound_state.tau_base.clone()
    before_upper = bound_state.tau_base.clone()
    active_edges = bound_state.cone_edges
    before_lower[active_edges] = bound_state.lower[active_edges]
    before_upper[active_edges] = bound_state.upper[active_edges]
    after_lower = before_lower.clone()
    after_upper = before_upper.clone()
    after_lower[tightened_edges] = tau_new[tightened_edges]
    after_upper[tightened_edges] = tau_new[tightened_edges]

    inserted_ids, _ = _lookup_edge_ids(
        runtime.new_state.num_vertices,
        runtime.new_state.edge_code,
        runtime.expanded_update["edge_inserts"],
    )
    refill_tau_base, refill_upper_cap, refill_plus, refill_minus = edge_window_metrics(
        old_state=runtime.base_state,
        new_state=runtime.new_state,
        old_index=runtime.old_index,
        new_index=runtime.new_index,
        edge_ids=refill_pool,
        inserted_ids=inserted_ids,
    )
    refill_lower, refill_upper = _narrow_window_bounds(
        refill_tau_base,
        refill_upper_cap,
        allow_decrease=runtime.allow_candidate_decrease,
    )
    before_lower[refill_pool] = refill_lower
    before_upper[refill_pool] = refill_upper
    after_lower[refill_pool] = refill_lower
    after_upper[refill_pool] = refill_upper

    critical_k = torch.div(refill_lower + refill_upper + 1, 2, rounding_mode="floor")
    support_order, left_edge, right_edge, source_local = _sorted_pack(runtime.new_index, refill_pool)
    sorted_refill_pool = refill_pool[support_order]
    sorted_critical_k = critical_k[support_order]
    support_started_at = time.perf_counter()
    smin_before_sorted, smax_before_sorted = estimate_support_bounds_for_packed_edges(
        source_local=source_local,
        left_edge=left_edge,
        right_edge=right_edge,
        edge_count=sorted_refill_pool.numel(),
        k_values=sorted_critical_k,
        lower=before_lower,
        upper=before_upper,
    )
    smin_after_sorted, smax_after_sorted = estimate_support_bounds_for_packed_edges(
        source_local=source_local,
        left_edge=left_edge,
        right_edge=right_edge,
        edge_count=sorted_refill_pool.numel(),
        k_values=sorted_critical_k,
        lower=after_lower,
        upper=after_upper,
    )
    support_elapsed = time.perf_counter() - support_started_at
    if round_profile is not None:
        round_profile.t_support += support_elapsed
    smin_before = torch.empty_like(smin_before_sorted)
    smax_before = torch.empty_like(smax_before_sorted)
    smin_after = torch.empty_like(smin_after_sorted)
    smax_after = torch.empty_like(smax_after_sorted)
    smin_before[support_order] = smin_before_sorted
    smax_before[support_order] = smax_before_sorted
    smin_after[support_order] = smin_after_sorted
    smax_after[support_order] = smax_after_sorted
    window_open = refill_lower < refill_upper
    delta_active = (refill_plus > 0) | (refill_minus > 0)
    support_uncertain_before = (smin_before < (critical_k - 2)) & (smax_before >= (critical_k - 2))
    support_uncertain_after = (smin_after < (critical_k - 2)) & (smax_after >= (critical_k - 2))
    support_changed = (smin_before != smin_after) | (smax_before != smax_after)
    if runtime.expanded_update["edge_inserts"].numel() == 0:
        tightened_mask = torch.zeros((runtime.new_state.num_edges,), device=device, dtype=torch.bool)
        tightened_mask[tightened_edges] = True
        shared_tightened_sorted = torch.zeros((sorted_refill_pool.numel(),), device=device, dtype=torch.bool)
        if source_local.numel() > 0:
            tri_hits = tightened_mask[left_edge] | tightened_mask[right_edge]
            if torch.any(tri_hits):
                shared_tightened_sorted = torch.bincount(
                    source_local[tri_hits],
                    minlength=sorted_refill_pool.numel(),
                ) > 0
        shared_tightened = torch.empty_like(shared_tightened_sorted)
        shared_tightened[support_order] = shared_tightened_sorted
        affected_mask = bound_state.seed_neighbor_mask[refill_pool] | shared_tightened
        crossable = delta_active | support_uncertain_before | support_uncertain_after | support_changed | affected_mask
    else:
        affected_mask = bound_state.seed_neighbor_mask[refill_pool] | support_changed
        crossable = delta_active | support_uncertain_before | support_uncertain_after | support_changed
    return refill_pool[window_open & crossable & affected_mask]


def execute_prepared_dcd(
    runtime: DCDPreparedRuntime,
    *,
    max_rounds: int = 1000,
    enable_refinement: bool = False,
    max_candidate_rounds: int = 64,
    max_insert_refill_rounds: int = 2,
    max_insert_rescue_rounds: int = 4,
    insert_rescue_saturation_threshold: float = 0.30,
    insert_full_closure_threshold: int = 15_000,
    insert_rescue_over_refill_factor: int = 4,
) -> DCDResult:
    new_state = runtime.new_state
    new_index = runtime.new_index
    bound_state = _clone_bound_state(runtime.bound_state)
    used_fallback = False
    fallback_triggered = False
    delete_stats = PhaseStats()
    insert_stats = PhaseStats()
    tau_new = bound_state.tau_base.clone()
    explicit_index = runtime.explicit_index
    candidate_triangle_count = int(explicit_index.edges_of_tri.size(0)) if explicit_index is not None else 0
    profile = DCDProfile()
    candidate_rounds = 0
    refilled_mask = torch.zeros((runtime.new_state.num_edges,), device=device, dtype=torch.bool)
    insert_refills_since_rescue = 0
    insert_rescue_rounds = 0
    is_insert_phase = runtime.expanded_update["edge_inserts"].numel() > 0
    previous_cone_size = int(bound_state.cone_edges.numel())

    while True:
        candidate_rounds += 1
        current_cone_size = int(bound_state.cone_edges.numel())
        round_profile = DCDRoundProfile(
            round_id=candidate_rounds,
            cone_edges=current_cone_size,
            candidate_triangle_count=candidate_triangle_count,
            delta_cone=0 if candidate_rounds == 1 else max(current_cone_size - previous_cone_size, 0),
        )
        profile.rounds.append(round_profile)
        previous_cone_size = current_cone_size

        if candidate_rounds > max_candidate_rounds and not fallback_triggered:
            used_fallback = True
            fallback_triggered = True
            expand_started_at = time.perf_counter()
            bound_state = build_initial_bounds(
                old_state=runtime.base_state,
                new_state=runtime.new_state,
                old_index=runtime.old_index,
                new_index=runtime.new_index,
                inserted_pairs=runtime.expanded_update["edge_inserts"],
                deleted_pairs=runtime.expanded_update["edge_deletes"],
                candidate_hops=None,
            )
            _inject_inserted_bounds(bound_state, runtime.new_state, runtime.expanded_update["edge_inserts"])
            expand_elapsed = time.perf_counter() - expand_started_at
            round_profile.t_expand += expand_elapsed
            profile.t_expand += expand_elapsed
            refilled_mask.zero_()

        exact_edges = _exact_candidate_edges(bound_state)
        explicit_index = _refresh_explicit_index(
            runtime=runtime,
            explicit_index=explicit_index,
            exact_edges=exact_edges,
            profile=profile,
            round_profile=round_profile,
        )
        candidate_triangle_count = int(explicit_index.edges_of_tri.size(0)) if explicit_index is not None else 0
        round_profile.candidate_triangle_count = candidate_triangle_count

        if enable_refinement:
            delete_stats = refine_phase(
                triangle_index=new_index,
                lower=bound_state.lower,
                upper=bound_state.upper,
                candidate_mask=bound_state.candidate_mask,
                allow_raise_lower=False,
                max_rounds=max_rounds,
            )
            insert_stats = refine_phase(
                triangle_index=new_index,
                lower=bound_state.lower,
                upper=bound_state.upper,
                candidate_mask=bound_state.candidate_mask,
                allow_raise_lower=True,
                max_rounds=max_rounds,
            )
        else:
            delete_stats = PhaseStats()
            insert_stats = PhaseStats()

        exact_candidate_mask = torch.zeros_like(bound_state.candidate_mask)
        if exact_edges.numel() > 0:
            exact_candidate_mask[exact_edges] = True
        fixed_tau = bound_state.tau_base.clone()
        resolved_mask = bound_state.candidate_mask & ~exact_candidate_mask
        if torch.any(resolved_mask):
            fixed_tau[resolved_mask] = bound_state.lower[resolved_mask]

        exact_started_at = time.perf_counter()
        tau_new = solve_candidate_truss_exact(
            triangle_index=new_index,
            candidate_mask=exact_candidate_mask,
            fixed_tau=fixed_tau,
            upper=bound_state.upper,
            explicit_index=explicit_index,
        )
        exact_elapsed = time.perf_counter() - exact_started_at
        round_profile.t_exact += exact_elapsed
        profile.t_exact += exact_elapsed

        if is_insert_phase and (runtime.uses_full_insert_closure or runtime.uses_triangle_closed_insert_cone):
            break

        if not is_insert_phase:
            refill_started_at = time.perf_counter()
            support_before = round_profile.t_support
            refill_edges = _refill_candidate_edges(
                runtime=runtime,
                bound_state=bound_state,
                tau_new=tau_new,
                refilled_mask=refilled_mask,
                round_profile=round_profile,
            )
            refill_elapsed = time.perf_counter() - refill_started_at
            round_profile.t_refill += refill_elapsed
            round_profile.refill_edges = int(refill_edges.numel())
            profile.t_refill += refill_elapsed
            profile.t_support += round_profile.t_support - support_before
            if refill_edges.numel() == 0:
                break
            refilled_mask[refill_edges] = True
            expand_started_at = time.perf_counter()
            expanded_candidate_edges = torch.unique(torch.cat((bound_state.cone_edges, refill_edges)))
            bound_state = _rebuild_bound_state(runtime, bound_state, expanded_candidate_edges)
            expand_elapsed = time.perf_counter() - expand_started_at
            round_profile.t_expand += expand_elapsed
            profile.t_expand += expand_elapsed
            continue

        if insert_refills_since_rescue < max_insert_refill_rounds:
            refill_started_at = time.perf_counter()
            support_before = round_profile.t_support
            refill_edges = _refill_candidate_edges(
                runtime=runtime,
                bound_state=bound_state,
                tau_new=tau_new,
                refilled_mask=refilled_mask,
                round_profile=round_profile,
            )
            refill_elapsed = time.perf_counter() - refill_started_at
            round_profile.t_refill += refill_elapsed
            round_profile.refill_edges = int(refill_edges.numel())
            profile.t_refill += refill_elapsed
            profile.t_support += round_profile.t_support - support_before

            rescue_started_at = time.perf_counter()
            rescue_edges = _rescue_candidate_edges(runtime, bound_state, tau_new)
            rescue_elapsed = time.perf_counter() - rescue_started_at
            round_profile.t_rescue += rescue_elapsed
            round_profile.rescue_edges = int(rescue_edges.numel())
            profile.t_rescue += rescue_elapsed
            prefer_rescue = (
                rescue_edges.numel() > 0
                and (
                    refill_edges.numel() == 0
                    or int(refill_edges.numel()) > insert_rescue_over_refill_factor * int(rescue_edges.numel())
                )
            )
            if prefer_rescue and insert_rescue_rounds < max_insert_rescue_rounds:
                insert_rescue_rounds += 1
                insert_refills_since_rescue = 0
                expand_started_at = time.perf_counter()
                expanded_candidate_edges = torch.unique(torch.cat((bound_state.cone_edges, rescue_edges)))
                bound_state = _rebuild_bound_state(runtime, bound_state, expanded_candidate_edges)
                expand_elapsed = time.perf_counter() - expand_started_at
                round_profile.t_expand += expand_elapsed
                profile.t_expand += expand_elapsed
                continue
            if refill_edges.numel() > 0:
                refilled_mask[refill_edges] = True
                insert_refills_since_rescue += 1
                expand_started_at = time.perf_counter()
                expanded_candidate_edges = torch.unique(torch.cat((bound_state.cone_edges, refill_edges)))
                bound_state = _rebuild_bound_state(runtime, bound_state, expanded_candidate_edges)
                expand_elapsed = time.perf_counter() - expand_started_at
                round_profile.t_expand += expand_elapsed
                profile.t_expand += expand_elapsed
                continue

        saturated_edges = _saturated_candidate_edges(bound_state, tau_new)
        saturation_ratio = float(saturated_edges.numel()) / float(max(int(bound_state.cone_edges.numel()), 1))
        rescue_started_at = time.perf_counter()
        rescue_edges = _rescue_candidate_edges(runtime, bound_state, tau_new)
        rescue_elapsed = time.perf_counter() - rescue_started_at
        round_profile.t_rescue += rescue_elapsed
        round_profile.rescue_edges = max(round_profile.rescue_edges, int(rescue_edges.numel()))
        profile.t_rescue += rescue_elapsed
        if (
            insert_rescue_rounds >= max_insert_rescue_rounds
            or saturation_ratio <= insert_rescue_saturation_threshold
            or rescue_edges.numel() == 0
        ):
            if int(bound_state.cone_edges.numel()) >= insert_full_closure_threshold:
                used_fallback = True
                fallback_triggered = True
                expand_started_at = time.perf_counter()
                bound_state = build_initial_bounds(
                    old_state=runtime.base_state,
                    new_state=runtime.new_state,
                    old_index=runtime.old_index,
                    new_index=runtime.new_index,
                    inserted_pairs=runtime.expanded_update["edge_inserts"],
                    deleted_pairs=runtime.expanded_update["edge_deletes"],
                    candidate_hops=None,
                )
                _inject_inserted_bounds(bound_state, runtime.new_state, runtime.expanded_update["edge_inserts"])
                expand_elapsed = time.perf_counter() - expand_started_at
                round_profile.t_expand += expand_elapsed
                profile.t_expand += expand_elapsed

                exact_edges = _exact_candidate_edges(bound_state)
                explicit_index = _refresh_explicit_index(
                    runtime=runtime,
                    explicit_index=None,
                    exact_edges=exact_edges,
                    profile=profile,
                    round_profile=round_profile,
                )
                candidate_triangle_count = int(explicit_index.edges_of_tri.size(0)) if explicit_index is not None else 0
                round_profile.candidate_triangle_count = candidate_triangle_count
                exact_candidate_mask = torch.zeros_like(bound_state.candidate_mask)
                if exact_edges.numel() > 0:
                    exact_candidate_mask[exact_edges] = True
                fixed_tau = bound_state.tau_base.clone()
                resolved_mask = bound_state.candidate_mask & ~exact_candidate_mask
                if torch.any(resolved_mask):
                    fixed_tau[resolved_mask] = bound_state.lower[resolved_mask]
                exact_started_at = time.perf_counter()
                tau_new = solve_candidate_truss_exact(
                    triangle_index=new_index,
                    candidate_mask=exact_candidate_mask,
                    fixed_tau=fixed_tau,
                    upper=bound_state.upper,
                    explicit_index=explicit_index,
                )
                exact_elapsed = time.perf_counter() - exact_started_at
                round_profile.t_exact += exact_elapsed
                profile.t_exact += exact_elapsed
            break

        insert_rescue_rounds += 1
        insert_refills_since_rescue = 0
        expand_started_at = time.perf_counter()
        expanded_candidate_edges = torch.unique(torch.cat((bound_state.cone_edges, rescue_edges)))
        bound_state = _rebuild_bound_state(runtime, bound_state, expanded_candidate_edges)
        expand_elapsed = time.perf_counter() - expand_started_at
        round_profile.t_expand += expand_elapsed
        profile.t_expand += expand_elapsed

    bound_state.lower = tau_new.clone()
    bound_state.upper = tau_new.clone()

    new_state = GraphState(
        row_ptr=new_state.row_ptr,
        columns=new_state.columns,
        tau=tau_new,
        vertex_labels=new_state.vertex_labels,
        edge_code=new_state.edge_code,
        bidirectional_view=new_state.bidirectional_view,
        edge_src_index=new_state.edge_src_index,
    )
    return DCDResult(
        state=new_state,
        tau_new=tau_new,
        lower=bound_state.lower,
        upper=bound_state.upper,
        candidate_mask=bound_state.candidate_mask,
        seed_edges=bound_state.seed_edges,
        cone_edges=bound_state.cone_edges,
        candidate_triangle_count=candidate_triangle_count,
        candidate_rounds=candidate_rounds,
        delete_stats=delete_stats,
        insert_stats=insert_stats,
        used_fallback=used_fallback,
        profile=profile,
    )
