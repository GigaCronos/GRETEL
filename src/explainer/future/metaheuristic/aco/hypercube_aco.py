import hashlib
from collections import deque
import math
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Deque, Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta


Modification = Tuple[str, Tuple[int, int]]
BinarySolution = np.ndarray
PheromoneDeposit = Dict[str, object]
AntCheckpoint = Tuple[BinarySolution, int]

PHEROMONE_MASS_SMOOTHING = 1e-12


# -----------------------------------------------------------------------------
# ProcessPool worker support
# -----------------------------------------------------------------------------
# Workers construct forward trajectories only. They never build GraphInstance
# objects and never call the oracle. Oracle calls, cache updates, best-solution
# updates and pheromone deposits remain serialized in the main process.
_PROCESS_WORKER_CONTEXT = None


def _init_process_worker(worker_context: Dict[str, object]):
    global _PROCESS_WORKER_CONTEXT
    _PROCESS_WORKER_CONTEXT = worker_context


def _process_pool_construct_forward_ant(
    task: Tuple[List[PheromoneDeposit], int, int]
) -> List[AntCheckpoint]:
    if _PROCESS_WORKER_CONTEXT is None:
        raise RuntimeError("Process worker context was not initialized")

    pheromone_deposits, iteration, ant_id = task
    return _construct_forward_ant_from_context(
        ctx=_PROCESS_WORKER_CONTEXT,
        pheromone_deposits=pheromone_deposits,
        iteration=iteration,
        ant_id=ant_id,
    )


def _anneal_parameter(
    initial_value: float,
    final_value: float,
    current_iter: int,
    max_iter: int,
    decay_rate: float,
) -> float:
    if max_iter <= 1:
        return initial_value

    progress = current_iter / max(1, max_iter - 1)
    return final_value + (initial_value - final_value) * math.exp(-decay_rate * progress)


def _construct_forward_ant_from_context(
    ctx: Dict[str, object],
    pheromone_deposits: List[PheromoneDeposit],
    iteration: int,
    ant_id: int,
) -> List[AntCheckpoint]:
    seed = int(ctx["random_seed"]) + 1000003 * iteration + 9176 * ant_id
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed % (2**32 - 1))

    epsilon = _anneal_parameter(
        float(ctx["explorer_epsilon0"]),
        float(ctx["explorer_epsilonfinal"]),
        iteration,
        int(ctx["num_iterations"]),
        float(ctx["explorer_epsilon_decay_rate"]),
    )

    return _worker_construct_forward_ant_path(
        ctx=ctx,
        pheromone_deposits=pheromone_deposits,
        epsilon=epsilon,
        rng=rng,
        np_rng=np_rng,
    )


def _worker_prepare_pheromone_arrays(
    pheromone_deposits: List[PheromoneDeposit],
    m: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not pheromone_deposits:
        return (
            np.empty((0, m), dtype=np.int8),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    centers = np.vstack(
        [np.asarray(deposit["center"], dtype=np.int8) for deposit in pheromone_deposits]
    )
    powers = np.asarray(
        [float(deposit["power"]) for deposit in pheromone_deposits],
        dtype=np.float64,
    )
    edit_distances = np.asarray(
        [int(deposit["edit_distance"]) for deposit in pheromone_deposits],
        dtype=np.int64,
    )
    return centers, powers, edit_distances


def _worker_compute_deposit_weights(
    ctx: Dict[str, object],
    valid_powers: np.ndarray,
    valid_hdist: np.ndarray,
) -> np.ndarray:
    alpha_pheromone = float(ctx["alpha_pheromone"])
    beta_compatibility = float(ctx["beta_compatibility"])
    alpha_scale = float(ctx["alpha_scale"])
    min_sigma = float(ctx["min_sigma"])

    powered_pheromone = np.maximum(valid_powers, 0.0) ** alpha_pheromone

    if beta_compatibility <= 0.0:
        compatibility = np.ones_like(valid_powers, dtype=np.float64)
    else:
        sigmas = np.maximum(alpha_scale * valid_powers, min_sigma)
        compatibility = np.exp(
            -beta_compatibility * valid_hdist / (2.0 * sigmas * sigmas)
        )

    return powered_pheromone * compatibility


def _worker_compute_candidate_masses(
    ctx: Dict[str, object],
    candidates: List[int],
    centers: np.ndarray,
    powers: np.ndarray,
    hdist: np.ndarray,
) -> np.ndarray:
    if len(powers) == 0:
        return np.ones(len(candidates), dtype=np.float64)

    valid = powers > 0.0
    if not np.any(valid):
        return np.ones(len(candidates), dtype=np.float64)

    valid_powers = powers[valid]
    valid_hdist = hdist[valid]
    valid_centers = centers[valid]

    deposit_weights = _worker_compute_deposit_weights(
        ctx=ctx,
        valid_powers=valid_powers,
        valid_hdist=valid_hdist,
    )

    candidate_array = np.asarray(candidates, dtype=np.int64)
    candidate_bits = valid_centers[:, candidate_array]
    masses = np.dot(candidate_bits.T.astype(np.float64), deposit_weights)

    if not np.all(np.isfinite(masses)) or float(np.sum(masses)) <= 0.0:
        return np.ones(len(candidates), dtype=np.float64)

    return masses


def _worker_sample_dimension_from_candidates(
    ctx: Dict[str, object],
    candidates: List[int],
    centers: np.ndarray,
    powers: np.ndarray,
    hdist: np.ndarray,
    epsilon: float,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Optional[int]:
    if not candidates:
        return None

    if rng.random() < epsilon or len(powers) == 0:
        return rng.choice(candidates)

    masses = _worker_compute_candidate_masses(
        ctx=ctx,
        candidates=candidates,
        centers=centers,
        powers=powers,
        hdist=hdist,
    )

    weights = np.maximum(masses, PHEROMONE_MASS_SMOOTHING)
    total = float(np.sum(weights))

    if not np.isfinite(total) or total <= 0.0:
        return rng.choice(candidates)

    probabilities = weights / total
    candidate_array = np.asarray(candidates, dtype=np.int64)
    return int(np_rng.choice(candidate_array, p=probabilities))


def _worker_sample_next_dimension(
    ctx: Dict[str, object],
    available_add_dims: Set[int],
    available_remove_dims: Set[int],
    centers: np.ndarray,
    powers: np.ndarray,
    hdist: np.ndarray,
    epsilon: float,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Optional[int]:
    if not available_add_dims and not available_remove_dims:
        return None

    if available_add_dims and available_remove_dims:
        candidates_set = available_add_dims if rng.random() < 0.5 else available_remove_dims
    elif available_add_dims:
        candidates_set = available_add_dims
    else:
        candidates_set = available_remove_dims

    return _worker_sample_dimension_from_candidates(
        ctx=ctx,
        candidates=list(candidates_set),
        centers=centers,
        powers=powers,
        hdist=hdist,
        epsilon=epsilon,
        rng=rng,
        np_rng=np_rng,
    )


def _worker_sample_adaptive_depth(
    ctx: Dict[str, object],
    pheromone_deposits: List[PheromoneDeposit],
    m: int,
    epsilon: float,
) -> int:
    hard_max = m
    max_path_length = ctx.get("max_path_length")
    if max_path_length is not None:
        hard_max = min(hard_max, int(max_path_length))

    if hard_max <= 0:
        return 0

    hard_min = 1

    valid_depths = []
    valid_weights = []
    for deposit in pheromone_deposits:
        depth = int(deposit.get("edit_distance", 0))
        power = float(deposit.get("power", 0.0))
        if depth > 0 and power > 0.0:
            valid_depths.append(depth)
            valid_weights.append(power)

    # Without deposits there is no learned reduced counterfactual scale yet.
    # Keep the previous behavior: use the configured hard maximum.
    if not valid_depths:
        return hard_max

    depths = np.asarray(valid_depths, dtype=np.float64)
    weights = np.asarray(valid_weights, dtype=np.float64)

    avg_depth = float(np.average(depths, weights=weights))

    # Forward search is degraded by explorer epsilon in favor of exploration,
    # while deposits come from backward-reduced counterfactuals. Therefore the
    # forward depth is expanded deterministically according to the current
    # explorer epsilon instead of being sampled from a depth distribution.
    bounded_epsilon = min(max(float(epsilon), 0.0), 0.999999)
    exploration_multiplier = 1.0 / (1.0 - bounded_epsilon)
    target_depth = int(math.ceil(avg_depth * exploration_multiplier) * 2.5)

    return max(hard_min, min(target_depth, hard_max))


def _worker_get_checkpoint_query_probability(
    ctx: Dict[str, object],
    edit_distance: int,
) -> float:
    p0 = min(max(float(ctx["oracle_query_probability"]), 0.0), 1.0)
    p_min = min(max(float(ctx["min_oracle_query_probability"]), 0.0), p0)
    scale = max(1.0, float(ctx["oracle_query_decay_scale"]))

    schedule = str(ctx["oracle_query_probability_schedule"]).lower()
    if schedule == "constant":
        p = p0
    elif schedule == "exp":
        p = p0 * math.exp(-float(edit_distance) / scale)
    else:
        p = p0 / (1.0 + float(edit_distance) / scale)

    return min(max(p, p_min), 1.0)


def _worker_should_query_checkpoint(
    ctx: Dict[str, object],
    edit_distance: int,
    max_steps: int,
    rng: random.Random,
) -> bool:
    if edit_distance == max_steps:
        return True
    return rng.random() < _worker_get_checkpoint_query_probability(ctx, edit_distance)


def _worker_construct_forward_ant_path(
    ctx: Dict[str, object],
    pheromone_deposits: List[PheromoneDeposit],
    epsilon: float,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> List[AntCheckpoint]:
    m = len(ctx["modification_space"])
    if m == 0:
        return []

    max_steps = _worker_sample_adaptive_depth(
        ctx=ctx,
        pheromone_deposits=pheromone_deposits,
        m=m,
        epsilon=epsilon,
    )
    if max_steps <= 0:
        return []

    solution = np.zeros(m, dtype=np.int8)
    checkpoints: List[AntCheckpoint] = []

    centers, powers, _ = _worker_prepare_pheromone_arrays(pheromone_deposits, m)
    num_deposits = len(powers)
    hdist = np.zeros(num_deposits, dtype=np.float64)

    available_add_dims: Set[int] = set(ctx["add_dims"])
    available_remove_dims: Set[int] = set(ctx["remove_dims"])

    for step in range(max_steps):
        dim = _worker_sample_next_dimension(
            ctx=ctx,
            available_add_dims=available_add_dims,
            available_remove_dims=available_remove_dims,
            centers=centers,
            powers=powers,
            hdist=hdist,
            epsilon=epsilon,
            rng=rng,
            np_rng=np_rng,
        )

        if dim is None:
            break

        solution[dim] = 1
        available_add_dims.discard(dim)
        available_remove_dims.discard(dim)

        if num_deposits > 0:
            hdist += (centers[:, dim] != 1)

        edit_distance = step + 1
        if _worker_should_query_checkpoint(ctx, edit_distance, max_steps, rng):
            checkpoints.append((solution.copy(), edit_distance))

    return checkpoints


class _BackwardRemovalQueue:
    """Static priority agenda for backward removal attempts.

    Priorities are computed once at the beginning of a backward run. The queue
    is not a dynamic state of the current solution: accepted removals disappear
    permanently, failed singleton removals disappear permanently, and only
    failed multi-element batches are reinserted at the front for immediate
    retry at a smaller batch size.

    On each pop, explorer_epsilon can force a random non-front extraction while
    preserving the relative order of the remaining items.
    """

    def __init__(
        self,
        dim_priorities: Optional[List[Tuple[int, float]]] = None,
        explorer_epsilon: float = 0.0,
        rng: Optional[random.Random] = None,
    ):
        self.items: Deque[int] = deque()
        self.explorer_epsilon = min(max(float(explorer_epsilon), 0.0), 1.0)
        self.rng = rng if rng is not None else random.Random()

        if not dim_priorities:
            return

        normalized_items = [
            (int(dim), float(priority), order)
            for order, (dim, priority) in enumerate(dim_priorities)
        ]

        normalized_items.sort(key=lambda item: (item[1], item[2]))
        self.items = deque(dim for dim, _, _ in normalized_items)

    def __len__(self) -> int:
        return len(self.items)

    def pop(self) -> Optional[int]:
        if not self.items:
            return None

        if len(self.items) > 1 and self.rng.random() < self.explorer_epsilon:
            random_idx = self.rng.randrange(1, len(self.items))
            self.items.rotate(-random_idx)
            dim = self.items.popleft()
            self.items.rotate(random_idx)
            return dim

        return self.items.popleft()

    def reinsert_batch(self, dims: List[int]):
        # dims is assumed to already be in the desired priority order.
        for dim in reversed(dims):
            self.items.appendleft(int(dim))


class HypercubeACOExplainer(Explainer, metaclass=ExplainerTransformMeta):
    """
    Hypercube ACO explainer for graph counterfactuals.

    Each binary dimension represents one possible edge edit relative to the
    original graph. Forward ants construct trajectories by activating one edit
    at a time. When a trajectory reaches a counterfactual checkpoint, a serial
    backward refinement removes batches of active edits while preserving the
    counterfactual prediction. The final refined counterfactual deposits one
    pheromone center in the binary hypercube.
    """

    # Log accumulated ACROSS instances (not reset per instance).
    # Each entry is (cache_hits, oracle_calls_used) for one explained instance.
    # To average over several instances/runs, read this attribute from the
    # notebook and clear it with HypercubeACOExplainer._cache_log.clear() before
    # each experiment.
    _cache_log: list = []

    def init(self):
        params = self.local_config.get("parameters", {})

        self.num_iterations = params.get("num_iterations", 10)
        self.num_workers = params.get("num_workers", 4)
        self.random_seed = int(params.get("random_seed", 0))
        self.rng = random.Random(self.random_seed)

        self.initial_num_ants = int(params.get("initial_num_ants", 1))
        self.final_num_ants = int(params.get("final_num_ants", 12))

        self.explorer_epsilon0 = params.get("explorer_epsilon0", 0.8)
        self.explorer_epsilonfinal = params.get("explorer_epsilonfinal", 0)
        self.explorer_epsilon_decay_rate = params.get("explorer_epsilon_decay_rate", 5.0)

        self.alpha_pheromone = params.get("alpha_pheromone", 1.0)
        self.beta_compatibility = params.get("beta_compatibility", 1.0)
        self.alpha_scale = params.get("alpha_scale", 1.0)
        self.min_sigma = params.get("min_sigma", 1e-3)

        self.max_path_length = params.get("max_path_length", None)

        self.oracle_query_probability = params.get("oracle_query_probability", 0.20)
        self.oracle_query_probability_schedule = params.get("oracle_query_probability_schedule", "inverse")
        self.min_oracle_query_probability = params.get("min_oracle_query_probability", 0.01)
        self.oracle_query_decay_scale = params.get("oracle_query_decay_scale", 50.0)

        self.max_oracle_calls = params.get("max_oracle_calls", params.get("oracle_call_hard_limit", None))
        self.solution_hash_bits = min(512, max(64, int(params.get("solution_hash_bits", 128))))

        self.rho = params.get("rho", 0.25)
        self.deposit_scale = params.get("deposit_scale", 2.0)
        self.score_gamma = params.get("score_gamma", 2.0)
        self.max_deposits = params.get("max_deposits", 256)
        self.early_stop_patience = params.get("early_stop_patience", 8)

        self.backward_initial_batch_size = int(params.get("backward_initial_batch_size", 5))
        self.backward_batch_growth = float(params.get("backward_batch_growth", 1.25))
        self.backward_batch_shrink = float(params.get("backward_batch_shrink", 0.5))
        self.enable_backward_refinement = bool(params.get("enable_backward_refinement", True))

        self.modification_space: List[Modification] = []
        self.add_dims: List[int] = []
        self.remove_dims: List[int] = []
        self.pheromone_deposits: List[PheromoneDeposit] = []
        self.current_iteration = 0
        self.oracle_calls_used = 0
        self.solution_prediction_cache: Dict[int, object] = {}
        self.cache_hits = 0
        self._last_best_score = -1.0
        self._last_best_solution = None
        self._last_best_graph = None

    def real_fit(self):
        pass

    def explain(self, instance: GraphInstance):
        best_graph, _, _ = self._run_hypercube_aco(instance)
        type(self)._cache_log.append((self.cache_hits, self.oracle_calls_used))
        return best_graph if best_graph is not None else instance

    # -------------------------------------------------------------------------
    # Main algorithm
    # -------------------------------------------------------------------------

    def _run_hypercube_aco(
        self,
        G_orig: GraphInstance,
    ) -> Tuple[Optional[GraphInstance], Optional[BinarySolution], float]:
        self._reset_state(G_orig)

        original_label = self._query_oracle(G_orig)
        if original_label is None:
            return None, None, -1.0

        best_solution: Optional[BinarySolution] = None
        best_graph: Optional[GraphInstance] = None
        best_score = -1.0
        no_improve_counter = 0

        use_parallel = self.num_workers is not None and self.num_workers > 1 and self.final_num_ants > 1
        executor = None
        if use_parallel:
            executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                initializer=_init_process_worker,
                initargs=(self._make_worker_context(),),
            )

        try:
            for iteration in range(self.num_iterations):
                if self._oracle_budget_reached():
                    break

                self.current_iteration = iteration
                self._evaporate_pheromones()
                pheromone_snapshot = self._make_pheromone_snapshot()
                ant_count = self._get_forward_ant_count(iteration)

                ant_paths = self._run_forward_ants(
                    executor=executor,
                    pheromone_snapshot=pheromone_snapshot,
                    ant_count=ant_count,
                )

                improved = self._evaluate_forward_ant_paths(
                    G_orig=G_orig,
                    original_label=original_label,
                    iteration_ant_paths=ant_paths,
                    best_score=best_score,
                    best_solution=best_solution,
                    best_graph=best_graph,
                )

                if self._last_best_score > best_score:
                    best_score = self._last_best_score
                    best_solution = self._last_best_solution
                    best_graph = self._last_best_graph
                    improved = True

                if improved:
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1

                if no_improve_counter >= self.early_stop_patience:
                    break
        finally:
            if executor is not None:
                executor.shutdown()

        return best_graph, best_solution, best_score

    def _reset_state(self, G_orig: GraphInstance):
        self.modification_space = self._generate_modifications(G_orig)
        self.add_dims, self.remove_dims = self._split_modification_dims(self.modification_space)
        self.pheromone_deposits = []
        self.current_iteration = 0
        self.oracle_calls_used = 0
        self.solution_prediction_cache = {}
        self.cache_hits = 0
        self._last_best_score = -1.0
        self._last_best_solution = None
        self._last_best_graph = None

    def _make_pheromone_snapshot(self) -> List[PheromoneDeposit]:
        return [
            {
                "center": deposit["center"].copy(),
                "power": float(deposit["power"]),
                "edit_distance": int(deposit["edit_distance"]),
            }
            for deposit in self.pheromone_deposits
        ]

    def _make_worker_context(self) -> Dict[str, object]:
        return {
            "modification_space": self.modification_space,
            "add_dims": list(self.add_dims),
            "remove_dims": list(self.remove_dims),
            "max_path_length": self.max_path_length,
            "oracle_query_probability": self.oracle_query_probability,
            "oracle_query_probability_schedule": self.oracle_query_probability_schedule,
            "min_oracle_query_probability": self.min_oracle_query_probability,
            "oracle_query_decay_scale": self.oracle_query_decay_scale,
            "alpha_pheromone": self.alpha_pheromone,
            "beta_compatibility": self.beta_compatibility,
            "alpha_scale": self.alpha_scale,
            "min_sigma": self.min_sigma,
            "explorer_epsilon0": self.explorer_epsilon0,
            "explorer_epsilonfinal": self.explorer_epsilonfinal,
            "explorer_epsilon_decay_rate": self.explorer_epsilon_decay_rate,
            "num_iterations": self.num_iterations,
            "random_seed": self.random_seed,
        }

    def _get_forward_ant_count(self, iteration: int) -> int:
        if self.num_iterations <= 1:
            return max(0, int(self.final_num_ants))

        progress = iteration / max(1, self.num_iterations - 1)
        value = self.initial_num_ants + (self.final_num_ants - self.initial_num_ants) * progress
        return max(0, int(round(value)))

    def _run_forward_ants(
        self,
        executor: Optional[ProcessPoolExecutor],
        pheromone_snapshot: List[PheromoneDeposit],
        ant_count: int,
    ) -> List[List[AntCheckpoint]]:
        if ant_count <= 0:
            return []

        if executor is None:
            context = self._make_worker_context()
            return [
                _construct_forward_ant_from_context(
                    ctx=context,
                    pheromone_deposits=pheromone_snapshot,
                    iteration=self.current_iteration,
                    ant_id=ant_id,
                )
                for ant_id in range(ant_count)
            ]

        tasks = [
            (pheromone_snapshot, self.current_iteration, ant_id)
            for ant_id in range(ant_count)
        ]
        return list(executor.map(_process_pool_construct_forward_ant, tasks))

    def _evaluate_forward_ant_paths(
        self,
        G_orig: GraphInstance,
        original_label,
        iteration_ant_paths: List[List[AntCheckpoint]],
        best_score: float,
        best_solution: Optional[BinarySolution],
        best_graph: Optional[GraphInstance],
    ) -> bool:
        improved = False

        for ant_idx, checkpoints in enumerate(iteration_ant_paths):
            if self._oracle_budget_reached():
                break

            for solution, _ in checkpoints:
                if self._oracle_budget_reached():
                    break

                score, changed = self._evaluate_solution(G_orig, original_label, solution)
                if score is None:
                    break
                if not changed:
                    continue

                if self.enable_backward_refinement:
                    final_solution, final_score, _ = self._run_backward_from_forward_cf(
                        G_orig=G_orig,
                        original_label=original_label,
                        initial_solution=solution,
                        initial_score=score,
                    )
                else:
                    final_solution, final_score = solution, score

                self._deposit_solution(final_solution, final_score)

                if final_score > best_score:
                    best_score = final_score
                    best_solution = final_solution.copy()
                    best_graph = self._apply_binary_solution(G_orig, final_solution)
                    improved = True

                self._last_best_score = best_score
                self._last_best_solution = best_solution
                self._last_best_graph = best_graph
                break

        self._last_best_score = best_score
        self._last_best_solution = best_solution
        self._last_best_graph = best_graph
        return improved

    # -------------------------------------------------------------------------
    # Serial backward refinement
    # -------------------------------------------------------------------------

    def _run_backward_from_forward_cf(
        self,
        G_orig: GraphInstance,
        original_label,
        initial_solution: BinarySolution,
        initial_score: float,
    ) -> Tuple[BinarySolution, float, bool]:
        current_solution = np.asarray(initial_solution, dtype=np.int8).copy()
        current_score = float(initial_score)
        improved = False

        explorer_epsilon = _anneal_parameter(
            float(self.explorer_epsilon0),
            float(self.explorer_epsilonfinal),
            self.current_iteration,
            int(self.num_iterations),
            float(self.explorer_epsilon_decay_rate),
        )
        queue = self._build_backward_removal_queue(current_solution, explorer_epsilon)
        k = max(1, int(self.backward_initial_batch_size))
        growth = max(1.0, float(self.backward_batch_growth))
        shrink = min(max(float(self.backward_batch_shrink), 0.0), 0.999999)

        while len(queue) > 0 and not self._oracle_budget_reached():
            batch_size = min(k, len(queue), int(np.sum(current_solution)))
            if batch_size <= 0:
                break

            batch = []
            for _ in range(batch_size):
                dim = queue.pop()
                if dim is None:
                    break
                if current_solution[dim] == 1:
                    batch.append(dim)

            if not batch:
                continue

            candidate = current_solution.copy()
            candidate[batch] = 0

            score, changed = self._evaluate_solution(G_orig, original_label, candidate)
            if score is None:
                break

            if changed:
                current_solution = candidate
                current_score = float(score)
                improved = True
                k = max(k + 1, int(math.ceil(k * growth)))
            else:
                k = max(1, int(math.floor(k * shrink)))
                if len(batch) > 1:
                    queue.reinsert_batch(batch)

        return current_solution, current_score, improved

    def _build_backward_removal_queue(
        self,
        solution: BinarySolution,
        explorer_epsilon: float,
    ) -> _BackwardRemovalQueue:
        active_dims = np.flatnonzero(solution == 1).astype(np.int64)

        if len(active_dims) == 0:
            return _BackwardRemovalQueue(explorer_epsilon=explorer_epsilon, rng=self.rng)

        keep_masses = self._compute_backward_keep_masses(solution, active_dims)
        dim_priorities = [
            (int(dim), max(float(keep_mass), PHEROMONE_MASS_SMOOTHING))
            for dim, keep_mass in zip(active_dims, keep_masses)
        ]

        return _BackwardRemovalQueue(
            dim_priorities,
            explorer_epsilon=explorer_epsilon,
            rng=self.rng,
        )

    def _compute_backward_keep_masses(
        self,
        solution: BinarySolution,
        active_dims: np.ndarray,
    ) -> np.ndarray:
        if not self.pheromone_deposits:
            return np.ones(len(active_dims), dtype=np.float64)

        centers = np.vstack(
            [np.asarray(deposit["center"], dtype=np.int8) for deposit in self.pheromone_deposits]
        )
        powers = np.asarray(
            [float(deposit["power"]) for deposit in self.pheromone_deposits],
            dtype=np.float64,
        )

        valid = powers > 0.0
        if not np.any(valid):
            return np.ones(len(active_dims), dtype=np.float64)

        valid_centers = centers[valid]
        valid_powers = powers[valid]
        valid_hdist = np.sum(valid_centers != solution, axis=1).astype(np.float64)
        deposit_weights = self._compute_deposit_weights(valid_powers, valid_hdist)

        candidate_bits = valid_centers[:, active_dims]
        keep_masses = np.dot(candidate_bits.T.astype(np.float64), deposit_weights)

        if not np.all(np.isfinite(keep_masses)) or float(np.sum(keep_masses)) <= 0.0:
            return np.ones(len(active_dims), dtype=np.float64)

        return keep_masses

    # -------------------------------------------------------------------------
    # Oracle cache and solution scoring
    # -------------------------------------------------------------------------

    def _oracle_budget_reached(self) -> bool:
        if self.max_oracle_calls is None:
            return False
        return self.oracle_calls_used >= int(self.max_oracle_calls)

    def _query_oracle(self, G: GraphInstance):
        if self._oracle_budget_reached():
            return None

        pred = self.oracle.predict(G)
        self.oracle_calls_used += 1
        return pred

    def _hash_solution(self, solution: BinarySolution) -> int:
        bits = max(64, int(self.solution_hash_bits))
        digest_size = max(8, (bits + 7) // 8)

        compact = np.ascontiguousarray(solution.astype(np.uint8, copy=False))
        packed = np.packbits(compact)

        hasher = hashlib.blake2b(digest_size=digest_size)
        hasher.update(len(solution).to_bytes(8, byteorder="little", signed=False))
        hasher.update(packed.tobytes())

        value = int.from_bytes(hasher.digest(), byteorder="little", signed=False)
        if digest_size * 8 > bits:
            value &= (1 << bits) - 1
        return value

    def _get_oracle_prediction_for_solution(
        self,
        G_orig: GraphInstance,
        original_label,
        solution: BinarySolution,
    ):
        if not np.any(solution):
            return original_label

        solution_hash = self._hash_solution(solution)
        if solution_hash in self.solution_prediction_cache:
            self.cache_hits += 1
            return self.solution_prediction_cache[solution_hash]

        if self._oracle_budget_reached():
            return None

        graph_candidate = self._apply_binary_solution(G_orig, solution)
        pred = self._query_oracle(graph_candidate)
        if pred is None:
            return None

        self.solution_prediction_cache[solution_hash] = pred
        return pred

    def _evaluate_solution(
        self,
        G_orig: GraphInstance,
        original_label,
        solution: BinarySolution,
    ) -> Tuple[Optional[float], bool]:
        pred = self._get_oracle_prediction_for_solution(G_orig, original_label, solution)
        if pred is None:
            return None, False

        if pred == original_label:
            return 0.0, False

        edit_distance = int(np.sum(solution))
        m = max(1, len(solution))
        score = (1.0 - edit_distance / (m + 1)) ** self.score_gamma
        return max(score, 1e-6), True

    # -------------------------------------------------------------------------
    # Pheromone updates
    # -------------------------------------------------------------------------

    def _evaporate_pheromones(self):
        if not self.pheromone_deposits:
            return

        for deposit in self.pheromone_deposits:
            deposit["power"] *= (1.0 - self.rho)

        self.pheromone_deposits.sort(key=lambda x: x["power"], reverse=True)
        self.pheromone_deposits = self.pheromone_deposits[: self.max_deposits]

    def _deposit_solution(self, solution: BinarySolution, score: float):
        if score <= 0.0:
            return

        self.pheromone_deposits.append(
            {
                "center": solution.copy(),
                "power": self.deposit_scale * float(score),
                "edit_distance": int(np.sum(solution)),
            }
        )

        if len(self.pheromone_deposits) > self.max_deposits:
            self.pheromone_deposits.sort(key=lambda x: x["power"], reverse=True)
            self.pheromone_deposits = self.pheromone_deposits[: self.max_deposits]

    def _compute_deposit_weights(
        self,
        valid_powers: np.ndarray,
        valid_hdist: np.ndarray,
    ) -> np.ndarray:
        powered_pheromone = np.maximum(valid_powers, 0.0) ** float(self.alpha_pheromone)

        if float(self.beta_compatibility) <= 0.0:
            compatibility = np.ones_like(valid_powers, dtype=np.float64)
        else:
            sigmas = np.maximum(float(self.alpha_scale) * valid_powers, float(self.min_sigma))
            compatibility = np.exp(
                -float(self.beta_compatibility) * valid_hdist / (2.0 * sigmas * sigmas)
            )

        return powered_pheromone * compatibility

    # -------------------------------------------------------------------------
    # Graph modifications
    # -------------------------------------------------------------------------

    def _generate_modifications(self, G: GraphInstance) -> List[Modification]:
        data = G.data
        num_nodes = len(data)
        modifications: List[Modification] = []

        if G.directed:
            for u in range(num_nodes):
                for v in range(num_nodes):
                    if u == v:
                        continue
                    if data[u, v] > 0.5:
                        modifications.append(("remove_edge", (u, v)))
                    else:
                        modifications.append(("add_edge", (u, v)))
        else:
            for u in range(num_nodes):
                for v in range(u + 1, num_nodes):
                    if data[u, v] > 0.5:
                        modifications.append(("remove_edge", (u, v)))
                    else:
                        modifications.append(("add_edge", (u, v)))

        self.rng.shuffle(modifications)
        return modifications

    def _split_modification_dims(
        self,
        modifications: List[Modification],
    ) -> Tuple[List[int], List[int]]:
        add_dims: List[int] = []
        remove_dims: List[int] = []

        for idx, (mod_type, _) in enumerate(modifications):
            if mod_type == "add_edge":
                add_dims.append(idx)
            elif mod_type == "remove_edge":
                remove_dims.append(idx)
            else:
                raise ValueError(f"Unknown modification type: {mod_type}")

        return add_dims, remove_dims

    def _apply_binary_solution(self, G: GraphInstance, solution: BinarySolution) -> GraphInstance:
        data_new = self._apply_solution_to_matrix(G, solution)
        return GraphInstance(
            id=G.id,
            label=G.label,
            data=data_new,
            node_features=G.node_features,
            directed=G.directed,
        )

    def _apply_solution_to_matrix(self, G: GraphInstance, solution: BinarySolution) -> np.ndarray:
        data_new = np.copy(G.data)

        for idx, bit in enumerate(solution):
            if bit != 1:
                continue

            mod_type, (u, v) = self.modification_space[idx]
            if mod_type == "remove_edge":
                data_new[u, v] = 0
                if not G.directed:
                    data_new[v, u] = 0
            elif mod_type == "add_edge":
                data_new[u, v] = 1
                if not G.directed:
                    data_new[v, u] = 1
            else:
                raise ValueError(f"Unknown modification type: {mod_type}")

        return data_new
