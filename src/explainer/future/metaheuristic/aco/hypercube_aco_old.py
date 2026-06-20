import hashlib
import math
import random
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta


Modification = Tuple[str, Tuple[int, int]]
BinarySolution = np.ndarray
PheromoneDeposit = Dict[str, object]

# A checkpoint is a candidate solution generated at some intermediate point of
# an ant trajectory. Oracle calls are performed only on checkpoints, and always
# outside the worker threads.
#   solution: binary vector in {0, 1}^m
#   edit_distance: number of active edits in the solution
#   edge_delta: number of edges relative to the original graph
#               (+ means denser, - means sparser)
AntCheckpoint = Tuple[BinarySolution, int, int]



# -----------------------------------------------------------------------------
# ProcessPool worker support
# -----------------------------------------------------------------------------
# Workers only need static construction data: the modification space, edge deltas, 
# sampling parameters, and a per-iteration read-only pheromone snapshot. 
# Oracle calls remain in the main process.
_PROCESS_WORKER_CONTEXT = None


def _init_process_worker(worker_context: Dict[str, object]):
    """Stores static ant-construction data once per worker process."""
    global _PROCESS_WORKER_CONTEXT
    _PROCESS_WORKER_CONTEXT = worker_context


def _process_pool_construct_ant(task: Tuple[List[PheromoneDeposit], float, int, int, str]):
    """
    ProcessPool entry point.

    The static context is provided by _init_process_worker. The task contains
    only per-iteration/per-ant data, keeping IPC overhead lower than sending the
    full modification space for every ant.
    """
    if _PROCESS_WORKER_CONTEXT is None:
        raise RuntimeError("Process worker context was not initialized")

    pheromone_deposits, epsilon, iteration, ant_id, ant_mode = task
    return _construct_ant_from_context(
        ctx=_PROCESS_WORKER_CONTEXT,
        pheromone_deposits=pheromone_deposits,
        epsilon=epsilon,
        iteration=iteration,
        ant_id=ant_id,
        ant_mode=ant_mode,
    )


def _construct_ant_from_context(
    ctx: Dict[str, object],
    pheromone_deposits: List[PheromoneDeposit],
    epsilon: float,
    iteration: int,
    ant_id: int,
    ant_mode: str,
) -> List[AntCheckpoint]:
    """Constructs one ant output without accessing the explainer instance."""
    seed = 1000003 * iteration + 9176 * ant_id + 12345
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed % (2**32 - 1))

    if ant_mode == "exploit":
        return _worker_construct_exploiter_solution(
            ctx=ctx,
            pheromone_deposits=pheromone_deposits,
            rng=rng,
            np_rng=np_rng,
        )

    explorer_epsilon = anneal_parameter(ctx["explorer_epsilon"], 0, iteration, ctx["num_iterations"], 10)

    ctx.get("explorer_epsilon")
    effective_epsilon = float(explorer_epsilon) if explorer_epsilon is not None else epsilon

    return _worker_construct_explorer_ant_path(
        ctx=ctx,
        pheromone_deposits=pheromone_deposits,
        epsilon=effective_epsilon,
        rng=rng,
        np_rng=np_rng,
    )


def _worker_prepare_pheromone_arrays(
    pheromone_deposits: List[PheromoneDeposit],
    m: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not pheromone_deposits:
        return (
            np.empty((0, m), dtype=np.int8),
            np.empty(0, dtype=np.float64),
        )

    centers = np.vstack(
        [np.asarray(deposit["center"], dtype=np.int8) for deposit in pheromone_deposits]
    )
    powers = np.asarray(
        [float(deposit["power"]) for deposit in pheromone_deposits],
        dtype=np.float64,
    )

    return centers, powers


def _worker_compute_deposit_weights(
    ctx: Dict[str, object],
    valid_powers: np.ndarray,
    valid_hdist: np.ndarray,
) -> np.ndarray:
    """
    Shared deposit-weight rule for both explorer and exploiter ants.

    alpha_pheromone controls the strength of pheromone power.
    beta_compatibility controls how strongly partial-solution mismatch reduces
    the influence of a deposit.
    """
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
    """
    Computes pheromone masses for explorer candidate edits.

    Explorer ants do not decide bit 0 vs bit 1. They choose the next edit to
    activate in the trajectory. A candidate receives mass from compatible
    deposits whose center has that candidate bit active. Deposit contribution is
    weighted by alpha_pheromone and beta_compatibility through
    _worker_compute_deposit_weights(...).
    """
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

    if not np.all(np.isfinite(masses)):
        return np.ones(len(candidates), dtype=np.float64)

    if float(np.sum(masses)) <= 0.0:
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

    weights = np.maximum(masses, float(ctx["pheromone_mass_smoothing"]))
    total = float(np.sum(weights))

    if not np.isfinite(total) or total <= 0.0:
        return rng.choice(candidates)

    probabilities = weights / total
    candidate_array = np.asarray(candidates, dtype=np.int64)
    return int(np_rng.choice(candidate_array, p=probabilities))


def _worker_sample_move_type(
    ctx: Dict[str, object],
    current_edge_delta: int,
    rng: random.Random,
) -> str:
    policy = str(ctx["move_type_policy"]).lower()

    if policy == "sigmoid_to_zero":
        lam = float(ctx["density_balance_lambda"])
        x = lam * float(current_edge_delta)

        if x >= 0:
            exp_neg = math.exp(-x)
            p_add = exp_neg / (1.0 + exp_neg)
        else:
            p_add = 1.0 / (1.0 + math.exp(x))
    else:
        p_add = 0.5

    min_p = min(max(float(ctx["min_move_type_probability"]), 0.0), 0.5)
    p_add = min(max(p_add, min_p), 1.0 - min_p)

    return "add_edge" if rng.random() < p_add else "remove_edge"


def _worker_sample_next_dimension(
    ctx: Dict[str, object],
    available_add_dims: Set[int],
    available_remove_dims: Set[int],
    current_edge_delta: int,
    centers: np.ndarray,
    powers: np.ndarray,
    hdist: np.ndarray,
    epsilon: float,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> Optional[int]:
    if not available_add_dims and not available_remove_dims:
        return None

    if not bool(ctx["use_add_remove_balancing"]):
        candidates = list(available_add_dims | available_remove_dims)
        return _worker_sample_dimension_from_candidates(
            ctx=ctx,
            candidates=candidates,
            centers=centers,
            powers=powers,
            hdist=hdist,
            epsilon=epsilon,
            rng=rng,
            np_rng=np_rng,
        )

    move_type = _worker_sample_move_type(ctx, current_edge_delta, rng)

    if move_type == "add_edge":
        primary = available_add_dims
        fallback = available_remove_dims
    else:
        primary = available_remove_dims
        fallback = available_add_dims

    candidates_set = primary if primary else fallback
    if not candidates_set:
        return None

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


def _worker_sample_ant_depth(
    ctx: Dict[str, object],
    m: int,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> int:
    hard_max = m
    max_modifications = ctx.get("max_modifications")
    max_path_length = ctx.get("max_path_length")

    if max_modifications is not None:
        hard_max = min(hard_max, int(max_modifications))
    if max_path_length is not None:
        hard_max = min(hard_max, int(max_path_length))

    hard_min = max(1, min(int(ctx["min_path_length"]), hard_max))

    if hard_max <= 0:
        return 0

    strategy = str(ctx["depth_strategy"]).lower()

    if strategy == "geometric":
        p = min(max(float(ctx["geometric_depth_p"]), 1e-9), 1.0)
        sampled = int(np_rng.geometric(p))
        return max(hard_min, min(sampled, hard_max))

    r = rng.random()
    p_shallow = max(0.0, float(ctx["prob_shallow"]))
    p_medium = max(0.0, float(ctx["prob_medium"]))
    p_deep = max(0.0, float(ctx["prob_deep"]))
    total = p_shallow + p_medium + p_deep

    if total <= 0.0:
        p_shallow, p_medium, p_deep = 0.50, 0.30, 0.20
        total = 1.0

    p_shallow /= total
    p_medium /= total

    if r < p_shallow:
        upper = min(int(ctx["shallow_depth"]), hard_max)
        lower = hard_min
    elif r < p_shallow + p_medium:
        lower = max(hard_min, min(int(ctx["shallow_depth"]) + 1, hard_max))
        upper = min(int(ctx["medium_depth"]), hard_max)
    else:
        lower = max(hard_min, min(int(ctx["medium_depth"]) + 1, hard_max))
        deep_depth = ctx.get("deep_depth")
        upper = hard_max if deep_depth is None else min(int(deep_depth), hard_max)

    if upper < lower:
        lower, upper = hard_min, hard_max

    return rng.randint(lower, upper)


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

    p = _worker_get_checkpoint_query_probability(ctx, edit_distance)
    return rng.random() < p


def _worker_construct_explorer_ant_path(
    ctx: Dict[str, object],
    pheromone_deposits: List[PheromoneDeposit],
    epsilon: float,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> List[AntCheckpoint]:
    m = len(ctx["modification_space"])
    solution = np.zeros(m, dtype=np.int8)
    checkpoints: List[AntCheckpoint] = []

    if m == 0:
        return checkpoints

    max_steps = _worker_sample_ant_depth(ctx=ctx, m=m, rng=rng, np_rng=np_rng)
    if max_steps <= 0:
        return checkpoints

    centers, powers = _worker_prepare_pheromone_arrays(pheromone_deposits, m)
    num_deposits = len(powers)
    hdist = np.zeros(num_deposits, dtype=np.float64)

    available_add_dims: Set[int] = set(ctx["add_dims"])
    available_remove_dims: Set[int] = set(ctx["remove_dims"])
    modification_edge_deltas = ctx["modification_edge_deltas"]

    current_edge_delta = 0

    for step in range(max_steps):
        dim = _worker_sample_next_dimension(
            ctx=ctx,
            available_add_dims=available_add_dims,
            available_remove_dims=available_remove_dims,
            current_edge_delta=current_edge_delta,
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
        current_edge_delta += int(modification_edge_deltas[dim])

        if num_deposits > 0:
            # Explorer compatibility is computed over the active path prefix.
            # Each trajectory step fixes the selected dimension to 1, so deposits
            # that do not contain this edit become less compatible.
            hdist += (centers[:, dim] != 1)

        edit_distance = step + 1
        if _worker_should_query_checkpoint(
            ctx=ctx,
            edit_distance=edit_distance,
            max_steps=max_steps,
            rng=rng,
        ):
            checkpoints.append((solution.copy(), edit_distance, current_edge_delta))

    return checkpoints


def _worker_compute_bit_masses(
    ctx: Dict[str, object],
    dim: int,
    centers: np.ndarray,
    powers: np.ndarray,
    hdist: np.ndarray,
) -> Tuple[float, float]:
    if len(powers) == 0:
        return 1.0, 1.0

    valid = powers > 0.0
    if not np.any(valid):
        return 1.0, 1.0

    valid_powers = powers[valid]
    valid_hdist = hdist[valid]
    valid_centers_dim = centers[valid, dim]

    deposit_weights = _worker_compute_deposit_weights(
        ctx=ctx,
        valid_powers=valid_powers,
        valid_hdist=valid_hdist,
    )

    mass1 = float(np.sum(deposit_weights[valid_centers_dim == 1]))
    mass0 = float(np.sum(deposit_weights[valid_centers_dim == 0]))

    if not math.isfinite(mass0) or not math.isfinite(mass1):
        return 1.0, 1.0

    if mass0 <= 0.0 and mass1 <= 0.0:
        return 1.0, 1.0

    return mass0, mass1


def _worker_sample_exploiter_bit(
    ctx: Dict[str, object],
    dim: int,
    centers: np.ndarray,
    powers: np.ndarray,
    hdist: np.ndarray,
    rng: random.Random,
) -> int:
    mass0, mass1 = _worker_compute_bit_masses(
        ctx=ctx,
        dim=dim,
        centers=centers,
        powers=powers,
        hdist=hdist,
    )

    score0 = max(mass0, float(ctx["pheromone_mass_smoothing"]))
    score1 = max(mass1, float(ctx["pheromone_mass_smoothing"]))
    denom = score0 + score1

    if not math.isfinite(denom) or denom <= 0.0:
        # Exploiters should normally only run once deposits exist. If numerical
        # degeneracy still happens, use a neutral fallback instead of an undefined
        # random base probability.
        return 1 if rng.random() < 0.5 else 0

    p_one = score1 / denom
    return 1 if rng.random() < p_one else 0


def _worker_construct_exploiter_solution(
    ctx: Dict[str, object],
    pheromone_deposits: List[PheromoneDeposit],
    rng: random.Random,
    np_rng: np.random.Generator,
) -> List[AntCheckpoint]:
    m = len(ctx["modification_space"])
    solution = np.zeros(m, dtype=np.int8)

    if m == 0:
        return []

    centers, powers = _worker_prepare_pheromone_arrays(pheromone_deposits, m)
    num_deposits = len(powers)
    hdist = np.zeros(num_deposits, dtype=np.float64)

    order = np_rng.permutation(m)
    active_count = 0
    current_edge_delta = 0
    max_modifications = ctx.get("max_modifications")
    modification_edge_deltas = ctx["modification_edge_deltas"]

    for dim in order:
        dim = int(dim)

        if max_modifications is not None and active_count >= int(max_modifications):
            bit = 0
        else:
            bit = _worker_sample_exploiter_bit(
                ctx=ctx,
                dim=dim,
                centers=centers,
                powers=powers,
                hdist=hdist,
                rng=rng,
            )

        solution[dim] = bit

        if bit == 1:
            active_count += 1
            current_edge_delta += int(modification_edge_deltas[dim])

        if num_deposits > 0:
            hdist += (centers[:, dim] != bit)

    return [(solution, active_count, current_edge_delta)]


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def anneal_parameter(initial_value, final_value, current_iter, max_iter, epsilon_decay_rate):
    if max_iter <= 1:
        return final_value

    progress = current_iter / max(1, max_iter - 1)

    return final_value + (initial_value - final_value) * math.exp(
        -epsilon_decay_rate * progress
    )


class HypercubeACOExplainer(Explainer, metaclass=ExplainerTransformMeta):
    """
    Trajectory-based Binary Hypercube ACO explainer for graph counterfactuals.

    Each dimension of the hypercube still corresponds to one possible edge edit
    relative to the original graph:
      - x_j = 1 -> apply modification j
      - x_j = 0 -> do not apply modification j

    The main difference from the previous bit-by-bit full-vector construction is
    that an ant now builds an incremental trajectory:

        G_0 -> G_1 -> G_2 -> ... -> G_k

    where each step activates exactly one new modification. The algorithm does
    not wait until the final vector to evaluate the candidate. Instead, each ant
    produces intermediate checkpoints, and the main thread queries the oracle on
    those checkpoints in trajectory order.

    This matters because a trajectory can cross the decision boundary and later
    move back to the original class. Querying only the final vector can therefore
    miss valid counterfactuals.

    Oracle calls are centralized through _get_oracle_prediction_for_solution().
    This method hashes every binary solution, checks a prediction cache, enforces
    a hard global oracle-call limit, and records whether the solution changes the
    original class or not. No other method should call self.oracle.predict(...)
    for candidate counterfactual solutions.

    Pheromones are deposited only by confirmed counterfactual solutions.
    Non-counterfactual candidates do not deposit pheromones.

    The population is split into two ant types:
      - explorer ants: trajectory-based, add/remove-aware, checkpointed;
      - exploiter ants: classic bit-by-bit hypercube construction guided by
        pheromone deposits.

    Ant construction may run concurrently, but oracle calls, best-solution
    updates, and pheromone deposits are serialized in the main thread. This keeps
    oracle interaction deterministic and easy to parallelize later if needed.
    """

    def init(self):
        params = self.local_config.get("parameters", {})

        self.num_ants = params.get("num_ants", 12)
        self.num_iterations = params.get("num_iterations", 30)
        self.num_workers = params.get("num_workers", 4)

        # ------------------------------------------------------------------
        # Exploration / exploitation
        # ------------------------------------------------------------------
        # The global epsilon now controls the population split:
        #   high epsilon -> mostly explorer ants;
        #   low epsilon  -> mostly exploiter ants.
        #
        # It is no longer the activation probability of exploiter bits.
        # Explorer ants still have their own local sampling epsilon below.
        self.epsilon0 = params.get("epsilon0", 0.9)
        self.epsilonfinal = params.get("epsilonfinal", 0.05)
        self.epsilon = self.epsilon0
        self.epsilon_decay_rate = params.get("epsilon_decay_rate", 5.0)

        # Local epsilon for explorer ants only. It controls whether an explorer
        # samples a concrete edge edit uniformly inside the chosen add/remove
        # group or according to pheromone masses. The default is 0.0 because the
        # global epsilon already controls how many explorer ants are created.
        self.explorer_epsilon = params.get("explorer_epsilon", 0.5)

        # ------------------------------------------------------------------
        # Pheromone weighting controls
        # ------------------------------------------------------------------
        # ACO-style pheromone exponent. This boosts or attenuates deposit power.
        # New preferred name: alpha_pheromone. Older aliases are accepted to avoid
        # breaking existing experiment configs.
        self.alpha_pheromone = params.get(
            "alpha_pheromone",
            params.get("pheromone_power_alpha", 1.0),
        )

        # Compatibility exponent. Larger values make deposits lose influence
        # faster when their center disagrees with the partial solution/path built
        # so far. New preferred name: beta_compatibility.
        self.beta_compatibility = params.get(
            "beta_compatibility",
            params.get("deposit_compatibility_beta", 1.0),
        )

        # Small smoothing term used when a candidate edit or bit has zero
        # pheromone mass. This prevents hard exclusion of unseen modifications.
        self.pheromone_mass_smoothing = params.get(
            "pheromone_mass_smoothing", 1e-12
        )

        # ------------------------------------------------------------------
        # Trajectory depth and oracle checkpoints
        # ------------------------------------------------------------------
        # Each ant samples a maximum path length. This is what lets different
        # ants explore different Hamming distances from the original instance.
        #
        # Supported strategies:
        #   - "mixed":      shallow / medium / deep buckets.
        #   - "geometric":  favor shallow paths but occasionally go deep.
        self.depth_strategy = params.get(
            "depth_strategy", params.get("trajectory_depth_strategy", "mixed")
        )
        self.min_path_length = params.get("min_path_length", 1)
        self.max_path_length = params.get(
            "max_path_length", params.get("max_modifications", None)
        )

        # Bucket parameters for depth_strategy == "mixed".
        self.shallow_depth = params.get("shallow_depth", 20)
        self.medium_depth = params.get("medium_depth", 100)
        self.deep_depth = params.get("deep_depth", self.max_path_length)
        self.prob_shallow = params.get("prob_shallow", 0.65)
        self.prob_medium = params.get("prob_medium", 0.25)
        self.prob_deep = params.get("prob_deep", 0.10)

        # Parameter for depth_strategy == "geometric".
        self.geometric_depth_p = params.get("geometric_depth_p", 0.08)

        # Oracle checkpoint probability schedule.
        #
        # The base probability is used near the original instance. As edit distance
        # grows, the probability can decay so long trajectories do not consume hundreds
        # or thousands of oracle calls.
        #
        # p0 = oracle_query_probability
        # Supported policies:
        #   - "constant": same probability at all distances.
        #   - "inverse":  p(d) = max(p_min, p0 / (1 + d / scale)).
        #   - "exp":      p(d) = max(p_min, p0 * exp(-d / scale)).
        self.oracle_query_probability = params.get("oracle_query_probability", 0.20)

        self.oracle_query_probability_schedule = params.get(
            "oracle_query_probability_schedule", "inverse"
        )
        self.min_oracle_query_probability = params.get(
            "min_oracle_query_probability", 0.01
        )
        self.oracle_query_decay_scale = params.get(
            "oracle_query_decay_scale", 50.0
        )

        # ------------------------------------------------------------------
        # Oracle-call budget and solution cache
        # ------------------------------------------------------------------
        # Maximum number of actual oracle calls allowed during one explain() run.
        # None means unlimited. This budget includes the initial prediction on
        # the original instance.
        self.max_oracle_calls = params.get(
            "max_oracle_calls", params.get("oracle_call_hard_limit", None)
        )

        # Number of bits used when hashing binary solutions. The value is forced
        # to be at least 64. 128 is the default to make accidental collisions very
        # unlikely while still storing hashes as Python integers.
        self.solution_hash_bits = max(64, int(params.get("solution_hash_bits", 128)))

        # Runtime fields initialized in _reset_state().
        self.oracle_calls_used = 0
        self.solution_prediction_cache: Dict[int, object] = {}

        # ------------------------------------------------------------------
        # Add/remove movement policy
        # ------------------------------------------------------------------
        # When add/remove balancing is enabled, the algorithm first selects 
        # the move type, then uses pheromones only to select the concrete 
        # edge edit inside that type.
        #
        # This avoids the cardinality bias where sparse graphs have many more
        # possible add_edge edits than remove_edge edits.
        self.use_add_remove_balancing = params.get("use_add_remove_balancing", True)

        # Supported policies:
        #   - "balanced":        P(add_edge) = 0.5 regardless of edge_delta.
        #   - "sigmoid_to_zero": softly biases edge_delta back toward 0:
        #                         P(add) < 0.5 when edge_delta > 0,
        #                         P(add) > 0.5 when edge_delta < 0.
        self.move_type_policy = params.get("move_type_policy", "balanced")
        self.density_balance_lambda = params.get("density_balance_lambda", 0.0)

        # Clamp P(add_edge) so both directions keep some chance of being sampled.
        self.min_move_type_probability = params.get("min_move_type_probability", 0.05)

        # ------------------------------------------------------------------
        # Pheromone dynamics
        # ------------------------------------------------------------------
        self.rho = params.get("rho", 0.25)
        self.deposit_scale = params.get("deposit_scale", 2.0)
        self.p_min = params.get("p_min", 1e-4)
        self.score_gamma = params.get("score_gamma", 2.0)

        # Gaussian-like compatibility over the active part of the partial path.
        # hdist is measured only over edits already selected by the ant, because
        # unselected edits are not fixed to 0 until the trajectory stops.
        self.alpha_scale = params.get("alpha_scale", 1.0)
        self.min_sigma = params.get("min_sigma", 1e-3)

        # Archive management.
        self.max_deposits = params.get("max_deposits", 256)

        # Search controls.
        self.max_modifications = params.get("max_modifications", None)
        self.early_stop_patience = params.get("early_stop_patience", 8)

        self.modification_space: List[Modification] = []
        self.modification_edge_deltas: np.ndarray = np.empty(0, dtype=np.int8)
        self.add_dims: List[int] = []
        self.remove_dims: List[int] = []

        self.pheromone_deposits: List[PheromoneDeposit] = []
        self.current_iteration = 0

    def real_fit(self):
        pass

    def explain(self, instance: GraphInstance):
        best_graph, best_solution, _ = self._run_hypercube_aco(instance)
        return best_graph if best_graph is not None else instance

    # -------------------------------------------------------------------------
    # Main algorithm
    # -------------------------------------------------------------------------

    def _run_hypercube_aco(
        self, G_orig: GraphInstance
    ) -> Tuple[Optional[GraphInstance], Optional[BinarySolution], float]:
        self._reset_state(G_orig)

        original_label = self._query_oracle(G_orig)
        if original_label is None:
            return None, None, -1.0

        best_solution: Optional[BinarySolution] = None
        best_graph: Optional[GraphInstance] = None
        best_score = -1.0
        no_improve_counter = 0

        use_parallel = (
            self.num_workers is not None
            and self.num_workers > 1
            and self.num_ants > 1
        )

        executor = None
        if use_parallel:
            # ProcessPoolExecutor requires all worker-callable code to be top-level
            # and pickleable. The worker initializer receives only static
            # construction data; per-iteration pheromone snapshots are passed as
            # task arguments. Oracle calls stay serialized in the main process.
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
                
                self.epsilon = anneal_parameter(
                    self.epsilon0, self.epsilonfinal, iteration, self.num_iterations, self.epsilon_decay_rate
                )

                self._evaporate_pheromones()

                # Read-only snapshot used by all ants in this iteration.
                pheromone_snapshot = self._make_pheromone_snapshot()

                # Parallel phase: construct ant trajectories and checkpoints.
                # No oracle calls are executed inside workers.
                iteration_ant_paths = self._run_iteration_ants_parallel(
                    executor=executor,
                    pheromone_snapshot=pheromone_snapshot,
                    epsilon=self.epsilon,
                )

                # Serial phase: evaluate each ant path in trajectory order. When
                # a path reaches the first confirmed CF, deposit pheromone on that
                # first CF and stop evaluating later checkpoints from the same ant.
                improved = False

                for checkpoints in iteration_ant_paths:
                    if self._oracle_budget_reached():
                        break

                    for solution, edit_distance, edge_delta in checkpoints:
                        if self._oracle_budget_reached():
                            break

                        score, changed = self._evaluate_solution(
                            G_orig=G_orig,
                            original_label=original_label,
                            solution=solution,
                        )

                        if score is None:
                            # Hard oracle-call limit reached before this solution
                            # could be evaluated.
                            break

                        if not changed:
                            continue

                        if score > best_score:
                            best_score = score
                            best_solution = solution.copy()
                            best_graph = self._apply_binary_solution(G_orig, solution)
                            improved = True

                        # Only confirmed counterfactuals deposit pheromones.
                        self._deposit_solution(
                            solution=solution,
                            score=score,
                            edit_distance=edit_distance,
                            edge_delta=edge_delta,
                        )

                        # This emulates an online trajectory search: once this ant
                        # found a CF, it would have stopped there.
                        break

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
        self.modification_edge_deltas = self._generate_modification_edge_deltas(
            self.modification_space
        )
        self.add_dims, self.remove_dims = self._split_modification_dims(
            self.modification_space
        )
        self.pheromone_deposits = []
        self.current_iteration = 0

        self.oracle_calls_used = 0
        self.solution_prediction_cache = {}

    def _make_pheromone_snapshot(self) -> List[PheromoneDeposit]:
        return [
            {
                "center": deposit["center"].copy(),
                "power": float(deposit["power"]),
                "score": float(deposit.get("score", 0.0)),
                "edit_distance": int(deposit.get("edit_distance", 0)),
                "edge_delta": int(deposit.get("edge_delta", 0)),
            }
            for deposit in self.pheromone_deposits
        ]

    def _make_worker_context(self) -> Dict[str, object]:
        """
        Builds the static, pickleable context used by ProcessPool workers.

        This context must not contain self, the oracle, GraphInstance objects, or
        any other repository object that may be expensive or impossible to pickle.
        Workers only construct binary solutions/checkpoints; graph materialization
        and oracle calls remain in the main process.
        """
        return {
            "modification_space": self.modification_space,
            "modification_edge_deltas": np.asarray(
                self.modification_edge_deltas, dtype=np.int8
            ),
            "add_dims": list(self.add_dims),
            "remove_dims": list(self.remove_dims),
            "max_modifications": self.max_modifications,
            "max_path_length": self.max_path_length,
            "min_path_length": self.min_path_length,
            "depth_strategy": self.depth_strategy,
            "shallow_depth": self.shallow_depth,
            "medium_depth": self.medium_depth,
            "deep_depth": self.deep_depth,
            "prob_shallow": self.prob_shallow,
            "prob_medium": self.prob_medium,
            "prob_deep": self.prob_deep,
            "geometric_depth_p": self.geometric_depth_p,
            "oracle_query_probability": self.oracle_query_probability,
            "oracle_query_probability_schedule": self.oracle_query_probability_schedule,
            "min_oracle_query_probability": self.min_oracle_query_probability,
            "oracle_query_decay_scale": self.oracle_query_decay_scale,
            "use_add_remove_balancing": self.use_add_remove_balancing,
            "move_type_policy": self.move_type_policy,
            "density_balance_lambda": self.density_balance_lambda,
            "min_move_type_probability": self.min_move_type_probability,
            "alpha_pheromone": self.alpha_pheromone,
            "beta_compatibility": self.beta_compatibility,
            "num_iterations": self.num_iterations,
            # Compatibility aliases kept for older worker/helper code.,
            "pheromone_mass_smoothing": self.pheromone_mass_smoothing,
            "alpha_scale": self.alpha_scale,
            "min_sigma": self.min_sigma,
            "explorer_epsilon": self.explorer_epsilon,
        }

    # -------------------------------------------------------------------------
    # Parallel ant construction
    # -------------------------------------------------------------------------

    def _run_iteration_ants_parallel(
        self,
        executor: Optional[ProcessPoolExecutor],
        pheromone_snapshot: List[PheromoneDeposit],
        epsilon: float,
    ) -> List[List[AntCheckpoint]]:
        ant_modes = self._get_ant_modes_for_iteration(pheromone_snapshot)

        if executor is None:
            worker_context = self._make_worker_context()
            return [
                _construct_ant_from_context(
                    ctx=worker_context,
                    pheromone_deposits=pheromone_snapshot,
                    epsilon=epsilon,
                    iteration=self.current_iteration,
                    ant_id=ant_id,
                    ant_mode=ant_modes[ant_id],
                )
                for ant_id in range(self.num_ants)
            ]

        tasks = [
            (
                pheromone_snapshot,
                epsilon,
                self.current_iteration,
                ant_id,
                ant_modes[ant_id],
            )
            for ant_id in range(self.num_ants)
        ]

        # executor.map preserves task order, which keeps evaluation deterministic
        # even though construction happens in multiple processes.
        return list(executor.map(_process_pool_construct_ant, tasks))

    def _get_ant_modes_for_iteration(
        self,
        pheromone_deposits: List[PheromoneDeposit],
    ) -> List[str]:
        """
        Splits the population into explorer and exploiter ants.

        If there are no pheromone deposits yet, all ants are explorers because
        exploitation has no information to exploit. Once at least one CF has
        deposited pheromone, a fixed fraction of ants remains exploratory while
        the rest uses the old bit-by-bit pheromone construction.
        """
        if self.num_ants <= 0:
            return []

        if not pheromone_deposits:
            return ["explore" for _ in range(self.num_ants)]

        explorer_fraction = min(max(float(self.epsilon), 0.0), 1.0)
        num_explorers = int(round(self.num_ants * explorer_fraction))

        num_exploiters = self.num_ants - num_explorers
        modes = ["explore"] * num_explorers + ["exploit"] * num_exploiters

        # Deterministic per-iteration shuffle so parallel execution does not
        # make the population assignment depend on scheduling.
        rng = random.Random(99991 * self.current_iteration + 17)
        rng.shuffle(modes)
        return modes

    # -------------------------------------------------------------------------
    # Oracle cache, hard budget, and solution evaluation
    # -------------------------------------------------------------------------

    def _oracle_budget_reached(self) -> bool:
        if self.max_oracle_calls is None:
            return False
        return self.oracle_calls_used >= int(self.max_oracle_calls)

    def _query_oracle(self, G: GraphInstance):
        """
        Centralized low-level oracle call.

        This is the only method that is allowed to increment oracle_calls_used.
        It enforces the hard oracle-call limit. If the budget is exhausted, it
        returns None and does not call self.oracle.predict(...).
        """
        if self._oracle_budget_reached():
            return None

        pred = self.oracle.predict(G)
        self.oracle_calls_used += 1
        return pred

    def _hash_solution(self, solution: BinarySolution) -> int:
        """
        Hashes a binary solution into an integer with at least 64 bits.

        The hash also includes the solution length. This prevents accidental
        collisions between equal byte payloads interpreted at different lengths.
        For example, an empty/short packed representation cannot collide just
        because of missing trailing zeros.

        The result is a Python int. Python integers are arbitrary precision, so
        using 128 bits by default is safe.
        """
        bits = max(64, int(self.solution_hash_bits))
        digest_size = max(8, (bits + 7) // 8)

        compact = np.ascontiguousarray(solution.astype(np.uint8, copy=False))
        packed = np.packbits(compact)

        hasher = hashlib.blake2b(digest_size=digest_size)
        hasher.update(len(solution).to_bytes(8, byteorder="little", signed=False))
        hasher.update(packed.tobytes())

        value = int.from_bytes(hasher.digest(), byteorder="little", signed=False)

        # If solution_hash_bits is not a multiple of 8, mask the extra high bits.
        if digest_size * 8 > bits:
            value &= (1 << bits) - 1

        return value

    def _get_oracle_prediction_for_solution(
        self,
        G_orig: GraphInstance,
        original_label,
        solution: BinarySolution,
    ):
        """
        Returns the oracle prediction for a candidate binary solution.

        This is the single high-level interface for candidate-solution
        prediction. It guarantees:
          - identical binary solutions are not queried twice;
          - known CF and non-CF solution hashes are stored separately;
          - the hard oracle-call limit is respected;
          - graph materialization is skipped when the prediction is cached or
            the oracle budget is already exhausted.

        Returns:
          - prediction object if available;
          - None if the hard oracle-call limit prevents evaluation.
        """
        solution_hash = self._hash_solution(solution)

        if solution_hash in self.solution_prediction_cache:
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
        pred = self._get_oracle_prediction_for_solution(
            G_orig=G_orig,
            original_label=original_label,
            solution=solution,
        )

        if pred is None:
            # This means the hard oracle-call limit was reached before this
            # solution could be evaluated.
            return None, False

        changed = pred != original_label
        if not changed:
            return 0.0, False

        edit_distance = int(np.sum(solution))
        m = max(1, len(solution))
        edit_ratio = edit_distance / (m + 1)

        # High score for compact counterfactuals, with visible differences even
        # when the modification space is large.
        score = (1.0 - edit_ratio) ** self.score_gamma
        score = max(score, 1e-6)

        return score, True

    # -------------------------------------------------------------------------
    # Pheromone updates
    # -------------------------------------------------------------------------

    def _evaporate_pheromones(self):
        if not self.pheromone_deposits:
            return

        kept = []
        for deposit in self.pheromone_deposits:
            deposit["power"] *= (1.0 - self.rho)
            if deposit["power"] >= self.p_min:
                kept.append(deposit)

        self.pheromone_deposits = kept

    def _deposit_solution(
        self,
        solution: BinarySolution,
        score: float,
        edit_distance: int,
        edge_delta: int,
    ):
        if score <= 0.0:
            return

        delta_power = self.deposit_scale * score

        self.pheromone_deposits.append(
            {
                "center": solution.copy(),
                "power": delta_power,
                "score": score,
                "edit_distance": int(edit_distance),
                "edge_delta": int(edge_delta),
            }
        )

        # Keep the archive bounded.
        if len(self.pheromone_deposits) > self.max_deposits:
            self.pheromone_deposits.sort(
                key=lambda x: (x["power"], x.get("score", 0.0)), reverse=True
            )
            self.pheromone_deposits = self.pheromone_deposits[: self.max_deposits]

    # -------------------------------------------------------------------------
    # Graph modifications
    # -------------------------------------------------------------------------

    def _generate_modifications(self, G: GraphInstance) -> List[Modification]:
        data = G.data
        num_nodes = len(data)
        modifications: List[Modification] = []

        if G.is_directed:
            for u in range(num_nodes):
                for v in range(num_nodes):
                    if u == v:
                        continue
                    if data[u][v] > 0.5:
                        modifications.append(("remove_edge", (u, v)))
                    else:
                        modifications.append(("add_edge", (u, v)))
        else:
            for u in range(num_nodes):
                for v in range(u + 1, num_nodes):
                    if data[u][v] > 0.5:
                        modifications.append(("remove_edge", (u, v)))
                    else:
                        modifications.append(("add_edge", (u, v)))

        random.shuffle(modifications)
        return modifications

    def _generate_modification_edge_deltas(
        self,
        modifications: List[Modification],
    ) -> np.ndarray:
        """
        Converts each modification into its edge-count effect.

        add_edge    -> +1 edge relative to the original graph.
        remove_edge -> -1 edge relative to the original graph.

        For undirected graphs, each modification still represents one logical
        edge edit because the modification space only includes pairs u < v.
        """
        deltas = np.zeros(len(modifications), dtype=np.int8)

        for idx, (mod_type, _) in enumerate(modifications):
            if mod_type == "add_edge":
                deltas[idx] = 1
            elif mod_type == "remove_edge":
                deltas[idx] = -1
            else:
                raise ValueError(f"Unknown modification type: {mod_type}")

        return deltas

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

    def _apply_binary_solution(
        self, G: GraphInstance, solution: BinarySolution
    ) -> GraphInstance:
        data_new = self._apply_solution_to_matrix(G, solution)

        return GraphInstance(
            id=G.id,
            label=G.label,
            data=data_new,
            node_features=G.node_features,
        )

    def _apply_solution_to_matrix(
        self, G: GraphInstance, solution: BinarySolution
    ) -> np.ndarray:
        data_new = np.copy(G.data)

        for idx, bit in enumerate(solution):
            if bit != 1:
                continue

            mod_type, (u, v) = self.modification_space[idx]

            if mod_type == "remove_edge":
                data_new[u][v] = 0
                if not G.is_directed:
                    data_new[v][u] = 0
            elif mod_type == "add_edge":
                data_new[u][v] = 1
                if not G.is_directed:
                    data_new[v][u] = 1
            else:
                raise ValueError(f"Unknown modification type: {mod_type}")

        return data_new
