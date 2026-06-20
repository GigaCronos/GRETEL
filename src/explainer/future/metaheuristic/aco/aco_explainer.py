import random
import bisect
import math
from typing import List, Tuple, Dict, Optional
import numpy as np
import heapq

from src.core.explainer_base import Explainer
from src.dataset.instances.graph import GraphInstance
from src.explainer.future.utils.explainer_transform import ExplainerTransformMeta

class ACOExplainer(Explainer, metaclass=ExplainerTransformMeta):

    def init(self):
        self.num_ants = self.local_config['parameters'].get('num_ants', 10)
        self.num_iterations = self.local_config['parameters'].get('num_iterations', 25)
        self.alpha = self.local_config['parameters'].get('alpha', 1.0)
        self.beta = self.local_config['parameters'].get('beta', 2.0)
        self.rho = self.local_config['parameters'].get('rho', 0.3)
        self.q = self.local_config['parameters'].get('q', 2.0)
        self.max_modifications = self.local_config['parameters'].get('max_modifications', 5)
        self.early_stop_patience = self.local_config['parameters'].get('early_stop_patience', 5)

        # enfoque greedy-levy
        self.levy_ratio = 5

        self.epsilon0 = self.local_config['parameters'].get('epsilon0', 0.9)
        self.epsilon = self.epsilon0
        self.epsilonfinal = self.local_config['parameters'].get('epsilonfinal', 0.1)

        self.levy0 = self.local_config['parameters'].get('levy0', 0.5)
        self.levy_threshold = self.levy0
        self.levyfinal = self.local_config['parameters'].get('levyfinal', 0.1)

        self.lambda_fid = self.local_config['parameters'].get('lambda_fid', 1.0)
        self.lambda_prox = self.local_config['parameters'].get('lambda_prox', 0.3)

        self.tau0 = 1e-6
        self.tau = {}
        self.modification_space = []
        self.node_degrees = {}
        self.last_update = {}
        self.currentIter = 0
        self.r_min = 0
        self.r_max = 0
        self.use_softmax_when_exploiting = False #False para explotar con argmax o True para explotar con probabily_random_rule

    def real_fit(self):
        pass

    def init_variables(self, G_orig: GraphInstance):
        self.epsilon = self.epsilon0
        self.levy_threshold = self.levy0
        self.node_degrees = {}
        self.last_update = {}
        self.tau0 = 1e-6
        self.tau = {}
        self.modification_space = []
        self.currentIter = 0

        self.generate_modifications(G_orig)
        for mod in self.modification_space:
            self.tau[mod] = 1.0
            self.last_update[mod] = 0
        #self.node_degrees = dict(enumerate(np.sum(G_orig.data, axis=1).astype(int)))
        self.levy_ratio = 0.03 * len(self.modification_space)

        self.compute_dynamic_r_range()

    def explain(self, instance: GraphInstance):
        best_graph, best_solution = self.aco_run(instance)
        return best_graph if best_graph is not None else instance

    def weighted_sample(self, items, k):
        weights = np.array([self.get_pheromone(item) for item in items], dtype=np.float64)
        if weights.sum() == 0:
            weights = np.ones(len(items)) / len(items)
        else:
            weights /= weights.sum()
        elems = np.empty(len(items), dtype=object)
        elems[:] = items
        return list(np.random.choice(elems, size=k, replace=False, p=weights))

    def aco_run(self, G_orig: GraphInstance):
        original_label = self.oracle.predict(G_orig)

        best_solution = None
        best_score = -float('inf')
        best_graph = None
        no_improve_counter = 0

        self.init_variables(G_orig)

        for iteration in range(self.num_iterations):
            self.currentIter = iteration
            self.epsilon = self.anneal_parameter(self.epsilon0, self.epsilonfinal, iteration, self.num_iterations)
            self.levy_threshold = self.anneal_parameter(self.levy0, self.levyfinal, iteration, self.num_iterations)

            candidate_mods = self.select_subset_modifications()
            all_solutions = []
            all_scores = []
            improved = False

            for _ in range(self.num_ants):
                solution_path, G_candidate, pred = self.build_solution(G_orig, original_label, candidate_mods)
                distance = self.graph_distance(G_candidate, G_orig)
                changed = int(pred != original_label)
                if changed:
                    score = changed + (1 / (1 + distance))
                else:
                    # Penalización si no cambia la predicción
                    score = 0.1 * (1 / (1 + distance))

                all_solutions.append(solution_path)
                all_scores.append(score)

                if changed and score > best_score:
                    best_score = score
                    best_solution = solution_path
                    best_graph = GraphInstance(id=G_candidate.id, label=0, data=G_candidate.data.copy(), node_features=G_candidate.node_features.copy())
                    improved = True

            self.update_pheromones(all_solutions, all_scores)

            if improved:
                no_improve_counter = 0
            else:
                no_improve_counter += 1

            if no_improve_counter >= self.early_stop_patience:
                break

        return best_graph, best_solution

    def build_solution(self, G_orig: GraphInstance, original_label, modifications):
        G_current = GraphInstance(id=G_orig.id, label=0, data=G_orig.data.copy(), node_features=G_orig.node_features)
        solution_path = []
        pred_current = 0
        for _ in range(self.max_modifications):
            available_mods = [m for m in modifications if m not in solution_path]
            if not available_mods:
                break
            selected_mod = self.select_modification(available_mods, G_current, original_label)
            G_current = self.apply_modification_inplace(G_current, selected_mod)
            solution_path.append(selected_mod)
            pred_current = self.oracle.predict(G_current)
            if pred_current != original_label:                
                break

        return solution_path, G_current, pred_current

    def compute_dynamic_r_range(self):
        m = len(self.modification_space)
        self.r_min = max(2, int(0.015 * m))
        self.r_max = max(self.r_min + 3, int(0.25 * m))       

    def sample_subset_size(self):
        n = len(self.modification_space)
        r_values = list(range(self.r_min, min(self.r_max + 1, n)))
        weights = np.array([n - r for r in r_values], dtype=np.float64)
        weights /= weights.sum()
        return random.choices(r_values, weights=weights, k=1)[0]

    def select_subset_modifications(self):
        r = self.sample_subset_size()
        # Devuelve las r modificaciones con mayor valor de feromona τ
        top_r = heapq.nlargest(r, self.modification_space, key=self.get_pheromone)
        return top_r

    def select_modification(self, modifications, G, original_label):
        if random.random() < self.epsilon:
            # ε-greedy: con probabilidad ε se explora (aleatorio)
            a = random.choice(self.modification_space)
            if random.random() < self.levy_threshold:
                a = self.levy_jump(a, self.modification_space, self.levy_ratio)
            return a
        else:
            #con 1−ε se explota (mejor opción)
            # Explotación: decide si usar argmax o muestreo proporcional
            probs = self.compute_probabilities(modifications, G, original_label)

            if self.use_softmax_when_exploiting:
                # Exploitation guided but stochastic
                return random.choices(modifications, weights=probs, k=1)[0]
            else:
                # Exploitation strict
                return modifications[probs.index(max(probs))]

    def levy_jump(self, chosen, actions, A):
        n = len(actions)
        idx = actions.index(chosen)
        step = int(math.floor((random.paretovariate(alpha=1.5)) * A))
        return actions[(idx + step) % n]

    def anneal_parameter(self, initial_value, final_value, current_iter, max_iter):
        progress = current_iter / max_iter
        return initial_value * (1 - progress) + final_value * progress

    def update_pheromones(self, elite_paths: List[List], scores: List[float]):
        for path, score in zip(elite_paths, scores):
            for mod in path:
                self.tau[mod] = self.get_pheromone(mod) + self.q * score

    def get_pheromone(self, mod):
        last_update_iter = self.last_update.get(mod, 0)
        delta_iters = self.currentIter - last_update_iter

        # Si nunca se actualizó o está desactualizado, aplicar evaporación perezosa
        if delta_iters > 0:
            decay = (1 - self.rho) ** delta_iters
            old_val = self.tau.get(mod, self.tau0)
            new_val = max(old_val * decay, self.tau0)
            self.tau[mod] = new_val
            self.last_update[mod] = self.currentIter

        return self.tau.get(mod, self.tau0)

    def compute_probabilities(self, modifications, G: GraphInstance, original_label):
        weights = []
        total = 0
        for m in modifications:
            tau_val = self.get_pheromone(m)
            eta_val = self.compute_heuristic(m, G, original_label)
            tau_term = tau_val ** self.alpha if tau_val > 0 else 0
            eta_term = eta_val ** self.beta if eta_val > 0 else 0
            term = tau_term * eta_term
            weights.append(term)
            total += term

        return [w / total for w in weights]

    def compute_heuristic(self, mod, G: GraphInstance, original_label):
        G_tmp = self.apply_modification_inplace(G, mod)
        tmp_label = self.oracle.predict(G_tmp)
        changed = int(tmp_label != original_label)

        distance = self.graph_distance(G, G_tmp)
        return self.lambda_fid * changed + self.lambda_prox * (1 / (1 + distance))

    def generate_modifications(self, G: GraphInstance):
        data = G.data
        num_nodes = len(data)

        for u in range(num_nodes):
            for v in range(u, num_nodes):
                if u!=v:
                    if data[u][v] > 0.5:
                        self.modification_space.append(("remove_edge", (u, v)))
                    else:
                        self.modification_space.append(("add_edge", (u, v)))

        random.shuffle(self.modification_space)

    def apply_modification_inplace(self, G: GraphInstance, mod):
        data_new = self.apply_modification_to_matrices(G, mod)
        return GraphInstance(id=G.id, label=0, data=data_new, node_features=G.node_features)

    def apply_modification_to_matrices(self, G: GraphInstance, mod):
        data_new = G.data.copy()
        mod_type, data = mod
        u, v = data
        if mod_type == "remove_edge":
            data_new[u, v] = 0
            data_new[v, u] = 0
        elif mod_type == "add_edge":
            data_new[u, v] = 1
            data_new[v, u] = 1
        return data_new

    def proxy_dist_change(self, mod):
        """
        Estima el impacto estructural de una modificación usando el grado de los nodos involucrados.
        - Más impacto si los nodos son centrales.
        - Penaliza menos las remociones que las adiciones.
        """
        mod_type, data = mod

        if mod_type in {"add_edge", "remove_edge"}:
            u, v = data

            deg_u = self.node_degrees.get(u, 1)
            deg_v = self.node_degrees.get(v, 1)

            centrality_score = (deg_u + deg_v) / 2

            base_cost = 1.0 if mod_type == "add_edge" else 0.8

            return base_cost * centrality_score

        else:
            return 1.0  # fallback para otros tipos
        
    def graph_distance(self, G1: GraphInstance, G2: GraphInstance):
        diff = np.abs(G1.data - G2.data)
        return np.sum(diff) / 2 if not G1.is_directed else np.sum(diff)
