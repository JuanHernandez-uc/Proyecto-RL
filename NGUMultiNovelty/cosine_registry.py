import numpy as np

class CosineRegistry:
    """
    Agrupa visitas por clusters en el espacio z (normalizado) usando un umbral de similitud coseno.
    Guarda qué agentes han visitado cada cluster (por episodio).
    """
    def __init__(self, sim_thresh=0.90, tau_mode="k", k=2, n_agents=3, dtype="float32"):
        self.sim_thresh = sim_thresh
        self.tau_mode = tau_mode  # "any"|"k"|"all"
        self.k = k
        self.n_agents = n_agents
        self.dtype = dtype
        self._centroids = []     # list[np.ndarray] (no normalizados aquí; normalizamos en consulta)
        self._agents_seen = []   # list[set[int]]

    def reset(self):
        self._centroids.clear()
        self._agents_seen.clear()

    def _find_cluster(self, z):
        """Devuelve el índice del cluster con cos(z, c) >= sim_thresh, o None."""
        if not self._centroids:
            return None
        C = np.stack(self._centroids, axis=0)  # [K, P]
        # normalizar (z y centroids) para coseno
        z = z / (np.linalg.norm(z) + 1e-8)
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
        sims = Cn @ z
        idx = int(np.argmax(sims))
        return idx if sims[idx] >= self.sim_thresh else None

    def visits(self, z):
        idx = self._find_cluster(z)
        return 0 if idx is None else len(self._agents_seen[idx])

    def register(self, agent_id, z):
        z = np.asarray(z, dtype=self.dtype)
        idx = self._find_cluster(z)
        if idx is None:
            self._centroids.append(z)
            self._agents_seen.append({agent_id})
        else:
            self._agents_seen[idx].add(agent_id)

    def tau_star(self):
        if self.tau_mode == "any": return 1
        if self.tau_mode == "all": return self.n_agents
        return max(1, min(self.k, self.n_agents))

    def is_team_novel(self, z):
        return self.visits(z) < self.tau_star()
