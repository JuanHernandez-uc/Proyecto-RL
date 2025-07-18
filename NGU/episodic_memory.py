import numpy as np

class EpisodicMemory:
    def __init__(self):
        self.memory = []

    def reset(self):
        self.memory = []

    def add(self, embedding):
        self.memory.append(embedding)

    def get_intrinsic_reward(self, embedding):
        """
        Calcula la recompensa intrínseca del embedding actual.
        Se define como la distancia mínima (euclideana) respecto a los embeddings almacenados.
        Si la memoria está vacía, se devuelve una novedad inicial proporcional a la norma del embedding.
        """

        ## Si la memoria está vacía, la novedad es proporcional a la norma del embedding
        if not self.memory:
            return np.linalg.norm(embedding) * 1.5

        ## Calculamos todas las distancias euclideanas al embedding actual
        distances = [np.linalg.norm(embedding - mem) for mem in self.memory]
        min_dist = float(np.min(distances))

        ## Como recompensa intrínseca usamos la distancia mínima
        return min_dist