import torch
import numpy as np

class FastReplayBuffer:
    def __init__(self, size, obs_dim, n_actions, device = "cpu"):
        self.size = size
        self.device = device

        ## Prealoca memoria para cada componente de la transición
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.nextobs = np.zeros_like(self.obs)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

        ## Puntero de inserción y bandera para saber si se llenó
        self.pos = 0
        self.full = False

    def add(self, state, action, reward, next_state, done):
        self.obs[self.pos] = state
        self.nextobs[self.pos] = next_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        ## Avanza la posición en el buffer (circularmente) y marca como lleno si dio la vuelta
        self.pos = (self.pos + 1) % self.size
        self.full |= self.pos == 0

    def sample(self, batch):
        limit = self.size if self.full else self.pos
        indexes = np.random.randint(0, limit, size = batch)

        states = torch.tensor(self.obs[indexes], dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions[indexes], device=self.device)
        rewards = torch.tensor(self.rewards[indexes], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(self.nextobs[indexes], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[indexes],   dtype=torch.float32, device=self.device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size if self.full else self.pos