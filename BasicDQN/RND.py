import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict


class RunningMeanStd:
    
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        #Creo que deberian ir tambien aca weas del NGU según el paper, al menos el embedding pero no me hace sentido xd
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, batch: np.ndarray):
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
    
    @property
    def std(self):
        return np.sqrt(self.var + self.epsilon)


class RNDNetwork(nn.Module):
    
    def __init__(self, input_channels: int = 1, output_dim: int = 128):
        super().__init__()
        
        # Convolutional encoder (Mnih et al., 2015)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc = nn.Linear(64 * 7 * 7, output_dim)
        
    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RND:
    def __init__(
        self,
        input_shape: Tuple[int, int, int],  # (C, H, W)
        output_dim: int = 128,
        learning_rate: float = 0.0005,  # From NGU paper
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        clip_range: float = 5.0, 
        proportion_of_exp_for_predictor: float = 0.25,  # From RND paper
        max_reward_scale: float = 5.0  # L parameter from NGU paper
    ):
        self.device = device
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.clip_range = clip_range
        self.proportion_of_exp_for_predictor = proportion_of_exp_for_predictor
        self.max_reward_scale = max_reward_scale
        
        self.target_network = RNDNetwork(input_shape[0], output_dim).to(device)
        self.predictor_network = RNDNetwork(input_shape[0], output_dim).to(device)
        
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        self.optimizer = torch.optim.Adam(
            self.predictor_network.parameters(), 
            lr=learning_rate,
            eps=1e-4  # From NGU paper
        )
        
        self.obs_rms = RunningMeanStd(shape=(1, *input_shape))
        
        self.reward_rms = RunningMeanStd(shape=())
        
        self._initialized = False
        
    def _normalize_observation(self, obs: np.ndarray):
        normalized = (obs - self.obs_rms.mean) / self.obs_rms.std
        return np.clip(normalized, -self.clip_range, self.clip_range)
    
    def initialize_normalization(self, random_obs: np.ndarray):
        #Initialize observation normalization with random observations
        self.obs_rms.update(random_obs)
        self._initialized = True
    
    def compute_intrinsic_reward(self, obs: np.ndarray):
        if not self._initialized:
            raise ValueError("Must call initialize_normalization first")
        
        if obs.ndim == 3:
            obs = obs[np.newaxis, ...]
        
        # Normalize observation and then tensorize it ou yeh
        obs_norm = self._normalize_observation(obs)
        obs_tensor = torch.FloatTensor(obs_norm).to(self.device)
        
        with torch.no_grad():
            target_features = self.target_network(obs_tensor)
            predictor_features = self.predictor_network(obs_tensor)
            # Compute MSE
            rnd_error = F.mse_loss(predictor_features, target_features, reduction='none')
            rnd_error = rnd_error.mean(dim=1).cpu().numpy() 
        
        if len(rnd_error) == 1:
            rnd_error = rnd_error[0]
        
        # Esta formula es la que se usa en NGU para calcular α_t pero no es la misma que en RND paper
        # α_t = 1 + (err(x_t) - μ_e) / σ_e
        normalized_error = (rnd_error - self.reward_rms.mean) / self.reward_rms.std
        alpha_t = 1.0 + normalized_error
        
        return float(rnd_error), float(alpha_t)
    
    def modulate_episodic_reward(self, episodic_reward: float, obs: np.ndarray):
        _, alpha_t = self.compute_intrinsic_reward(obs)
        modulation_factor = min(max(alpha_t, 1.0), self.max_reward_scale)
        
        return episodic_reward * modulation_factor
    
    def update_predictor(self, obs_batch: np.ndarray):

        batch_size = obs_batch.shape[0]
        train_indices = np.random.choice(
            batch_size, 
            size=int(batch_size * self.proportion_of_exp_for_predictor),
            replace=False
        )
        obs_batch = obs_batch[train_indices]
        
        self.obs_rms.update(obs_batch)
        
        obs_norm = self._normalize_observation(obs_batch)
        obs_tensor = torch.FloatTensor(obs_norm).to(self.device)
        
        with torch.no_grad():
            target_features = self.target_network(obs_tensor)
        
        predictor_features = self.predictor_network(obs_tensor)
        
        # Compute loss
        loss = F.mse_loss(predictor_features, target_features)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            errors = F.mse_loss(predictor_features, target_features, reduction='none')
            errors = errors.mean(dim=1).cpu().numpy()
            self.reward_rms.update(errors)
        
        return loss.item()
    
    def save(self, path: str):
        """Save RND state."""
        state = {
            'predictor_state_dict': self.predictor_network.state_dict(),
            'target_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_rms_mean': self.obs_rms.mean,
            'obs_rms_var': self.obs_rms.var,
            'obs_rms_count': self.obs_rms.count,
            'reward_rms_mean': self.reward_rms.mean,
            'reward_rms_var': self.reward_rms.var,
            'reward_rms_count': self.reward_rms.count,
        }
        torch.save(state, path)
    
    def load(self, path: str):
        state = torch.load(path, map_location=self.device)
        self.predictor_network.load_state_dict(state['predictor_state_dict'])
        self.target_network.load_state_dict(state['target_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.obs_rms.mean = state['obs_rms_mean']
        self.obs_rms.var = state['obs_rms_var']
        self.obs_rms.count = state['obs_rms_count']
        self.reward_rms.mean = state['reward_rms_mean']
        self.reward_rms.var = state['reward_rms_var']
        self.reward_rms.count = state['reward_rms_count']
        self._initialized = True

class NGUWithRND:
    
    def __init__(self, observation_shape: Tuple[int, int, int], **kwargs):
        self.rnd = RND(observation_shape)
        
        # Aca deberiamos poner las weas de NGU, como el embedding, rl loss, etc.
        
    def compute_intrinsic_reward(self, obs: np.ndarray, episodic_reward: float):
        return self.rnd.modulate_episodic_reward(episodic_reward, obs)
    
    def train_step(self, obs_batch: np.ndarray):
        losses = {}
        
        rnd_loss = self.rnd.update_predictor(obs_batch)
        losses['rnd_loss'] = rnd_loss
        
        # Aquí deberíamos agregar el resto de las pérdidas de NGU
        
        return losses


#Esto no tengo idea si es necesario, pero lo dejo por si acaso 
def scale_proportion_for_parallel_envs(base_proportion: float, base_envs: int, current_envs: int):
    """
    Scale the proportion of experience used for predictor training based on 
    number of parallel environments (as mentioned in RND paper).
    """
    return base_proportion * base_envs / current_envs


if __name__ == "__main__":
    obs_shape = (1, 84, 84)
    rnd = RND(obs_shape)
    
    random_obs = np.random.randn(1000, *obs_shape).astype(np.float32)
    rnd.initialize_normalization(random_obs)
    
    obs = np.random.randn(*obs_shape).astype(np.float32)
    episodic_reward = 1.0
    
    rnd_error, alpha_t = rnd.compute_intrinsic_reward(obs)
    modulated_reward = rnd.modulate_episodic_reward(episodic_reward, obs)
    
    print(f"RND Error: {rnd_error:.4f}")
    print(f"Alpha_t: {alpha_t:.4f}")
    print(f"Episodic Reward: {episodic_reward:.4f}")
    print(f"Modulated Reward: {modulated_reward:.4f}")
    
    batch_obs = np.random.randn(128, *obs_shape).astype(np.float32)
    loss = rnd.update_predictor(batch_obs)
    print(f"Training Loss: {loss:.4f}")