import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, batch):
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
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
        return np.sqrt(self.var + 1e-8)

class RNDNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class RND:
    def __init__(self, input_dim, output_dim = 128, lr = 1e-4, max_alpha = 5.0, device = "cpu"):
        self.device = device
        self.max_alpha = max_alpha

        self.target = RNDNetwork(input_dim, output_dim).to(device)
        self.predictor = RNDNetwork(input_dim, output_dim).to(device)

        for param in self.target.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)
        self.reward_rms = RunningMeanStd(shape=())

    def compute_alpha_and_rnd(self, obs_tensor):
        with torch.no_grad():
            target_feat = self.target(obs_tensor)
            pred_feat = self.predictor(obs_tensor)
            rnd_error = F.mse_loss(pred_feat, target_feat, reduction='none').mean(dim=1)

        # α_t = 1 + (error - mean) / std
        rnd_np = rnd_error.cpu().numpy()
        self.reward_rms.update(rnd_np)
        mean, std = self.reward_rms.mean, self.reward_rms.std
        alpha = 1.0 + (rnd_np - mean) / std
        alpha = np.clip(alpha, 1.0, self.max_alpha)
        return rnd_error, torch.tensor(alpha, dtype=torch.float32, device=self.device)

    def modulate_reward(self, episodic_reward, obs_tensor):
        _, alpha = self.compute_alpha_and_rnd(obs_tensor)
        return episodic_reward * alpha

    def update(self, obs_tensor):
        with torch.no_grad():
            target_features = self.target(obs_tensor)

        predictor_features = self.predictor(obs_tensor)
        loss = F.mse_loss(predictor_features, target_features)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
