# inverse_dynamics.py
import torch
import torch.nn as nn

class InverseDynamicsModel(nn.Module):
    def __init__(self, embed_dim, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2 * embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, emb_s, emb_next_s):
        x = torch.cat([emb_s, emb_next_s], dim=1)
        return self.fc(x)
