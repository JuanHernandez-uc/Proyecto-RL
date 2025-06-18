import torch.nn as nn

class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embed_dim=64):
        super(EmbeddingNet, self).__init__()
        # Definimos una MLP simple que produce vectores de embedding de dimensión embed_dim.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        return self.net(x)
