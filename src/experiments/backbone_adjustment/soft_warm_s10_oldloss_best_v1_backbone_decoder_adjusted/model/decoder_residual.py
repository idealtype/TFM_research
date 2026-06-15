import torch.nn as nn


class ResidualDecoder(nn.Module):
    def __init__(self, embed_dim, horizon, mlp_units, activation="ReLU", dropout=0.0):
        super().__init__()
        hidden_dim = int(mlp_units[0]) if mlp_units else embed_dim
        self.hidden_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
        )
        self.output_layer = nn.Linear(hidden_dim, horizon)
        self.residual_layer = nn.Linear(embed_dim, horizon)

    def forward(self, emb):
        return self.output_layer(self.hidden_layer(emb)) + self.residual_layer(emb)
