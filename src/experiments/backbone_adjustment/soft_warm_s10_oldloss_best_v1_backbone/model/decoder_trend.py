import torch
import torch.nn as nn


def _build_mlp(input_dim, hidden_layers, activation, dropout):
    activations = {"ReLU": nn.ReLU, "SiLU": nn.SiLU, "GELU": nn.GELU}
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activations[activation]())
        layers.append(nn.Dropout(dropout))
        last_dim = hidden_dim
    return nn.Sequential(*layers), last_dim


class TrendDecoder(nn.Module):
    def __init__(self, embed_dim, horizon, n_knots, mlp_units, activation="ReLU", dropout=0.0):
        super().__init__()
        t = torch.linspace(0.0, 1.0, horizon, dtype=torch.float32)
        changepoints = torch.arange(1, n_knots + 1, dtype=torch.float32) / (n_knots + 1)
        self.register_buffer("t", t)
        self.register_buffer("changepoints", changepoints)
        self.register_buffer("past_changepoint", (t.unsqueeze(1) >= changepoints.unsqueeze(0)).to(torch.float32))

        self.mlp, last_dim = _build_mlp(embed_dim, mlp_units, activation, dropout)
        self.forecast_head = nn.Linear(last_dim, n_knots + 2)

    def forward(self, emb):
        out = self.forecast_head(self.mlp(emb))
        intercept = out[:, :1]
        base_slope = out[:, 1:2]
        slope_delta = out[:, 2:]
        k_t = slope_delta @ self.past_changepoint.T
        gamma = -self.changepoints.unsqueeze(0) * slope_delta
        m_t = gamma @ self.past_changepoint.T
        trend = intercept + (base_slope + k_t) * self.t.unsqueeze(0) + m_t
        return trend, slope_delta
