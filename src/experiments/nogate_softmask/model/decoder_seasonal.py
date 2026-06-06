import torch.nn as nn
import torch

# HEAD capacity = actual order (daily:10, weekly:4, monthly:2, yearly:8)
# Identical to fine_mask.
N_FOURIER_TERMS = {"daily": 10, "weekly": 4, "monthly": 2, "yearly": 8}

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


class SeasonalDecoder(nn.Module):
    """Seasonal decoder without a learned gate.

    It keeps the soft-mask physics-only basis, but applies the predicted
    Fourier coefficients directly. Sparsity is imposed by the training loss on
    the coefficients themselves.
    """

    def __init__(self, embed_dim, mlp_units, activation="ReLU", dropout=0.0):
        super().__init__()
        self.mlp_daily,   last_d = _build_mlp(embed_dim, mlp_units, activation, dropout)
        self.mlp_weekly,  last_w = _build_mlp(embed_dim, mlp_units, activation, dropout)
        self.mlp_monthly, last_m = _build_mlp(embed_dim, mlp_units, activation, dropout)
        self.mlp_yearly,  last_y = _build_mlp(embed_dim, mlp_units, activation, dropout)

        self.forecast_head_daily   = nn.Linear(last_d, 2 * N_FOURIER_TERMS["daily"])
        self.forecast_head_weekly  = nn.Linear(last_w, 2 * N_FOURIER_TERMS["weekly"])
        self.forecast_head_monthly = nn.Linear(last_m, 2 * N_FOURIER_TERMS["monthly"])
        self.forecast_head_yearly  = nn.Linear(last_y, 2 * N_FOURIER_TERMS["yearly"])

    def predict_coefficients(self, emb):
        return {
            "daily":   self.forecast_head_daily(self.mlp_daily(emb)),
            "weekly":  self.forecast_head_weekly(self.mlp_weekly(emb)),
            "monthly": self.forecast_head_monthly(self.mlp_monthly(emb)),
            "yearly":  self.forecast_head_yearly(self.mlp_yearly(emb)),
        }

    def forward(self, emb, daily_basis, weekly_basis, monthly_basis, yearly_basis,
                return_coefficients=False):
        coeffs = self.predict_coefficients(emb)
        s_d = torch.bmm(daily_basis,   coeffs["daily"].unsqueeze(-1)).squeeze(-1)
        s_w = torch.bmm(weekly_basis,  coeffs["weekly"].unsqueeze(-1)).squeeze(-1)
        s_m = torch.bmm(monthly_basis, coeffs["monthly"].unsqueeze(-1)).squeeze(-1)
        s_y = torch.bmm(yearly_basis,  coeffs["yearly"].unsqueeze(-1)).squeeze(-1)

        seasonal = s_d + s_w + s_m + s_y

        if return_coefficients:
            return seasonal, coeffs
        return seasonal
