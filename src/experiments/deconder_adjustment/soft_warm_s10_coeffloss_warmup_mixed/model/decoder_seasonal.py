import torch
import torch.nn as nn

# HEAD capacity = actual order (daily:10, weekly:4, monthly:2, yearly:8)
# Identical to fine_mask.
N_FOURIER_TERMS = {"daily": 10, "weekly": 4, "monthly": 2, "yearly": 8}

# Gate dimension per family — must match K_MAX in common.py
_K_MAX = {"daily": 10, "weekly": 4, "monthly": 2, "yearly": 8}
_GATE_SLICES = {
    "daily":   (0,  10),
    "weekly":  (10, 14),
    "monthly": (14, 16),
    "yearly":  (16, 24),
}
N_GATES = 24  # sum of K_MAX values


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


class HarmonicGatingNetwork(nn.Module):
    """Lightweight MLP: emb → 24 per-harmonic gates ∈ [0, 1].

    Gate layout (contiguous):
      indices  0-9  : daily   k=1..10
      indices 10-13 : weekly  k=1..4
      indices 14-15 : monthly k=1..2
      indices 16-23 : yearly  k=1..8

    Initialization: all zeros → sigmoid(0) = 0.5 (neutral start).
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_GATES),
        )
        # Neutral init: last layer weight=0, bias=0 → gate starts at 0.5
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """Returns gates of shape (B, 24) with values in [0, 1]."""
        return torch.sigmoid(self.net(emb))


class SeasonalDecoder(nn.Module):
    """Seasonal decoder with learned per-harmonic soft gating.

    Architecture (vs fine_mask):
      SAME : four family MLPs + forecast heads (predict Fourier coefficients)
      NEW  : HarmonicGatingNetwork — predicts 24 gates from emb
             gates are applied element-wise to coefficients before bmm with basis

    Basis passed in must be the soft_mask physics-only basis
    (fd < P/k, no context_span condition). The gates learn the context-span
    discrimination that was hard-coded in fine_mask.
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

        # Independent lightweight gating network
        self.gating_network = HarmonicGatingNetwork(embed_dim)

    def predict_coefficients(self, emb):
        return {
            "daily":   self.forecast_head_daily(self.mlp_daily(emb)),
            "weekly":  self.forecast_head_weekly(self.mlp_weekly(emb)),
            "monthly": self.forecast_head_monthly(self.mlp_monthly(emb)),
            "yearly":  self.forecast_head_yearly(self.mlp_yearly(emb)),
        }

    @staticmethod
    def _apply_gate(coef: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Expand per-harmonic gate to per-coefficient and multiply.

        coef  : (B, 2*K)  — [a1, b1, a2, b2, ..., aK, bK]
        gate  : (B, K)    — one scalar per harmonic
        return: (B, 2*K)  — gated coefficients
        """
        gate_expanded = gate.repeat_interleave(2, dim=1)  # (B, 2*K)
        return coef * gate_expanded

    def forward(self, emb, daily_basis, weekly_basis, monthly_basis, yearly_basis,
                return_coefficients=False):
        coeffs = self.predict_coefficients(emb)
        gates  = self.gating_network(emb)  # (B, 24)

        # Slice gates per family
        g_d = gates[:, _GATE_SLICES["daily"][0]:   _GATE_SLICES["daily"][1]]
        g_w = gates[:, _GATE_SLICES["weekly"][0]:  _GATE_SLICES["weekly"][1]]
        g_m = gates[:, _GATE_SLICES["monthly"][0]: _GATE_SLICES["monthly"][1]]
        g_y = gates[:, _GATE_SLICES["yearly"][0]:  _GATE_SLICES["yearly"][1]]

        # Apply gates: element-wise on coefficients, then bmm with basis
        coef_d = self._apply_gate(coeffs["daily"],   g_d)
        coef_w = self._apply_gate(coeffs["weekly"],  g_w)
        coef_m = self._apply_gate(coeffs["monthly"], g_m)
        coef_y = self._apply_gate(coeffs["yearly"],  g_y)

        s_d = torch.bmm(daily_basis,   coef_d.unsqueeze(-1)).squeeze(-1)
        s_w = torch.bmm(weekly_basis,  coef_w.unsqueeze(-1)).squeeze(-1)
        s_m = torch.bmm(monthly_basis, coef_m.unsqueeze(-1)).squeeze(-1)
        s_y = torch.bmm(yearly_basis,  coef_y.unsqueeze(-1)).squeeze(-1)

        seasonal = s_d + s_w + s_m + s_y

        if return_coefficients:
            # Return pre-gate coefficients (raw predictions) + gates separately
            return seasonal, coeffs, gates
        return seasonal, gates
