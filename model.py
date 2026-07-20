"""
model.py — Model A: classical fully-connected Physics-Informed Neural
Network. Input is the raw (x, t) pair (NO quantum feature map — this is
precisely the component Model B will replace, so keep this file minimal
and self-contained for a clean A/B comparison).
"""
import torch
import torch.nn as nn


_ACTIVATIONS = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class PINN(nn.Module):
    """
    Fully-connected network mapping (x, t) -> u(x, t).

    Architecture is configurable via `layers`, e.g. [2, 64, 64, 64, 64, 64, 1].
    Uses tanh activations by default (standard choice for PINNs — smooth,
    infinitely differentiable, which matters because we need 2nd-order
    derivatives of the network output for the PDE residual).
    """

    def __init__(self, layers, activation="tanh", init="xavier_normal"):
        super().__init__()
        assert layers[0] == 2, "Input layer must have width 2 (x, t)."
        assert layers[-1] == 1, "Output layer must have width 1 (u)."

        act_cls = _ACTIVATIONS.get(activation)
        if act_cls is None:
            raise ValueError(f"Unknown activation '{activation}'. "
                              f"Choices: {list(_ACTIVATIONS)}")

        modules = []
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            self._init_weights(linear, init)
            modules.append(linear)
            if i < len(layers) - 2:
                modules.append(act_cls())
        self.net = nn.Sequential(*modules)

    @staticmethod
    def _init_weights(linear: nn.Linear, init: str):
        if init == "xavier_normal":
            nn.init.xavier_normal_(linear.weight)
        elif init == "xavier_uniform":
            nn.init.xavier_uniform_(linear.weight)
        elif init == "kaiming_normal":
            nn.init.kaiming_normal_(linear.weight, nonlinearity="tanh")
        else:
            raise ValueError(f"Unknown init scheme '{init}'.")
        nn.init.zeros_(linear.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x, t: shape (N, 1) each. Returns u: shape (N, 1)."""
        xt = torch.cat([x, t], dim=1)
        return self.net(xt)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg) -> PINN:
    model = PINN(cfg.LAYERS, activation=cfg.ACTIVATION, init=cfg.INIT)
    return model.to(device=cfg.DEVICE, dtype=cfg.DTYPE)
