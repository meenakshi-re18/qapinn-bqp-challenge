# model_a_preprocessed.py
import torch
import torch.nn as nn
from model import PINN

class PreprocessedPINN(nn.Module):
    def __init__(self, layers, activation="tanh", init="xavier_normal", preprocess_dim=4):
        super().__init__()
        self.preproc = nn.Sequential(
            nn.Linear(2, preprocess_dim), nn.Tanh(),
            nn.Linear(preprocess_dim, 2), nn.Tanh(),
        )
        self.body = PINN(layers, activation=activation, init=init)

    def forward(self, x, t):
        xt = self.preproc(torch.cat([x, t], dim=1))
        return self.body(xt[:, 0:1], xt[:, 1:2])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def build_model(cfg):
    m = PreprocessedPINN(cfg.LAYERS, activation=cfg.ACTIVATION, init=cfg.INIT,
                          preprocess_dim=getattr(cfg, "CLASSICAL_PREPROCESS_DIM", 4))
    return m.to(device=cfg.DEVICE, dtype=cfg.DTYPE)