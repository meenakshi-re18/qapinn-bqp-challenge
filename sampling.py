"""
sampling.py — generates training points for the PINN:
  - Interior collocation points (PDE residual enforced here)
  - Initial-condition points (t = 0)
  - Boundary-condition points (x = x_min and x = x_max)

Latin Hypercube Sampling (LHS) is used by default for the collocation
points because it gives better coverage of the 2D (x,t) domain than
plain uniform random sampling for the same point budget — this matters
because the shock region is a thin, specific area of the domain that
needs adequate coverage.
"""
import math
import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube


def _lhs(n, d, seed):
    sampler = LatinHypercube(d=d, seed=seed)
    return sampler.random(n)  # in [0,1)^d


def sample_collocation_points(cfg, seed=None):
    """Interior (x,t) points where the PDE residual is enforced."""
    seed = cfg.SEED if seed is None else seed
    n = cfg.N_COLLOCATION

    if cfg.SAMPLING_STRATEGY == "lhs":
        u = _lhs(n, 2, seed)
    else:
        rng = np.random.default_rng(seed)
        u = rng.random((n, 2))

    x = cfg.X_MIN + (cfg.X_MAX - cfg.X_MIN) * u[:, 0]
    t = cfg.T_MIN + (cfg.T_MAX - cfg.T_MIN) * u[:, 1]
    return x.astype(np.float32), t.astype(np.float32)


def sample_ic_points(cfg, seed=None):
    """Points on t = 0 with the known initial condition u(x,0) = -sin(pi x)."""
    seed = cfg.SEED if seed is None else seed
    rng = np.random.default_rng(seed + 1)
    x = rng.uniform(cfg.X_MIN, cfg.X_MAX, size=cfg.N_IC).astype(np.float32)
    t = np.zeros_like(x)
    u = -np.sin(math.pi * x).astype(np.float32)
    return x, t, u


def sample_bc_points(cfg, seed=None):
    """Points on x = x_min and x = x_max, where u = 0 (Dirichlet)."""
    seed = cfg.SEED if seed is None else seed
    rng = np.random.default_rng(seed + 2)
    t = rng.uniform(cfg.T_MIN, cfg.T_MAX, size=cfg.N_BC).astype(np.float32)

    x_left = np.full_like(t, cfg.X_MIN)
    x_right = np.full_like(t, cfg.X_MAX)

    x_bc = np.concatenate([x_left, x_right])
    t_bc = np.concatenate([t, t])
    u_bc = np.zeros_like(x_bc)
    return x_bc, t_bc, u_bc


def to_tensor(*arrays, device, dtype):
    return [
        torch.tensor(a, device=device, dtype=dtype).reshape(-1, 1)
        for a in arrays
    ]


def build_training_tensors(cfg):
    """Convenience wrapper: returns all training tensors, ready for the model."""
    x_f, t_f = sample_collocation_points(cfg)
    x_ic, t_ic, u_ic = sample_ic_points(cfg)
    x_bc, t_bc, u_bc = sample_bc_points(cfg)

    x_f, t_f = to_tensor(x_f, t_f, device=cfg.DEVICE, dtype=cfg.DTYPE)
    x_ic, t_ic, u_ic = to_tensor(x_ic, t_ic, u_ic, device=cfg.DEVICE, dtype=cfg.DTYPE)
    x_bc, t_bc, u_bc = to_tensor(x_bc, t_bc, u_bc, device=cfg.DEVICE, dtype=cfg.DTYPE)

    return {
        "x_f": x_f, "t_f": t_f,
        "x_ic": x_ic, "t_ic": t_ic, "u_ic": u_ic,
        "x_bc": x_bc, "t_bc": t_bc, "u_bc": u_bc,
    }
