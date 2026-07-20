"""
evaluate.py — quantitative evaluation of a trained PINN against the
high-resolution numerical reference solution.

Metrics computed (identical set will be used for Model B, so the
head-to-head comparison table is apples-to-apples):

  - L2 relative error over the full (x,t) grid
  - Max absolute error over the full grid
  - L2 relative error restricted to the "shock region"
    (|x| < SHOCK_X_WINDOW, t > SHOCK_T_MIN) — this is the hard part
    of the domain and the primary metric of scientific interest.
  - Mean/max PDE residual on a held-out evaluation grid (physics
    consistency check, independent of the reference solution).
  - Parameter count and training time (efficiency/reliability metrics).
"""
import json
import os
import numpy as np
import torch

from pinn_losses import pde_residual


@torch.no_grad()
def predict_on_grid(model, x_grid, t_grid, device, dtype):
    """
    x_grid: (nx,) ndarray
    t_grid: (nt,) ndarray
    Returns U_pred: (nt, nx) ndarray, matching reference solution layout.
    """
    X, T = np.meshgrid(x_grid, t_grid)  # shapes (nt, nx)
    x_flat = torch.tensor(X.reshape(-1, 1), device=device, dtype=dtype)
    t_flat = torch.tensor(T.reshape(-1, 1), device=device, dtype=dtype)

    u_flat = model(x_flat, t_flat).cpu().numpy().reshape(X.shape)
    return u_flat


def compute_pde_residual_grid(model, x_grid, t_grid, nu, device, dtype):
    """PDE residual evaluated on a grid (requires grad, so no @torch.no_grad)."""
    X, T = np.meshgrid(x_grid, t_grid)
    x_flat = torch.tensor(X.reshape(-1, 1), device=device, dtype=dtype)
    t_flat = torch.tensor(T.reshape(-1, 1), device=device, dtype=dtype)

    f = pde_residual(model, x_flat, t_flat, nu)
    return f.detach().cpu().numpy().reshape(X.shape)


def evaluate_model(model, cfg, x_ref, t_ref, U_ref, training_stats=None, logger=None):
    """
    Full evaluation pipeline. Returns a metrics dict and also writes it to
    cfg.METRICS_PATH as JSON.
    """
    model.eval()

    U_pred = predict_on_grid(model, x_ref, t_ref, cfg.DEVICE, cfg.DTYPE)

    # ---------- Global L2 relative error ----------
    l2_error = np.linalg.norm(U_pred - U_ref) / np.linalg.norm(U_ref)
    max_abs_error = np.max(np.abs(U_pred - U_ref))
    mean_abs_error = np.mean(np.abs(U_pred - U_ref))

    # ---------- Shock-region error ----------
    X, T = np.meshgrid(x_ref, t_ref)
    shock_mask = (np.abs(X) < cfg.SHOCK_X_WINDOW) & (T > cfg.SHOCK_T_MIN)
    if shock_mask.sum() > 0:
        shock_l2_error = (np.linalg.norm(U_pred[shock_mask] - U_ref[shock_mask])
                           / np.linalg.norm(U_ref[shock_mask]))
        shock_max_error = np.max(np.abs(U_pred[shock_mask] - U_ref[shock_mask]))
    else:
        shock_l2_error, shock_max_error = None, None

    # ---------- PDE residual on evaluation grid (physics consistency) ----------
    model.train()  # need grads enabled for autograd through the model
    f_grid = compute_pde_residual_grid(model, x_ref, t_ref, cfg.NU, cfg.DEVICE, cfg.DTYPE)
    model.eval()
    residual_mean_abs = float(np.mean(np.abs(f_grid)))
    residual_max_abs = float(np.max(np.abs(f_grid)))
    residual_rmse = float(np.sqrt(np.mean(f_grid ** 2)))

    # ---------- Fourier spectrum recovery (spectral-bias diagnostic) ----------
    # Compare spatial frequency content of predicted vs reference solution
    # at the final time slice (where the shock is sharpest / highest freq).
    u_pred_final = U_pred[-1, :]
    u_ref_final = U_ref[-1, :]
    spec_pred = np.abs(np.fft.rfft(u_pred_final))
    spec_ref = np.abs(np.fft.rfft(u_ref_final))
    spectrum_l2_error = (np.linalg.norm(spec_pred - spec_ref)
                          / (np.linalg.norm(spec_ref) + 1e-12))

    metrics = {
        "l2_relative_error": float(l2_error),
        "max_abs_error": float(max_abs_error),
        "mean_abs_error": float(mean_abs_error),
        "shock_region_l2_relative_error": (float(shock_l2_error)
                                            if shock_l2_error is not None else None),
        "shock_region_max_abs_error": (float(shock_max_error)
                                        if shock_max_error is not None else None),
        "pde_residual_mean_abs": residual_mean_abs,
        "pde_residual_max_abs": residual_max_abs,
        "pde_residual_rmse": residual_rmse,
        "fourier_spectrum_l2_relative_error": float(spectrum_l2_error),
        "num_trainable_parameters": model.count_parameters(),
    }

    if training_stats is not None:
        metrics["training"] = {
            "adam_time_sec": training_stats.get("adam_time_sec"),
            "lbfgs_time_sec": training_stats.get("lbfgs_time_sec"),
            "total_time_sec": training_stats.get("total_time_sec"),
            "best_loss": training_stats.get("best_loss"),
        }

    os.makedirs(os.path.dirname(cfg.METRICS_PATH), exist_ok=True)
    with open(cfg.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    if logger:
        logger.info("Evaluation metrics:")
        for k, v in metrics.items():
            if k != "training":
                logger.info(f"  {k}: {v}")
        logger.info(f"Metrics saved to {cfg.METRICS_PATH}")

    return metrics, U_pred
