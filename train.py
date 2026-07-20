"""
train.py — trains Model A (classical PINN) on the Burgers' equation.

Two-phase schedule (standard for PINNs):
  Phase 1: Adam       — fast, robust global descent from random init.
  Phase 2: L-BFGS      — quasi-Newton fine-tuning; squeezes out the last
                          bit of accuracy once Adam has found a good basin.

Both phases optimize the exact same combined loss (PDE + IC + BC), on the
exact same fixed set of collocation/IC/BC points, so the comparison to
Model B stays fair as long as Model B uses this same file unmodified.
"""
import os
import csv
import time
import torch

from pinn_losses import compute_losses
from utils import get_logger, Timer


def _log_row(csv_path, row, header):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def train(model, cfg, batch, logger=None):
    """
    Trains `model` in place. Returns a dict with training-time metrics:
    total wall-clock time, final losses, and the loss-history CSV path.
    """
    logger = logger or get_logger("train")
    cfg.ensure_dirs()

    header = ["epoch", "phase", "total", "pde", "ic", "bc", "lr", "elapsed_sec"]
    csv_path = cfg.LOSS_HISTORY_PATH
    if os.path.exists(csv_path):
        os.remove(csv_path)  # fresh run, fresh log

    t_start = time.perf_counter()

    # ------------------------------------------------------------
    # Phase 1: Adam
    # ------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ADAM_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.ADAM_LR_DECAY_STEP, gamma=cfg.ADAM_LR_DECAY_GAMMA
    )

    best_loss = float("inf")
    logger.info(f"Phase 1/2: Adam — {cfg.ADAM_EPOCHS} epochs, lr={cfg.ADAM_LR}")

    for epoch in range(1, cfg.ADAM_EPOCHS + 1):
        optimizer.zero_grad()
        losses = compute_losses(model, batch, cfg.NU, cfg.LOSS_WEIGHTS)
        losses["total"].backward()

        if cfg.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)

        optimizer.step()
        scheduler.step()

        total_val = losses["total"].item()
        if total_val < best_loss:
            best_loss = total_val
            _save_checkpoint(model, cfg, epoch, total_val, phase="adam")

        if epoch % cfg.LOG_EVERY == 0 or epoch == 1 or epoch == cfg.ADAM_EPOCHS:
            elapsed = time.perf_counter() - t_start
            lr_now = optimizer.param_groups[0]["lr"]
            logger.info(
                f"[Adam {epoch:5d}/{cfg.ADAM_EPOCHS}] "
                f"total={total_val:.3e} pde={losses['pde'].item():.3e} "
                f"ic={losses['ic'].item():.3e} bc={losses['bc'].item():.3e} "
                f"lr={lr_now:.2e} t={elapsed:.1f}s"
            )
            _log_row(csv_path,
                      [epoch, "adam", total_val, losses["pde"].item(),
                       losses["ic"].item(), losses["bc"].item(), lr_now, elapsed],
                      header)

    adam_time = time.perf_counter() - t_start

    # ------------------------------------------------------------
    # Phase 2: L-BFGS (optional but recommended — standard for PINNs)
    # ------------------------------------------------------------
    lbfgs_time = 0.0
    if cfg.USE_LBFGS:
        logger.info(f"Phase 2/2: L-BFGS — up to {cfg.LBFGS_MAX_ITER} iterations")
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=cfg.LBFGS_MAX_ITER,
            history_size=cfg.LBFGS_HISTORY_SIZE,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
        )

        iteration_counter = {"n": 0}
        t_lbfgs_start = time.perf_counter()

        def closure():
            lbfgs.zero_grad()
            losses = compute_losses(model, batch, cfg.NU, cfg.LOSS_WEIGHTS)
            losses["total"].backward()

            iteration_counter["n"] += 1
            n = iteration_counter["n"]
            if n % cfg.LOG_EVERY == 0 or n == 1:
                elapsed = time.perf_counter() - t_lbfgs_start
                logger.info(
                    f"[LBFGS {n:5d}] total={losses['total'].item():.3e} "
                    f"pde={losses['pde'].item():.3e} ic={losses['ic'].item():.3e} "
                    f"bc={losses['bc'].item():.3e} t={elapsed:.1f}s"
                )
                _log_row(csv_path,
                          [cfg.ADAM_EPOCHS + n, "lbfgs", losses["total"].item(),
                           losses["pde"].item(), losses["ic"].item(),
                           losses["bc"].item(), lbfgs.param_groups[0]["lr"], elapsed],
                          header)
            return losses["total"]

        lbfgs.step(closure)
        lbfgs_time = time.perf_counter() - t_lbfgs_start

        final_losses = compute_losses(model, batch, cfg.NU, cfg.LOSS_WEIGHTS)
        if final_losses["total"].item() < best_loss:
            best_loss = final_losses["total"].item()
        _save_checkpoint(model, cfg, cfg.ADAM_EPOCHS + iteration_counter["n"],
                          final_losses["total"].item(), phase="lbfgs")

    total_time = time.perf_counter() - t_start
    logger.info(f"Training complete. Total time: {total_time:.1f}s "
                f"(Adam: {adam_time:.1f}s, LBFGS: {lbfgs_time:.1f}s). "
                f"Best total loss: {best_loss:.3e}")

    return {
        "adam_time_sec": adam_time,
        "lbfgs_time_sec": lbfgs_time,
        "total_time_sec": total_time,
        "best_loss": best_loss,
        "loss_history_csv": csv_path,
    }


def _save_checkpoint(model, cfg, epoch, loss_val, phase):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "phase": phase,
        "loss": loss_val,
        "model_state_dict": model.state_dict(),
        "config": cfg.as_dict(),
    }, cfg.CHECKPOINT_PATH)
