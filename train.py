"""
train.py — trains a PINN (Model A or Model B — this file is shared, see
report Section 5.1) on the Burgers' equation.

Two-phase schedule (standard for PINNs):
  Phase 1: Adam       — fast, robust global descent from random init.
  Phase 2: L-BFGS      — quasi-Newton fine-tuning; squeezes out the last
                          bit of accuracy once Adam has found a good basin.

Both phases optimize the exact same combined loss (PDE + IC + BC), on the
exact same fixed set of collocation/IC/BC points, so the comparison to
Model B stays fair as long as Model B uses this same file unmodified.

Reliability features (added after observing Model B plateau early in
training — see module-level note at QUANTUM GRADIENT DIAGNOSTIC below):
  - Resume support: training can be interrupted and continued from the
    last saved state (model + optimizer + scheduler + epoch), so a long
    run is never lost to a crash, sleep, or manual interruption.
  - Periodic checkpointing: saves every cfg.CHECKPOINT_EVERY epochs
    regardless of whether loss improved, as a safety net independent of
    the "best loss so far" checkpoint.
  - Quantum gradient-norm diagnostic: for any model with a `.qlayer`
    attribute (i.e. Model B), logs the gradient norm on the quantum
    circuit's variational weights every LOG_EVERY epochs. A gradient
    norm that collapses toward zero early in training, while loss stays
    flat, is the specific signature of a barren plateau (McClean et al.,
    2018) — not evidence of convergence. This number is what
    distinguishes "the model converged" from "the model's gradients
    vanished" and should be reported alongside the loss curve for
    Model B, not left out.
  - Early stopping is OPT-IN (cfg.EARLY_STOPPING_PATIENCE, default None
    = disabled) rather than automatic, specifically so a flat loss curve
    doesn't silently truncate training before the gradient-norm
    diagnostic has had a chance to reveal whether it's a barren plateau.
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


def _quantum_grad_norm(model):
    """
    Returns the L2 norm of the gradient on the quantum circuit's
    variational weights, or None if `model` has no quantum layer
    (i.e. this is Model A). Call AFTER loss.backward(), BEFORE
    optimizer.step() / zero_grad().
    """
    qlayer = getattr(model, "qlayer", None)
    if qlayer is None:
        return None
    grad = getattr(qlayer.weights, "grad", None)
    if grad is None:
        return None
    return grad.norm().item()


def _checkpoint_state(model, optimizer, scheduler, epoch, phase, loss_val, cfg):
    return {
        "epoch": epoch,
        "phase": phase,
        "loss": loss_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "config": cfg.as_dict(),
    }


def _save_checkpoint(model, cfg, epoch, loss_val, phase, optimizer=None, scheduler=None, path=None):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    torch.save(_checkpoint_state(model, optimizer, scheduler, epoch, phase, loss_val, cfg),
               path or cfg.CHECKPOINT_PATH)


def _latest_checkpoint_path(cfg):
    """Separate from cfg.CHECKPOINT_PATH (which holds the BEST loss seen)
    so periodic/resume checkpointing never clobbers the best-model file."""
    return cfg.CHECKPOINT_PATH.replace(".pt", "_latest.pt")


def try_resume(model, optimizer, scheduler, cfg, logger):
    """
    If cfg.RESUME is True and a 'latest' checkpoint exists, loads model,
    optimizer, and scheduler state in place. Returns the epoch to resume
    FROM (0 if no resume happened, so the training loop starts at 1 as
    normal).
    """
    if not getattr(cfg, "RESUME", False):
        return 0

    path = _latest_checkpoint_path(cfg)
    if not os.path.exists(path):
        logger.info(f"--resume requested but no checkpoint found at {path}; starting fresh.")
        return 0

    ckpt = torch.load(path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt.get("scheduler_state_dict") is not None and scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    resume_epoch = ckpt["epoch"]
    logger.info(f"Resumed from {path}: epoch {resume_epoch}, loss={ckpt['loss']:.3e}")
    return resume_epoch


def train(model, cfg, batch, logger=None):
    """
    Trains `model` in place. Returns a dict with training-time metrics:
    total wall-clock time, final losses, and the loss-history CSV path.
    """
    logger = logger or get_logger("train")
    cfg.ensure_dirs()

    header = ["epoch", "phase", "total", "pde", "ic", "bc", "lr", "quantum_grad_norm", "elapsed_sec"]
    csv_path = cfg.LOSS_HISTORY_PATH
    resuming = getattr(cfg, "RESUME", False) and os.path.exists(_latest_checkpoint_path(cfg))
    if os.path.exists(csv_path) and not resuming:
        os.remove(csv_path)  # fresh run, fresh log (unless resuming an existing one)

    t_start = time.perf_counter()

    # ------------------------------------------------------------
    # Phase 1: Adam
    # ------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ADAM_LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.ADAM_LR_DECAY_STEP, gamma=cfg.ADAM_LR_DECAY_GAMMA
    )

    resume_epoch = try_resume(model, optimizer, scheduler, cfg, logger)
    start_epoch = resume_epoch + 1

    best_loss = float("inf")
    patience = getattr(cfg, "EARLY_STOPPING_PATIENCE", None)
    min_delta = getattr(cfg, "EARLY_STOPPING_MIN_DELTA", 1e-6)
    epochs_since_improvement = 0

    logger.info(f"Phase 1/2: Adam — epochs {start_epoch}..{cfg.ADAM_EPOCHS}, lr={cfg.ADAM_LR}"
                + (f" | early stopping: patience={patience}, min_delta={min_delta}" if patience else
                   " | early stopping: disabled"))

    stopped_early = False
    epoch = start_epoch - 1
    for epoch in range(start_epoch, cfg.ADAM_EPOCHS + 1):
        optimizer.zero_grad()
        losses = compute_losses(model, batch, cfg.NU, cfg.LOSS_WEIGHTS)
        losses["total"].backward()

        if cfg.GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)

        q_grad_norm = _quantum_grad_norm(model)  # read BEFORE optimizer.step()

        optimizer.step()
        scheduler.step()

        total_val = losses["total"].item()
        if total_val < best_loss - min_delta:
            best_loss = total_val
            epochs_since_improvement = 0
            _save_checkpoint(model, cfg, epoch, total_val, phase="adam",
                              optimizer=optimizer, scheduler=scheduler)
        else:
            epochs_since_improvement += 1

        # Periodic safety-net checkpoint, independent of "best so far" —
        # this is what --resume loads from, so an interrupted run never
        # loses more than CHECKPOINT_EVERY epochs of progress.
        if epoch % cfg.CHECKPOINT_EVERY == 0:
            _save_checkpoint(model, cfg, epoch, total_val, phase="adam",
                              optimizer=optimizer, scheduler=scheduler,
                              path=_latest_checkpoint_path(cfg))

        if epoch % cfg.LOG_EVERY == 0 or epoch == start_epoch or epoch == cfg.ADAM_EPOCHS:
            elapsed = time.perf_counter() - t_start
            lr_now = optimizer.param_groups[0]["lr"]
            grad_str = f" q_grad_norm={q_grad_norm:.3e}" if q_grad_norm is not None else ""
            logger.info(
                f"[Adam {epoch:5d}/{cfg.ADAM_EPOCHS}] "
                f"total={total_val:.3e} pde={losses['pde'].item():.3e} "
                f"ic={losses['ic'].item():.3e} bc={losses['bc'].item():.3e} "
                f"lr={lr_now:.2e}{grad_str} t={elapsed:.1f}s"
            )
            _log_row(csv_path,
                      [epoch, "adam", total_val, losses["pde"].item(),
                       losses["ic"].item(), losses["bc"].item(), lr_now,
                       q_grad_norm if q_grad_norm is not None else "", elapsed],
                      header)

        if patience and epochs_since_improvement >= patience:
            logger.info(f"Early stopping: no improvement > {min_delta} for {patience} epochs "
                        f"(stopped at epoch {epoch}/{cfg.ADAM_EPOCHS}).")
            stopped_early = True
            break

    # Always leave a 'latest' checkpoint at the true end of Adam phase,
    # so a subsequent --resume (e.g. to run more LBFGS) has something to load.
    _save_checkpoint(model, cfg, epoch, best_loss, phase="adam",
                      optimizer=optimizer, scheduler=scheduler,
                      path=_latest_checkpoint_path(cfg))

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
                q_grad_norm = _quantum_grad_norm(model)
                grad_str = f" q_grad_norm={q_grad_norm:.3e}" if q_grad_norm is not None else ""
                logger.info(
                    f"[LBFGS {n:5d}] total={losses['total'].item():.3e} "
                    f"pde={losses['pde'].item():.3e} ic={losses['ic'].item():.3e} "
                    f"bc={losses['bc'].item():.3e}{grad_str} t={elapsed:.1f}s"
                )
                _log_row(csv_path,
                          [cfg.ADAM_EPOCHS + n, "lbfgs", losses["total"].item(),
                           losses["pde"].item(), losses["ic"].item(),
                           losses["bc"].item(), lbfgs.param_groups[0]["lr"],
                           q_grad_norm if q_grad_norm is not None else "", elapsed],
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
                f"Best total loss: {best_loss:.3e}"
                + (" [stopped early]" if stopped_early else ""))

    return {
        "adam_time_sec": adam_time,
        "lbfgs_time_sec": lbfgs_time,
        "total_time_sec": total_time,
        "best_loss": best_loss,
        "loss_history_csv": csv_path,
        "stopped_early": stopped_early,
        "final_epoch": epoch,
    }
