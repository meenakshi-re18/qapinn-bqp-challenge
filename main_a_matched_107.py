"""
main.py — end-to-end pipeline for Model A (Classical PINN baseline).

Usage:
    python main.py
    python main.py --adam_epochs 4000 --no_lbfgs      # quick run
    python main.py --collocation 20000                # denser sampling

Pipeline:
    1. Set random seed (reproducibility)
    2. Get/compute reference (ground truth) solution
    3. Sample training points (collocation, IC, BC)
    4. Build model
    5. Train (Adam -> L-BFGS)
    6. Evaluate against reference solution
    7. Generate figures
    8. Print final summary table
"""
import argparse
import json

from config_a_matched_107 import Config #from config import Config-default
from utils import set_seed, get_logger
from reference_solution import get_reference_solution
from sampling import build_training_tensors
from model import build_model
from train import train
from evaluate import evaluate_model
from visualize import generate_all_figures, figures_dir_for


def parse_args():
    p = argparse.ArgumentParser(description="Train Model A: Classical PINN for Burgers' equation")
    p.add_argument("--adam_epochs", type=int, default=None)
    p.add_argument("--lbfgs_iters", type=int, default=None)
    p.add_argument("--no_lbfgs", action="store_true")
    p.add_argument("--collocation", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                    help="Continue from the last periodic checkpoint if one exists.")
    p.add_argument("--early_stopping_patience", type=int, default=None,
                    help="Stop Adam phase after this many epochs with no improvement. "
                         "Off by default.")
    return p.parse_args()


def apply_overrides(cfg, args):
    if args.adam_epochs is not None:
        cfg.ADAM_EPOCHS = args.adam_epochs
    if args.lbfgs_iters is not None:
        cfg.LBFGS_MAX_ITER = args.lbfgs_iters
    if args.no_lbfgs:
        cfg.USE_LBFGS = False
    if args.collocation is not None:
        cfg.N_COLLOCATION = args.collocation
    if args.seed is not None:
        cfg.SEED = args.seed
    if args.resume:
        cfg.RESUME = True
    if args.early_stopping_patience is not None:
        cfg.EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(Config, args)
    cfg.ensure_dirs()

    logger = get_logger("model_a", log_path=f"{cfg.LOGS_DIR}/{cfg.MODEL_NAME}_run.log")
    logger.info("=" * 70)
    logger.info("MODEL A — Classical PINN baseline — Burgers' Equation")
    logger.info("=" * 70)
    logger.info(f"Device: {cfg.DEVICE}")

    set_seed(cfg.SEED)

    # 1. Reference solution (ground truth, cached)
    x_ref, t_ref, U_ref = get_reference_solution(cfg, logger=logger)

    # 2. Training data
    logger.info(f"Sampling training points: {cfg.N_COLLOCATION} collocation, "
                f"{cfg.N_IC} IC, {cfg.N_BC}x2 BC "
                f"(strategy={cfg.SAMPLING_STRATEGY})")
    batch = build_training_tensors(cfg)

    # 3. Model
    model = build_model(cfg)
    logger.info(f"Model architecture: {cfg.LAYERS}, activation={cfg.ACTIVATION}")
    logger.info(f"Trainable parameters: {model.count_parameters():,}")

    # 4. Train
    training_stats = train(model, cfg, batch, logger=logger)

    # 5. Evaluate
    metrics, U_pred = evaluate_model(model, cfg, x_ref, t_ref, U_ref,
                                      training_stats=training_stats, logger=logger)

    # 6. Visualize
    logger.info("Generating figures...")
    generate_all_figures(cfg, x_ref, t_ref, U_ref, U_pred)
    logger.info(f"Figures saved to {figures_dir_for(cfg)}/")
    
    # 7. Summary
    logger.info("=" * 70)
    logger.info("SUMMARY — Model A (Classical PINN)")
    logger.info("=" * 70)
    logger.info(f"L2 relative error (global):        {metrics['l2_relative_error']:.4e}")
    logger.info(f"L2 relative error (shock region):  {metrics['shock_region_l2_relative_error']:.4e}")
    logger.info(f"Max absolute error:                {metrics['max_abs_error']:.4e}")
    logger.info(f"PDE residual RMSE:                 {metrics['pde_residual_rmse']:.4e}")
    logger.info(f"Fourier spectrum L2 error:          {metrics['fourier_spectrum_l2_relative_error']:.4e}")
    logger.info(f"Total training time:                {training_stats['total_time_sec']:.1f}s")
    logger.info(f"Trainable parameters:                {metrics['num_trainable_parameters']:,}")
    logger.info(f"Full metrics JSON: {cfg.METRICS_PATH}")
    logger.info("Model A complete. This metrics JSON is the baseline Model B must beat.")


if __name__ == "__main__":
    main()
