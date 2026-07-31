"""
main_b.py — end-to-end pipeline for Model B (Quantum-Assisted PINN).

Mirrors main.py exactly in structure — the only functional difference is
importing model_b/config_b instead of model/config — per the project's
controlled A/B design (report Section 5.1).

Usage:
    python main_b.py
    python main_b.py --adam_epochs 500 --no_lbfgs --collocation 500   # quick run
    python main_b.py --qubits 6 --depth 4 --encoding amplitude --entanglement full
    python main_b.py --match_model_a   # use Model A's exact training budget
                                          (10000 collocation pts, 8000 Adam
                                          epochs) instead of Model B's reduced
                                          default — see config_b.py docstring
"""
import argparse
import os

from config_b import ConfigB
from utils import set_seed, get_logger
from reference_solution import get_reference_solution
from sampling import build_training_tensors
from model_b import build_model
from train import train
from evaluate import evaluate_model
from visualize import generate_all_figures, figures_dir_for

def parse_args():
    p = argparse.ArgumentParser(description="Train Model B: QAPINN for Burgers' equation")
    p.add_argument("--adam_epochs", type=int, default=None)
    p.add_argument("--lbfgs_iters", type=int, default=None)
    p.add_argument("--no_lbfgs", action="store_true")
    p.add_argument("--collocation", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--qubits", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--encoding", type=str, default=None,
                    choices=["angle", "angle_reuploading", "amplitude"])
    p.add_argument("--entanglement", type=str, default=None,
                    choices=["linear", "circular", "full"])
    p.add_argument("--input_mode", type=str, default=None,
                    choices=["direct", "preprocessed"])
    p.add_argument("--match_model_a", action="store_true",
                    help="Use Model A's exact training budget instead of "
                         "Model B's reduced default (much slower).")
    p.add_argument("--resume", action="store_true",
                    help="Continue from the last periodic checkpoint if one exists.")
    p.add_argument("--early_stopping_patience", type=int, default=None,
                    help="Stop Adam phase after this many epochs with no improvement. "
                         "Off by default — see train.py docstring on barren plateaus "
                         "before enabling this for Model B.")
    p.add_argument("--run_name", type=str, default=None, help="Unique name for this experiment.")
    
    return p.parse_args()


def apply_overrides(cfg, args):
    if args.match_model_a:
        cfg.N_COLLOCATION = 10000
        cfg.ADAM_EPOCHS = 8000
        cfg.ADAM_LR_DECAY_STEP = 2000
        cfg.LBFGS_MAX_ITER = 2000
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
    if args.qubits is not None:
        cfg.N_QUBITS = args.qubits
    if args.depth is not None:
        cfg.CIRCUIT_DEPTH = args.depth
    if args.encoding is not None:
        cfg.ENCODING = args.encoding
    if args.entanglement is not None:
        cfg.ENTANGLEMENT = args.entanglement
    if args.input_mode is not None:
        cfg.INPUT_MODE = args.input_mode
    if args.resume:
        cfg.RESUME = True
    if args.early_stopping_patience is not None:
        cfg.EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    if args.run_name is not None:
        cfg.MODEL_NAME = args.run_name
        cfg.CHECKPOINT_PATH = os.path.join(cfg.CHECKPOINT_DIR, f"{cfg.MODEL_NAME}.pt")
        cfg.METRICS_PATH = os.path.join(cfg.METRICS_DIR, f"{cfg.MODEL_NAME}_metrics.json")
        cfg.LOSS_HISTORY_PATH = os.path.join(cfg.LOGS_DIR, f"{cfg.MODEL_NAME}_loss_history.csv")

    return cfg


def run(cfg, logger=None):
    """Runs the full Model B pipeline for a given config. Returns metrics dict.
    Factored out so ablation_runner.py can call this directly per variant."""
    logger = logger or get_logger(cfg.MODEL_NAME, log_path=f"{cfg.LOGS_DIR}/{cfg.MODEL_NAME}_run.log")
    cfg.ensure_dirs()

    logger.info(f"Device: {cfg.DEVICE}")
    logger.info(f"Quantum config: {cfg.N_QUBITS} qubits, depth={cfg.CIRCUIT_DEPTH}, "
                f"encoding={cfg.ENCODING}, entanglement={cfg.ENTANGLEMENT}, "
                f"input_mode={cfg.INPUT_MODE}")

    set_seed(cfg.SEED)

    x_ref, t_ref, U_ref = get_reference_solution(cfg, logger=logger)

    logger.info(f"Sampling training points: {cfg.N_COLLOCATION} collocation, "
                f"{cfg.N_IC} IC, {cfg.N_BC}x2 BC")
    batch = build_training_tensors(cfg)

    model = build_model(cfg)
    logger.info(f"Trainable parameters: {model.count_parameters():,}")

    training_stats = train(model, cfg, batch, logger=logger)

    metrics, U_pred = evaluate_model(model, cfg, x_ref, t_ref, U_ref,
                                      training_stats=training_stats, logger=logger)

    logger.info("Generating figures...")
    generate_all_figures(cfg, x_ref, t_ref, U_ref, U_pred)
    logger.info(f"Figures saved to {figures_dir_for(cfg)}/")

    return metrics


def main():
    args = parse_args()
    cfg = apply_overrides(ConfigB, args)
    cfg.ensure_dirs()

    logger = get_logger("model_b", log_path=f"{cfg.LOGS_DIR}/{cfg.MODEL_NAME}_run.log")
    logger.info("=" * 70)
    logger.info("MODEL B — Quantum-Assisted PINN (QAPINN) — Burgers' Equation")
    logger.info("=" * 70)

    metrics = run(cfg, logger=logger)

    logger.info("=" * 70)
    logger.info("SUMMARY — Model B (QAPINN)")
    logger.info("=" * 70)
    logger.info(f"L2 relative error (global):        {metrics['l2_relative_error']:.4e}")
    logger.info(f"L2 relative error (shock region):  {metrics['shock_region_l2_relative_error']:.4e}")
    logger.info(f"Max absolute error:                {metrics['max_abs_error']:.4e}")
    logger.info(f"PDE residual RMSE:                 {metrics['pde_residual_rmse']:.4e}")
    logger.info(f"Fourier spectrum L2 error:          {metrics['fourier_spectrum_l2_relative_error']:.4e}")
    logger.info(f"Trainable parameters:                {metrics['num_trainable_parameters']:,}")
    logger.info(f"Full metrics JSON: {cfg.METRICS_PATH}")
    logger.info("Compare this against outputs/metrics/model_a_classical_pinn_metrics.json")


if __name__ == "__main__":
    main()
