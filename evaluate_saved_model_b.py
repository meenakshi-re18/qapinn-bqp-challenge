"""
evaluate_saved_model_b.py — regenerates metrics/figures for an already-
trained Model B checkpoint, without retraining. Use this if training
completed (or got far enough) but the run crashed before evaluate_model()
ran — e.g. the train.py checkpoint bug fixed alongside this script.

IMPORTANT CAVEAT — read before trusting the output:
cfg.CHECKPOINT_PATH holds the BEST loss seen during the ADAM phase only.
If your run reached L-BFGS before crashing, L-BFGS's refinement was never
persisted to disk (it only exists in the crashed process's memory, which
is gone) — so the metrics this script produces reflect Adam-only
training, not the full Adam+L-BFGS pipeline your methodology section
describes. For Model A, L-BFGS moved the loss from 2.023e-4 to 2.022e-4
(negligible) — so this is likely a fine substitute, but it is a real,
disclosable difference from "the pipeline as documented," not an exact
recovery of what the crashed run would have produced. State this
explicitly wherever these numbers are used in the report, or re-run
main_b.py (now fixed) if you want a clean Adam+LBFGS result instead.

Usage:
    python evaluate_saved_model_b.py
    python evaluate_saved_model_b.py --checkpoint outputs/checkpoints/model_b_qapinn_latest.pt
"""
import argparse
import torch

from config_b import ConfigB
from model_b import build_model
from reference_solution import get_reference_solution
from evaluate import evaluate_model
from visualize import generate_all_figures, figures_dir_for
from utils import get_logger


def main():
    p = argparse.ArgumentParser(description="Re-evaluate a saved Model B checkpoint")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint (.pt). Defaults to ConfigB.CHECKPOINT_PATH "
                         "(the best-loss checkpoint from the Adam phase).")
    args = p.parse_args()

    cfg = ConfigB
    cfg.ensure_dirs()
    logger = get_logger("evaluate_saved_model_b")

    checkpoint_path = args.checkpoint or cfg.CHECKPOINT_PATH
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=cfg.DEVICE)
    logger.info(f"Checkpoint is from epoch {ckpt['epoch']} (phase='{ckpt['phase']}'), "
                f"recorded loss={ckpt['loss']:.4e}")
    if ckpt["phase"] != "lbfgs":
        logger.warning(
            "This checkpoint is from the Adam phase, not L-BFGS. If your run reached "
            "L-BFGS before crashing, that refinement is NOT reflected in these metrics "
            "— see this script's module docstring before reporting these numbers as the "
            "final Adam+L-BFGS result."
        )

    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded model with {model.count_parameters()} trainable parameters.")

    x_ref, t_ref, U_ref = get_reference_solution(cfg, logger=logger)

    # training_stats=None here means the metrics JSON's "training" block will
    # be absent/None for time-based fields — this script doesn't know how
    # long the original (crashed) run actually took. Fill that in manually
    # in the JSON afterward if you have it from the run's console log.
    metrics, U_pred = evaluate_model(model, cfg, x_ref, t_ref, U_ref,
                                      training_stats=None, logger=logger)

    logger.info("Generating figures...")
    generate_all_figures(cfg, x_ref, t_ref, U_ref, U_pred)
    logger.info(f"Done. Metrics: {cfg.METRICS_PATH} | Figures: {figures_dir_for(cfg)}/")

if __name__ == "__main__":
    main()
