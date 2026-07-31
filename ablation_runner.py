"""
ablation_runner.py — runs the QAPINN ablation studies (qubit count,
encoding, circuit depth, entanglement) and aggregates every run's metrics
into a single summary table, ready to drop into the report's Section 6.4.

This directly implements the experiment list from the project plan:
    Experiment 2: qubit count       (2, 4, 6)
    Experiment 3: feature map       (angle, angle_reuploading, amplitude)
    Experiment 4: circuit depth     (1, 2, 4)
    Experiment 5: entanglement      (linear, circular, full)
Experiments 1 (A vs B), 6 (Fourier analysis), 7-9 (time/params/memory) and
the Model A baseline are already produced by main.py / main_b.py's normal
metrics output — this script only covers the four circuit-design sweeps
that require multiple distinct training runs.

Each run gets its own MODEL_NAME (via config_b.make_variant_config), so
checkpoints/metrics/logs never collide and can all be inspected afterward.

Usage:
    python ablation_runner.py                       # run everything below
    python ablation_runner.py --group qubits         # run one group only
    python ablation_runner.py --epochs 800 --collocation 1500   # slower, higher-fidelity sweep

NOTE ON COST: this runs many independent trainings back-to-back. The
per-run epoch/collocation budget defaults LOWER than main_b.py's already-
reduced default, specifically so the full sweep is tractable on a laptop
CPU in a reasonable amount of time. This is appropriate for an ablation
study (you're looking for *trends* across configurations, not squeezing
out the last decimal of accuracy on any single one) — but it must be
stated as such in the report, exactly like the Model A vs. B budget
difference (see config_b.py docstring and report Section 5.1).
"""
import argparse
import csv
import json
import os
import time

from config_b import make_variant_config
from utils import set_seed, get_logger
from reference_solution import get_reference_solution
from main_b import run


# ----------------------------------------------------------------------
# Ablation groups. Each entry overrides exactly ONE architectural axis
# relative to the baseline config, so results are attributable to that
# axis alone. Baseline: 4 qubits, depth 3, angle_reuploading, circular.
# ----------------------------------------------------------------------
BASELINE = dict(N_QUBITS=4, CIRCUIT_DEPTH=3, ENCODING="angle_reuploading", ENTANGLEMENT="circular")

GROUPS = {
    "qubits": [
        {**BASELINE, "N_QUBITS": 2},
        {**BASELINE, "N_QUBITS": 4},
        {**BASELINE, "N_QUBITS": 6},
    ],
    "encoding": [
        {**BASELINE, "ENCODING": "angle"},
        {**BASELINE, "ENCODING": "angle_reuploading"},
        {**BASELINE, "ENCODING": "amplitude"},
    ],
    "depth": [
        {**BASELINE, "CIRCUIT_DEPTH": 1},
        {**BASELINE, "CIRCUIT_DEPTH": 2},
        {**BASELINE, "CIRCUIT_DEPTH": 4},
    ],
    "entanglement": [
        {**BASELINE, "ENTANGLEMENT": "linear"},
        {**BASELINE, "ENTANGLEMENT": "circular"},
        {**BASELINE, "ENTANGLEMENT": "full"},
    ],
}


def run_name_for(variant: dict) -> str:
    return (f"nq{variant['N_QUBITS']}_d{variant['CIRCUIT_DEPTH']}_"
            f"{variant['ENCODING']}_{variant['ENTANGLEMENT']}")


def run_group(group_name: str, epochs: int, collocation: int, use_lbfgs: bool,
              logger) -> list:
    results = []
    for variant in GROUPS[group_name]:
        run_name = run_name_for(variant)
        logger.info(f"--- Ablation [{group_name}] run: {run_name} ---")

        cfg = make_variant_config(
            run_name,
            **variant,
            ADAM_EPOCHS=epochs,
            N_COLLOCATION=collocation,
            USE_LBFGS=use_lbfgs,
            LOG_EVERY=max(epochs // 5, 1),
        )
        cfg.ensure_dirs()

        t0 = time.time()
        try:
            metrics = run(cfg, logger=logger)
            wall_time = time.time() - t0
            results.append({
                "group": group_name,
                "run_name": run_name,
                **variant,
                "l2_relative_error": metrics["l2_relative_error"],
                "shock_region_l2_relative_error": metrics["shock_region_l2_relative_error"],
                "pde_residual_rmse": metrics["pde_residual_rmse"],
                "fourier_spectrum_l2_relative_error": metrics["fourier_spectrum_l2_relative_error"],
                "num_trainable_parameters": metrics["num_trainable_parameters"],
                "wall_time_sec": wall_time,
                "status": "ok",
            })
        except Exception as e:
            logger.error(f"Run {run_name} FAILED: {e}")
            results.append({
                "group": group_name, "run_name": run_name, **variant,
                "status": f"failed: {e}",
            })
    return results


def save_summary(all_results: list, out_dir: str = "outputs/metrics"):
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "ablation_summary.json")
    csv_path = os.path.join(out_dir, "ablation_summary.csv")

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if all_results:
        fieldnames = sorted({k for r in all_results for k in r.keys()})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    return json_path, csv_path


def main():
    p = argparse.ArgumentParser(description="Run QAPINN ablation studies")
    p.add_argument("--group", type=str, default="all",
                    choices=["all", "qubits", "encoding", "depth", "entanglement"])
    p.add_argument("--epochs", type=int, default=300,
                    help="Adam epochs per ablation run (kept low by default; see module docstring)")
    p.add_argument("--collocation", type=int, default=800)
    p.add_argument("--use_lbfgs", action="store_true", default=False)
    args = p.parse_args()

    logger = get_logger("ablation_runner", log_path="outputs/logs/ablation_runner.log")
    logger.info("=" * 70)
    logger.info("QAPINN ABLATION STUDIES")
    logger.info(f"Budget per run: {args.epochs} Adam epochs, {args.collocation} collocation points, "
                f"LBFGS={'on' if args.use_lbfgs else 'off'}")
    logger.info("=" * 70)

    # Pre-compute the reference solution once, up front, so every run reuses
    # the same cached ground truth instead of recomputing it N times.
    from config_b import ConfigB
    get_reference_solution(ConfigB, logger=logger)

    groups_to_run = list(GROUPS.keys()) if args.group == "all" else [args.group]

    all_results = []
    t_start = time.time()
    for group_name in groups_to_run:
        all_results.extend(run_group(group_name, args.epochs, args.collocation,
                                      args.use_lbfgs, logger))

    json_path, csv_path = save_summary(all_results)

    logger.info("=" * 70)
    logger.info(f"Ablation studies complete in {time.time() - t_start:.1f}s")
    logger.info(f"Summary JSON: {json_path}")
    logger.info(f"Summary CSV:  {csv_path}")
    logger.info("=" * 70)
    for r in all_results:
        if r.get("status") == "ok":
            logger.info(f"[{r['group']:12s}] {r['run_name']:35s} "
                        f"L2={r['l2_relative_error']:.4f}  "
                        f"shock_L2={r['shock_region_l2_relative_error']:.4f}  "
                        f"fourier={r['fourier_spectrum_l2_relative_error']:.4f}")
        else:
            logger.info(f"[{r['group']:12s}] {r['run_name']:35s} FAILED")


if __name__ == "__main__":
    main()
