# exp1_matched_seeds.py
import argparse, csv, json, os, time

from config_a_matched import Config as ConfigA85
from config_a_matched_107 import Config as ConfigA107
from utils import set_seed, get_logger
from reference_solution import get_reference_solution
from sampling import build_training_tensors
from model import build_model
from train import train
from evaluate import evaluate_model
from visualize import generate_all_figures

VARIANTS = {"85params": ConfigA85, "107params": ConfigA107}

def make_seeded_config(base_cfg, seed):
    name = f"{base_cfg.MODEL_NAME}_seed{seed}"
    attrs = {
        "SEED": seed, "MODEL_NAME": name,
        "CHECKPOINT_PATH": os.path.join(base_cfg.CHECKPOINT_DIR, f"{name}.pt"),
        "METRICS_PATH": os.path.join(base_cfg.METRICS_DIR, f"{name}_metrics.json"),
        "LOSS_HISTORY_PATH": os.path.join(base_cfg.LOGS_DIR, f"{name}_loss_history.csv"),
    }
    return type(f"Config_{name}", (base_cfg,), attrs)

def run_one(cfg, logger):
    cfg.ensure_dirs()
    set_seed(cfg.SEED)
    x_ref, t_ref, U_ref = get_reference_solution(cfg, logger=logger)
    batch = build_training_tensors(cfg)
    model = build_model(cfg)
    logger.info(f"[{cfg.MODEL_NAME}] params={model.count_parameters()}, seed={cfg.SEED}")
    stats = train(model, cfg, batch, logger=logger)
    metrics, U_pred = evaluate_model(model, cfg, x_ref, t_ref, U_ref, training_stats=stats, logger=logger)
    generate_all_figures(cfg, x_ref, t_ref, U_ref, U_pred)
    return metrics

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 99])
    args = p.parse_args()
    logger = get_logger("exp1_matched_seeds", log_path="outputs/logs/exp1_matched_seeds.log")

    results = []
    for variant, base_cfg in VARIANTS.items():
        for seed in args.seeds:
            cfg = make_seeded_config(base_cfg, seed)
            t0 = time.time()
            m = run_one(cfg, logger)
            results.append({"variant": variant, "seed": seed, "model_name": cfg.MODEL_NAME,
                             "global_l2": m["l2_relative_error"],
                             "shock_l2": m["shock_region_l2_relative_error"],
                             "fourier_l2": m["fourier_spectrum_l2_relative_error"],
                             "params": m["num_trainable_parameters"],
                             "wall_time_sec": time.time() - t0})

    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/exp1_matched_seeds_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    with open("outputs/metrics/exp1_matched_seeds_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys())); w.writeheader(); w.writerows(results)
    logger.info("Done: outputs/metrics/exp1_matched_seeds_summary.{json,csv}")

if __name__ == "__main__":
    main()