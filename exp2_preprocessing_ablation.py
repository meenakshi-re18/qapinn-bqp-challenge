# exp2_preprocessing_ablation.py
import argparse, json, os
from config_a_preprocessed_107 import Config
from utils import set_seed, get_logger
from reference_solution import get_reference_solution
from sampling import build_training_tensors
from model_a_preprocessed import build_model
from train import train
from evaluate import evaluate_model
from visualize import generate_all_figures

# Fill these in with your actual existing metrics filenames:
EXISTING = {
    "A_classical_no_preproc_85": "outputs/metrics/model_a_matched_85params_metrics.json",
    "C_quantum_no_preproc_85":  "outputs/metrics/amplitude_matched_budget_metrics.json",
    "D_quantum_preproc_107":    "outputs/metrics/preprocessed_amplitude_metrics.json",
}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    logger = get_logger("exp2", log_path="outputs/logs/exp2_preprocessing_ablation.log")

    cfg = Config; cfg.SEED = args.seed; cfg.ensure_dirs()
    set_seed(cfg.SEED)
    x_ref, t_ref, U_ref = get_reference_solution(cfg, logger=logger)
    batch = build_training_tensors(cfg)
    model = build_model(cfg)
    logger.info(f"Cell B params: {model.count_parameters()} (target: 107)")
    stats = train(model, cfg, batch, logger=logger)
    m_b, U_pred = evaluate_model(model, cfg, x_ref, t_ref, U_ref, training_stats=stats, logger=logger)
    generate_all_figures(cfg, x_ref, t_ref, U_ref, U_pred)

    table = {"B_classical_preproc_107": {k: m_b[k] for k in
             ("l2_relative_error","shock_region_l2_relative_error",
              "fourier_spectrum_l2_relative_error","num_trainable_parameters")}}
    for label, path in EXISTING.items():
        if os.path.exists(path):
            with open(path) as f: table[label] = json.load(f)
        else:
            logger.warning(f"{label}: not found at {path} — fix path or fill in manually")

    with open("outputs/metrics/exp2_2x2_summary.json", "w") as f:
        json.dump(table, f, indent=2)
    logger.info("Saved outputs/metrics/exp2_2x2_summary.json")

if __name__ == "__main__":
    main()