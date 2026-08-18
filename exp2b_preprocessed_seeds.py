# exp2b_preprocessed_seeds.py
import argparse, os
from config_a_preprocessed_107 import Config
from utils import set_seed, get_logger
from reference_solution import get_reference_solution
from sampling import build_training_tensors
from model_a_preprocessed import build_model
from train import train
from evaluate import evaluate_model
from visualize import generate_all_figures

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()

    cfg = Config
    cfg.SEED = args.seed
    cfg.MODEL_NAME = f"model_a_preprocessed_107params_seed{args.seed}"
    cfg.CHECKPOINT_PATH = os.path.join(cfg.CHECKPOINT_DIR, f"{cfg.MODEL_NAME}.pt")
    cfg.METRICS_PATH = os.path.join(cfg.METRICS_DIR, f"{cfg.MODEL_NAME}_metrics.json")
    cfg.LOSS_HISTORY_PATH = os.path.join(cfg.LOGS_DIR, f"{cfg.MODEL_NAME}_loss_history.csv")
    cfg.ensure_dirs()

    logger = get_logger(cfg.MODEL_NAME, log_path=f"{cfg.LOGS_DIR}/{cfg.MODEL_NAME}_run.log")
    set_seed(cfg.SEED)
    x_ref, t_ref, U_ref = get_reference_solution(cfg, logger=logger)
    batch = build_training_tensors(cfg)
    model = build_model(cfg)
    logger.info(f"params={model.count_parameters()}, seed={cfg.SEED}")
    stats = train(model, cfg, batch, logger=logger)
    metrics, U_pred = evaluate_model(model, cfg, x_ref, t_ref, U_ref, training_stats=stats, logger=logger)
    generate_all_figures(cfg, x_ref, t_ref, U_ref, U_pred)

if __name__ == "__main__":
    main()