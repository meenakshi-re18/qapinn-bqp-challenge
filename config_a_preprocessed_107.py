# config_a_preprocessed_107.py
import os
from config_a_matched import Config as ConfigA85Matched   # reuses the 85-param body

class Config(ConfigA85Matched):
    CLASSICAL_PREPROCESS_DIM = 4          # +22 params -> 85 + 22 = 107 total
    MODEL_NAME = "model_a_preprocessed_107params"
    CHECKPOINT_PATH = os.path.join(ConfigA85Matched.CHECKPOINT_DIR, f"{MODEL_NAME}.pt")
    METRICS_PATH = os.path.join(ConfigA85Matched.METRICS_DIR, f"{MODEL_NAME}_metrics.json")
    LOSS_HISTORY_PATH = os.path.join(ConfigA85Matched.LOGS_DIR, f"{MODEL_NAME}_loss_history.csv")