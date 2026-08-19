import os
from config_b import ConfigB

class ConfigDirect109(ConfigB):
    INPUT_MODE = "direct"
    ENCODING = "amplitude"
    N_QUBITS = 4
    CIRCUIT_DEPTH = 3
    ENTANGLEMENT = "circular"
    CLASSICAL_HEAD_LAYERS = [12, 1]   # 4->12->1 head: 73 params; +36 quantum = 109

    N_COLLOCATION = 10000
    ADAM_EPOCHS = 8000
    ADAM_LR_DECAY_STEP = 2000
    USE_LBFGS = True
    LBFGS_MAX_ITER = 2000

    MODEL_NAME = "model_b_qapinn_direct_109params"
    CHECKPOINT_PATH = os.path.join(ConfigB.CHECKPOINT_DIR, f"{MODEL_NAME}.pt")
    METRICS_PATH = os.path.join(ConfigB.METRICS_DIR, f"{MODEL_NAME}_metrics.json")
    LOSS_HISTORY_PATH = os.path.join(ConfigB.LOGS_DIR, f"{MODEL_NAME}_loss_history.csv")