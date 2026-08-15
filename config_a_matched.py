"""
config.py — Model A: Classical PINN Baseline for Burgers' Equation
QAPINN Project (BQP Challenge)

Central configuration. Model B (quantum-assisted) should reuse this file
(or a subclass of it) so both experiments share identical physics,
sampling, optimizer, and evaluation settings — the only difference
allowed between Model A and Model B is the input encoding / feature map.
"""
import os
import math
import torch


class Config:
    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    SEED = 42

    # ------------------------------------------------------------------
    # Physical domain — Raissi et al. (2017) Burgers' benchmark
    # u_t + u*u_x - nu*u_xx = 0,  x in [-1, 1], t in [0, 1]
    # u(x, 0) = -sin(pi * x)
    # u(-1, t) = u(1, t) = 0
    # ------------------------------------------------------------------
    X_MIN, X_MAX = -1.0, 1.0
    T_MIN, T_MAX = 0.0, 1.0
    NU = 0.01 / math.pi  # viscosity -> sharp shock forms near x=0, t->1

    # ------------------------------------------------------------------
    # Training point sampling
    # ------------------------------------------------------------------
    N_COLLOCATION = 10000     # interior PDE residual (collocation) points
    N_IC = 200                # initial-condition points along t=0
    N_BC = 200                # boundary points PER boundary (x=-1 and x=1)
    SAMPLING_STRATEGY = "lhs"  # 'lhs' (Latin Hypercube) or 'random'
    RESAMPLE_EVERY = 0        # 0 = fixed points for whole training;
                               # >0 = resample collocation pts every N epochs

    # ------------------------------------------------------------------
    # Network architecture
    # ------------------------------------------------------------------
    LAYERS = [2, 7, 7, 1] #[2, 64, 64, 64, 64, 64, 1]-default  # (x,t) -> hidden... -> u
    ACTIVATION = "tanh"
    INIT = "xavier_normal"

    # ------------------------------------------------------------------
    # Training schedule
    # ------------------------------------------------------------------
    ADAM_EPOCHS = 8000
    ADAM_LR = 1e-3
    ADAM_LR_DECAY_STEP = 2000
    ADAM_LR_DECAY_GAMMA = 0.5

    USE_LBFGS = True
    LBFGS_MAX_ITER = 2000
    LBFGS_HISTORY_SIZE = 50

    LOSS_WEIGHTS = {"pde": 1.0, "ic": 1.0, "bc": 1.0}
    GRAD_CLIP_NORM = 0.0      # 0 = disabled

    LOG_EVERY = 200           # epochs between console/CSV log lines
    CHECKPOINT_EVERY = 500    # epochs between periodic (resume-safe) checkpoint saves

    RESUME = False            # if True, continue from the last periodic checkpoint if one exists
    EARLY_STOPPING_PATIENCE = None   # None/0 = disabled (default). If set, stop after this many
                                       # epochs with no improvement > EARLY_STOPPING_MIN_DELTA.
                                       # Left OFF by default — see train.py module docstring for why
                                       # a flat loss curve on a quantum model should be diagnosed
                                       # (barren plateau?) before being treated as "converged."
    EARLY_STOPPING_MIN_DELTA = 1e-6

    # ------------------------------------------------------------------
    # Reference (ground-truth) solution — high-resolution numerical solve
    # used ONLY for evaluation, never for training (keeps PINN unsupervised
    # in the interior, as intended).
    # ------------------------------------------------------------------
    REF_NX = 401
    REF_NT = 201
    REF_SOLVER_METHOD = "BDF"   # stiff ODE integrator for method-of-lines

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    SHOCK_X_WINDOW = 0.1        # |x| < this counts as "shock region"
    SHOCK_T_MIN = 0.6           # t > this counts as "shock region"

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    OUTPUT_DIR = "outputs"
    CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
    METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
    REFERENCE_DIR = os.path.join(OUTPUT_DIR, "reference")
    FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
    LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

    MODEL_NAME = "model_a_matched_85params" #"model_a_classical_pinn"-default
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}.pt")
    METRICS_PATH = os.path.join(METRICS_DIR, f"{MODEL_NAME}_metrics.json")
    LOSS_HISTORY_PATH = os.path.join(LOGS_DIR, f"{MODEL_NAME}_loss_history.csv")
    REFERENCE_CACHE_PATH = os.path.join(REFERENCE_DIR, "burgers_reference.npz")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32

    @classmethod
    def ensure_dirs(cls):
        for d in [cls.OUTPUT_DIR, cls.CHECKPOINT_DIR, cls.METRICS_DIR,
                  cls.REFERENCE_DIR, cls.FIGURES_DIR, cls.LOGS_DIR]:
            os.makedirs(d, exist_ok=True)

    @classmethod
    def as_dict(cls):
        """Serializable snapshot of hyperparameters (for reproducibility logs)."""
        out = {}
        for k, v in vars(cls).items():
            if k.startswith("_") or callable(v) or isinstance(v, classmethod):
                continue
            if isinstance(v, torch.dtype):
                v = str(v)
            out[k] = v
        return out
