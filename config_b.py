"""
config_b.py — Model B: Quantum-Assisted PINN (QAPINN) configuration.

Subclasses Config so that physics, sampling defaults, optimizer schedule,
and evaluation protocol are inherited unchanged from Model A (config.py) —
per the project's controlled A/B design (report Section 5.1). Only
quantum-circuit-specific settings are added here, plus a reduced default
training budget (see NOTE below).

NOTE on the reduced training budget — a disclosed, deliberate deviation:
Quantum circuit simulation is substantially more expensive per sample
than a classical MLP of comparable size: every PDE-residual evaluation
differentiates through a simulated parameterized circuit, twice (for
u_xx), for every collocation point, every epoch. Model B's defaults below
(N_COLLOCATION, ADAM_EPOCHS) are therefore reduced from Model A's. This
must be stated explicitly in the report (Section 5.1) rather than left
implicit — an undisclosed difference in training budget would confound
any claim about the effect of the quantum encoding itself. Increase these
back to Model A's exact values via CLI flags if your hardware allows it.
"""
import os
from config import Config


class ConfigB(Config):
    # ------------------------------------------------------------------
    # Quantum circuit architecture
    # ------------------------------------------------------------------
    N_QUBITS = 4
    CIRCUIT_DEPTH = 3                  # number of variational layers
    ENCODING = "angle_reuploading"     # 'angle' | 'angle_reuploading' | 'amplitude'
    ENTANGLEMENT = "circular"          # 'linear' | 'circular' | 'full'
    QUANTUM_DEVICE = "default.qubit"   # PennyLane simulator device
    DIFF_METHOD = "backprop"           # fast + supports the 2nd derivatives
                                        # the PDE residual needs; use
                                        # 'parameter-shift' only if targeting
                                        # real hardware (much slower, and
                                        # 2nd-order derivatives need extra care)

    # ------------------------------------------------------------------
    # Hybrid architecture: where the quantum layer sits (see report 5.3
    # for the open design decision this resolves)
    # ------------------------------------------------------------------
    INPUT_MODE = "direct"              # 'direct': quantum layer replaces the classical input layer entirely
                                        # 'preprocessed': a small classical layer runs before the quantum layer
    CLASSICAL_PREPROCESS_DIM = 4       # only used if INPUT_MODE == 'preprocessed'
    CLASSICAL_HEAD_LAYERS = [8, 1]     # classical readout: n_qubits -> 8 -> 1

    # ------------------------------------------------------------------
    # Reduced default training budget — see module docstring
    # ------------------------------------------------------------------
    N_COLLOCATION = 4000
    ADAM_EPOCHS = 3000
    ADAM_LR_DECAY_STEP = 750
    USE_LBFGS = True
    LBFGS_MAX_ITER = 300
    LOG_EVERY = 100

    # ------------------------------------------------------------------
    # Paths (distinct from Model A so nothing overwrites the baseline)
    # ------------------------------------------------------------------
    MODEL_NAME = "model_b_qapinn"
    CHECKPOINT_PATH = os.path.join(Config.CHECKPOINT_DIR, f"{MODEL_NAME}.pt")
    METRICS_PATH = os.path.join(Config.METRICS_DIR, f"{MODEL_NAME}_metrics.json")
    LOSS_HISTORY_PATH = os.path.join(Config.LOGS_DIR, f"{MODEL_NAME}_loss_history.csv")


def make_variant_config(run_name: str, **overrides) -> type:
    """
    Factory for ablation-study configs: returns a fresh ConfigB subclass
    with `run_name` baked into all output paths (so multiple ablation
    runs never overwrite each other's checkpoints/metrics/logs), plus
    any hyperparameter overrides applied.

    Example:
        cfg = make_variant_config("qubits4_angle_circular", N_QUBITS=4)
    """
    attrs = dict(overrides)
    model_name = f"model_b_qapinn_{run_name}"
    attrs["MODEL_NAME"] = model_name
    attrs["CHECKPOINT_PATH"] = os.path.join(Config.CHECKPOINT_DIR, f"{model_name}.pt")
    attrs["METRICS_PATH"] = os.path.join(Config.METRICS_DIR, f"{model_name}_metrics.json")
    attrs["LOSS_HISTORY_PATH"] = os.path.join(Config.LOGS_DIR, f"{model_name}_loss_history.csv")
    return type(f"ConfigB_{run_name}", (ConfigB,), attrs)
