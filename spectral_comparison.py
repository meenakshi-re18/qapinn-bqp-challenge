# spectral_comparison.py
import argparse
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reference_solution import get_reference_solution
from evaluate import predict_on_grid
from config_a_matched import Config as ConfigA85Matched
from config_b import make_variant_config
from config_b_direct_109 import ConfigDirect109
import model as model_a_mod
import model_b as model_b_mod

# Post-hoc visualization fix only -- no retraining, no architecture change.
# These two configs reconstruct the EXACT settings each checkpoint was
# trained with (amplitude encoding, correct INPUT_MODE), which the
# previous version of this script got wrong for these two curves.

CfgDirect85 = make_variant_config(
    "spectral_eval_direct85",
    ENCODING="amplitude", N_QUBITS=4, CIRCUIT_DEPTH=3,
    ENTANGLEMENT="circular", INPUT_MODE="direct",
)  # CLASSICAL_HEAD_LAYERS=[8,1] inherited from ConfigB -> 36 + 49 = 85 params

CfgPreproc107 = make_variant_config(
    "spectral_eval_preproc107",
    ENCODING="amplitude", N_QUBITS=4, CIRCUIT_DEPTH=3,
    ENTANGLEMENT="circular", INPUT_MODE="preprocessed",
)  # CLASSICAL_PREPROCESS_DIM=4, CLASSICAL_HEAD_LAYERS=[8,1] inherited -> 36+22+49=107

MODELS = [
    ("Classical (matched, 85p)", ConfigA85Matched, model_a_mod,
     "outputs/checkpoints/model_a_matched_85params.pt"),

    ("QAPINN direct (85p)", CfgDirect85, model_b_mod,
     "outputs/checkpoints/amplitude_matched_budget.pt"),

    ("QAPINN direct (109p)", ConfigDirect109, model_b_mod,
     "outputs/checkpoints/model_b_qapinn_direct_109params_seed42.pt"),

    ("QAPINN preprocessed (107p)", CfgPreproc107, model_b_mod,
     "outputs/checkpoints/preprocessed_amplitude.pt"),
]

def load_model(cfg, model_mod, ckpt_path):
    model = model_mod.build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model.count_parameters()

def main():
    p = argparse.ArgumentParser(); p.add_argument("--t_index", type=int, default=-1)
    args = p.parse_args()

    from config import Config as BaseConfig
    x_ref, t_ref, U_ref = get_reference_solution(BaseConfig)
    n = len(x_ref)
    freqs = np.fft.rfftfreq(n, d=(x_ref[1] - x_ref[0]))
    spec_ref = np.abs(np.fft.rfft(U_ref[args.t_index]))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(freqs[1:], spec_ref[1:], "k-", linewidth=2.5, label="Reference")
    for label, cfg, model_mod, ckpt_path in MODELS:
        model, n_params = load_model(cfg, model_mod, ckpt_path)
        print(f"{label}: {ckpt_path} -> {n_params} params "
              f"(encoding={getattr(cfg,'ENCODING','n/a')}, "
              f"input_mode={getattr(cfg,'INPUT_MODE','n/a')}, "
              f"qubits={getattr(cfg,'N_QUBITS','n/a')}, "
              f"depth={getattr(cfg,'CIRCUIT_DEPTH','n/a')}, "
              f"entanglement={getattr(cfg,'ENTANGLEMENT','n/a')})")
        U_pred = predict_on_grid(model, x_ref, t_ref, cfg.DEVICE, cfg.DTYPE)
        spec_pred = np.abs(np.fft.rfft(U_pred[args.t_index]))
        ax.loglog(freqs[1:], spec_pred[1:], "--", linewidth=1.8, label=label)

    ax.set_xlabel("Spatial frequency"); ax.set_ylabel("|Fourier amplitude|")
    ax.set_title(f"Spectral recovery comparison (t = {t_ref[args.t_index]:.2f})")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("outputs/figures/spectral_comparison_all_models.png", dpi=180)
    print("Saved outputs/figures/spectral_comparison_all_models.png")

if __name__ == "__main__":
    main()