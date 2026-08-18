# spectral_comparison.py
import argparse, importlib
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reference_solution import get_reference_solution
from evaluate import predict_on_grid

# label, config module, cfg class name, model module, checkpoint path
MODELS = [
    ("Classical (matched, 85p)", "config_a_matched", "Config", "model", "outputs/checkpoints/model_a_matched_85params.pt"),
    ("QAPINN direct (85p)", "config_b", "ConfigB", "model_b", "outputs/checkpoints/amplitude_matched_budget.pt"),
    ("QAPINN preprocessed (107p)", "config_b", "ConfigB", "model_b", "outputs/checkpoints/model_b_qapinn.pt"),
]

def load_model(cfg_mod, cfg_name, model_mod, ckpt_path):
    cfg = getattr(importlib.import_module(cfg_mod), cfg_name)
    build_model = importlib.import_module(model_mod).build_model
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg

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
    for label, cfg_mod, cfg_name, model_mod, ckpt_path in MODELS:
        model, cfg = load_model(cfg_mod, cfg_name, model_mod, ckpt_path)
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