"""
visualize.py — figures used to visually validate the PINN and to build
the "shock resolution" story for the report / GitHub README:

  1. Heatmap comparison: reference vs prediction vs absolute error
  2. Line-plot snapshots at fixed t (t=0.25, 0.5, 0.75, 0.99) showing
     shock formation and how well the PINN tracks it
  3. Training loss curve (log scale, all components)
  4. Fourier spectrum comparison at final time (spectral-bias diagnostic)

FIX (this version): figures are now saved into a per-model subdirectory
(outputs/figures/<MODEL_NAME>/...) instead of a single shared flat
outputs/figures/ folder. Previously every run — Model A, Model B, and
every ablation variant — wrote to the exact same 4 filenames
(heatmap_comparison.png, snapshots.png, loss_history.png,
fourier_spectrum.png), so each new run silently overwrote the previous
one's figures. Metrics/checkpoints/logs were already safe because those
paths were built from cfg.MODEL_NAME (see config.py / config_b.py) —
this brings figures in line with that same pattern, using the same
cfg.MODEL_NAME that's already unique per run (including every ablation
variant, via make_variant_config).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


def plot_heatmap_comparison(x, t, U_ref, U_pred, save_path, model_label="Model"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    extent = [x.min(), x.max(), t.min(), t.max()]

    im0 = axes[0].imshow(U_ref, extent=extent, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[0].set_title("Reference (numerical ground truth)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(U_pred, extent=extent, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[1].set_title(f"{model_label}: prediction")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    err = np.abs(U_pred - U_ref)
    im2 = axes[2].imshow(err, extent=extent, origin="lower", aspect="auto", cmap="inferno")
    axes[2].set_title("Absolute error |pred - ref|")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("t")

    fig.suptitle(f"Burgers' Equation — {model_label} vs Reference Solution", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_snapshots(x, t, U_ref, U_pred, save_path, snapshot_times=(0.25, 0.5, 0.75, 0.99), model_label="Model"):
    fig, axes = plt.subplots(1, len(snapshot_times), figsize=(4.2 * len(snapshot_times), 3.6),
                              sharey=True)
    if len(snapshot_times) == 1:
        axes = [axes]

    for ax, t_target in zip(axes, snapshot_times):
        idx = np.argmin(np.abs(t - t_target))
        ax.plot(x, U_ref[idx], "k-", linewidth=2, label="Reference")
        ax.plot(x, U_pred[idx], "r--", linewidth=2, label=model_label)
        ax.set_title(f"t = {t[idx]:.2f}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("u(x,t)")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Solution snapshots — shock formation over time", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_loss_history(csv_path, save_path, model_label="Model"):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(df["epoch"], df["total"], label="Total", linewidth=2)
    ax.semilogy(df["epoch"], df["pde"], label="PDE residual", alpha=0.8)
    ax.semilogy(df["epoch"], df["ic"], label="IC", alpha=0.8)
    ax.semilogy(df["epoch"], df["bc"], label="BC", alpha=0.8)

    if "phase" in df.columns and (df["phase"] == "lbfgs").any():
        transition_epoch = df.loc[df["phase"] == "lbfgs", "epoch"].min()
        ax.axvline(transition_epoch, color="gray", linestyle=":", label="Adam -> L-BFGS")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title(f"Training loss history — {model_label}")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_fourier_spectrum(x, U_ref, U_pred, save_path, t_index=-1, model_label="Model"):
    u_ref_final = U_ref[t_index]
    u_pred_final = U_pred[t_index]

    n = len(x)
    freqs = np.fft.rfftfreq(n, d=(x[1] - x[0]))
    spec_ref = np.abs(np.fft.rfft(u_ref_final))
    spec_pred = np.abs(np.fft.rfft(u_pred_final))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(freqs[1:], spec_ref[1:], "k-", linewidth=2, label="Reference")
    ax.loglog(freqs[1:], spec_pred[1:], "r--", linewidth=2, label=model_label)
    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel("|Fourier amplitude|")
    ax.set_title("Fourier spectrum recovery near shock (final time slice)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


# Maps a config's MODEL_NAME to a clean display label, so every plot
# title/legend automatically shows the right model without main.py or
# main_b.py needing to pass anything extra.
_MODEL_LABELS = {
    "model_a_classical_pinn": "Model A (Classical PINN)",
    "model_b_qapinn": "Model B (QAPINN)",
}


def _infer_model_label(cfg):
    name = getattr(cfg, "MODEL_NAME", "")
    if name in _MODEL_LABELS:
        return _MODEL_LABELS[name]
    if name.startswith("model_b_qapinn"):   # ablation-runner variant configs
        return f"QAPINN ({name.replace('model_b_qapinn_', '')})"
    return name or "Model"


def figures_dir_for(cfg):
    """
    Per-model figures subdirectory: outputs/figures/<MODEL_NAME>/
    Shared by generate_all_figures() and by main.py/main_b.py/
    evaluate_saved_model_b.py so log messages report the real path.
    """
    return os.path.join(cfg.FIGURES_DIR, cfg.MODEL_NAME)


def generate_all_figures(cfg, x_ref, t_ref, U_ref, U_pred, model_label=None):
    fig_dir = figures_dir_for(cfg)
    os.makedirs(fig_dir, exist_ok=True)
    model_label = model_label or _infer_model_label(cfg)
    plot_heatmap_comparison(x_ref, t_ref, U_ref, U_pred,
                             os.path.join(fig_dir, "heatmap_comparison.png"),
                             model_label=model_label)
    plot_snapshots(x_ref, t_ref, U_ref, U_pred,
                   os.path.join(fig_dir, "snapshots.png"),
                   model_label=model_label)
    plot_loss_history(cfg.LOSS_HISTORY_PATH,
                       os.path.join(fig_dir, "loss_history.png"),
                       model_label=model_label)
    plot_fourier_spectrum(x_ref, U_ref, U_pred,
                          os.path.join(fig_dir, "fourier_spectrum.png"),
                          model_label=model_label)
    return fig_dir
