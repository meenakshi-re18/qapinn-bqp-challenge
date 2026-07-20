"""
visualize.py — figures used to visually validate the PINN and to build
the "shock resolution" story for the report / GitHub README:

  1. Heatmap comparison: reference vs prediction vs absolute error
  2. Line-plot snapshots at fixed t (t=0.25, 0.5, 0.75, 0.99) showing
     shock formation and how well the PINN tracks it
  3. Training loss curve (log scale, all components)
  4. Fourier spectrum comparison at final time (spectral-bias diagnostic)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt


def plot_heatmap_comparison(x, t, U_ref, U_pred, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharey=True)
    extent = [x.min(), x.max(), t.min(), t.max()]

    im0 = axes[0].imshow(U_ref, extent=extent, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[0].set_title("Reference (numerical ground truth)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(U_pred, extent=extent, origin="lower", aspect="auto", cmap="RdBu_r")
    axes[1].set_title("Model A: Classical PINN prediction")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    err = np.abs(U_pred - U_ref)
    im2 = axes[2].imshow(err, extent=extent, origin="lower", aspect="auto", cmap="inferno")
    axes[2].set_title("Absolute error |pred - ref|")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("t")

    fig.suptitle("Burgers' Equation — Model A vs Reference Solution", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_snapshots(x, t, U_ref, U_pred, save_path, snapshot_times=(0.25, 0.5, 0.75, 0.99)):
    fig, axes = plt.subplots(1, len(snapshot_times), figsize=(4.2 * len(snapshot_times), 3.6),
                              sharey=True)
    if len(snapshot_times) == 1:
        axes = [axes]

    for ax, t_target in zip(axes, snapshot_times):
        idx = np.argmin(np.abs(t - t_target))
        ax.plot(x, U_ref[idx], "k-", linewidth=2, label="Reference")
        ax.plot(x, U_pred[idx], "r--", linewidth=2, label="Model A (PINN)")
        ax.set_title(f"t = {t[idx]:.2f}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("u(x,t)")
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Solution snapshots — shock formation over time", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_loss_history(csv_path, save_path):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(df["epoch"], df["total"], label="Total", linewidth=2)
    ax.semilogy(df["epoch"], df["pde"], label="PDE residual", alpha=0.8)
    ax.semilogy(df["epoch"], df["ic"], label="IC", alpha=0.8)
    ax.semilogy(df["epoch"], df["bc"], label="BC", alpha=0.8)

    # mark the Adam -> LBFGS transition, if present
    if "phase" in df.columns and (df["phase"] == "lbfgs").any():
        transition_epoch = df.loc[df["phase"] == "lbfgs", "epoch"].min()
        ax.axvline(transition_epoch, color="gray", linestyle=":", label="Adam -> L-BFGS")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training loss history — Model A (Classical PINN)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_fourier_spectrum(x, U_ref, U_pred, save_path, t_index=-1):
    u_ref_final = U_ref[t_index]
    u_pred_final = U_pred[t_index]

    n = len(x)
    freqs = np.fft.rfftfreq(n, d=(x[1] - x[0]))
    spec_ref = np.abs(np.fft.rfft(u_ref_final))
    spec_pred = np.abs(np.fft.rfft(u_pred_final))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(freqs[1:], spec_ref[1:], "k-", linewidth=2, label="Reference")
    ax.loglog(freqs[1:], spec_pred[1:], "r--", linewidth=2, label="Model A (PINN)")
    ax.set_xlabel("Spatial frequency")
    ax.set_ylabel("|Fourier amplitude|")
    ax.set_title("Fourier spectrum recovery near shock (final time slice)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def generate_all_figures(cfg, x_ref, t_ref, U_ref, U_pred):
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)
    plot_heatmap_comparison(x_ref, t_ref, U_ref, U_pred,
                             os.path.join(cfg.FIGURES_DIR, "heatmap_comparison.png"))
    plot_snapshots(x_ref, t_ref, U_ref, U_pred,
                   os.path.join(cfg.FIGURES_DIR, "snapshots.png"))
    plot_loss_history(cfg.LOSS_HISTORY_PATH,
                       os.path.join(cfg.FIGURES_DIR, "loss_history.png"))
    plot_fourier_spectrum(x_ref, U_ref, U_pred,
                          os.path.join(cfg.FIGURES_DIR, "fourier_spectrum.png"))
