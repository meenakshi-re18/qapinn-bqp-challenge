# Quantum-Assisted Physics-Informed Neural Networks for Aerospace Shock-Wave Modeling

**BQP Challenge 2026 — WISER Summer Program**

An empirical investigation of how quantum feature-map design influences learning behavior
and frequency representation in Quantum-Assisted Physics-Informed Neural Networks (QAPINNs),
using the viscous Burgers' equation as a mathematical proxy for shock-wave phenomena in
hypersonic aerospace flow.

> Working title under revision — see `report/` for the current framing. The project's
> contribution shifted over the course of the work from "is QAPINN better than a classical
> PINN" toward "which architectural components of a QAPINN actually influence performance,
> and how do representation, optimization, and physics constraints interact."

## Project Status

- [x] **Model A** — Classical PINN baseline — complete
- [x] **Model B** — Quantum-Assisted PINN (QAPINN) baseline — complete
- [x] **Ablation studies** (qubit count, encoding, depth, entanglement) — complete; encoding is the only axis with a clean (unconfounded) signal — see report Section 6.4/8.1
- [x] **Amplitude encoding + L-BFGS follow-up** — complete; substantial improvement over the baseline configuration, including in the shock region
- [ ] **Matched-budget validation** (amplitude encoding at Model A's exact training budget) — in progress
- [ ] **Collocation-density sensitivity sweep** — in progress
- [ ] **Technical report** — Model A and Model B core sections complete; Discussion being restructured around explicit hypotheses; Abstract/Conclusion pending final experiments
- [ ] **Presentation slides** — pending

## Team

| Member | Role |
|---|---|
| Krishna Priya Kaku | Classical PINN & Neural Network Lead |
| Meenakshi R | Quantum & QAPINN Lead |
| Mallampati Geethika | Comparative Analysis, Evaluation & Documentation Lead |

## The Research Question

Does a quantum feature map, with richer Fourier expressivity, help a PINN capture the sharp
shock in Burgers' equation better than a classical PINN? We treated this as genuinely
falsifiable rather than assumed — and the honest answer, based on results so far, is more
nuanced than yes/no: the evaluated QAPINN configurations did not match the classical
baseline's accuracy, but controlled experiments identified *which* architectural choices
meaningfully affect performance (encoding, optimizer phase) and which largely don't (qubit
count, circuit depth, entanglement pattern) — see the report's Discussion for the full,
hypothesis-by-hypothesis breakdown.

We are not building a CFD solver or a hypersonic flow simulator. Burgers' equation is a
standard, well-characterized benchmark that produces a sharp shock — a simplified
mathematical proxy for shock formation in transonic/hypersonic aerospace flow.

## Problem

Viscous Burgers' equation (Raissi et al., 2019 benchmark):

```
u_t + u·u_x - ν·u_xx = 0,      x ∈ [-1, 1], t ∈ [0, 1]
u(x, 0) = -sin(πx)
u(-1, t) = u(1, t) = 0
ν = 0.01/π
```

## Repository Structure

```
config.py               Model A hyperparameters (domain, network, training, paths)
config_b.py              Model B hyperparameters — subclasses config.py (quantum circuit settings)
utils.py                  Seeding + logging helpers
reference_solution.py     High-res numerical ground truth (method-of-lines, BDF) — shared by A & B
sampling.py                LHS collocation / IC / BC point generation — shared by A & B
model.py                    Model A: classical MLP PINN
model_b.py                   Model B: quantum-assisted PINN (PennyLane VQC; angle, angle re-uploading,
                              and amplitude encoding; linear/circular/full entanglement)
pinn_losses.py                 PDE residual (autograd) + IC + BC losses — shared by A & B
train.py                        Adam -> L-BFGS training loop, resume support, periodic checkpointing,
                                 quantum gradient-norm diagnostics — shared by A & B
evaluate.py                      L2 error, shock-region error, PDE residual, Fourier metrics — shared
visualize.py                      Heatmaps, snapshots, loss curves, spectrum plots — saved per-model
                                   to outputs/figures/<MODEL_NAME>/
main.py                            Runs Model A end-to-end
main_b.py                           Runs Model B end-to-end; supports --run_name for one-off
                                     experiments so they don't overwrite the baseline's output files
evaluate_saved_model_b.py           Re-evaluates a saved Model B checkpoint without retraining
ablation_runner.py                  Runs the qubit/encoding/depth/entanglement sweeps
report/                              Technical report (Word doc)
```

Model A and Model B intentionally share every file except `model.py` / `model_b.py` — that's
what makes the A vs. B comparison a controlled experiment rather than two unrelated
implementations.

## Setup

```bash
pip install -r requirements.txt
```

## Run Model A (classical baseline)

```bash
python main.py                                                   # full run (~10-20 min CPU)
python main.py --adam_epochs 200 --lbfgs_iters 50 --collocation 1000   # quick smoke test (~30s)
```

## Run Model B (quantum-assisted)

```bash
python main_b.py                                                 # official baseline (reduced budget, see config_b.py)
python main_b.py --qubits 6 --depth 4 --encoding amplitude --entanglement full
python main_b.py --match_model_a                                 # Model A's exact training budget (slow)
python ablation_runner.py                                        # full qubit/encoding/depth/entanglement sweep
python evaluate_saved_model_b.py                                 # re-evaluate a saved checkpoint without retraining
```

**Always pass `--run_name <something>` for any run that isn't meant to replace the official
baseline** — e.g. `python main_b.py --encoding amplitude --run_name amplitude_followup`.
Without it, a new run silently overwrites the previous one's checkpoint, metrics, and logs,
since they all share the same default filenames.

## Results So Far

The classical baseline (Model A) substantially outperforms the initial Model B configuration.
Controlled ablations found that **encoding choice** is the architectural factor with the
largest, cleanest observed effect — qubit count, circuit depth, and entanglement pattern
showed comparatively little influence. A follow-up experiment replacing the baseline's
encoding with **amplitude encoding** produced a large improvement, including in the
shock region specifically (the hardest part of the domain), using the same 85 trainable
parameters as every other Model B configuration — i.e. the improvement came from a better
representation, not a larger model.

| Metric | Model A | Model B (baseline config) | Model B (amplitude + L-BFGS, reduced budget) |
|---|---|---|---|
| L2 relative error (global) | 4.01% | 98.36% | 55.47% |
| L2 relative error (shock region) | 11.83% | 99.85% | 89.12% |
| Fourier spectrum L2 error | 1.35% | 88.90% | 53.99% |
| Trainable parameters | 16,897 | 85 | 85 |

A matched-training-budget validation of the amplitude-encoding result (against Model A's
exact collocation/epoch budget) is in progress — see Project Status above.

**Known gap**: the original Model B baseline configuration's raw checkpoint file was
overwritten by a later run before being separately named; its full results remain
documented in the report (metrics, tables, figures) but are not independently
re-verifiable from the current checkpoint file alone. Re-running `python main_b.py`
with no arguments (same seed, same code) should reproduce it closely.

Full metrics, figures, and analysis: `outputs/metrics/`, `outputs/figures/`, and the
technical report in `report/`.

## Full Technical Report

See [`report/QAPINN_Research_Report_Draft.docx`](report/QAPINN_Research_Report_Draft.docx)
for complete methodology, mathematical background, results, and analysis.

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
- Rahaman, N., et al. (2019). On the Spectral Bias of Neural Networks. *ICML*.
- Schuld, M., Sweke, R., & Meyer, J. J. (2021). Effect of data encoding on the expressive power of variational quantum-machine-learning models. *Physical Review A*, 103, 032430.
- Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055–A3081.

## Design Notes

- **LHS sampling** instead of uniform random: better coverage of the thin shock region for a fixed point budget.
- **Autograd-based PDE residual**, not finite differences: the network satisfies the PDE at *any* point, not just grid points.
- **Adam → L-BFGS**: standard two-phase PINN schedule. Repeatedly found to matter more for Model B than for Model A — L-BFGS consistently recovered accuracy Adam alone couldn't reach, most dramatically when paired with amplitude encoding.
- **Reference solution is independent numerical ground truth** (method-of-lines + implicit BDF), never used in training — evaluation is never contaminated by what the network was trained on.
- **Model B's default training budget is deliberately reduced** from Model A's (quantum circuit simulation is more expensive per point) — a disclosed deviation, documented in `config_b.py` and the report, being directly tested via the matched-budget experiment above rather than left as an assumption.
- **Every output file is named from `cfg.MODEL_NAME`** (checkpoints, metrics, logs, and — as of this update — figures, saved to their own subfolder) so that no two runs collide by default.