# Quantum-Assisted Physics-Informed Neural Networks for Aerospace Shock-Wave Modeling

**BQP Challenge 2026 — WISER Summer Program**

Investigating whether a quantum Fourier feature map can reduce spectral bias in
Physics-Informed Neural Networks (PINNs), using the viscous Burgers' equation
as a mathematical proxy for shock-wave phenomena in hypersonic aerospace flow.

## Project Status

- [x] **Model A** — Classical PINN baseline — complete
- [ ] **Model B** — Quantum-Assisted PINN (QAPINN) — in progress
- [ ] **Ablation studies** (qubit count, encoding, depth, entanglement) — in progress
- [ ] **Technical report** — Model A sections complete, Model B sections pending
- [ ] **Presentation slides** — pending

## Team

| Member | Role |
|---|---|
| Krishna Priya Kaku | Classical PINN & Neural Network Lead |
| Meenakshi R | Quantum & QAPINN Lead |
| Mallampati Geethika | Comparative Analysis, Evaluation & Documentation Lead |

## The Hypothesis

Classical PINN → spectral bias → poor shock prediction. Does a quantum feature
map, with richer Fourier expressivity, help represent the high-frequency shock
region of Burgers' equation better than a classical PINN? Either outcome
(yes or no) is scientifically valid — the hypothesis is falsifiable by design.

We are not building a CFD solver or a hypersonic flow simulator. Burgers'
equation is used as a standard, well-characterized benchmark that produces a
sharp shock — a simplified mathematical proxy for shock formation in
transonic/hypersonic aerospace flow.

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
model_b.py                   Model B: quantum-assisted PINN (PennyLane VQC)
pinn_losses.py                 PDE residual (autograd) + IC + BC losses — shared by A & B
train.py                        Adam -> L-BFGS training loop — shared by A & B
evaluate.py                      L2 error, shock-region error, PDE residual, Fourier metrics — shared
visualize.py                      Heatmaps, snapshots, loss curves, spectrum plots — shared
main.py                            Runs Model A end-to-end
main_b.py                           Runs Model B end-to-end
ablation_runner.py                  Runs the qubit/encoding/depth/entanglement sweeps
report/                              Technical report (Word doc)
```

Model A and Model B intentionally share every file except `model.py` /
`model_b.py` — that's what makes the A vs. B comparison a controlled
experiment rather than two unrelated implementations.

## Setup

```bash
pip install -r requirements.txt
```

## Run Model A (classical baseline)

```bash
python main.py                                                   # full run (~10-20 min CPU)
python main.py --adam_epochs 200 --lbfgs_iters 50 --collocation 1000   # quick smoke test (~30s)
```

## Run Model B (quantum-assisted) — once pushed

```bash
python main_b.py                                                 # default reduced budget (see config_b.py)
python main_b.py --qubits 6 --depth 4 --encoding amplitude --entanglement full
python ablation_runner.py                                        # full qubit/encoding/depth/entanglement sweep
```

## Results So Far — Model A Baseline

| Metric | Value |
|---|---|
| L2 relative error (global) | 4.01% |
| L2 relative error (shock region) | 11.83% |
| Fourier spectrum L2 error | 1.35% (concentrated at high frequencies — see report) |
| Trainable parameters | 16,897 |

Full metrics, figures, and analysis: `outputs/metrics/model_a_classical_pinn_metrics.json`,
`outputs/figures/`, and the technical report in `report/`.

## Full Technical Report

See [`report/QAPINN_Research_Report_Draft.docx`](report/QAPINN_Research_Report_Draft.docx)
for complete methodology, mathematical background, results, and analysis.

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
- Rahaman, N., et al. (2019). On the Spectral Bias of Neural Networks. *ICML*.
- Schuld, M., Sweke, R., & Meyer, J. J. (2021). Effect of data encoding on the expressive power of variational quantum-machine-learning models. *Physical Review A*, 103, 032430.

## Design Notes

- **LHS sampling** instead of uniform random: better coverage of the thin shock region for a fixed point budget.
- **Autograd-based PDE residual**, not finite differences: the network satisfies the PDE at *any* point, not just grid points.
- **Adam → L-BFGS**: standard two-phase PINN schedule.
- **Reference solution is independent numerical ground truth** (method-of-lines + implicit BDF), never used in training — evaluation is never contaminated by what the network was trained on.
- **Model B's training budget is deliberately reduced** from Model A's (quantum circuit simulation is more expensive per point) — this is a disclosed deviation, documented in `config_b.py` and in the report, not hidden.