# Quantum-Assisted Physics-Informed Neural Networks for Aerospace Shock-Wave Modeling

## A Mechanistic Investigation of Representation, Optimization, and Spectral Bias

**BQP Challenge 2026 – WISER Summer Program**

---

## Overview

This repository contains the complete implementation, experiments, technical report, and presentation for our submission to the **BQP WISER Summer Challenge 2026**.

The project investigates whether a **Quantum-Assisted Physics-Informed Neural Network (QAPINN)** can reduce spectral bias and improve shock-region accuracy on the viscous Burgers' equation compared to a classical Physics-Informed Neural Network (PINN).

Rather than attempting to demonstrate an unconditional quantum advantage, this work systematically studies **which architectural and optimization choices actually influence QAPINN performance** through controlled experiments.

---

## Project Status

- [x] Classical PINN (Model A) implementation
- [x] Quantum-Assisted PINN (Model B) implementation
- [x] Controlled architecture comparison
- [x] Ablation study (encoding, qubits, depth, entanglement)
- [x] Optimizer × representation analysis
- [x] Collocation-density sensitivity study
- [x] Matched-budget validation
- [x] Final technical report
- [x] Final presentation
- [x] Repository prepared for BQP WISER Summer Challenge 2026 submission

---

## Team

| Author | Role |
|---------|------|
| **Meenakshi R.** | Quantum & QAPINN Research Lead |
| **Krishna Priya Kaku** | Classical PINN & Numerical Modelling Lead |
| **Mallampati Geethika** | Comparative Analysis & Documentation Lead |

---

## The Research Question

> Can a quantum feature map, used as the input representation of a Physics-Informed Neural Network, reduce spectral bias and improve shock-region accuracy on the viscous Burgers' equation?

If not,

> Which architectural and optimization factors explain the observed behavior?

We are not building a CFD solver or a hypersonic flow simulator. Burgers' equation is a
standard, well-characterized benchmark that produces a sharp shock — a simplified
mathematical proxy for shock formation in transonic/hypersonic aerospace flow.

---

## Problem

Viscous Burgers' equation (Raissi et al., 2019 benchmark):

```
u_t + u·u_x - ν·u_xx = 0,      x ∈ [-1, 1], t ∈ [0, 1]
u(x, 0) = -sin(πx)
u(-1, t) = u(1, t) = 0
ν = 0.01/π
```
---

# Scientific Contributions

This work makes the following contributions:

- Developed a controlled comparison framework where the Classical PINN and QAPINN share every component except the input representation.

- Performed systematic controlled ablations across:
  - Data encoding
  - Optimizer
  - Qubit count
  - Circuit depth
  - Entanglement
  - Collocation density

- Demonstrated that **representation and optimization strategy** have substantially greater influence on QAPINN performance than increasing circuit resources.

- Showed that **encoding choice** is the dominant architectural factor under the evaluated settings.

- Identified an optimizer–representation interaction in which amplitude encoding benefits significantly from L-BFGS refinement whereas angle re-uploading shows minimal improvement.

---

## Repository Structure

```text
QAPINN/
│
├── config.py
├── config_b.py
├── model.py
├── model_b.py
├── train.py
├── evaluate.py
├── visualize.py
├── sampling.py
├── reference_solution.py
├── pinn_losses.py
├── utils.py
│
├── main.py
├── main_b.py
├── ablation_runner.py
├── evaluate_saved_model_b.py
│
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   ├── metrics/
│   └── reference/
│
├── report/
│
├── presentation/
│
├── requirements.txt
│
└── README.md
```

Detailed descriptions of the major source files and experimental workflow are provided in the accompanying technical report.

---

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
```

## Controlled Ablation Study

```bash
python ablation_runner.py                                        # full qubit/encoding/depth/entanglement sweep
```

## Evaluate a Saved Model

```bash
python evaluate_saved_model_b.py                                 # re-evaluate a saved checkpoint without retraining
```

**Always pass `--run_name <something>` for any run that isn't meant to replace the official
baseline** — e.g. `python main_b.py --encoding amplitude --run_name amplitude_followup`.
Without it, a new run silently overwrites the previous one's checkpoint, metrics, and logs,
since they all share the same default filenames.

---

# Experimental Summary

Two models were investigated.

## Model A

- Classical PINN
- Fully-connected MLP
- 16,897 trainable parameters
- Adam + L-BFGS
- 10,000 collocation points

---

## Model B

- Quantum-Assisted PINN
- Direct quantum input representation
- 4-qubit variational circuit
- 85 trainable parameters
- PennyLane implementation

---


## Main Findings

The controlled experiments showed that:

- Encoding strategy produced the largest measurable improvement.

- Increasing qubit count alone did not consistently improve performance.

- Increasing circuit depth alone did not consistently improve performance.

- Changing entanglement topology alone produced little measurable effect.

- Amplitude encoding substantially outperformed the default angle-based encodings under identical experimental settings.

- L-BFGS refinement benefited amplitude encoding much more strongly than angle re-uploading.

- Representation and optimization strategy were more influential than circuit size under the evaluated benchmark.

| Metric | Model A | Model B (baseline config) | Model B (amplitude + L-BFGS, reduced budget) |
|---|---|---|---|
| L2 relative error (global) | 4.01% | 98.36% | 55.47% |
| L2 relative error (shock region) | 11.83% | 99.85% | 89.12% |
| Fourier spectrum L2 error | 1.35% | 88.90% | 53.99% |
| Trainable parameters | 16,897 | 85 | 85 |

These results demonstrate that representation and optimization strategy had a substantially greater influence on QAPINN performance than increasing quantum circuit resources. Although the classical PINN remained the strongest performer on the evaluated benchmark, controlled experiments identified encoding choice as the dominant architectural factor affecting performance.

---

## Experimental Outputs

Generated outputs are available under:

outputs/

including

- checkpoints
- metrics
- logs
- figures
- reference solutions

All figures presented in the report and presentation were generated directly from these outputs.

---

# Submission Documents


## Presentation

- [Presentation (PDF)](presentation/QAPINN_Research_Presentation.pdf)

- [Presentation (PPTX)](presentation/QAPINN_Research_Presentation.pptx)

---

## Technical Report

- [Technical Report (PDF)](report/QAPINN_Research_Report.pdf)

- [Technical Report (DOCX)](report/QAPINN_Research_Report.docx)

---

# Reproducibility

The repository contains all code, configurations, outputs, figures, and documentation required to reproduce the reported experiments.

### Software

| Component | Version |
|------------|----------|
| Python | 3.13.7 |
| PyTorch | 2.13.0 (CPU) |
| PennyLane | 0.45.1 |
| PennyLane-Qiskit | 0.45.0 |

---

### Hardware

| Component | Specification |
|------------|---------------|
| Operating System | Windows 11 Home |
| Processor | Intel Core i7-1355U |
| RAM | 16 GB |
| Backend | PennyLane `default.qubit` |
| GPU | CPU-only |

---

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

---

# Acknowledgement

This work was completed as part of the **BQP WISER Summer Challenge 2026**, exploring hybrid quantum-classical machine learning for scientific computing and Physics-Informed Neural Networks.

---

# License

This repository is released for academic and research purposes.

Please cite the accompanying report if this work is used in future research.

---

## Contact

**Meenakshi R.**

Department of Artificial Intelligence and Data Science

Panimalar Engineering College

-GitHub: https://github.com/meenakshi-re18/qapinn-bqp-challenge

For questions regarding the implementation, experiments, or reproducibility, please feel free to reach out.

