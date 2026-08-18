# Quantum-Assisted Physics-Informed Neural Networks for Aerospace Shock-Wave Modeling

## A Mechanistic Investigation of Representation, Optimization, and Spectral Learning

**BQP Challenge 2026 – WISER Summer Program**

---

## Overview

This repository contains the implementation, experiments, technical report, presentation, and extended validation studies for our work on **Quantum-Assisted Physics-Informed Neural Networks (QAPINNs)** for shock-wave modeling using the viscous Burgers' equation.

The project investigates whether a **quantum input representation, integrated into a Physics-Informed Neural Network (PINN), changes how the model learns sharp, high-gradient regions and frequency-dependent structure**.

Rather than treating quantum integration as an attempt to demonstrate unconditional quantum advantage, this study focuses on a more controlled question:

> **How do representation, preprocessing, model capacity, optimization, and quantum-circuit design interact to influence QAPINN learning behaviour?**

The viscous Burgers' equation is used as a controlled mathematical benchmark containing a sharp shock structure. An independently generated numerical reference solution is used for evaluation.

---

## The Research Question

The initial research question was:

> **Can a quantum feature map, used as the input representation of a Physics-Informed Neural Network, improve learning of the shock region and frequency-dependent structure in the viscous Burgers' equation?**

As the experiments progressed, the question became more mechanistic:

> **When QAPINN behaviour changes, is the change caused by the quantum representation itself, by preprocessing, by model capacity, by optimization, or by their interaction?**

The study therefore emphasizes controlled comparisons rather than simply asking whether a quantum model produces a lower overall error.

---

## Problem

We study the viscous Burgers' equation:

```text
u_t + u·u_x - ν·u_xx = 0

x ∈ [-1, 1]
t ∈ [0, 1]

u(x, 0) = -sin(πx)

u(-1, t) = u(1, t) = 0

ν = 0.01/π
```

The equation develops a sharp shock structure, making it a useful controlled benchmark for investigating how neural networks represent high-gradient regions.

This project is **not intended to be a complete CFD solver or a hypersonic-flow simulator**. Burgers' equation is used as a simplified mathematical benchmark for studying learning behaviour relevant to scientific machine learning.

---

## Why QAPINN?

A standard PINN uses a classical neural network to approximate:

```text
(x, t) → u(x, t)
```

and trains the network using the governing PDE together with initial and boundary conditions.

Our QAPINN investigates a hybrid representation in which the input is passed through a quantum circuit before the final classical prediction:

```text
(x, t)
   │
   ▼
Quantum Input Encoding
   │
   ▼
Variational Quantum Circuit
   │
   ▼
Classical Neural Network
   │
   ▼
u(x, t)
   │
   ▼
PDE Residual + Boundary/Initial Loss
```

A second configuration introduces a small classical preprocessing stage before quantum encoding:

```text
(x, t)
   │
   ▼
Classical Preprocessing
   │
   ▼
Quantum Encoding
   │
   ▼
Variational Quantum Circuit
   │
   ▼
Classical Head
   │
   ▼
u(x, t)
```

The purpose is to investigate whether these different representations change the learning behaviour of the PINN.

---

## Research Workflow

```text
Problem Definition
        │
        ▼
Classical PINN Baseline
        │
        ▼
Quantum-Assisted PINN
        │
        ▼
Encoding & Architecture Ablations
        │
        ▼
Optimizer × Representation Analysis
        │
        ▼
Collocation-Density Study
        │
        ▼
Parameter-Matched Validation
        │
        ▼
3-Seed Preprocessing Ablation
        │
        ▼
Frequency-Domain Analysis
        │
        ▼
Next: Fixed-Capacity Preprocessing Study
```

---

## Key Contributions

This work investigates:

- Classical PINN and QAPINN architectures for the same PDE benchmark.
- Direct and preprocessed quantum input representations.
- Different quantum encoding strategies.
- Qubit count, circuit depth, and entanglement topology.
- Adam versus Adam + L-BFGS optimization.
- Collocation-point density.
- Parameter-matched classical baselines.
- Repeated-seed validation.
- Preprocessing effects on QAPINN performance.
- Frequency-domain / Fourier-spectrum behaviour.
- Spatial reconstruction of shock formation over time.

A central goal is to distinguish **quantum representation effects** from effects caused by **model capacity, preprocessing, and optimization**.

---

## Team

| Author | Role |
|---|---|
| **Meenakshi R.** | Quantum & QAPINN Research Lead |
| **Krishna Priya Kaku** | Classical PINN & Numerical Modelling Lead |
| **Mallampati Geethika** | Comparative Analysis & Documentation Lead |

---

# Original WISER Study

The original BQP WISER study established the baseline Classical PINN and QAPINN implementations and investigated:

- Encoding strategy
- Qubit count
- Circuit depth
- Entanglement topology
- Optimizer choice
- Collocation density
- Direct quantum input representation
- Amplitude encoding with L-BFGS refinement

The original study found that the classical PINN remained substantially stronger on the evaluated benchmark, while representation and optimization choices had large effects on QAPINN behaviour.

The original WISER results are retained in the repository for historical comparison and reproducibility.

---

## Original Experimental Summary

### Model A — Classical PINN

- Fully connected MLP
- 16,897 trainable parameters
- Adam + L-BFGS
- 10,000 collocation points in the main baseline

### Model B — QAPINN

- Hybrid quantum-classical PINN
- Quantum input representation
- Variational quantum circuit
- 85 trainable parameters in the evaluated baseline configuration
- PennyLane implementation

### Original WISER Results

| Metric | Classical PINN | QAPINN — Baseline | QAPINN — Amplitude + L-BFGS |
|---|---:|---:|---:|
| Global L2 error | 4.01% | 98.36% | 55.47% |
| Shock-region L2 error | 11.83% | 99.85% | 89.12% |
| Fourier-spectrum L2 error | 1.35% | 88.90% | 53.99% |
| Trainable parameters | 16,897 | 85 | 85 |

These values describe the original WISER configurations and are kept separate from the later parameter-matched experiments.

---

# Extended Experimental Validation

Following the original study, additional experiments were conducted to investigate questions raised by the initial results.

The extended experiments are maintained on the separate branch:

```text
experiments/indoml-round2
```

The branch contains:

1. Parameter-matched classical validation.
2. Three-seed validation.
3. A 2×2 preprocessing experiment.
4. Frequency-domain spectral comparison.
5. Spatial snapshot comparison.

These experiments extend the mechanistic investigation without modifying the original validated WISER baseline.

---

## Experiment 1 — Parameter-Matched Classical Validation

The original QAPINN configurations operated at substantially smaller parameter counts than the full classical PINN.

To separate model-capacity effects from representation effects, smaller classical baselines were constructed at parameter counts corresponding to the QAPINN configurations.

The extended validation uses three random seeds:

```text
42
7
99
```

Repeated seeds reduce dependence on a single initialization and provide an estimate of run-to-run variability.

---

## Experiment 2 — 2×2 Preprocessing Ablation

The main extended experiment compares four configurations:

|  | No preprocessing | Preprocessing |
|---|---:|---:|
| **Classical** | 85 parameters | 107 parameters |
| **QAPINN** | 85 parameters | 107 parameters |

The four cells are:

- **A:** Classical, no preprocessing, 85 parameters
- **B:** Classical, preprocessing, 107 parameters
- **C:** QAPINN, direct input, 85 parameters
- **D:** QAPINN, preprocessing, 107 parameters

The QAPINN configurations use the amplitude + L-BFGS configuration used in the extended comparison.

The current 2×2 summary was extended to three seeds for the reported comparison.

---

## Current 3-Seed Results

All values are relative L2 errors reported as percentages.

**Lower is better.**

| Configuration | Parameters | Global L2 | Shock-region L2 | Fourier L2 |
|---|---:|---:|---:|---:|
| Classical — no preprocessing | 85 | **18.33 ± 2.38%** | **34.92 ± 15.66%** | **5.79 ± 2.33%** |
| Classical — preprocessing | 107 | 25.30 ± 5.39% | 36.54 ± 11.69% | 9.44 ± 2.81% |
| QAPINN — direct input | 85 | 51.32 ± 3.30% | 84.72 ± 4.39% | 43.72 ± 7.44% |
| QAPINN — preprocessed input | 107 | 43.90 ± 0.75% | 72.46 ± 3.56% | 24.66 ± 0.75% |

---

# Main Findings

## 1. Classical PINN Remains Stronger Overall

At the tested parameter budgets, the classical configurations have lower global, shock-region, and Fourier errors than the corresponding QAPINN configurations.

Therefore:

> **The current study does not claim quantum advantage.**

Instead, the results reinforce the need for controlled classical baselines when evaluating QAPINNs.

---

## 2. Preprocessing Improves the Tested QAPINN Configuration

Within the QAPINN configurations:

| Metric | Direct QAPINN — 85p | Preprocessed QAPINN — 107p |
|---|---:|---:|
| Global L2 | 51.32 ± 3.30% | **43.90 ± 0.75%** |
| Shock-region L2 | 84.72 ± 4.39% | **72.46 ± 3.56%** |
| Fourier L2 | 43.72 ± 7.44% | **24.66 ± 0.75%** |

The preprocessed configuration therefore shows lower error across all three reported metrics.

The largest relative reduction occurs in the Fourier-spectrum error.

However, this comparison changes **both preprocessing and parameter count**:

```text
Direct QAPINN
85 parameters
      │
      │ preprocessing + additional parameters
      ▼
Preprocessed QAPINN
107 parameters
```

Therefore, the observed improvement cannot yet be attributed to preprocessing alone.

---

## 3. The Classical Comparison Does Not Show the Same Improvement

For the classical configurations:

| Metric | Classical — 85p | Classical — 107p |
|---|---:|---:|
| Global L2 | **18.33 ± 2.38%** | 25.30 ± 5.39% |
| Shock-region L2 | **34.92 ± 15.66%** | 36.54 ± 11.69% |
| Fourier L2 | **5.79 ± 2.33%** | 9.44 ± 2.81% |

The 107-parameter preprocessed classical configuration does not show the same improvement pattern as the preprocessed QAPINN.

This is an observed difference between the tested configurations, but it does **not** establish that the difference is caused by the quantum representation.

---

# Important Experimental Limitation

The current preprocessing comparison contains a capacity confound.

The QAPINN changes from:

```text
85 → 107 trainable parameters
```

when preprocessing is introduced.

Therefore, the current experiment cannot independently determine the contribution of:

- preprocessing,
- parameter capacity,
- quantum representation,
- optimization dynamics,
- or interactions between these factors.

This is an unresolved experimental question.

---

# Frequency-Domain Analysis

A frequency-domain comparison was performed using:

```text
outputs/figures/spectral_comparison_all_models.png
```

The comparison includes the reference solution and the evaluated classical and QAPINN configurations.

The direct QAPINN shows pronounced deviations from the reference spectrum, including irregular behaviour in parts of the low-to-mid frequency range.

The preprocessed QAPINN has a substantially lower Fourier-spectrum error:

```text
43.72 ± 7.44%
        ↓
24.66 ± 0.75%
```

The spectrum also indicates that the preprocessed QAPINN does not simply recover all reference high-frequency content. Some high-frequency components are attenuated relative to the reference.

Therefore, the current result should **not** be interpreted as proof that preprocessing removes spectral bias or that the quantum representation learns high-frequency information better.

A cautious interpretation is:

> **Preprocessing substantially changes the frequency distribution learned by the QAPINN and reduces its Fourier-spectrum error, but the resulting spectrum also shows attenuation of higher-frequency components relative to the reference.**

This motivates a more quantitative frequency-domain investigation.

---

# Spatial-Domain Analysis

Solution snapshots were examined at:

```text
t = 0.25
t = 0.50
t = 0.75
t = 0.99
```

The current visual comparisons show different reconstruction behaviour.

### Classical — 85 Parameters

The parameter-matched classical model tracks the reference solution substantially more closely, including the shock transition.

### Direct QAPINN — 85 Parameters

The direct QAPINN shows stronger smoothing and amplitude mismatch around the shock region.

### Preprocessed QAPINN — 107 Parameters

The preprocessed QAPINN improves over the direct QAPINN in overall reconstruction, but the shock remains smoother and less sharply resolved than the reference.

This leads to an important distinction:

> **A lower Fourier-spectrum error does not necessarily imply better localized shock reconstruction.**

Frequency-domain and spatial-domain metrics therefore need to be interpreted together.

---

# What We Can and Cannot Conclude

## Supported by the Current Experiments

- The classical PINN remains substantially stronger overall at the tested parameter budgets.
- The direct QAPINN has substantially higher errors than the parameter-matched classical model.
- The tested preprocessed QAPINN configuration has lower global, shock-region, and Fourier errors than the tested direct QAPINN configuration.
- The largest relative QAPINN improvement occurs in the Fourier-spectrum metric.
- The preprocessed QAPINN still under-resolves the sharp shock.
- Preprocessing changes the learned QAPINN frequency spectrum substantially.
- The current evidence does not establish quantum advantage.

## Not Established Yet

We cannot currently conclude that:

- quantum representations outperform classical representations;
- preprocessing alone causes the QAPINN improvement;
- the improvement is caused by the additional parameters;
- preprocessing removes spectral bias;
- the QAPINN learns high-frequency shock information better;
- the observed spectral changes are uniquely caused by the quantum representation.

These require additional controlled experiments.

---

# Evolving Research Question

The project began with:

> **Can a quantum feature map help a PINN learn shock regions better?**

The current evidence leads to a more precise question:

> **Under controlled parameter capacity, how do representation and preprocessing interact to change what a PINN learns in the spatial and frequency domains?**

This reframes the study from simply comparing:

```text
Quantum vs Classical
```

toward understanding the interaction between:

```text
Representation
      +
Preprocessing
      +
Model Capacity
      +
Optimization
      +
Frequency Content
```

---

# Next Experiments

## 1. Fixed-Parameter Preprocessing Ablation

The immediate next experiment is to keep the total number of trainable parameters fixed while varying preprocessing:

```text
Preprocessing OFF
        vs.
Preprocessing ON
```

for the same parameter budget.

This will help separate the effect of preprocessing from the effect of model capacity.

---

## 2. Classical Capacity-Controlled Comparison

Additional classical architectures can be constructed at the same parameter budgets as the QAPINN configurations.

The goal is to determine whether observed behaviour can be explained by:

- parameter capacity,
- architecture,
- representation,
- or their interaction.

---

## 3. Quantitative Frequency-Domain Analysis

The current spectrum provides qualitative evidence.

The next stage will quantify frequency-dependent behaviour using:

- low-frequency error,
- mid-frequency error,
- high-frequency error,
- spectral energy distribution,
- frequency-dependent reconstruction error.

This will help determine which parts of the spectrum are improved, attenuated, or redistributed.

---

# Reproducibility

The original WISER experiments and the extended experiments are maintained through Git branches and separate experiment files.

The extended validation is contained in:

```text
experiments/indoml-round2
```

### Important Experiment Scripts

```text
exp1_matched_seeds.py
exp2_preprocessing_ablation.py
exp2b_preprocessed_seeds.py
exp2_full_summary.py
spectral_comparison.py
model_a_preprocessed.py
config_a_preprocessed_107.py
```

### Important Extended Metrics

```text
outputs/metrics/
├── exp1_matched_seeds_summary.json
├── exp1_matched_seeds_summary.csv
├── exp2_2x2_summary.json
└── exp2_2x2_summary_3seed.json
```

Seed-level metric files are also stored under:

```text
outputs/metrics/
```

The main spectral comparison is:

```text
outputs/figures/spectral_comparison_all_models.png
```

---

# Repository Structure

```text
QAPINN/
│
├── config.py
├── config_a_matched.py
├── config_a_matched_107.py
├── config_a_preprocessed_107.py
├── config_b.py
│
├── model.py
├── model_b.py
├── model_a_preprocessed.py
│
├── train.py
├── evaluate.py
├── visualize.py
├── sampling.py
├── reference_solution.py
├── pinn_losses.py
├── utils.py
│
├── main.py
├── main_a_matched.py
├── main_a_matched_107.py
├── main_b.py
├── ablation_runner.py
│
├── exp1_matched_seeds.py
├── exp2_preprocessing_ablation.py
├── exp2b_preprocessed_seeds.py
├── exp2_full_summary.py
├── spectral_comparison.py
│
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   ├── metrics/
│   └── reference/
│
├── report/
├── presentation/
├── requirements.txt
└── README.md
```

---

# Development Environment

| Component | Version / Specification |
|---|---|
| Python | 3.13.7 |
| PyTorch | 2.13.0 (CPU) |
| PennyLane | 0.45.1 |
| PennyLane-Qiskit | 0.45.0 |
| NumPy | Compatible version |
| SciPy | Compatible version |
| Matplotlib | Compatible version |
| Operating System | Windows 11 |
| Quantum Backend | PennyLane `default.qubit` simulator |
| GPU | None — CPU execution |
| Validation Seeds | 42, 7, 99 |

---

# Setup

```bash
pip install -r requirements.txt
```

---

# Run Model A — Classical PINN

```bash
python main.py
```

### Quick Smoke Test

```bash
python main.py --adam_epochs 200 --lbfgs_iters 50 --collocation 1000
```

---

# Run Model B — QAPINN

```bash
python main_b.py
```

### Example Amplitude-Encoding Configuration

```bash
python main_b.py --qubits 6 --depth 4 --encoding amplitude --entanglement full
```

---

# Controlled Ablation Study

```bash
python ablation_runner.py
```

This performs the configured encoding, qubit-count, circuit-depth, and entanglement experiments.

---

# Experimental Outputs

Generated outputs are organized under:

```text
outputs/
├── checkpoints/
├── figures/
├── logs/
├── metrics/
└── reference/
```

The reference solution is generated numerically and is used only for evaluation.

---

# Data and Reference Solution

This research is equation-driven rather than based on a conventional machine-learning dataset.

The viscous Burgers' equation serves as the mathematical benchmark.

The reference solution is generated numerically using the project's numerical solver and is used only for evaluation. It is not supplied to the neural network during physics-informed training.

The networks are trained using:

- the governing PDE residual;
- initial conditions;
- boundary conditions;
- collocation points.

---

# Software and Libraries

The implementation uses:

- Python
- PyTorch
- PennyLane
- PennyLane-Qiskit
- NumPy
- SciPy
- Matplotlib
- Git
- GitHub

---

# Original WISER Status

- [x] Classical PINN implementation
- [x] QAPINN implementation
- [x] Controlled architecture comparison
- [x] Encoding ablation
- [x] Qubit-count experiments
- [x] Circuit-depth experiments
- [x] Entanglement experiments
- [x] Optimizer × representation analysis
- [x] Collocation-density sensitivity study
- [x] Technical report
- [x] Final presentation
- [x] BQP WISER Summer Challenge 2026 submission

---

# Extended Validation Status

- [x] Three-seed matched validation
- [x] 2×2 preprocessing experiment
- [x] Three-seed validation of the 2×2 configurations
- [x] Frequency-domain comparison
- [x] Spatial snapshot comparison
- [ ] Fixed-parameter preprocessing ablation
- [ ] Quantitative frequency-band analysis
- [ ] Final mechanistic interpretation

---

# Design Notes

## Physics-Informed Training

The network is trained using the governing PDE residual together with initial and boundary conditions.

## LHS Sampling

Latin Hypercube Sampling is used to improve coverage of the input domain.

## Automatic Differentiation

The PDE residual is calculated using automatic differentiation rather than finite-difference approximations of the neural network.

## Adam → L-BFGS

A two-stage optimization strategy is used. Adam provides the initial optimization stage and L-BFGS provides refinement.

## Independent Reference Solution

The numerical reference solution is generated independently and is used only for evaluation.

## Parameter-Matched Validation

Smaller classical models are constructed to compare QAPINN behaviour under similar parameter budgets.

## Repeated Seeds

Three random seeds are used in the extended validation to reduce dependence on a single initialization.

## Spectral Analysis

Fourier-spectrum comparisons are used alongside spatial-domain error metrics to investigate frequency-dependent learning behaviour.

## Capacity Confound

The current preprocessing comparison changes both the preprocessing architecture and the number of trainable parameters. A fixed-capacity ablation is therefore required before attributing the observed improvement to preprocessing itself.

---

# References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
- Rahaman, N., et al. (2019). On the Spectral Bias of Neural Networks. *ICML*.
- Schuld, M., Sweke, R., & Meyer, J. J. (2021). Effect of data encoding on the expressive power of variational quantum-machine-learning models. *Physical Review A*, 103, 032430.
- Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055–A3081.

---

# Current Research Position

The current evidence does **not** support the claim that quantum input provides a general advantage for PINNs.

Instead, the experiments indicate that:

> **Representation, preprocessing, model capacity, optimization, and frequency-dependent learning interact in determining QAPINN behaviour.**

The ongoing investigation therefore focuses on understanding **when, why, and under what controlled conditions a quantum component changes scientific machine-learning behaviour**, rather than attempting to demonstrate quantum advantage by default.

---

# Acknowledgement

This work was completed as part of the **BQP WISER Summer Challenge 2026**, exploring hybrid quantum-classical machine learning for scientific computing and Physics-Informed Neural Networks.

---

# License

This repository is released for academic and research purposes.

Please cite the accompanying technical report if this work is used in future research.
