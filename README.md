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
Parameter-Matched Classical Validation
        │
        ▼
3-Seed Preprocessing Ablation
        │
        ▼
3-Seed Direct 109p QAPINN Validation
        │
        ▼
Capacity-Effect Decomposition
        │
        ▼
Frequency-Domain Analysis
        │
        ▼
Next: Exact Fixed-Capacity Quantum Ablation

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
| Classical — no preprocessing | 107 | 28.72 ± 6.99% | 66.04 ± 21.85% | 10.15 ± 4.82% |
| Classical — preprocessing | 107 | 25.30 ± 5.39% | 36.54 ± 11.69% | 9.44 ± 2.81% |
| QAPINN — direct input | 85 | 51.32 ± 3.30% | 84.72 ± 4.39% | 43.72 ± 7.44% |
| QAPINN — direct input | 109 | 45.98 ± 1.44% | 76.02 ± 3.36% | 29.92 ± 3.08% |
| QAPINN — preprocessed input | 107 | **43.90 ± 0.75%** | **72.46 ± 3.56%** | **24.66 ± 0.75%** |

The 109-parameter direct QAPINN uses the same 4-qubit, depth-3, amplitude-encoding, circular-entanglement circuit as the other extended QAPINN configurations, with a larger classical head (4 → 12 → 1) and no preprocessing.

The classical — no preprocessing, 107 parameters row is an exact parameter match with the classical — preprocessing, 107 parameters row directly below it — both have exactly 107 trainable parameters and differ only in whether the classical preprocessing block is present. This is the one exact, non-confounded preprocessing comparison in the current results; see Finding 3.

It is used as an **approximately capacity-matched comparison point** to the 107-parameter preprocessed QAPINN. An exact 107-parameter 4 → h → 1 head is not achievable with a single hidden layer, so 109 parameters is the closest available configuration for this comparison.

---

# Main Findings

## 1. Classical PINN Remains Stronger Overall

At the tested parameter budgets, the classical configurations have lower global, shock-region, and Fourier errors than the corresponding QAPINN configurations.

Therefore:

> **The current study does not claim quantum advantage.**

Instead, the results reinforce the need for controlled classical baselines when evaluating QAPINNs.

---

## 2. Capacity Explains a Substantial Part of the Original QAPINN Improvement

A new experiment adds a **Direct QAPINN configuration at 109 parameters**, used as an approximately capacity-matched comparison point to the 107-parameter preprocessed QAPINN.

| Metric | Direct QAPINN — 85p | Direct QAPINN — 109p | Preprocessed QAPINN — 107p |
|---|---:|---:|---:|
| Global L2 | 51.32 ± 3.30% | 45.98 ± 1.44% | **43.90 ± 0.75%** |
| Shock-region L2 | 84.72 ± 4.39% | 76.02 ± 3.36% | **72.46 ± 3.56%** |
| Fourier L2 | 43.72 ± 7.44% | 29.92 ± 3.08% | **24.66 ± 0.75%** |

The original 85p → 107p QAPINN difference can be decomposed into two observed steps.

**Step 1 — Direct 85p → Direct 109p: capacity effect without preprocessing**

- Global L2: −5.34 percentage points
- Shock-region L2: −8.70 percentage points
- Fourier L2: −13.80 percentage points

**Step 2 — Direct 109p → Preprocessed 107p: residual difference after approximately matching capacity**

- Global L2: −2.08 percentage points
- Shock-region L2: −3.56 percentage points
- Fourier L2: −5.26 percentage points

Increasing model capacity therefore accounts for a substantial part of the improvement originally observed between the 85p direct and 107p preprocessed QAPINNs, with the capacity effect consistent in direction across all three metrics.

A smaller residual difference remains after approximately controlling for capacity. The Fourier-spectrum ranges (29.92 ± 3.08% versus 24.66 ± 0.75%) are clearly separated, making the Fourier-spectrum reduction the strongest-supported residual difference in this comparison.

The residual global difference (45.98 ± 1.44% versus 43.90 ± 0.75%) and shock-region difference (76.02 ± 3.36% versus 72.46 ± 3.56%) have overlapping mean ± SD ranges at n = 3 seeds and should therefore be treated as weaker evidence rather than established effects.

We describe this as an **observed error-reduction decomposition**, not formal causal attribution. The 109p versus 107p comparison is approximately, rather than exactly, parameter-matched, and three seeds provide limited statistical power.

Therefore:

> **A substantial part of the original 85p → 107p QAPINN improvement is explained by increased model capacity. A smaller residual difference remains after approximately matching capacity, with the clearest remaining signal appearing in Fourier-spectrum error.**

---

## 3. At Fixed Capacity, Classical Preprocessing Has a Large Effect

An exact, non-confounded classical preprocessing comparison is available at 107 parameters:

| Metric | Classical — 107p, no preprocessing | Classical — 107p, preprocessing | Change |
|---|---:|---:|---:|
| Global L2 | 28.72 ± 6.99% | **25.30 ± 5.39%** | −3.42 pp |
| Shock-region L2 | 66.04 ± 21.85% | **36.54 ± 11.69%** | −29.50 pp |
| Fourier L2 | 10.15 ± 4.82% | **9.44 ± 2.81%** | −0.71 pp |

Both configurations have exactly 107 trainable parameters, so this isolates the classical preprocessing effect without any change in capacity. At fixed capacity, preprocessing produces a large reduction in classical shock-region error (−29.50 percentage points), with smaller reductions in global and Fourier error. This is a larger effect than the residual QAPINN preprocessing effect described in Finding 2 — though the two are not directly comparable, since this classical comparison is an exact 107p-vs-107p match, while the QAPINN comparison is an approximate 109p-vs-107p match.

A separate pattern is visible from the full six-row table above: increasing classical capacity **without** preprocessing (85p → 107p, no preprocessing) does not improve performance — global, shock-region, and Fourier error all increase (18.33%→28.72%, 34.92%→66.04%, 5.79%→10.15%). This is the opposite direction from the QAPINN capacity effect in Finding 2, where increasing capacity (85p→109p, no preprocessing) improved all three metrics. This asymmetry is an observed pattern in the current results, not something the current experiments explain.

The earlier framing of this comparison (85p classical, no preprocessing, vs. 107p classical, preprocessing) mixed a negative capacity effect with a positive preprocessing effect, which partially cancelled and made the classical preprocessing benefit look smaller than it is. The exact 107p-vs-107p comparison above is the correct comparison for isolating classical preprocessing.

This does **not** establish that the classical and QAPINN preprocessing effects share the same mechanism, or that either is caused by preprocessing acting on the quantum representation specifically.

Note that the classical fixed-capacity comparison above is an **exact** parameter match, whereas the quantum-side 109p versus 107p comparison in Finding 2 is only an **approximate** parameter match, because no exact 107-parameter single-hidden-layer 4 → h → 1 head exists.

---

# Important Experimental Limitation

The original 85p direct → 107p preprocessed QAPINN comparison was confounded by both preprocessing and increased parameter capacity, since introducing preprocessing increased the QAPINN from 85 to 107 trainable parameters. The equivalent classical comparison (85p no-preprocessing vs. 107p preprocessing) shares the same confound.

This confound is resolved differently on each side of the study:

- **Classical side — resolved exactly.** An exact, fixed-capacity comparison exists at 107 parameters: classical — no preprocessing (28.72 ± 6.99% global) versus classical — preprocessing (25.30 ± 5.39% global), both with exactly 107 trainable parameters. This isolates the classical preprocessing effect with no capacity change, and shows a large effect on shock-region error in particular (−29.50 percentage points). **This comparison is not missing from the current results** — see Finding 3.

- **Quantum side — resolved approximately.** The Direct QAPINN — 109p configuration provides an approximately, not exactly, capacity-matched comparison point to the 107p preprocessed QAPINN (a 2-parameter difference, since no exact 107-parameter single-hidden-layer 4 → h → 1 head exists). The residual Fourier-spectrum difference is clearly separated across the observed mean ± SD ranges, while the global and shock-region differences remain less clearly distinguished from seed-to-seed variability at n = 3.

The remaining limitations are therefore:

- The quantum-side 109p vs. 107p comparison is approximate, not exact, unlike the classical-side 107p vs. 107p comparison, which is exact.
- The residual QAPINN global and shock-region differences have not been distinguished from seed-to-seed variability at n = 3 seeds.
- The current experiments do not establish that the classical and QAPINN preprocessing effects share a common cause, or that any observed effect is uniquely attributable to the quantum representation.
- Why classical capacity alone (85p → 107p, no preprocessing) increases error while QAPINN capacity alone (85p → 109p, no preprocessing) decreases error is an open question the current experiments do not explain.

---

# Frequency-Domain Analysis

To examine how the models represent spatial frequencies, we compare the Fourier
amplitude spectrum of the predicted solution with the numerical reference
solution at t = 1.00, using the corrected spectral comparison
(`spectral_comparison.py`, fixed to reconstruct each checkpoint with the exact
encoding and input mode it was trained with).

The corrected spectral comparison shows that the three QAPINN configurations
learn substantially different frequency profiles. The direct 85-parameter
QAPINN exhibits the strongest spectral attenuation over the intermediate and
higher-frequency range. Increasing the direct QAPINN capacity to 109 parameters
changes the spectrum substantially and reduces the Fourier-spectrum error from
43.72 ± 7.44% to 29.92 ± 3.08%.

The approximately capacity-matched preprocessed 107-parameter QAPINN produces
a further reduction in Fourier-spectrum error to 24.66 ± 0.75%. Its spectral
profile also differs visibly from the direct 109-parameter model — tracking
the reference furthest into the frequency range of the three QAPINN
configurations before a region of irregular behaviour around frequencies 8–12
— indicating that preprocessing changes the learned frequency representation
beyond the capacity change alone.

Spectral amplitude alone is not a direct proxy for accuracy: at the highest
frequencies the reference spectrum itself decays sharply, so the classical
model's comparatively higher amplitude in this range does not necessarily
indicate closer agreement with the reference than the more strongly attenuated
QAPINN curves. The correct comparison is distance from the reference spectrum,
not amplitude in isolation.

These results should not be interpreted as evidence that preprocessing removes
spectral bias or universally improves high-frequency learning. Furthermore, the
109-parameter direct model and 107-parameter preprocessed model are only
approximately capacity-matched, and the comparison uses three random seeds.

The frequency-domain results therefore support a narrower conclusion:
increased capacity produces a substantial change in QAPINN spectral behaviour,
while preprocessing is associated with an additional, smaller reduction in
Fourier-spectrum error after approximately controlling for capacity.

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
- Increasing QAPINN capacity from 85 to 109 parameters, without preprocessing, produces consistent error reductions across global, shock-region, and Fourier metrics.
- This capacity increase accounts for a substantial part of the original 85p → 107p QAPINN improvement.
- A smaller residual difference remains between the approximately capacity-matched direct QAPINN (109p) and preprocessed QAPINN (107p).
- At exactly matched capacity (107 parameters), classical preprocessing produces a large reduction in shock-region error (66.04% → 36.54%) and smaller reductions in global and Fourier error.
- Increasing classical capacity without preprocessing (85p → 107p) increases global, shock-region, and Fourier error — the opposite direction from the QAPINN capacity effect (85p → 109p, which decreases all three).
- The residual Fourier-spectrum difference is the clearest remaining signal in the current three-seed comparison.
- The preprocessed QAPINN still under-resolves the sharp shock.
- Preprocessing changes the learned QAPINN frequency spectrum substantially.
- The current evidence does not establish quantum advantage.

## Not Established Yet

We cannot currently conclude that:

- quantum representations outperform classical representations;
- preprocessing universally improves QAPINN configurations;
- the residual global or shock-region differences between the approximately capacity-matched 109p direct and 107p preprocessed QAPINNs are established effects at n = 3 seeds;
- preprocessing removes spectral bias;
- the QAPINN learns high-frequency shock information better;
- the observed spectral changes are uniquely caused by the quantum representation;
- the 109p versus 107p quantum comparison is an exact parameter match;
- the observed capacity/residual decomposition constitutes formal causal attribution.
- that the classical and QAPINN preprocessing effects share the same underlying mechanism, or why classical and QAPINN capacity affect error in opposite directions;

Further controlled experiments, particularly stronger fixed-capacity quantum-side preprocessing ablations and larger seed counts, are required before making stronger claims.

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

## 1. Stronger Fixed-Capacity Quantum Preprocessing Ablation

The next major experiment is to construct a quantum-side comparison in which the total number of trainable parameters is held exactly fixed while preprocessing is varied.

The goal is:

```text
Same QAPINN parameter budget

        │
        ├── Preprocessing OFF
        │
        └── Preprocessing ON

```
This would isolate preprocessing more cleanly than the current 109p versus 107p approximate match.

## 2. Larger Seed Validation

The current decomposition uses three seeds:
```text
42
7
99
```
Additional seeds would provide a stronger estimate of run-to-run variability and help determine whether the residual global, shock-region, and Fourier differences persist beyond the current n = 3 comparison.

## 3. Quantitative Frequency-Domain Analysis

The current spectrum comparison provides frequency-domain evidence, but the next stage should quantify frequency-dependent behaviour using:

- low-frequency error;
- mid-frequency error;
- high-frequency error;
- spectral energy distribution;
- frequency-dependent reconstruction error.

This will help determine which parts of the spectrum are improved, attenuated, or redistributed.

## 4. Mechanistic Interpretation

The final stage is to connect the controlled parameter, preprocessing, spatial, and frequency-domain experiments into a unified explanation of QAPINN behaviour.

The objective is not to establish quantum advantage by default, but to determine:

```text
What changes?
      ↓
When does it change?
      ↓
Which factor explains the change?
      ↓
Does the effect survive capacity and seed controls?

```
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
- [x] Three-seed direct 109-parameter QAPINN validation
- [x] Frequency-domain comparison
- [x] Spatial snapshot comparison
- [x] Capacity-effect decomposition
- [x] Classical fixed-capacity preprocessing comparison
- [ ] Exact fixed-parameter quantum-side preprocessing ablation
- [ ] Quantitative frequency-band analysis
- [ ] Larger-seed validation
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

The original 85p direct → 107p preprocessed QAPINN comparison changed both preprocessing and parameter capacity.

The extended validation now separates these effects more carefully:

- a 3-seed 85p → 109p direct QAPINN comparison measures the effect of increased capacity without preprocessing;
- a 109p direct → 107p preprocessed comparison provides an approximately capacity-matched residual comparison;
- the classical side also contains fixed-capacity 107-parameter configurations with and without preprocessing.

The quantum-side 109p versus 107p comparison is not an exact parameter match, so the residual difference should be interpreted cautiously rather than as formal causal evidence.

An exact fixed-capacity quantum-side preprocessing ablation remains an open experiment.

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
