# exp2_full_summary.py — aggregates all four cells (A,B,C,D), 3 seeds each, into mean±SD
import json, statistics as st

FILES = {
    "A_classical_no_preproc_85": [
        "outputs/metrics/model_a_matched_85params_seed42_metrics.json",
        "outputs/metrics/model_a_matched_85params_seed7_metrics.json",
        "outputs/metrics/model_a_matched_85params_seed99_metrics.json",
    ],
    "B_classical_preproc_107": [
        "outputs/metrics/model_a_preprocessed_107params_metrics.json",       # seed 42
        "outputs/metrics/model_a_preprocessed_107params_seed7_metrics.json",
        "outputs/metrics/model_a_preprocessed_107params_seed99_metrics.json",
    ],
    "C_quantum_no_preproc_85": [
        "outputs/metrics/amplitude_matched_budget_metrics.json",             # seed 42
        "outputs/metrics/amplitude_seed7_metrics.json",
        "outputs/metrics/amplitude_seed99_metrics.json",
    ],
    "D_quantum_preproc_107": [
        "outputs/metrics/preprocessed_amplitude_metrics.json",               # seed 42
        "outputs/metrics/preprocessed_amplitude_seed7_metrics.json",
        "outputs/metrics/preprocessed_amplitude_seed99_metrics.json",
    ],
}
KEYS = ["l2_relative_error", "shock_region_l2_relative_error", "fourier_spectrum_l2_relative_error"]

summary = {}
for cell, paths in FILES.items():
    vals = {k: [] for k in KEYS}
    for path in paths:
        with open(path) as f:
            m = json.load(f)
        for k in KEYS:
            vals[k].append(m[k])
    summary[cell] = {
        f"{k}_mean_pct": round(st.mean(vals[k]) * 100, 2) for k in KEYS
    } | {
        f"{k}_sd_pct": round(st.pstdev(vals[k]) * 100, 2) for k in KEYS
    }

with open("outputs/metrics/exp2_2x2_summary_3seed.json", "w") as f:
    json.dump(summary, f, indent=2)

for cell, s in summary.items():
    print(cell, s)