# UniNILM — Training and Supplementary Evaluation Code

Code accompanying the paper *UniNILM: A Lightweight Multi-Output Network with Bidirectional Mamba and Day-Type Adaptive Gating for Edge NILM*. The repository contains the training/benchmark script for all evaluated methods and a set of supplementary evaluation scripts covering on/off (event) detection, robustness to data-transmission drops, concurrent-load analysis, and cross-household generalization.

---

## Repository Layout

```
UniNILM/
├── README.md                          # This file
├── baseline_experiments_v3.py         # Main training/benchmark script (all 9 methods)
├── evaluation/                        # Supplementary evaluation scripts
│   ├── concurrent_analysis.py         # Concurrent-load analysis (AMPds2 week4)
│   ├── concurrent_analysis_baselines.py  # Concurrent-load analysis for Mamba2NILM / Seq2Seq
│   ├── drop_concurrent_all.py         # Drop robustness + concurrent-load analysis, remaining methods
│   ├── make_concurrent_figure.py      # Figure: MAE vs. number of concurrently active appliances
│   ├── compute_onoff_metrics.py       # On/off metrics with a fixed 10 W threshold
│   ├── compute_onoff_calibrated.py    # On/off metrics with per-appliance calibrated thresholds
│   ├── onoff_extended.py              # Calibrated on/off metrics for CL-MDTFN and HMoE
│   ├── onoff_all_methods.py           # Calibrated on/off metrics for all 9 methods
│   ├── atp_itp_calibrated.py          # Event-level ATP/ITP under calibrated thresholds
│   ├── dropout_sim.py                 # UniNILM robustness to data-transmission drops
│   ├── dropout_mamba.py               # Mamba2NILM robustness to drops
│   ├── cross_rerun.py                 # Cross-household (UKDALE House1 -> House5) rerun
│   └── verify_ukdale_mae.py           # Per-appliance UKDALE MAE sanity check
├── data/                              # Datasets (not included; see "How to Run")
├── trained_models/                    # Checkpoints (not included; produced by training)
└── results/                           # JSON outputs and figures (created at run time)
```

---

## Main Script: `baseline_experiments_v3.py`

This is the core script of the repository. It defines the data pipeline, all model classes, and the unified training/evaluation protocol used to produce the main comparison tables.

**Data pipeline.** `NILMDataset` loads the aligned aggregate-power file (`SP` column) together with the per-appliance columns, builds sliding windows of length `SEQ_LENGTH = 480` (8 hours at 1-minute resolution), and computes the day-type feature vector `[weekday, weekend flag, weekday flag, hour, minute]` (5 dimensions) from the Unix timestamps. Normalization statistics are computed on the training split only. The split is unified across all methods: the first 4 weeks (40,320 samples) for training, the next 4 weeks for validation, and the remainder for testing; the 8-week setting doubles the training and validation partitions while keeping the test partition unchanged.

**Model classes.** `BiMamba` (UniNILM), `Mamba2NILM`, `CLMDTFN`, `HMoE`, `Seq2Seq`, `Seq2PointSeq2Seq`, `DUNILM`, `FFSTT`, and `AttentionNILM`. Per-appliance methods are trained as one model per appliance, multi-output methods as a single model over all appliances.

**Training protocol.** Masked MSE loss (NaN ground-truth entries are ignored), AdamW with `lr = 1e-4` and `weight_decay = 1e-2`, cosine annealing with warm restarts (`T_0 = 10`), 40 epochs, batch size 128 for multi-output models and 64 for per-appliance models, and automatic mixed precision on GPU (disabled for the models that are numerically sensitive to it). The best checkpoint is selected on the validation split, and the test split is evaluated once with that checkpoint. Checkpoints are written to `trained_models/` and a JSON summary of all results to `trained_models/baseline_results_v2.json`.

**Usage.**

```bash
# Train and evaluate all methods
python baseline_experiments_v3.py

# A single method, retraining from scratch
python baseline_experiments_v3.py --methods BiMamba --force-train
```

Optional arguments: `--methods` (subset of method names), `--prefix` (log prefix), `--force-train` (ignore existing checkpoints).

---

## Supplementary Evaluation Scripts

### 1) Concurrent-load analysis

| Script | Description |
|---|---|
| `concurrent_analysis.py` | MAE per number of concurrently active appliances (10 W on-threshold, non-overlapping 480-step windows). Reports fine-grained values (0–8 concurrent appliances) and grouped values (isolated 0–1 / light 2–3 / busy ≥ 4). |
| `concurrent_analysis_baselines.py` | Same analysis for Mamba2NILM (multi-output) and Seq2Seq (per-appliance). |
| `drop_concurrent_all.py` | Same analysis for the remaining methods (CL-MDTFN, HMoE, DU-NILM, FFSTT, Attention-NILM, Seq2Point), plus drop robustness. |
| `make_concurrent_figure.py` | Builds `results/concurrent_analysis.pdf` and `.png` from the JSON outputs. |

### 2) On/off detection metrics

| Script | Description |
|---|---|
| `compute_onoff_metrics.py` | Step-level F1/precision/recall and event-level ATP/ITP at a fixed 10 W threshold. |
| `compute_onoff_calibrated.py` | Per-appliance thresholds selected on the **validation set** (grid search 5–100 W, F1-maximizing) and then applied to the test set. |
| `onoff_extended.py` | Same calibrated protocol for CL-MDTFN and HMoE. |
| `onoff_all_methods.py` | Calibrated on/off metrics for all 9 methods; Seq2Point is evaluated at the window center (stride 60). |
| `atp_itp_calibrated.py` | Event-level ATP/ITP/miss/FP counts under the calibrated thresholds. |

### 3) Drop robustness

| Script | Description |
|---|---|
| `dropout_sim.py` | Randomly zeroes 5/10/20% of the input time steps at inference and measures MAE degradation on the test split. |
| `dropout_mamba.py` | Same protocol for Mamba2NILM. |
| `drop_concurrent_all.py` | Drop robustness for the remaining methods. |

### 4) Cross-household generalization

| Script | Description |
|---|---|
| `cross_rerun.py` | UKDALE House 1 (training) to House 5 (testing) cross-household experiment. Normalization statistics come from the first 4 weeks of House 5, appliances that are missing in House 5 are excluded, and the MAE is weighted by the number of valid samples per appliance. |
| `verify_ukdale_mae.py` | Per-appliance MAE sanity check for the UKDALE model. |

---

## How to Run

**Dependencies.** Python 3.10+, PyTorch (CUDA recommended), NumPy, Pandas, Matplotlib, and `mamba_ssm` (required by the BiMamba and Mamba2NILM models).

**Data.** Dataset files are not distributed with this repository. Place the aligned CSVs under `data/`, or point the environment variables to them:

```bash
export AMPDS2_PATH=/path/to/ampds2_aligned.csv
export UKDALE_PATH=/path/to/house_5_prepared.csv
```

The aligned AMPds2 file is expected to contain a `unix_ts` column, the aggregate active power in `SP`, and the eight appliance columns `BME, DWE, FGE, TVE, RSE, DNE, HPE, CWE`. The UKDALE file follows the same layout with the five selected appliance columns.

**Checkpoints.** Pretrained weights are not included. Train the models with `baseline_experiments_v3.py`; checkpoints are written to `trained_models/` with the file names used by the evaluation scripts.

**Outputs.** Every evaluation script writes its JSON output and any generated figure to `results/`, which is created automatically.

```bash
python baseline_experiments_v3.py --methods BiMamba Seq2Seq
python evaluation/concurrent_analysis.py
python evaluation/compute_onoff_calibrated.py
python evaluation/cross_rerun.py
```

---

## Notes

1. Per-appliance threshold calibration (grid 5–100 W, step 5 W) is performed on the **validation split only**; the test split is used solely to report the metrics for the selected thresholds.
2. Seq2Point predicts the value at the center of the input window, so it is evaluated at the window center (stride 60).
3. In `cross_rerun.py`, `train_size = 40320` corresponds to the first 4 weeks of House 5 used for normalization statistics.
4. All scripts default to CPU when CUDA is unavailable, but full training runs are intended for a GPU.
