# conformal-med

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)] [![GitHub stars](https://img.shields.io/github/stars/habib-analyst/conformal-med.svg)]

**Conformal prediction for medical imaging: wrap any PyTorch classifier and get prediction sets with proven coverage — plus a measured look at how that coverage degrades under distribution shift.**

A 95%-accurate classifier still fails when it is confidently wrong on *your* patient. Conformal prediction adds distribution-free, finite-sample guarantees on top of any model, no retraining: "the true diagnosis is in this set with 90% probability." This toolkit adds the medical-imaging specifics — class-conditional coverage (rare diseases get the same guarantee), calibrated referral ("send to a radiologist when the set is ambiguous"), and a shift benchmark that measures the coverage gap.

## Quickstart

```python
from conformal_med import ConformalClassifier, auto_select

# pick the method by data regime (split-softmax / split-APS / class-conditional)
method, why = auto_select(n_calib=len(calib_ds), class_counts=class_counts,
                          n_classes=n_classes)

cc = ConformalClassifier(model, alpha=0.1, score="raps")   # 90% coverage target
cc.calibrate(calib_loader)                                 # split conformal calibration
sets = cc.predict_sets(test_loader)                        # bool tensor (n, K)
print(cc.coverage_report(test_loader))                     # coverage, set sizes, referral stats
```

Or run the full benchmark + demo:

```bash
pip install -r requirements.txt
python scripts/benchmark.py --datasets bloodmnist --epochs 5   # trains tiny CNN, ~3 min CPU
python scripts/demo.py
```

Expected output (`scripts/demo.py`, abridged — real run):

```
empirical coverage (nominal 90%):
  clean        coverage=0.993 mean_size=3.69
  noise@0.5    coverage=0.943 mean_size=3.46
  blur@1.0     coverage=0.742 mean_size=4.15  <-- COVERAGE GAP
```

## Measured results

Nominal coverage 90% (α=0.1). TinyCNN, 5 epochs, CPU-only, 0.34 wall-hours total. Full tables: `results/benchmark.md`.

**Clean test (does the guarantee hold?):**

| dataset | split-softmax (cov / size) | split-APS (cov / size) | class-cond (cov / size) | temp-scal ECE |
|---|---|---|---|---|
| bloodmnist | 0.893 / 1.12 | 0.995 / 3.63 | 0.917 / 1.22 | 0.019 |
| breastmnist | 0.897 / 1.35 | 0.994 / 1.99 | 0.942 / 1.65 | 0.044 |
| dermamnist | 0.913 / 1.92 | 0.978 / 3.56 | 0.893 / 2.60 | 0.054 |
| pneumoniamnist | **0.742** / 0.85 | **0.825** / 1.00 | **0.825** / 1.00 | 0.044 |

Three datasets validate the guarantee (0.89–0.91 ≈ nominal). **Pneumonia breaks it on the clean test set** — reported, not hidden: MedMNIST's own val/test splits violate exchangeability there (val accuracy 96.8% vs test 82.5%).

**Coverage gap under shift (split-softmax, clean → worst shift):**

| dataset | worst shift | coverage drop |
|---|---|---|
| bloodmnist | contrast@1.0 | 0.893 → **0.138** (gap 0.755) |
| breastmnist | noise@1.0 | 0.897 → 0.731 (gap 0.167) |
| dermamnist | noise@1.0 | 0.913 → 0.783 (gap 0.129) |
| pneumoniamnist | contrast@0.75 | 0.742 → 0.623 (gap 0.119) |

**APS over-coverage buys robustness** (bloodmnist, noise sweep): APS covers 0.995 clean with mean set size 3.63; at noise@1.0 it still covers 0.620 while softmax collapses to 0.219. The conservativeness of non-randomized APS is a robustness budget under shift — at the price of large sets.

**Baselines:** temperature scaling ECE degrades 0.019 → 0.300 (bloodmnist, clean → noise@1.0) — point calibration collapses exactly when you need it. MC Dropout referral is fixed at 20% by construction (not adaptive).

| Tests | **10/10** green |

## How it works

Given exchangeable calibration data and target miscoverage α, split conformal takes the ⌈(n+1)(1−α)⌉/n quantile of calibration nonconformity scores as threshold q̂. Then **P(y ∈ C(x)) ≥ 1−α** — finite-sample, distribution-free, no assumptions on the model. The shift benchmark exists because medical deployment *breaks* exchangeability; we measure the gap.

| Module | What |
|---|---|
| `scores` | softmax / APS / RAPS nonconformity scores |
| `calibrate` | split (inductive) + class-conditional (Mondrian) conformal |
| `predict` | `ConformalClassifier`: calibrate → predict sets → coverage report |
| `risk` | risk-controlling prediction sets (RCPS) — bound expected decision cost |
| `auto` | auto-select the method by data regime |
| `metrics` | coverage, set size, size-stratified coverage, referral calibration |
| `shifts` | graded synthetic shifts (noise/blur/contrast) for the shift benchmark |

Closest related work: [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) (general-purpose, not medical-scoped, no shift benchmark). Full writeup: `../hidden_files/papers/conformal-med/PAPER.md` in the research workspace.

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
