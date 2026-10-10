# conformal-med

**Stop reporting accuracy. Report what you can guarantee.**

`conformal-med` is a conformal prediction toolkit scoped to medical imaging: wrap *any* PyTorch image classifier and get **prediction sets with proven coverage** — "the true diagnosis is in this set with 90% probability" — plus risk-controlled referral ("send to a radiologist when the set is ambiguous").

Why you want this: a classifier that is 95% accurate still kills when it is confidently wrong on *your* patient. Conformal prediction gives distribution-free, finite-sample guarantees on top of any model, no retraining. This toolkit adds the medical-imaging specifics: class-conditional coverage (rare diseases get the same guarantee), a shift benchmark that *measures* how coverage degrades under noise/blur/contrast shift, and calibrated referral.

## 5-minute demo

```bash
pip install -r requirements.txt
python scripts/benchmark.py --datasets bloodmnist --epochs 5   # trains tiny CNN, ~3 min CPU
python scripts/demo.py
```

Expected output (abridged, real run):

```
dataset: bloodmnist (8 classes)
auto-selected method: split(raps)
calibrated on 500 validation images.

5 test images -> prediction sets (1 = in set):
  img 0: set=[0, 2, 3, 5]  top1=5  true=3  COVERED
  ...

empirical coverage (nominal 90%):
  clean        coverage=0.993 mean_size=3.69
  noise@0.5    coverage=0.943 mean_size=3.46
  blur@1.0     coverage=0.742 mean_size=4.15  <-- COVERAGE GAP
```

## Use it on your model

```python
from conformal_med import ConformalClassifier, auto_select

method, why = auto_select(n_calib=len(calib_ds),
                          class_counts=class_counts,
                          n_classes=n_classes)   # picks softmax/raps/class-conditional
cc = ConformalClassifier(model, alpha=0.1, score="raps")
cc.calibrate(calib_loader)
sets = cc.predict_sets(test_loader)      # bool tensor (n, K)
print(cc.coverage_report(test_loader))   # coverage, mean set size, referral stats
```

## What's inside

| Module | What |
|---|---|
| `scores` | softmax / APS / RAPS nonconformity scores |
| `calibrate` | split (inductive) + class-conditional (Mondrian) conformal |
| `risk` | risk-controlling prediction sets (RCPS) — bound expected decision cost |
| `auto` | auto-select the method by data regime |
| `metrics` | coverage, set size, size-stratified coverage, referral calibration |
| `shifts` | graded synthetic shifts (noise/blur/contrast) for the shift benchmark |

## Benchmark

`scripts/benchmark.py` trains small CNNs on MedMNIST (blood/breast/derma/pneumonia @28px, CPU) and measures, per shift severity: empirical coverage vs nominal 90%, mean set size, referral calibration — for 5 conformal variants vs temperature scaling (ECE) and MC Dropout baselines. Results in `results/benchmark.md`.

## Theory (30 seconds)

Given exchangeable calibration data and target miscoverage α, split conformal takes the ⌈(n+1)(1−α)⌉/n quantile of calibration nonconformity scores as threshold q̂. Then **P(y ∈ C(x)) ≥ 1−α** — finite-sample, distribution-free, no assumptions on the model. The benchmark exists because medical deployment *breaks* exchangeability; we measure the gap.

Closest related work: [MAPIE](https://github.com/scikit-learn-contrib/MAPIE) (general-purpose, not medical-scoped, no shift benchmark). Full writeup: `../hidden_files/papers/conformal-med/PAPER.md` (in the research workspace).

## License

MIT
