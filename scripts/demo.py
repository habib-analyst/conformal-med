#!/usr/bin/env python3
"""5-minute demo: guarantees, not just accuracy.

Loads a pretrained TinyCNN (weights/bloodmnist.pt), calibrates a conformal
predictor on 500 validation images, then shows prediction SETS on test
images — with the proven 90% coverage guarantee — plus what happens to
coverage when the input distribution shifts.

    python3 scripts/demo.py
"""
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conformal_med import ConformalClassifier, auto_select, metrics as M, shifts
from scripts.benchmark import load_medmnist, TinyCNN

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS = "bloodmnist"
ALPHA = 0.1


def main():
    t0 = time.time()
    print("== conformal-med 5-minute demo ==\n")
    (xtr, ytr), (xva, yva), (xte, yte), K = load_medmnist(DS)
    print(f"dataset: {DS} ({K} classes)")

    model = TinyCNN(xtr.shape[1], K)
    wpath = os.path.join(ROOT, "weights", f"{DS}.pt")
    assert os.path.exists(wpath), f"run scripts/benchmark.py first ({wpath} missing)"
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()

    # auto-select the conformal method for this data regime
    method, why = auto_select(len(yva),
                             class_counts=[int((yva == c).sum()) for c in range(K)],
                             alpha=ALPHA, n_classes=K)
    print(f"auto-selected method: {why}")

    cc = ConformalClassifier(model, alpha=ALPHA, score="raps")
    cc.calibrate(DataLoader(TensorDataset(xva[:500], yva[:500]), batch_size=256))
    print("calibrated on 500 validation images.\n")

    # show prediction sets on 5 test images
    probs, labels = cc._probs_labels(
        DataLoader(TensorDataset(xte[:5], yte[:5]), batch_size=5))
    sets = cc._cp.sets_from_probs(probs)
    print("5 test images -> prediction sets (1 = in set):")
    for i in range(5):
        members = sets[i].nonzero(as_tuple=False).flatten().tolist()
        top1 = int(probs[i].argmax())
        print(f"  img {i}: set={members}  top1={top1}  true={int(labels[i])}  "
              f"{'COVERED' if labels[i] in members else 'MISS'}")

    # coverage on clean test vs shifted test
    print("\nempirical coverage (nominal 90%):")
    for name, sev in [("clean", 0.0), ("noise@0.5", 0.5), ("blur@1.0", 1.0)]:
        if name == "clean":
            xs = xte[:2000]
        else:
            s, sv = name.split("@")
            xs = shifts.apply_shift(xte[:2000], s, float(sv))
        dl = DataLoader(TensorDataset(xs, yte[:2000]), batch_size=512)
        rep = cc.coverage_report(dl)
        flag = "" if rep["empirical_coverage"] >= 0.9 - 0.02 else "  <-- COVERAGE GAP"
        print(f"  {name:12s} coverage={rep['empirical_coverage']:.3f} "
              f"mean_size={rep['mean_set_size']:.2f}{flag}")

    print(f"\ndone in {time.time()-t0:.0f}s. "
          "Accuracy tells you the past; coverage guarantees the future.")


if __name__ == "__main__":
    main()
