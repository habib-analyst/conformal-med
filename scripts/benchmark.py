#!/usr/bin/env python3
"""Benchmark: conformal prediction under distribution shift on MedMNIST.

For each dataset (blood/breast/derma/pneumonia @28px):
  1. Train a small CNN (CPU, few epochs) — or load --weights.
  2. Calibrate 5 conformal variants on the val split.
  3. Evaluate on clean test + shifted test (noise/blur/contrast x severities):
     empirical coverage vs nominal 90%, mean set size, referral calibration.
  4. Baselines: temperature scaling (ECE), MC Dropout (entropy referral).

Writes results/benchmark.json + results/benchmark.md. Saves trained weights
to weights/<dataset>.pt for the 5-minute demo.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from conformal_med import (
    SplitConformal, ClassConditionalConformal, RiskControllingConformal,
    auto_select, metrics as M, shifts,
)
import medmnist
from medmnist import INFO

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class TinyCNN(nn.Module):
    def __init__(self, in_ch=1, n_classes=2, p_drop=0.0):
        super().__init__()
        self.p_drop = p_drop
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Dropout(p_drop),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_medmnist(name, root="~/.medmnist"):
    info = INFO[name]
    cls = getattr(medmnist, info["python_class"])
    # use numpy arrays directly (fast, no torchvision needed)
    import numpy as np
    train = cls(split="train", download=False, root=os.path.expanduser(root))
    val = cls(split="val", download=False, root=os.path.expanduser(root))
    test = cls(split="test", download=False, root=os.path.expanduser(root))

    def to_t(arr):
        t = torch.from_numpy(np.asarray(arr)).float()
        if t.dim() == 3:
            t = t.unsqueeze(1)  # (n,1,28,28)
        else:
            t = t.permute(0, 3, 1, 2)
        return t / 255.0

    def lab(arr):
        return torch.from_numpy(np.asarray(arr)).long().squeeze()

    n_classes = len(info["label"])
    return (
        (to_t(train.imgs), lab(train.labels)),
        (to_t(val.imgs), lab(val.labels)),
        (to_t(test.imgs), lab(test.labels)),
        n_classes,
    )


def train_cnn(x, y, n_classes, epochs, batch=128, seed=0, p_drop=0.0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, generator=g)
    model = TinyCNN(x.shape[1], n_classes, p_drop=p_drop)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    ce = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            ce(model(xb), yb).backward()
            opt.step()
    return model


@torch.no_grad()
def probs_of(model, x, mc_passes=1):
    was_train = model.training
    outs = []
    for _ in range(mc_passes):
        if mc_passes > 1:
            model.train()  # enable dropout
        outs.append(torch.softmax(model(x), dim=1))
    if was_train:
        model.train()
    else:
        model.eval()
    return torch.stack(outs).mean(0)


def shifted_loader(x, y, shift, sev, batch=512):
    xs = shifts.apply_shift(x, shift, sev)
    return DataLoader(TensorDataset(xs, y), batch_size=batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="bloodmnist,breastmnist,dermamnist,pneumoniamnist")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.weights, exist_ok=True)

    t0 = time.time()
    report = {"alpha": args.alpha, "epochs": args.epochs, "datasets": {}}

    for ds in args.datasets.split(","):
        ds = ds.strip()
        print(f"=== {ds} ===", flush=True)
        (xtr, ytr), (xva, yva), (xte, yte), K = load_medmnist(ds)
        wpath = os.path.join(args.weights, f"{ds}.pt")
        if os.path.exists(wpath):
            model = TinyCNN(xtr.shape[1], K)
            model.load_state_dict(torch.load(wpath, map_location="cpu"))
            print("  loaded weights", flush=True)
        else:
            model = train_cnn(xtr, ytr, K, args.epochs, seed=0)
            torch.save(model.state_dict(), wpath)
            print("  trained + saved weights", flush=True)
        model.eval()

        # calibration split = val
        calib_dl = DataLoader(TensorDataset(xva, yva), batch_size=512)

        methods = {
            "split-softmax": SplitConformal(alpha=args.alpha, score="softmax"),
            "split-aps": SplitConformal(alpha=args.alpha, score="aps"),
            "split-raps": SplitConformal(alpha=args.alpha, score="raps"),
            "class-cond": ClassConditionalConformal(K, alpha=args.alpha, score="softmax"),
            "risk-control": RiskControllingConformal(beta=0.1, delta=0.1),
        }
        auto_m, auto_name = auto_select(len(yva),
                                       class_counts=[int((yva == c).sum()) for c in range(K)],
                                       alpha=args.alpha, n_classes=K)
        methods[f"auto({auto_name})"] = auto_m
        for m in methods.values():
            m.calibrate(model, calib_dl)

        # temperature scaling baseline (fit on val)
        temp = fit_temperature(model, xva, yva)

        ds_rep = {"n_classes": K, "shifts": {}}
        for shift in ["clean"] + list(shifts.SHIFTS):
            sevs = [0.0] if shift == "clean" else shifts.SEVERITIES[1:]
            for sev in sevs:
                key = "clean" if shift == "clean" else f"{shift}@{sev}"
                dl = (DataLoader(TensorDataset(xte, yte), batch_size=512)
                      if shift == "clean" else shifted_loader(xte, yte, shift, sev))
                probs, labels = batch_probs(model, dl)
                row = {}
                for mname, m in methods.items():
                    sets = m.sets_from_probs(probs)
                    row[mname] = {
                        "coverage": round(M.coverage(sets, labels), 4),
                        "mean_size": round(M.mean_set_size(sets), 3),
                        "referral": M.referral_stats(sets, labels, probs)["referral_rate"],
                    }
                    row[mname]["referral_rate"] = round(row[mname].pop("referral"), 4)
                # baselines
                row["temp-scaling-ece"] = round(ece(probs / temp, labels), 4)
                mc = mc_dropout_referral(model, xte, yte)
                row["mc-dropout"] = mc
                ds_rep["shifts"][key] = row
                def fmt(k, v):
                    if isinstance(v, dict) and "coverage" in v:
                        return f"{k}={v['coverage']}/{v['mean_size']}"
                    if isinstance(v, dict):
                        return f"{k}=ref{v.get('referral_rate', '?')}"
                    return f"{k}={v}"
                print(f"  {key}: " + " ".join(fmt(k, v) for k, v in row.items()),
                      flush=True)
        report["datasets"][ds] = ds_rep

    report["wall_hours"] = round((time.time() - t0) / 3600, 2)
    with open(os.path.join(args.out, "benchmark.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(args.out, "benchmark.md"), "w") as f:
        f.write(md_summary(report))
    print(f"WROTE results/benchmark.json + benchmark.md ({report['wall_hours']}h)")


@torch.no_grad()
def batch_probs(model, dl):
    model.eval()
    ps, ys = [], []
    for x, y in dl:
        ps.append(torch.softmax(model(x), 1))
        ys.append(y)
    return torch.cat(ps), torch.cat(ys)


def fit_temperature(model, x, y, iters=50):
    temp = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([temp], lr=0.1, max_iter=iters)
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(x)

    def closure():
        opt.zero_grad()
        loss = ce(logits / temp.clamp(min=1e-3), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(temp.detach().clamp(min=1e-3))


def ece(probs, labels, bins=15):
    conf, pred = probs.max(dim=1)
    acc = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            e += (m.float().mean() *
                  abs(acc[m].mean() - conf[m].mean())).item()
    return e


@torch.no_grad()
def mc_dropout_referral(model, x, y, passes=10, top_frac=0.2):
    """MC Dropout with dropout enabled: refer the top_frac most uncertain."""
    model.train()  # enable dropout
    outs = torch.stack([torch.softmax(model(x), 1) for _ in range(passes)])
    model.eval()
    mean = outs.mean(0)
    entropy = -(mean * (mean + 1e-12).log()).sum(1)
    k = max(1, int(top_frac * len(y)))
    referred = torch.zeros(len(y), dtype=torch.bool)
    referred[entropy.topk(k).indices] = True
    top1 = mean.argmax(1)
    kept = ~referred
    return {
        "referral_rate": round(float(referred.float().mean()), 4),
        "kept_accuracy": round(float((top1[kept] == y[kept]).float().mean()), 4),
    }


def md_summary(report):
    L = ["# conformal-med benchmark", ""]
    L.append(f"alpha={report['alpha']} (nominal coverage {1-report['alpha']:.0%}), "
             f"epochs={report['epochs']}, wall {report['wall_hours']}h")
    for ds, d in report["datasets"].items():
        L += ["", f"## {ds} ({d['n_classes']} classes)", "",
              "| shift | method | coverage | mean size | referral |",
              "|---|---|---|---|---|"]
        for sk, row in d["shifts"].items():
            for m, v in row.items():
                if isinstance(v, dict) and "coverage" in v:
                    L.append(f"| {sk} | {m} | {v['coverage']} | "
                             f"{v['mean_size']} | {v['referral_rate']} |")
                elif isinstance(v, dict):
                    L.append(f"| {sk} | {m} | — | — | {v.get('referral_rate', '?')} |")
                else:
                    L.append(f"| {sk} | {m} | — | — | {v} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
