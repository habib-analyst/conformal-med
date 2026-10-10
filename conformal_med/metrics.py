"""Evaluation metrics for conformal prediction sets."""
import torch


def coverage(sets, labels):
    """Fraction of sets containing the true label."""
    return float(sets[torch.arange(len(labels)), labels].float().mean())


def mean_set_size(sets):
    return float(sets.float().sum(dim=1).mean())


def singleton_rate(sets):
    return float((sets.float().sum(dim=1) == 1).float().mean())


def empty_rate(sets):
    return float((sets.float().sum(dim=1) == 0).float().mean())


def size_stratified_coverage(sets, labels, probs):
    """Coverage conditioned on set size — diagnoses 'coverage by luck'
    (e.g. huge sets on hard cases masking misses on easy ones)."""
    sizes = sets.float().sum(dim=1).long()
    out = {}
    for s in sorted(set(sizes.tolist())):
        m = sizes == s
        out[int(s)] = {
            "n": int(m.sum()),
            "coverage": float(sets[m][torch.arange(int(m.sum())),
                                     labels[m]].float().mean()),
        }
    return out


def class_conditional_coverage(sets, labels, n_classes):
    out = {}
    for c in range(n_classes):
        m = labels == c
        if int(m.sum()) == 0:
            continue
        out[c] = float(sets[m][torch.arange(int(m.sum())),
                               labels[m]].float().mean())
    return out


def referral_stats(sets, labels, probs):
    """Referral = non-singleton set. Reports referral rate and the accuracy
    of the top-1 prediction on non-referred cases (what the clinician sees)."""
    sizes = sets.float().sum(dim=1)
    referred = sizes != 1
    top1 = probs.argmax(dim=1)
    kept = ~referred
    return {
        "referral_rate": float(referred.float().mean()),
        "kept_accuracy": float((top1[kept] == labels[kept]).float().mean())
        if kept.any() else float("nan"),
        "kept_n": int(kept.sum()),
    }
