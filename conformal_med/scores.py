"""Nonconformity score functions.

A nonconformity score s(x, y) measures how "strange" a label y is for input x.
Higher = stranger. Conformal prediction thresholds these scores on a
calibration set to build prediction sets with guaranteed coverage.
"""
import torch


def softmax_score(probs, labels):
    """1 - p(y|x). Simple, fast. probs: (n, K), labels: (n,)."""
    return 1.0 - probs[torch.arange(len(labels)), labels]


def aps_score(probs, labels, randomized=False, rng=None):
    """Adaptive Prediction Sets (Romano et al. 2020): cumulative mass of
    labels ranked at least as likely as the true label. Adapts set size to
    instance difficulty. randomized=True adds tie-breaking noise (smoother
    coverage at the cost of stochasticity)."""
    order = torch.argsort(probs, dim=1, descending=True)
    ranked = torch.gather(probs, 1, order)
    cumsum = torch.cumsum(ranked, dim=1)
    # rank position of the true label
    rank = (order == labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
    s = cumsum[torch.arange(len(labels)), rank]
    if randomized:
        u = torch.rand(len(labels)) if rng is None else torch.tensor(
            [rng.random() for _ in range(len(labels))])
        s = s - u * ranked[torch.arange(len(labels)), rank]
    return s


def raps_score(probs, labels, k_reg=2, lam_reg=0.01, **kw):
    """Regularized APS (Angelopoulos et al. 2021): penalizes large sets by
    adding lam_reg * max(0, rank - k_reg). Keeps sets small on easy data
    while preserving the adaptivity of APS."""
    order = torch.argsort(probs, dim=1, descending=True)
    ranked = torch.gather(probs, 1, order)
    cumsum = torch.cumsum(ranked, dim=1)
    rank = (order == labels.unsqueeze(1)).nonzero(as_tuple=False)[:, 1]
    s = cumsum[torch.arange(len(labels)), rank]
    s = s + lam_reg * torch.clamp(rank - k_reg, min=0).float()
    return s


SCORE_FNS = {
    "softmax": softmax_score,
    "aps": aps_score,
    "raps": raps_score,
}
