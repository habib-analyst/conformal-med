"""Tests for conformal_med. Core math validated on synthetic distributions."""
import math
import torch

from conformal_med import (
    softmax_score, aps_score, raps_score, SCORE_FNS,
    SplitConformal, ClassConditionalConformal, ConformalClassifier,
    RiskControllingConformal, auto_select,
)
from conformal_med import metrics as M
from conformal_med import shifts


def _synth(n=2000, K=4, seed=0, sharp=3.0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, K, generator=g) * sharp
    # true label = argmax with 80% prob, else random (controllable noise)
    top = logits.argmax(dim=1)
    flip = torch.rand(n, generator=g) < 0.2
    labels = top.clone()
    labels[flip] = torch.randint(0, K, (int(flip.sum()),), generator=g)
    probs = torch.softmax(logits, dim=1)
    return probs, labels


class FakeLoader:
    """Minimal loader yielding (logits, y); model is the identity on logits."""

    def __init__(self, logits, labels, batch=256):
        self.logits = logits
        self.labels = labels
        self.batch = batch

    def __iter__(self):
        n = len(self.labels)
        for i in range(0, n, self.batch):
            j = min(i + self.batch, n)
            yield self.logits[i:j], self.labels[i:j]


class FakeModel(torch.nn.Module):
    def forward(self, x):
        return x  # x is already logits


def test_scores_shapes_and_ranges():
    probs, labels = _synth()
    for name, fn in SCORE_FNS.items():
        s = fn(probs, labels)
        assert s.shape == (len(labels),), name
        assert torch.isfinite(s).all(), name
        assert (s >= 0).all(), name


def test_split_conformal_coverage():
    # exchangeable data: empirical coverage should be >= 1 - alpha (up to slack)
    probs, labels = _synth(n=4000, seed=1)
    logits = torch.log(probs + 1e-12)
    n_cal, n_test = 2000, 2000
    model = FakeModel()
    alpha = 0.1
    cp = SplitConformal(alpha=alpha, score="aps")
    cp.calibrate(model, FakeLoader(logits[:n_cal], labels[:n_cal]))
    sets = cp.predict_sets(model, FakeLoader(logits[n_cal:], labels[n_cal:]))
    cov = M.coverage(sets, labels[n_cal:])
    assert cov >= 1 - alpha - 0.03, f"coverage {cov} below nominal"
    assert M.mean_set_size(sets) < 4.0  # not degenerate full sets


def test_split_never_empty():
    probs, labels = _synth(n=600, seed=2)
    logits = torch.log(probs + 1e-12)
    model = FakeModel()
    cp = SplitConformal(alpha=0.05, score="aps").calibrate(
        model, FakeLoader(logits[:300], labels[:300]))
    sets = cp.predict_sets(model, FakeLoader(logits[300:], labels[300:]))
    assert M.empty_rate(sets) == 0.0


def test_class_conditional_per_class_coverage():
    probs, labels = _synth(n=4000, seed=3, K=3)
    logits = torch.log(probs + 1e-12)
    model = FakeModel()
    cp = ClassConditionalConformal(n_classes=3, alpha=0.1, score="softmax")
    cp.calibrate(model, FakeLoader(logits[:2000], labels[:2000]))
    probs_t, labels_t = probs[2000:], labels[2000:]
    sets = cp.sets_from_probs(probs_t)
    ccc = M.class_conditional_coverage(sets, labels_t, 3)
    for c, cov in ccc.items():
        assert cov >= 0.9 - 0.06, f"class {c} coverage {cov}"


def test_risk_control():
    probs, labels = _synth(n=3000, seed=4)
    logits = torch.log(probs + 1e-12)
    model = FakeModel()
    rc = RiskControllingConformal(beta=0.15, delta=0.1)
    rc.calibrate(model, FakeLoader(logits[:1500], labels[:1500]))
    assert rc.lam_hat is not None and 0 < rc.lam_hat <= 1.0
    probs_t, labels_t = probs[1500:], labels[1500:]
    sets = rc.sets_from_probs(probs_t)
    # default loss: confident-but-wrong singletons; risk should be modest
    top1 = sets.float().argmax(dim=1)
    confident = sets.sum(dim=1) == 1
    risk = float((confident & (top1 != labels_t)).float().mean())
    assert risk <= 0.25, f"risk {risk} too high"


def test_auto_select_rules():
    m, why = auto_select(n_calib=100, n_classes=4)
    assert why == "split(softmax)", why
    m, why = auto_select(n_calib=1000, n_classes=4,
                         class_counts=[1000, 1000, 1000, 10])
    assert why.startswith("class-conditional"), why
    m, why = auto_select(n_calib=1000, n_classes=4,
                         class_counts=[500, 500, 500, 500])
    assert why == "split(raps)", why
    m, why = auto_select(n_calib=1000, risk_control=True)
    assert isinstance(m, RiskControllingConformal)


def test_conformal_classifier_report():
    probs, labels = _synth(n=800, seed=5)
    logits = torch.log(probs + 1e-12)
    model = FakeModel()
    cc = ConformalClassifier(model, alpha=0.1, score="softmax")
    cc.calibrate(FakeLoader(logits[:400], labels[:400]))
    rep = cc.coverage_report(FakeLoader(logits[400:], labels[400:]))
    assert rep["nominal_coverage"] == 0.9
    assert rep["empirical_coverage"] >= 0.85
    assert rep["n"] == 400


def test_shifts_identity_at_zero():
    x = torch.rand(4, 1, 28, 28)
    for name in shifts.SHIFTS:
        y = shifts.apply_shift(x, name, 0.0)
        assert torch.allclose(x, y, atol=1e-5), name


def test_shifts_degrade_with_severity():
    torch.manual_seed(0)
    x = torch.rand(8, 1, 28, 28)
    for name in ("noise", "contrast"):
        d1 = (shifts.apply_shift(x, name, 0.25) - x).abs().mean().item()
        d2 = (shifts.apply_shift(x, name, 1.0) - x).abs().mean().item()
        assert d2 > d1, f"{name}: severity not monotone"


def test_metrics_basic():
    sets = torch.tensor([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.bool)
    labels = torch.tensor([0, 0, 2])
    probs = torch.tensor([[.5, .4, .1], [.7, .2, .1], [.2, .2, .6]])
    assert M.coverage(sets, labels) == 1.0
    assert abs(M.mean_set_size(sets) - 4 / 3) < 1e-6
    assert abs(M.singleton_rate(sets) - 2 / 3) < 1e-6
    assert M.empty_rate(sets) == 0.0
    r = M.referral_stats(sets, labels, probs)
    assert abs(r["referral_rate"] - 1 / 3) < 1e-6
    assert r["kept_accuracy"] == 1.0
