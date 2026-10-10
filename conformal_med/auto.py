"""Auto-select the conformal method by data regime.

Rules of thumb (documented, deterministic):
- n_calib < 200            -> softmax score (stable quantiles on few samples)
- class imbalance (min class freq < 5% of max) -> class-conditional
- else                     -> RAPS (adaptive sets, small on easy data)
- risk control requested    -> RiskControllingConformal
"""
from .calibrate import SplitConformal, ClassConditionalConformal
from .risk import RiskControllingConformal


def auto_select(n_calib, class_counts=None, alpha=0.1, risk_control=False,
                n_classes=None, device="cpu"):
    if risk_control:
        return RiskControllingConformal(device=device), "risk-controlling"
    if class_counts is not None:
        import numpy as np
        counts = np.asarray(class_counts, dtype=float)
        if counts.min() < 0.05 * counts.max() and n_classes is not None:
            return ClassConditionalConformal(
                n_classes, alpha=alpha, score="softmax", device=device), \
                "class-conditional(softmax)"
    if n_calib < 200:
        return SplitConformal(alpha=alpha, score="softmax", device=device), \
            "split(softmax)"
    return SplitConformal(alpha=alpha, score="raps", device=device), \
        "split(raps)"
