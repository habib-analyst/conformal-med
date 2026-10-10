"""conformal_med — distribution-free uncertainty with guarantees for medical imaging.

Stop reporting accuracy; report what you can guarantee.

Quick start:
    from conformal_med import ConformalClassifier
    cc = ConformalClassifier(model, alpha=0.1)   # 90% coverage target
    cc.calibrate(calib_loader)                    # split conformal calibration
    sets = cc.predict_sets(test_loader)           # prediction sets, not labels
    print(cc.coverage_report(test_loader))        # empirical coverage vs nominal
"""
from .scores import softmax_score, aps_score, raps_score, SCORE_FNS
from .calibrate import SplitConformal, ClassConditionalConformal
from .predict import ConformalClassifier
from .risk import RiskControllingConformal
from .auto import auto_select
from . import metrics, shifts

__all__ = [
    "softmax_score", "aps_score", "raps_score", "SCORE_FNS",
    "SplitConformal", "ClassConditionalConformal",
    "ConformalClassifier", "RiskControllingConformal",
    "auto_select", "metrics", "shifts",
]
__version__ = "0.1.0"
