"""High-level ConformalClassifier: wraps any PyTorch classifier."""
import torch

from .calibrate import SplitConformal, ClassConditionalConformal


class ConformalClassifier:
    """Drop-in uncertainty wrapper.

        cc = ConformalClassifier(model, alpha=0.1, score="aps")
        cc.calibrate(calib_loader)
        sets = cc.predict_sets(test_loader)          # bool (n, K)
        print(cc.coverage_report(test_loader))        # dict of metrics
    """

    def __init__(self, model, alpha=0.1, score="aps",
                 class_conditional=False, n_classes=None, device="cpu",
                 **score_kw):
        self.model = model
        self.alpha = alpha
        self.device = device
        if class_conditional:
            assert n_classes is not None
            self._cp = ClassConditionalConformal(
                n_classes, alpha=alpha, score=score, device=device, **score_kw)
        else:
            self._cp = SplitConformal(
                alpha=alpha, score=score, device=device, **score_kw)

    def calibrate(self, calib_loader):
        self._cp.calibrate(self.model, calib_loader)
        return self

    def predict_sets(self, loader):
        probs, labels = self._probs_labels(loader)
        return self._cp.sets_from_probs(probs)

    def _probs_labels(self, loader):
        self.model.eval()
        ps, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                ps.append(torch.softmax(self.model(x.to(self.device)), 1).cpu())
                ys.append(y)
        return torch.cat(ps), torch.cat(ys)

    def coverage_report(self, loader):
        """Empirical coverage, mean set size, empty/singleton rates."""
        from . import metrics as M
        probs, labels = self._probs_labels(loader)
        sets = self._cp.sets_from_probs(probs)
        return {
            "nominal_coverage": 1 - self.alpha,
            "empirical_coverage": M.coverage(sets, labels),
            "mean_set_size": M.mean_set_size(sets),
            "singleton_rate": M.singleton_rate(sets),
            "empty_rate": M.empty_rate(sets),
            "n": len(labels),
        }
