"""Split (inductive) conformal calibration + class-conditional (Mondrian) variant.

Theory (exchangeable calibration data, target miscoverage alpha):
    q_hat = quantile_{ceil((n+1)(1-alpha))/n} of calibration scores
    => P(y in C(x)) >= 1 - alpha   (marginal coverage guarantee)
Class-conditional: same, computed per true class => coverage holds per class.
"""
import torch

from .scores import SCORE_FNS


@torch.no_grad()
def _model_probs(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        x = x.to(device)
        ps.append(torch.softmax(model(x), dim=1).cpu())
        ys.append(y)
    return torch.cat(ps), torch.cat(ys)


class SplitConformal:
    """Standard split conformal predictor."""

    def __init__(self, alpha=0.1, score="aps", device="cpu", **score_kw):
        assert 0 < alpha < 1
        self.alpha = alpha
        self.score_fn = SCORE_FNS[score]
        self.score_name = score
        self.score_kw = score_kw
        self.device = device
        self.q_hat = None

    def calibrate(self, model, calib_loader):
        probs, labels = _model_probs(model, calib_loader, self.device)
        scores = self.score_fn(probs, labels, **self.score_kw)
        n = len(scores)
        level = min(1.0, (n + 1) * (1 - self.alpha) / n)
        self.q_hat = float(torch.quantile(scores, level))
        return self

    def predict_sets(self, model, loader):
        assert self.q_hat is not None, "calibrate() first"
        probs, _ = _model_probs(model, loader, self.device)
        return self.sets_from_probs(probs)

    def sets_from_probs(self, probs):
        """Build prediction sets directly from probability matrix (n, K)."""
        n, K = probs.shape
        order = torch.argsort(probs, dim=1, descending=True)
        ranked = torch.gather(probs, 1, order)
        cumsum = torch.cumsum(ranked, dim=1)
        if self.score_name == "softmax":
            # include labels with p >= 1 - q_hat
            include = probs >= (1.0 - self.q_hat)
        else:
            # include labels while cumulative mass <= q_hat (standard APS rule)
            include_ranked = cumsum <= self.q_hat + 1e-12
            # always include the top-1 label (never empty sets)
            include_ranked[:, 0] = True
            include = torch.zeros_like(probs, dtype=torch.bool)
            include.scatter_(1, order, include_ranked)
        return include


class ClassConditionalConformal:
    """Mondrian conformal: one threshold per class => P(y in C(x) | y=c) >= 1-alpha."""

    def __init__(self, n_classes, alpha=0.1, score="softmax", device="cpu", **kw):
        self.n_classes = n_classes
        self.alpha = alpha
        self.device = device
        self.q_hats = torch.full((n_classes,), float("inf"))
        self._inner = SplitConformal(alpha=alpha, score=score, device=device, **kw)

    def calibrate(self, model, calib_loader):
        probs, labels = _model_probs(model, calib_loader, self.device)
        scores = self._inner.score_fn(probs, labels, **self._inner.score_kw)
        for c in range(self.n_classes):
            sc = scores[labels == c]
            if len(sc) == 0:
                continue
            n = len(sc)
            level = min(1.0, (n + 1) * (1 - self.alpha) / n)
            self.q_hats[c] = float(torch.quantile(sc, level))
        return self

    def sets_from_probs(self, probs):
        n, K = probs.shape
        assert K == self.n_classes
        out = torch.zeros((n, K), dtype=torch.bool)
        for c in range(K):
            q = self.q_hats[c].item()
            if self._inner.score_name == "softmax":
                out[:, c] = probs[:, c] >= (1.0 - q)
            else:
                # per-class APS threshold on cumulative mass
                order = torch.argsort(probs, dim=1, descending=True)
                ranked = torch.gather(probs, 1, order)
                cumsum = torch.cumsum(ranked, dim=1)
                rank_of_c = (order == c).nonzero(as_tuple=False)[:, 1]
                s_c = cumsum[torch.arange(n), rank_of_c]
                out[:, c] = s_c <= q + 1e-12
        # never empty: fall back to top-1
        empty = ~out.any(dim=1)
        if empty.any():
            out[empty, probs[empty].argmax(dim=1)] = True
        return out
