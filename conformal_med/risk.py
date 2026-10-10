"""Risk-controlling prediction sets (RCPS, Angelopoulos et al. 2021).

Instead of controlling miscoverage P(y not in C), control the expected value
of a monotone loss L(C(x), y) <= beta with probability >= 1 - delta.
Use case: "refer for human review when the set is ambiguous" — control the
expected cost of wrong non-referred decisions.
"""
import torch

from .calibrate import _model_probs


class RiskControllingConformal:
    """Calibrate a threshold lambda on a nested family of sets.

    Family: C_lambda(x) = {labels with cumulative ranked mass <= lambda}.
    Loss must be monotone non-increasing in lambda (bigger sets = less loss).
    We binary-search the smallest lambda with an upper confidence bound on
    risk <= beta (Hoeffding-Bentkus style via simple Hoeffding bound).
    """

    def __init__(self, beta=0.1, delta=0.1, device="cpu"):
        self.beta = beta
        self.delta = delta
        self.device = device
        self.lam_hat = None

    @staticmethod
    def _sets(probs, lam):
        order = torch.argsort(probs, dim=1, descending=True)
        ranked = torch.gather(probs, 1, order)
        cumsum = torch.cumsum(ranked, dim=1)
        inc = cumsum <= lam + 1e-12
        inc[:, 0] = True
        out = torch.zeros_like(probs, dtype=torch.bool)
        out.scatter_(1, order, inc)
        return out

    def calibrate(self, model, calib_loader, loss_fn=None):
        """loss_fn(sets, labels) -> per-sample losses in [0,1], monotone in set size."""
        if loss_fn is None:
            # default: 1 if the top-1 prediction is wrong AND set is a singleton
            # (i.e. cost of a confident-but-wrong decision)
            def loss_fn(sets, labels):
                top1 = sets.float().argmax(dim=1)
                confident = sets.sum(dim=1) == 1
                return (confident & (top1 != labels)).float()
        probs, labels = _model_probs(model, calib_loader, self.device)
        n = len(labels)
        # Hoeffding UCB on the risk for a grid of lambdas, pick smallest passing
        grid = torch.linspace(0.05, 1.0, 40)
        for lam in grid:
            sets = self._sets(probs, float(lam))
            r_hat = float(loss_fn(sets, labels).mean())
            ucb = r_hat + ((0.5 * torch.log(torch.tensor(1.0 / self.delta)) / n).sqrt()).item()
            if ucb <= self.beta:
                self.lam_hat = float(lam)
                break
        if self.lam_hat is None:
            self.lam_hat = 1.0  # fall back to full sets
        return self

    def sets_from_probs(self, probs):
        assert self.lam_hat is not None, "calibrate() first"
        return self._sets(probs, self.lam_hat)
