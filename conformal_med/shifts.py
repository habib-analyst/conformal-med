"""Synthetic distribution shifts at graded severities (for the shift benchmark).

Conformal guarantees assume exchangeability; these shifts break it on
purpose so we can MEASURE the coverage gap. Shifts: gaussian noise,
gaussian blur, contrast scaling.
"""
import torch
import torch.nn.functional as F


def _to_float(x):
    x = x.float()
    return x / 255.0 if x.max() > 1.5 else x


def gaussian_noise(x, severity):
    """severity in [0,1] -> sigma = 0.35 * severity."""
    x = _to_float(x)
    return torch.clamp(x + torch.randn_like(x) * 0.35 * severity, 0, 1)


def gaussian_blur(x, severity):
    """severity in [0,1] -> kernel 3/5/7."""
    if severity <= 0:
        return x
    x = _to_float(x)
    k = [3, 5, 7][min(2, int(severity * 3))]
    pad = k // 2
    # depthwise blur per channel
    c = x.shape[1]
    kernel = torch.ones(1, 1, k, k, device=x.device) / (k * k)
    kernel = kernel.repeat(c, 1, 1, 1)
    return torch.clamp(F.conv2d(x, kernel, padding=pad, groups=c), 0, 1)


def contrast(x, severity):
    """severity in [0,1] -> contrast factor in [1, 0.2]."""
    x = _to_float(x)
    f = 1.0 - 0.8 * severity
    mean = x.mean(dim=(2, 3), keepdim=True)
    return torch.clamp((x - mean) * f + mean, 0, 1)


SHIFTS = {"noise": gaussian_noise, "blur": gaussian_blur, "contrast": contrast}
SEVERITIES = [0.0, 0.25, 0.5, 0.75, 1.0]


def apply_shift(x, name, severity):
    return SHIFTS[name](x, severity)
