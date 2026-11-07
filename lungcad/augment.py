"""S4 — dose and kernel augmentation, applied on GPU to normalised [0,1] batches.

Simulated dose reduction: image-domain Poisson–Gaussian noise. CT noise after reconstruction is
approximately Gaussian with variance that grows with attenuation (photon starvation through dense
tissue), so we draw  n ~ N(0, sqrt(a * I + b))  with I the normalised intensity, a (signal-dependent
term) and b (electronic term) sampled per volume. Kernel/reconstruction variation: Gaussian blur
(softer kernel) or unsharp masking (sharper kernel) with random strength. Both are cheap enough to
run every iteration on 24 GB.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

HU_RANGE = 1400.0  # HU_MAX - HU_MIN, converts HU-sigma <-> normalised-sigma


def _gauss3d(sigma: float, device, dtype):
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2); k = k / k.sum()
    return k, r


def separable_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """x: (B,C,D,H,W). Separable 3D Gaussian, reflect padding."""
    if sigma <= 0:
        return x
    k, r = _gauss3d(sigma, x.device, x.dtype)
    B, Cc = x.shape[:2]
    y = x
    for dim in (2, 3, 4):
        shape = [1, 1, 1, 1, 1]; shape[dim] = k.numel()
        w = k.view(shape).expand(Cc, 1, *shape[2:]).contiguous()
        pad = [0] * 6; pad[2 * (4 - dim)] = r; pad[2 * (4 - dim) + 1] = r
        y = F.conv3d(F.pad(y, pad, mode="reflect"), w, groups=Cc)
    return y


@torch.no_grad()
def dose_kernel_augment(x: torch.Tensor, p_noise: float = 0.5, p_kernel: float = 0.5,
                        sigma_hu: tuple = (10.0, 60.0), a_frac: float = 0.5,
                        blur_sigma: tuple = (0.3, 1.0), sharpen: tuple = (0.3, 1.0)) -> torch.Tensor:
    """Per-sample random dose (noise) and kernel (blur/sharpen) perturbation. x in [0,1], (B,C,D,H,W)."""
    B = x.shape[0]
    out = x.clone()
    for i in range(B):
        xi = out[i:i + 1]
        if torch.rand(()) < p_kernel:
            if torch.rand(()) < 0.5:
                s = float(torch.empty(()).uniform_(*blur_sigma))
                xi = separable_blur(xi, s)
            else:
                amt = float(torch.empty(()).uniform_(*sharpen))
                xi = xi + amt * (xi - separable_blur(xi, 0.8))
        if torch.rand(()) < p_noise:
            s = float(torch.empty(()).uniform_(*sigma_hu)) / HU_RANGE          # total sigma in normalised units
            var = s * s
            a = a_frac * var; b = (1 - a_frac) * var                            # signal-dependent + electronic
            sig = torch.sqrt(a * xi.clamp(0, 1) + b)
            xi = xi + torch.randn_like(xi) * sig
        out[i:i + 1] = xi.clamp_(0, 1)
    return out
