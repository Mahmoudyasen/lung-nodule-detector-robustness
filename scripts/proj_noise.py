"""Projection-domain low-dose simulation by noise-only insertion. Runs in the separate `astra` conda env.

Per axial slice (2D parallel-beam geometry: Siemens scanners rebin fan to parallel before FBP; ray spacing of the
SOMATOM Definition at the isocentre, as in the TCIA LDCT data):
  1. HU -> linear attenuation mu = MU_WATER * (1 + HU/1000), air clipped to 0
  2. forward-project -> line integrals p (ASTRA)
  3. noise-only sinogram for dose fraction a (Yu et al., JCAT 2012):
       var(p) = (1-a)/a * e^p / I0 * [1 + (1+a)/a * sigma_e^2 * e^p / I0]
     i.e. the EXTRA variance that turns the full-dose measurement into an a-dose measurement, incl. electronic noise
  4. FBP of the noise sinogram only (Ram-Lak) -> noise image in mu units -> HU, added to the original slice
Only the noise is reconstructed, so the original image (and its kernel) is untouched apart from the inserted noise.
FBP is linear, so its own reconstruction filter shapes the inserted noise texture (correlated, anisotropic, streaky
near dense structures), which is the point versus image-domain white noise. Real scans use tube-current modulation,
which keeps noise far more even along the body than a constant I0; LUNA16 has no per-slice mA, so modulation is
modelled per slice as a power law (tcm_factor) whose two parameters are fitted on the TCIA full/10 % pairs
(calibrate_tcia.py) at a fixed reference I0.
"""
from __future__ import annotations
import numpy as np
import astra

MU_WATER = 0.0192          # 1/mm, ~70 keV effective
SID, SDD = 595.0, 1085.6   # source-isocentre and source-detector distance, mm (Definition geometry)
N_DET, RAY_PITCH = 736, 1.2858 * SID / SDD   # channels; ray spacing at the isocentre after rebinning (~0.705 mm)
N_VIEWS = 1152             # parallel views over 180 degrees


class SliceGeom:
    """One slice geometry (rows x cols at pixel spacing (dy, dx) mm, centred on the isocentre)."""

    def __init__(self, rows: int, cols: int, dy: float, dx: float, cuda: bool = True):
        hy, hx = rows * dy / 2, cols * dx / 2
        self.vg = astra.create_vol_geom(rows, cols, -hx, hx, -hy, hy)
        ang = np.linspace(0, np.pi, N_VIEWS, endpoint=False)
        self.pg = astra.create_proj_geom("parallel", RAY_PITCH, N_DET, ang)
        self.cuda = cuda and astra.use_cuda()
        self.pid = astra.create_projector("cuda" if self.cuda else "linear", self.pg, self.vg)

    def fp(self, img: np.ndarray) -> np.ndarray:
        sid, s = astra.create_sino(img.astype(np.float32), self.pid); astra.data2d.delete(sid)
        return s

    def fbp(self, sino: np.ndarray, filt: str = "ram-lak") -> np.ndarray:
        """filt: ASTRA filter name, optionally with a frequency cutoff 'name:D' (D in (0, 1], ASTRA option FilterD)."""
        sid = astra.data2d.create("-sino", self.pg, sino.astype(np.float32))
        rid = astra.data2d.create("-vol", self.vg, 0)
        cfg = astra.astra_dict("FBP_CUDA" if self.cuda else "FBP")
        name, _, d = filt.partition(":")
        cfg.update(ReconstructionDataId=rid, ProjectionDataId=sid, ProjectorId=self.pid, FilterType=name)
        if d:
            cfg["option"] = {"FilterD": float(d)}
        alg = astra.algorithm.create(cfg); astra.algorithm.run(alg)
        out = astra.data2d.get(rid)
        astra.algorithm.delete(alg); astra.data2d.delete([sid, rid])
        return out

    def close(self):
        astra.projector.delete(self.pid)


def noise_var(p: np.ndarray, a: float, i0: float, sigma_e: float) -> np.ndarray:
    ep = np.exp(np.clip(p, 0, 20)) / i0
    return (1 - a) / a * ep * (1 + (1 + a) / a * sigma_e ** 2 * ep)


def body_mask(hu: np.ndarray) -> np.ndarray:
    """Patient body without air and lungs (HU > -400 inside the filled body outline): where noise SD is measured."""
    from scipy import ndimage
    body = ndimage.binary_fill_holes(ndimage.binary_opening(hu > -500, iterations=3))
    return body & (hu > -400)


def tcm_factor(sd_const: float, a: float, tcm) -> float:
    """Tube-current modulation as a power law fitted on TCIA (calibrate_tcia.py): with modulation the inserted-noise SD
    of a slice is c * s^beta, where s is the SD the same slice gets at the constant reference I0, both expressed at the
    10 % reference dose (noise SD scales with sqrt((1-a)/a)). Returns the factor that turns the constant-I0 noise image
    into the modulated one (beta = 1, c = 1: no modulation)."""
    c, beta = tcm
    s10 = sd_const * np.sqrt(9.0 / ((1 - a) / a))
    return float(c * s10 ** beta / s10)


def insert_noise_slice(hu: np.ndarray, geo: SliceGeom, a: float, i0: float, sigma_e: float, rng: np.random.Generator,
                       tcm=None, scale: float = 1.0, filt: str = "ram-lak") -> np.ndarray:
    """Return the HU noise image to ADD to `hu` so that it looks like an a-dose acquisition (tcm: (c, beta) or None).
    scale: extra SD factor, e.g. sqrt(reference thickness / slice thickness) (thinner slices are noisier at equal dose).
    filt: reconstruction filter of the inserted noise (match the image's kernel). The modulation factor is always computed
    on the Ram-Lak reconstruction (as calibrated), then applied to the `filt` reconstruction of the SAME projection noise,
    so a softer filter lowers the noise exactly as a softer kernel would at the same dose."""
    mu = np.clip(MU_WATER * (1 + hu.astype(np.float32) / 1000.0), 0, None)
    p = geo.fp(mu)
    dn = rng.standard_normal(p.shape).astype(np.float32) * np.sqrt(noise_var(p, a, i0, sigma_e)).astype(np.float32)
    n = geo.fbp(dn) / MU_WATER * 1000.0
    f = 1.0
    if tcm is not None:
        m = body_mask(hu)
        if m.sum() > 1000:
            f = tcm_factor(float(n[m].std()), a, tcm)
    if filt != "ram-lak":
        n = geo.fbp(dn, filt) / MU_WATER * 1000.0
    return n * f * scale


def insert_noise_volume(hu: np.ndarray, spacing_yx, a: float, i0: float, sigma_e: float, seed: int, cuda: bool = True,
                        tcm=None, scale: float = 1.0, filt: str = "ram-lak") -> np.ndarray:
    """hu: (z, y, x) original-resolution volume. Returns hu + noise (float32), slice by slice, deterministic in `seed`."""
    geo = SliceGeom(hu.shape[1], hu.shape[2], spacing_yx[0], spacing_yx[1], cuda=cuda)
    rng = np.random.default_rng(seed); out = np.empty(hu.shape, np.float32)
    try:
        for k in range(hu.shape[0]):
            out[k] = hu[k] + insert_noise_slice(hu[k], geo, a, i0, sigma_e, rng, tcm, scale, filt)
    finally:
        geo.close()
    return out


def _selftest():
    """FBP(FP(phantom)) must return the phantom (checks geometry and FBP scaling)."""
    n, d = 512, 0.75
    yy, xx = (np.mgrid[:n, :n] - n / 2 + 0.5) * d
    ph = np.where(xx ** 2 / 150 ** 2 + yy ** 2 / 110 ** 2 < 1, MU_WATER, 0).astype(np.float32)
    ph[(xx - 40) ** 2 + yy ** 2 < 20 ** 2] = MU_WATER * 0.2          # a lung-like hole
    for cuda in (False, True) if astra.use_cuda() else (False,):
        g = SliceGeom(n, n, d, d, cuda=cuda); rec = g.fbp(g.fp(ph)); g.close()
        inner = (xx ** 2 / 140 ** 2 + yy ** 2 / 100 ** 2 < 1) & ((xx - 40) ** 2 + yy ** 2 > 25 ** 2)
        print(f"selftest cuda={cuda}: water {rec[inner].mean() / MU_WATER * 1000 - 1000:+.1f} HU (expect 0), "
              f"hole {rec[(xx - 40) ** 2 + yy ** 2 < 15 ** 2].mean() / MU_WATER * 1000 - 1000:+.1f} HU (expect -800)")


if __name__ == "__main__":
    _selftest()
