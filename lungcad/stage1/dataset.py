"""Stage-1 patch dataset over the preprocessed LUNA16 archive (int16 LZ4 volumes + voxel annotation table).

Positive-biased sampling: with probability pos_frac a patch is placed so that a randomly chosen nodule lies fully
inside it (uniform jitter); otherwise the patch is placed uniformly at random inside the lung crop. Boxes are
axis-aligned cubes of the annotated diameter, in (z,y,x) corner-corner mode = MONAI StandardMode on our axis order.
Augmentation (CPU, in workers): flips on all three axes, in-plane transpose (with flips = all 8 dihedral symmetries),
isotropic zoom U(0.9,1.1) with p=0.3. Dose/kernel augmentation runs on GPU in the training loop.
"""
from __future__ import annotations
import zlib
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from .. import config as C
from ..preprocess import load_volume, normalize
from ..data.luna16_preprocess import stem_for


class Luna16DetPatches(Dataset):
    def __init__(self, index: pd.DataFrame, uids: list[str], annotations_voxel: pd.DataFrame, patch_size=(80, 192, 192),
                 pos_frac: float = 0.7, samples_per_scan: int = 4, augment: bool = True, zoom_p: float = 0.3, seed: int = 0):
        self.index = index.set_index("seriesuid")
        self.uids = [u for u in uids if u in self.index.index]
        self.ann = {k: g for k, g in annotations_voxel.groupby("seriesuid")}
        self.patch = np.array(patch_size); self.pos_frac = pos_frac; self.spc = samples_per_scan
        self.augment = augment; self.zoom_p = zoom_p
        self.rng = np.random.RandomState(seed)
        # per-item RNG salt: stable across processes (Python's hash() is randomised per interpreter start,
        # which made patch draws differ between runs) and dependent on the seed (seeds 0/1/2 now differ)
        self._salt = (zlib.crc32(self.uids[0].encode()) % 1000 + seed * 15_485_863) if self.uids else seed
        # shared memory: persistent DataLoader workers hold a copy of this dataset, so a plain int set in the main
        # process never reaches them (every epoch after worker start-up would replay the same patches)
        self._epoch = torch.zeros(1, dtype=torch.long).share_memory_()

    def __len__(self):
        return len(self.uids) * self.spc

    @property
    def epoch(self) -> int:
        return int(self._epoch[0])

    def set_epoch(self, e: int):
        self._epoch[0] = e

    def _boxes(self, uid):
        g = self.ann.get(uid)
        if g is None or len(g) == 0:
            return np.zeros((0, 6), np.float32)
        c = g[["vz", "vy", "vx"]].values; r = g[["rz_vox", "ry_vox", "rx_vox"]].values
        return np.concatenate([c - r, c + r], 1).astype(np.float32)

    def __getitem__(self, i):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.RandomState((self.epoch * 1_000_003 + i * 7919 + (wi.id if wi else 0) * 104729 + self._salt) % 2**31)
        uid = self.uids[i // self.spc]
        arr, meta = load_volume(stem_for(uid, int(self.index.loc[uid, "subset"])))
        boxes = self._boxes(uid)
        shape = np.array(arr.shape)
        zoom = 1.0
        if self.augment and rng.rand() < self.zoom_p:
            zoom = float(rng.uniform(0.9, 1.1))
        region = np.round(self.patch / zoom).astype(int)          # voxels read from the volume, resized to patch
        pad = np.maximum(region - shape, 0)
        if pad.any():
            arr = np.pad(arr, [(0, int(p)) for p in pad], constant_values=int(C.HU_MIN)); shape = np.array(arr.shape)
        # ---- choose the patch start
        if len(boxes) and rng.rand() < self.pos_frac:
            b = boxes[rng.randint(len(boxes))]
            ctr = (b[:3] + b[3:]) / 2; half = (b[3:] - b[:3]) / 2
            lo = np.ceil(ctr + half + 2 - region).astype(int); hi = np.floor(ctr - half - 2).astype(int)
            lo = np.maximum(lo, 0); hi = np.minimum(hi, shape - region)
            start = np.array([rng.randint(l, h + 1) if h >= l else int(np.clip(ctr[k] - region[k] // 2, 0, shape[k] - region[k])) for k, (l, h) in enumerate(zip(lo, hi))])
        else:
            start = np.array([rng.randint(0, s - r + 1) for s, r in zip(shape, region)])
        crop = arr[start[0]:start[0] + region[0], start[1]:start[1] + region[1], start[2]:start[2] + region[2]]
        bx = boxes - np.r_[start, start].astype(np.float32)
        img = torch.from_numpy(normalize(crop.astype(np.float32)))[None]          # (1, D, H, W) in [0,1]
        if zoom != 1.0:
            img = F.interpolate(img[None], size=tuple(int(p) for p in self.patch), mode="trilinear", align_corners=False)[0]
            bx = bx * (self.patch / region).astype(np.float32).repeat(2)[None] if len(bx) else bx
        size = self.patch.astype(np.float32)
        if len(bx):
            ctr = (bx[:, :3] + bx[:, 3:]) / 2
            keep = np.all((ctr >= 0) & (ctr < size), axis=1)
            bx = bx[keep]
            bx = np.concatenate([np.clip(bx[:, :3], 0, size), np.clip(bx[:, 3:], 0, size)], 1)
            bx = bx[np.all(bx[:, 3:] - bx[:, :3] >= 1.0, axis=1)]
        # ---- augmentation
        if self.augment:
            for ax in range(3):
                if rng.rand() < 0.5:
                    img = torch.flip(img, dims=[ax + 1])
                    if len(bx):
                        lo_, hi_ = bx[:, ax].copy(), bx[:, 3 + ax].copy()
                        bx[:, ax], bx[:, 3 + ax] = size[ax] - hi_, size[ax] - lo_
            if rng.rand() < 0.5 and self.patch[1] == self.patch[2]:
                img = img.transpose(2, 3).contiguous()
                if len(bx):
                    bx = bx[:, [0, 2, 1, 3, 5, 4]]
        boxes_t = torch.from_numpy(np.ascontiguousarray(bx, dtype=np.float32)).reshape(-1, 6)
        labels_t = torch.zeros(len(boxes_t), dtype=torch.long)
        return {"image": img.contiguous(), "box": boxes_t, "label": labels_t, "uid": uid}


def collate(batch):
    return {"image": torch.stack([b["image"] for b in batch]), "box": [b["box"] for b in batch],
            "label": [b["label"] for b in batch], "uid": [b["uid"] for b in batch]}


def load_full_volume(uid: str, subset: int):
    """Whole preprocessed scan as (1,1,D,H,W) float tensor in [0,1] + meta, for sliding-window inference."""
    arr, meta = load_volume(stem_for(uid, subset))
    return torch.from_numpy(normalize(arr.astype(np.float32)))[None, None], meta
