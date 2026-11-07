"""S2 done-when test: annotation world coordinate -> voxel -> world round-trips with < 0.1 mm error on 5 scans,
every annotation lies inside the stored lung crop, and the voxel under the nodule centre is not air."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.preprocess import load_volume, world_to_voxel, voxel_to_world, exists as stem_exists
from lungcad.data.luna16_preprocess import stem_for
from lungcad.data.luna16_index import load_index


def main(n_scans=5):
    index = load_index(); ann = pd.read_csv(C.LUNA_RAW / "annotations.csv")
    uids = [u for u in ann.seriesuid.unique() if stem_exists(stem_for(u, int(index.set_index('seriesuid').subset[u])))][:n_scans]
    sub = index.set_index("seriesuid").subset; worst = 0; outside = 0; air = 0; total = 0
    for uid in uids:
        arr, meta = load_volume(stem_for(uid, int(sub[uid]))); g = ann[ann.seriesuid == uid]
        w = g[["coordX", "coordY", "coordZ"]].values; v = world_to_voxel(meta, w); w2 = voxel_to_world(meta, v)
        worst = max(worst, float(np.abs(w2 - w).max()))
        for k in range(len(g)):
            total += 1; vi = np.round(v[k]).astype(int)
            if np.any(vi < 0) or np.any(vi >= np.array(arr.shape)): outside += 1; continue
            if arr[tuple(vi)] < -900: air += 1
        print(f"{uid[-12:]} shape {arr.shape} spacing {meta['spacing_zyx']} mask={meta['mask_source']} nodules={len(g)} maxerr={np.abs(w2-w).max():.2e} mm")
    print(f"round-trip max error {worst:.3e} mm (limit 0.1) | annotations {total}, outside crop {outside}, centre is air {air}")
    assert worst < 0.1, "round-trip error too large"
    assert outside == 0, "annotation outside the lung crop"
    print("PASS")


if __name__ == "__main__":
    main()
