"""Stage-1 training (S3 fold-0 gate, S5 full 10-fold). Resumable; every run writes run_manifest.json (git hash,
config hash, env freeze) next to its checkpoints. Model selection by validation CPM (FROC), not loss.

    python -m lungcad.stage1.train --config configs/stage1_retinanet.yaml --fold 0 --seed 0
"""
from __future__ import annotations
import argparse, json, math, os, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .. import config as C
from ..augment import dose_kernel_augment
from ..data.luna16_index import load_index, get_split
from ..version import write_run_manifest, run_tag
from .model import build_detector, set_inferer, count_params
from .dataset import Luna16DetPatches, collate
from .infer import infer_scans
from ..eval.froc import evaluate


def seed_all(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def lr_at(it, base, warmup, total, floor=0.01):
    if it < warmup:
        return base * (it + 1) / warmup
    p = (it - warmup) / max(1, total - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--fold", type=int, required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None); ap.add_argument("--resume", action="store_true"); ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="2 iterations + 2 val scans, for harness checks")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.epochs: cfg["train"]["epochs"] = a.epochs
    out = Path(a.out or (C.STAGE1 / f"fold{a.fold}" / f"seed{a.seed}")); out.mkdir(parents=True, exist_ok=True)
    seed_all(a.seed); torch.backends.cudnn.benchmark = True
    write_run_manifest(out, cfg, {"fold": a.fold, "seed": a.seed, "stage": 1})
    index = load_index(); split = get_split(index, a.fold)
    annv = pd.read_csv(C.LUNA_PRE / "annotations_voxel.csv")
    tr = cfg["train"]; ds_cfg = cfg["data"]
    ds = Luna16DetPatches(index, split["train"], annv, patch_size=ds_cfg["patch_size"], pos_frac=ds_cfg["pos_frac"],
                          samples_per_scan=ds_cfg["samples_per_scan"], augment=True, zoom_p=ds_cfg.get("zoom_p", 0.3), seed=a.seed)
    dl = DataLoader(ds, batch_size=tr["batch_size"], shuffle=True, num_workers=tr["workers"], collate_fn=collate,
                    pin_memory=True, persistent_workers=tr["workers"] > 0, drop_last=True, prefetch_factor=4 if tr["workers"] > 0 else None)
    det = build_detector(cfg["model"])
    set_inferer(det, cfg["infer"]["roi_size"], cfg["infer"]["overlap"], cfg["infer"]["sw_batch_size"])
    print(f"detector params {count_params(det.network)/1e6:.1f}M  train scans {len(split['train'])} val {len(split['val'])}  run {run_tag(cfg)}")
    if tr["optimizer"] == "sgd":
        opt = torch.optim.SGD(det.network.parameters(), lr=tr["lr"], momentum=0.9, weight_decay=tr["weight_decay"], nesterov=True)
    else:
        opt = torch.optim.AdamW(det.network.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=tr["amp"])
    iters_per_epoch = len(dl); total_iters = tr["epochs"] * iters_per_epoch
    state = {"epoch": 0, "it": 0, "best_cpm": -1.0, "best_epoch": -1}
    ck_last = out / "last.pt"
    if a.resume and ck_last.exists():
        ck = torch.load(ck_last, map_location="cuda", weights_only=False)
        det.network.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"]); state = ck["state"]
        print("resumed from", state)
    tb = SummaryWriter(out / "tb"); log = open(out / "train.log", "a")
    aug = cfg.get("augment", {}); val_uids = split["val"]
    if a.smoke:
        tr["epochs"] = state["epoch"] + 1; val_uids = val_uids[:2]
    t_start = time.time()
    for epoch in range(state["epoch"], tr["epochs"]):
        det.train(); ds.set_epoch(epoch); t0 = time.time(); losses = []
        for bi, batch in enumerate(dl):
            if a.smoke and bi >= 2: break
            lr = lr_at(state["it"], tr["lr"], tr["warmup_iters"], total_iters)
            for g in opt.param_groups: g["lr"] = lr
            img = batch["image"].cuda(non_blocking=True)
            if aug.get("dose_kernel", False):
                img = dose_kernel_augment(img, **aug.get("dose_kernel_params", {}))
            targets = [{"box": b.cuda(), "label": l.cuda()} for b, l in zip(batch["box"], batch["label"])]
            with torch.autocast("cuda", dtype=torch.float16, enabled=tr["amp"]):
                out_losses = det(img, targets)
                loss = out_losses[det.cls_key] + out_losses[det.box_reg_key]
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(det.network.parameters(), tr.get("grad_clip", 10.0))
            scaler.step(opt); scaler.update(); state["it"] += 1
            losses.append([loss.item(), out_losses[det.cls_key].item(), out_losses[det.box_reg_key].item()])
            if state["it"] % 20 == 0:
                tb.add_scalar("train/loss", loss.item(), state["it"]); tb.add_scalar("train/lr", lr, state["it"])
                tb.add_scalar("train/cls", losses[-1][1], state["it"]); tb.add_scalar("train/box", losses[-1][2], state["it"])
        state["epoch"] = epoch + 1
        m = np.mean(losses, 0) if losses else np.zeros(3)
        msg = f"epoch {epoch+1}/{tr['epochs']} loss {m[0]:.4f} cls {m[1]:.4f} box {m[2]:.4f} lr {lr:.2e} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.1f}G"
        # ---- validation by FROC/CPM on the val subset (full sliding-window inference)
        if (epoch + 1) % tr["val_every"] == 0 or epoch + 1 == tr["epochs"] or a.smoke:
            det.eval(); tv = time.time()
            cands = infer_scans(det, index, val_uids, cfg["infer"], amp=tr["amp"])
            res = evaluate(cands, val_uids, n_bootstrap=0, tag=f"val_e{epoch+1}")
            cpm = res["cpm"]; tb.add_scalar("val/cpm", cpm, epoch + 1); tb.add_scalar("val/max_sens", res["max_sensitivity"], epoch + 1)
            tb.add_scalar("val/fp_per_scan", res["n_fp"] / max(1, res["n_scans"]), epoch + 1)
            msg += f" | val CPM {cpm:.4f} sens@8 {res['sens_at_fp'][-1]:.3f} maxsens {res['max_sensitivity']:.3f} FP/scan {res['n_fp']/max(1,res['n_scans']):.0f} ({time.time()-tv:.0f}s)"
            with open(out / "val_history.jsonl", "a") as f: f.write(json.dumps({"epoch": epoch + 1, **res}) + "\n")
            if cpm > state["best_cpm"]:
                state["best_cpm"], state["best_epoch"] = cpm, epoch + 1
                torch.save({"model": det.network.state_dict(), "cfg": cfg, "state": state, "fold": a.fold, "seed": a.seed, "run_tag": run_tag(cfg)}, out / "best.pt")
                cands.to_csv(out / "val_candidates_best.csv", index=False); msg += " *best*"
        torch.save({"model": det.network.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "state": state, "cfg": cfg, "fold": a.fold, "seed": a.seed, "run_tag": run_tag(cfg)}, ck_last)
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
    with open(out / "done.json", "w") as f:
        json.dump({"best_cpm": state["best_cpm"], "best_epoch": state["best_epoch"], "epochs": state["epoch"], "hours": (time.time() - t_start) / 3600, "run_tag": run_tag(cfg)}, f, indent=1)
    print("finished", state)


if __name__ == "__main__":
    main()
