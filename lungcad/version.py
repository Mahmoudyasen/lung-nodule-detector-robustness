"""Version tagging: every checkpoint / candidate list carries git hash + config hash (Execution Guide v5 §2)."""
from __future__ import annotations
import hashlib, json, subprocess, sys, platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def git_hash() -> str:
    try:
        h = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short=10", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        # "-dirty" = uncommitted changes to anything that can affect a run; notebooks and reports are views only
        dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "--", ".", ":!notebooks", ":!report"],
                               capture_output=True).returncode != 0
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "nogit-" + source_hash()[:10]


def source_hash() -> str:
    h = hashlib.sha256()
    for p in sorted((ROOT / "lungcad").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def run_tag(cfg: dict) -> str:
    return f"g{git_hash()}-c{config_hash(cfg)}"


def env_freeze() -> dict:
    import torch, monai, numpy, SimpleITK
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "monai": monai.__version__,
            "numpy": numpy.__version__, "simpleitk": SimpleITK.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def write_run_manifest(out_dir, cfg: dict, extra: dict | None = None) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    man = {"git": git_hash(), "source_sha256": source_hash(), "config_hash": config_hash(cfg), "run_tag": run_tag(cfg),
           "config": cfg, "env": env_freeze(), **(extra or {})}
    with open(out_dir / "run_manifest.json", "w") as f:
        json.dump(man, f, indent=1, default=str)
    return out_dir / "run_manifest.json"
