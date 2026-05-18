"""Phase 2/3 — training entry point.

Reads a YAML config, loads the cached dataset (output of prep/prepare.py),
builds the StudentEncoder, and hands off to the Trainer.

Phase 2 (fp32 baseline): `--config configs/micro-fp32.yaml`
Phase 3 (QAT, future):   `--config configs/micro.yaml` once QAT is wired

Usage:
    python train.py --config configs/smoke-fp32.yaml      # ~30s pipeline check
    python train.py --config configs/micro-fp32.yaml      # the real run

To run without W&B:
    WANDB_MODE=disabled python train.py --config ...
"""

import argparse
import subprocess
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader

from config  import load_config
from data    import TernDataset, collate_fn, load_cache
from model   import StudentEncoder
from trainer import Trainer


def best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main(config_path: Path) -> None:
    cfg = load_config(config_path)
    if cfg.train is None:
        raise ValueError(f"Config {config_path} has no `train:` section.")
    tcfg = cfg.train

    # ── 1. Load cached dataset ───────────────────────────────────────────────
    print(f"→ Loading cache: {tcfg.cache_dir}/{tcfg.cache_name}.*")
    splits, manifest = load_cache(tcfg.cache_dir, tcfg.cache_name)
    print(f"  train: {len(splits['train']):,}  val: {len(splits['val']):,}  test: {len(splits['test']):,}")

    train_loader = DataLoader(
        TernDataset(splits["train"]),
        batch_size=tcfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TernDataset(splits["val"]),
        batch_size=tcfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ── 2. Build model ───────────────────────────────────────────────────────
    device = best_device()
    print(f"\n→ Building model (device={device})")
    model = StudentEncoder(
        vocab_size = tcfg.vocab_size,
        d_model    = tcfg.d_model,
        n_layers   = tcfg.n_layers,
        n_heads    = tcfg.n_heads,
        ffn_dim    = tcfg.ffn_dim,
        output_dim = tcfg.output_dim,
        dropout    = tcfg.dropout,
    )
    print(f"  parameters: {model.count_parameters():,}")

    # ── 2b. Warm-start (optional) ────────────────────────────────────────────
    # Loads model_state from another checkpoint. Phase 3 uses this to start
    # QAT from a converged fp32 baseline (much better than random init).
    # Optimizer/scheduler/RNG are NOT loaded — those reset for the new run.
    # Must happen BEFORE Trainer ctor swaps to BitLinear (if enable_qat=true),
    # so the BitLinear shadow weights inherit the fp32-trained values.
    if tcfg.init_from is not None:
        print(f"\n→ Warm-starting from {tcfg.init_from}")
        ckpt = torch.load(tcfg.init_from, weights_only=False, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"  loaded weights from epoch {ckpt['epoch']}")

    # ── 3. Run directory ─────────────────────────────────────────────────────
    short_sha = git_commit()[:7]
    full_run_name = f"{tcfg.run_name}-{short_sha}"
    run_dir = Path(tcfg.runs_dir) / full_run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  run dir: {run_dir}")

    # ── 4. W&B (levels 1-3: project, run, job_type) ──────────────────────────
    job_type = "train-qat" if tcfg.enable_qat else "train-fp32"
    wandb.init(
        project = cfg.wandb_project,
        group   = cfg.wandb_group,
        job_type= job_type,
        name    = full_run_name,
        tags    = cfg.wandb_tags,
        config  = {
            **tcfg.model_dump(mode="json"),
            "code_commit":   git_commit(),
            "data_manifest": manifest,
            "device":        device,
        },
    )

    # ── 5. Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, tcfg, device, run_dir)
    trainer.train()

    wandb.finish()
    print(f"\n✓ Done. Checkpoints in {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2/3 — train the student model")
    parser.add_argument("--config", type=Path, required=True, help="path to YAML config")
    args = parser.parse_args()
    main(args.config)
