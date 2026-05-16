"""Phase 1 — Data prep.

Reads a YAML config, mixes sources from HuggingFace, deduplicates, tokenizes,
runs the teacher model to produce target embeddings, splits into train/val/test,
and writes a versioned .pt cache + manifest.

Usage:
    python prep/prepare.py --config configs/smoke.yaml
    python prep/prepare.py --config configs/micro.yaml
    python prep/prepare.py --config configs/micro.yaml --force   # rebuild even if cache exists

To run without W&B (e.g. testing locally):
    WANDB_MODE=disabled python prep/prepare.py --config configs/smoke.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import wandb
from transformers import AutoTokenizer

from config import load_config
from data   import cache_exists, save_cache
from ingest import (
    build_loaders, build_manifest, compute_stats, deduplicate, git_commit,
    mix_sources, print_stats_dashboard, stratified_split, teacher_encode,
    tokenize_batch,
)


def main(config_path: Path, force: bool = False) -> None:
    cfg      = load_config(config_path)
    data_cfg = cfg.data

    # ── Bail if cache already exists ─────────────────────────────────────────
    if cache_exists(data_cfg.cache_dir, data_cfg.cache_name) and not force:
        print(f"✓ cache exists at {data_cfg.cache_dir}/{data_cfg.cache_name}.*")
        print(f"  pass --force to rebuild")
        return

    # ── W&B (project + group + job_type only — no artifacts yet) ─────────────
    short_sha = git_commit()[:7]
    wandb.init(
        project=cfg.wandb_project,
        group=cfg.wandb_group,
        job_type="data-prep",
        name=f"data-prep-{short_sha}",
        tags=cfg.wandb_tags,
        config=data_cfg.model_dump(mode="json"),
    )

    t0 = time.time()

    # ── 1. Mix sources ───────────────────────────────────────────────────────
    print(f"\n→ Loading sources (target {data_cfg.total_samples:,})")
    loaders = build_loaders(data_cfg.sources)
    raw, source_counts = mix_sources(loaders, data_cfg.total_samples, data_cfg.seed)
    print(f"  total loaded: {len(raw):,}")

    # ── 2. Dedup ─────────────────────────────────────────────────────────────
    print(f"\n→ Deduplicating")
    raw, n_removed = deduplicate(raw)
    print(f"  removed {n_removed:,} exact duplicates; {len(raw):,} unique remain")

    # ── 3. Tokenize ──────────────────────────────────────────────────────────
    print(f"\n→ Tokenizing with {data_cfg.tokenizer_id}")
    texts, source_names = zip(*raw)
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.tokenizer_id)
    tokens    = tokenize_batch(list(texts), tokenizer, data_cfg.max_len)
    print(f"  input_ids shape: {tokens['input_ids'].shape}")

    # ── 4. Teacher encode (the slow step) ────────────────────────────────────
    print(f"\n→ Teacher encoding with {data_cfg.teacher_id}")
    print(f"  ({len(texts):,} samples, batch_size={data_cfg.teacher_batch_size})")
    teacher_embs = teacher_encode(
        list(texts),
        teacher_id=data_cfg.teacher_id,
        revision=data_cfg.teacher_revision,
        batch_size=data_cfg.teacher_batch_size,
    )
    print(f"  teacher_embs shape: {teacher_embs.shape}")

    # ── 5. Pack into per-sample dicts ────────────────────────────────────────
    samples = [
        {
            "input_ids":      tokens["input_ids"][i],
            "attention_mask": tokens["attention_mask"][i],
            "teacher_emb":    teacher_embs[i],
            "source":         source_names[i],
        }
        for i in range(len(texts))
    ]

    # ── 6. Split (seeded, stratified per source) ─────────────────────────────
    print(f"\n→ Splitting  (val={data_cfg.val_ratio}, test={data_cfg.test_ratio}, seed={data_cfg.seed})")
    splits = stratified_split(samples, data_cfg.val_ratio, data_cfg.test_ratio, data_cfg.seed)
    split_counts = {k: len(v) for k, v in splits.items()}
    for name, n in split_counts.items():
        print(f"  {name}: {n:,}")

    # ── 7. Sanity stats (on the train split) ─────────────────────────────────
    print(f"\n→ Sanity stats (train split)")
    stats = compute_stats(splits["train"], tokenizer)
    print_stats_dashboard(stats)

    # ── 8. Save ──────────────────────────────────────────────────────────────
    print(f"→ Writing cache to {data_cfg.cache_dir}/")
    manifest = build_manifest(
        cfg=data_cfg,
        source_counts=source_counts,
        n_dedup_removed=n_removed,
        split_counts=split_counts,
        code_commit=git_commit(),
    )
    save_cache(splits, manifest, data_cfg.cache_dir, data_cfg.cache_name)

    # ── 9. Log summary to W&B ────────────────────────────────────────────────
    elapsed = time.time() - t0
    wandb.log({
        "stats/n_samples":              stats["n_samples"],
        "stats/n_dedup_removed":        n_removed,
        "stats/length_p50":             stats["length_p50"],
        "stats/length_p95":             stats["length_p95"],
        "stats/length_max":             stats["length_max"],
        "stats/teacher_emb_norm_mean":  stats["emb_norm_mean"],
        "stats/teacher_emb_norm_std":   stats["emb_norm_std"],
        "stats/wall_clock_seconds":     elapsed,
        **{f"split/{k}":  v for k, v in split_counts.items()},
        **{f"source/{s}": n for s, n in source_counts.items()},
    })
    wandb.finish()

    print(f"\n✓ Done in {elapsed/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 — build the prepared training dataset")
    parser.add_argument("--config", type=Path, required=True, help="path to YAML config")
    parser.add_argument("--force",  action="store_true",      help="rebuild even if cache exists")
    args = parser.parse_args()
    main(args.config, args.force)
