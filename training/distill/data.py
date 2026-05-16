"""Cross-phase data interface.

Phase 1 (`prep/prepare.py`) calls `save_cache` to write the prepared dataset.
Phases 2/3 (`train.py`) call `load_cache`, `TernDataset`, and `collate_fn` to
read it during training. Phase 4 (`evaluate.py`) calls `load_cache` to read
the held-out test split.

Cache layout on disk:

    <cache_dir>/<name>.train.pt       list[dict]  — train split
    <cache_dir>/<name>.val.pt         list[dict]  — validation split
    <cache_dir>/<name>.test.pt        list[dict]  — held out for Task 1 eval
    <cache_dir>/<name>.manifest.json  dict        — provenance + counts

Each sample dict:

    {
        "input_ids":      LongTensor (max_len,),
        "attention_mask": LongTensor (max_len,),
        "teacher_emb":    FloatTensor (384,)  L2-normalized,
        "source":         str                  e.g. "ms_marco"
    }
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


# ── Cache I/O ────────────────────────────────────────────────────────────────

def save_cache(
    splits:    dict[str, list[dict]],
    manifest:  dict,
    cache_dir: Path,
    name:      str,
) -> None:
    """Write per-split .pt files + manifest.json. Creates cache_dir if missing."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in splits.items():
        path = cache_dir / f"{name}.{split_name}.pt"
        torch.save(samples, path)
        size_mb = path.stat().st_size / 1e6
        print(f"  wrote {path}  ({len(samples):,} samples, {size_mb:.1f} MB)")

    manifest_path = cache_dir / f"{name}.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"  wrote {manifest_path}")


def load_cache(
    cache_dir: Path,
    name:      str,
) -> tuple[dict[str, list[dict]], dict]:
    """Read per-split .pt files + manifest.json."""
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / f"{name}.manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Did you run prep/prepare.py?"
        )

    with open(manifest_path) as f:
        manifest = json.load(f)

    splits: dict[str, list[dict]] = {}
    for split_name in ("train", "val", "test"):
        path = cache_dir / f"{name}.{split_name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        splits[split_name] = torch.load(path, weights_only=False)

    return splits, manifest


def cache_exists(cache_dir: Path, name: str) -> bool:
    """True if all expected cache files (manifest + 3 splits) are present."""
    cache_dir = Path(cache_dir)
    expected = [
        cache_dir / f"{name}.manifest.json",
        cache_dir / f"{name}.train.pt",
        cache_dir / f"{name}.val.pt",
        cache_dir / f"{name}.test.pt",
    ]
    return all(p.exists() for p in expected)


# ── Runtime (Phases 2/3) ─────────────────────────────────────────────────────

class TernDataset(Dataset):
    """Wraps one split (train / val / test). Indexed by row."""

    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """Stack per-sample dicts into a batched dict.

    The `source` field is a string per sample, so it's returned as a list rather
    than stacked. DataLoader is fine with mixed tensor/non-tensor outputs.
    """
    return {
        "input_ids":      torch.stack([s["input_ids"]      for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "teacher_emb":    torch.stack([s["teacher_emb"]    for s in batch]),
        "source":        [s["source"] for s in batch],
    }
