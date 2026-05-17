"""Trainer — the per-epoch training loop, encapsulated as a class.

Phase 2 (fp32 baseline) uses this directly. Phase 3 (QAT) will reuse it by
flipping `cfg.enable_qat = true` and adding BitLinear-swap + lambda-warmup
logic inside, gated by that flag. Same Trainer file, no parallel copies.

The Trainer owns:
    - the training step (forward + loss + backward + step)
    - the validation pass (per-epoch val/loss + val/spearman)
    - checkpointing (saves to runs/<run_name>/checkpoint_ep<N>.pt)
    - per-epoch W&B logging

It does NOT own:
    - W&B init/finish (that's the entry point's job — train.py)
    - model construction (entry point builds the model and hands it in)
    - DataLoader construction (entry point)
"""

import time
from pathlib import Path

import evaluate as hf_evaluate
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import TrainConfig
from loss   import distillation_loss


class Trainer:
    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          TrainConfig,
        device:       str,
        run_dir:      Path,
    ):
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.run_dir      = run_dir

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        total_steps  = len(train_loader) * cfg.epochs
        warmup_steps = int(total_steps * cfg.lr_warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self._spearman    = hf_evaluate.load("spearmanr")
        self._global_step = 0

        print(f"  optimizer: AdamW lr={cfg.lr} wd={cfg.weight_decay}")
        print(f"  schedule:  linear warmup over {warmup_steps:,} / {total_steps:,} steps ({100*cfg.lr_warmup_ratio:.0f}%)")
        print(f"  per-step W&B logging: every {cfg.log_every_n_steps} steps" if cfg.log_every_n_steps > 0 else "  per-step W&B logging: disabled")

    # ── Public ────────────────────────────────────────────────────────────────

    def train(self) -> None:
        print(f"\n→ Training {self.cfg.epochs} epochs")
        for epoch in range(self.cfg.epochs):
            t0 = time.time()

            train_metrics = self._train_epoch()
            val_metrics   = self._eval_epoch()
            elapsed = time.time() - t0

            print(
                f"Epoch {epoch+1:3d}/{self.cfg.epochs}  "
                f"loss={train_metrics['train/loss']:.4f}  "
                f"cos={train_metrics['train/cosine_sim']:.4f}  "
                f"val_loss={val_metrics['val/loss']:.4f}  "
                f"val_spearman={val_metrics['val/spearman']:.4f}  "
                f"({elapsed:.0f}s)"
            )

            wandb.log({
                "epoch":            epoch + 1,
                "epoch_seconds":    elapsed,
                **train_metrics,
                **val_metrics,
            })

            if (epoch + 1) % self.cfg.save_every == 0 or (epoch + 1) == self.cfg.epochs:
                self._save_checkpoint(epoch + 1)

    # ── Per-epoch ─────────────────────────────────────────────────────────────

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_cos  = 0.0

        for batch in self.train_loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            teacher_emb    = batch["teacher_emb"].to(self.device)

            student_emb = self.model(input_ids, attention_mask)
            loss        = distillation_loss(student_emb, teacher_emb)

            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()
            running_cos  += 1.0 - loss.item()
            self._global_step += 1

            # Per-step W&B logging — tells you the LR warmup curve, grad health, intra-epoch loss shape
            if self.cfg.log_every_n_steps > 0 and self._global_step % self.cfg.log_every_n_steps == 0:
                wandb.log({
                    "step":                  self._global_step,
                    "train_step/loss":       loss.item(),
                    "train_step/lr":         self.scheduler.get_last_lr()[0],
                    "train_step/grad_norm":  grad_norm.item(),
                })

        n = len(self.train_loader)
        return {
            "train/loss":       running_loss / n,
            "train/cosine_sim": running_cos  / n,
        }

    def _eval_epoch(self) -> dict[str, float]:
        """Validation pass — loss + Spearman + embedding-distribution diagnostics."""
        self.model.eval()
        total_loss = 0.0
        student_sims: list[float] = []
        teacher_sims: list[float] = []
        student_emb_chunks: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                teacher_emb    = batch["teacher_emb"].to(self.device)

                student_emb = self.model(input_ids, attention_mask)
                total_loss += distillation_loss(student_emb, teacher_emb).item()

                # Pairwise cosine within the batch (both already L2-normalized).
                # Upper triangle only — diagonal is self-similarity (always 1).
                s_sim = (student_emb @ student_emb.T).cpu()
                t_sim = (teacher_emb @ teacher_emb.T).cpu()
                idx   = torch.triu_indices(s_sim.size(0), s_sim.size(1), offset=1)
                student_sims.extend(s_sim[idx[0], idx[1]].tolist())
                teacher_sims.extend(t_sim[idx[0], idx[1]].tolist())

                student_emb_chunks.append(student_emb.cpu())

        result = self._spearman.compute(predictions=student_sims, references=teacher_sims)

        # Embedding-distribution diagnostics (collapse detector — matters more in Phase 3 QAT
        # but cheap to track in fp32 too, gives a baseline for what "healthy spread" looks like).
        all_embs       = torch.cat(student_emb_chunks, dim=0)
        embed_std_mean = all_embs.std(dim=0).mean().item()
        max_offdiag    = max(student_sims) if student_sims else 0.0

        return {
            "val/loss":                  total_loss / len(self.val_loader),
            "val/spearman":              result["spearmanr"],
            "embed/std_mean":            embed_std_mean,
            "embed/max_offdiag_cossim":  max_offdiag,
        }

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int) -> None:
        path = self.run_dir / f"checkpoint_ep{epoch}.pt"
        torch.save({
            "epoch":           epoch,
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config":          self.cfg.model_dump(mode="json"),
        }, path)
        print(f"  ↳ saved {path}")
