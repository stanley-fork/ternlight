# Phase 1 Milestones — Distillation Prototype

> This document breaks Phase 1 into small, observable steps. Each milestone has a clear goal, a "what you should see" outcome, and a decision point before moving forward. The full spec lives in [tern-phase1-prototype.md](../tern-phase1-prototype.md) — this is the working breakdown of how to get there.

---

## Milestone 0 — Environment & Sanity Check

**Goal:** Everything installs, GPU is visible, teacher model works, WandB is connected.

**Approach:** Notebook is fine here. This is pure exploration.

```
notebooks/00-environment-check.ipynb
```

**Steps:**
1. Install dependencies and verify GPU
2. Load teacher model (`all-MiniLM-L6-v2`) and encode 5 test strings
3. Print embedding shapes and a few cosine similarities between them
4. Log a test metric to WandB to confirm the connection

```python
import torch
from sentence_transformers import SentenceTransformer
import wandb

# GPU check
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# Teacher check
teacher = SentenceTransformer("all-MiniLM-L6-v2")
test_sentences = [
    "how do I reset my password",
    "I forgot my password",
    "my screen is black",
    "the display is broken",
    "quarterly earnings report",
]
embeddings = teacher.encode(test_sentences, normalize_embeddings=True)
print(f"Shape: {embeddings.shape}")  # expect (5, 384)

# Cosine similarities — pairs 0/1 and 2/3 should be high, 0/4 should be low
from sklearn.metrics.pairwise import cosine_similarity
sims = cosine_similarity(embeddings)
print(sims)

# WandB test
wandb.init(project="tern-distill", name="env-check")
wandb.log({"test_metric": 1.0})
wandb.finish()
```

**What you should see:**
- `embeddings.shape == (5, 384)`
- Cosine similarity between "reset password" / "forgot password" > 0.7
- Cosine similarity between "reset password" / "quarterly earnings" < 0.3
- A run appearing in your WandB dashboard

**Decision:** If teacher encodings look semantically sensible (similar strings score high, unrelated strings score low), proceed. If anything fails to install or the GPU isn't detected, stop and fix the environment.

---

## Milestone 1 — Data Pipeline

**Goal:** Produce a cached dataset of `(token_ids, teacher_embedding)` pairs on disk. This is the input to every training run.

**Approach:** Python script. This runs once and the output gets reused.

```
data/prepare.py
data/dataset.py
```

**Steps:**

1. Load a small slice of MS MARCO (~5,000 queries to start — not 50,000). Validate the pipeline before scaling.
2. Tokenize each query using `bert-base-uncased` WordPiece
3. Run teacher model on each query, save the embedding
4. Cache as a `.pt` file: list of `{"input_ids": tensor, "teacher_emb": tensor}`

```python
# data/prepare.py (simplified sketch)
from datasets import load_dataset
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
import torch

tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
teacher = SentenceTransformer("all-MiniLM-L6-v2")
teacher.eval()

ds = load_dataset("ms_marco", "v2.1", split="train[:5000]")
queries = [row["query"] for row in ds]

# Tokenize
encodings = [tokenizer.encode(q) for q in queries]
input_ids = [torch.tensor(e.ids[:128]) for e in encodings]

# Teacher embeddings — run in batches, this takes a few minutes
teacher_embs = teacher.encode(queries, batch_size=64,
                               normalize_embeddings=True,
                               show_progress_bar=True)
teacher_embs = torch.tensor(teacher_embs)

# Cache
data = [{"input_ids": ids, "teacher_emb": emb}
        for ids, emb in zip(input_ids, teacher_embs)]
torch.save(data, "data/cache/msmarco_5k.pt")
print(f"Saved {len(data)} samples")
```

**What you should see:**
- A `.pt` file on disk, ~50MB for 5,000 samples
- No shape mismatches — every `teacher_emb` should be `(384,)`
- A few minutes of GPU time for teacher encoding

**Decision:** Inspect 5 random samples. Print the raw text, the token IDs, and the teacher embedding norm. Everything should look consistent. Scale to 50,000 samples once you're happy.

---

## Milestone 2 — Float32 Student Baseline (No Ternary Yet)

**Goal:** Build and train the student architecture in full float32 — no BitLinear, no quantization. Prove the 2-layer architecture can learn to approximate teacher embeddings before introducing any ternary complexity.

**Why this step exists:** Ternary quantization adds instability. If you skip the float32 baseline and go straight to QAT, you won't know whether a bad result is caused by the architecture being too small or the quantization being too aggressive. The float32 baseline gives you a ceiling to compare against.

**Approach:** Python script. Short training run — 5 epochs is enough.

```
model/student.py     (architecture)
training/loss.py     (distillation loss only — keep it simple here)
train.py             (entrypoint)
config/micro-fp32.yaml
```

**The student model:**

```python
# model/student.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)
        self.layers = nn.ModuleList([
            TransformerLayer(cfg) for _ in range(cfg.n_layers)
        ])
        self.projection = nn.Linear(cfg.d_model, cfg.output_dim)  # 256 → 384, float32

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        # Mean pool over non-padding positions
        mask = attention_mask.unsqueeze(-1).float() if attention_mask is not None else 1.0
        pooled = (x * mask).sum(1) / mask.sum(1)
        return F.normalize(self.projection(pooled), dim=-1)
```

**Loss (distillation only for now):**

```python
# training/loss.py
import torch.nn.functional as F

def distillation_loss(student_emb, teacher_emb):
    return 1 - F.cosine_similarity(student_emb, teacher_emb, dim=-1).mean()
```

**WandB logging — add from the start:**

```python
import wandb
wandb.init(project="tern-distill", name="fp32-baseline-micro", config=cfg)

# In training loop, each epoch:
wandb.log({
    "train/loss": avg_loss,
    "train/cosine_sim": avg_cosine_sim,   # 1 - loss
    "epoch": epoch,
})
```

**What you should see:**
- Loss decreasing steadily within the first 3 epochs
- By epoch 5: cosine similarity between student and teacher output in the range 0.6–0.85
- WandB showing a smooth loss curve with no divergence

**Decision:** If loss is decreasing and cosine sim is climbing, the architecture works. This is your float32 ceiling — write down the epoch 5 cosine similarity number. The ternary model in Milestone 3 should reach at least 80% of this value to be considered viable.

---

## Milestone 3 — Add QAT (BitLinear + Warmup)

**Goal:** Replace `nn.Linear` with BitLinear in the attention and FFN layers. Add the float32 warmup phase before ternary kicks in. Observe the transition and monitor for weight collapse.

**This is the core novelty of the project.** Take it slowly. Read the BitLinear source before using it.

**Approach:** Python script.

```
model/bitlinear.py    (thin wrapper around schneiderkamplab/bitlinear)
config/micro.yaml     (adds warmup_epochs: 5)
```

**Swap in BitLinear:**

```python
# model/student.py — updated
from bitlinear import replace_modules  # schneiderkamplab/bitlinear

model = StudentEncoder(cfg)
# Replace all nn.Linear in attention + FFN with BitLinear
# Do NOT replace the output projection — that stays float32
replace_modules(model, exclude=["projection"])
```

**The warmup schedule:**

```python
# In training loop
for epoch in range(cfg.epochs):
    if epoch < cfg.warmup_epochs:
        # Disable ternary projection — train shadow weights in float32
        set_quantization_active(model, False)
    else:
        # QAT active — forward pass uses {-1, 0, +1} weights
        set_quantization_active(model, True)
```

**Monitoring — add to WandB logging:**

```python
# health check after each epoch
zero_fracs = []
for name, module in model.named_modules():
    if isinstance(module, BitLinear):
        w = module.weight
        scale = w.abs().mean()
        ternary = (w.abs() > 0.5 * scale).float()
        zero_frac = 1.0 - ternary.mean().item()
        zero_fracs.append(zero_frac)

avg_zero_frac = sum(zero_fracs) / len(zero_fracs)

wandb.log({
    "weights/zero_fraction": avg_zero_frac,
    "weights/zero_fraction_max": max(zero_fracs),
    "train/loss": avg_loss,
    "epoch": epoch,
})
```

**What you should see:**
- During warmup (epochs 0–4): loss curve looks similar to the float32 baseline — model is learning in float32
- At epoch 5 when QAT activates: expect a small loss spike, then recovery
- Zero weight fraction should start around 30–40% and stay stable — not climbing
- By epoch 15: loss should be approaching (not matching) the float32 baseline

**Go/No-Go checks (inline — don't wait for the full run):**
- After epoch 7: if `zero_fraction > 0.6` and loss is flat → stop, extend warmup, restart
- After epoch 10: if loss is still far above float32 baseline with no trend → flag as Risk 1 (weight collapse) from the phase 1 doc

**Decision:** This milestone is the highest-risk one. The WandB charts are your primary diagnostic tool. Keep the run small (20 epochs) until you're confident the QAT is stable, then extend.

---

## Milestone 4 — Full Loss + Full Training Run

**Goal:** Add the contrastive and variance loss terms. Run the full 30-epoch training. Hit the exit criteria from the phase 1 prototype doc.

**Approach:** Python scripts only from here. The training run will take hours — notebooks are fragile across long runs.

**Add the full loss:**

```python
# training/loss.py
def full_loss(student_emb, teacher_emb, weights):
    distill = 1 - F.cosine_similarity(student_emb, teacher_emb, dim=-1).mean()

    # Contrastive: push batch embeddings apart
    # Compute pairwise cosine similarity within batch, penalise off-diagonal
    sim_matrix = student_emb @ student_emb.T
    contrastive = sim_matrix.fill_diagonal_(0).pow(2).mean()

    # Variance: prevent embedding collapse (all outputs becoming similar)
    variance = F.relu(1 - student_emb.std(dim=0)).mean()

    return (
        weights.distillation * distill
      + weights.contrastive  * contrastive
      + weights.variance     * variance
    )
```

**WandB — log each loss term separately:**

```python
wandb.log({
    "loss/total": total_loss,
    "loss/distillation": distill_loss,
    "loss/contrastive": contrastive_loss,
    "loss/variance": variance_loss,
    "weights/zero_fraction": zero_frac,
    "embed/std_mean": student_emb.std(dim=0).mean().item(),
    "epoch": epoch,
})
```

**Checkpointing — save every 5 epochs:**

```python
if epoch % 5 == 0:
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
    }, f"runs/{run_name}/epoch_{epoch}.pt")
```

**What you should see:**
- All three loss terms decreasing (or at least not diverging)
- `embed/std_mean` staying above ~0.1 — if it collapses to near zero, the variance loss isn't working
- `loss/distillation` is the primary signal — it should dominate

**Decision:** At epoch 30, run eval (Milestone 5). If the training loss looks healthy but eval disappoints, check loss term weights — contrastive or variance may be crowding out distillation signal.

---

## Milestone 5 — Eval & Go/No-Go

**Goal:** Run the three eval tasks from the phase 1 prototype doc. Make the go/no-go decision.

**Approach:** `evaluate.py` — a standalone script that loads any checkpoint and runs the eval suite.

```bash
python evaluate.py --checkpoint runs/micro-qat/epoch_30.pt --config config/micro.yaml
```

**Eval pipeline (mirrors inference — ternary weights hardened, no straight-through):**

```python
# evaluate.py
model.eval()
harden_weights(model)  # materialize {-1, 0, +1}, remove shadow weights

# Task 1: Teacher alignment
# Task 2: STS proxy AUC
# Task 3: Recall@3 — run on both general and tech-domain corpus
```

**Log final eval metrics to WandB:**

```python
wandb.log({
    "eval/teacher_cosine_sim": task1_score,
    "eval/sts_auc": task2_score,
    "eval/recall_at_1": recall_1,
    "eval/recall_at_3": recall_3,
    "eval/recall_at_3_tech": recall_3_tech,
})
```

**The decision table:**

| Metric | Acceptable | Marginal → try d_model=384 | No-go |
|---|---|---|---|
| Task 1 cosine sim | > 0.75 | 0.60–0.75 | < 0.60 |
| Task 2 STS AUC | > 0.80 | 0.70–0.80 | < 0.70 |
| Task 3 Recall@3 | > 0.70 | 0.55–0.70 | < 0.55 |
| Zero weight fraction | < 40% | 40–60% | > 60% |

**If marginal:** Change four lines in `config/small.yaml` (d_model: 384, ffn_dim: 1536) and rerun from Milestone 3. Same pipeline, bigger model.

**If acceptable:** Phase 1 is done. Save the final checkpoint. Phase 2 begins with the `.bin` export.

---

## Project Structure

Following the layout from `design.md`:

```
tern-distill-prototype/
│
├── design.md                      # reference decisions (already exists)
├── milestones.md                  # this file
│
├── config/
│   ├── micro.yaml                 # d_model=256, QAT — primary prototype
│   ├── micro-fp32.yaml            # d_model=256, float32 — Milestone 2 baseline
│   └── small.yaml                 # d_model=384 — fallback if micro is marginal
│
├── data/
│   ├── prepare.py                 # download, tokenize, cache teacher embeddings
│   ├── dataset.py                 # TernDataset — loads .pt cache
│   └── cache/                     # .pt files, gitignored
│
├── model/
│   ├── bitlinear.py               # wrapper around schneiderkamplab/bitlinear
│   └── student.py                 # TernStudent encoder
│
├── training/
│   ├── loss.py                    # distillation, contrastive, variance losses
│   ├── trainer.py                 # TernTrainer extending HF Trainer
│   └── health.py                  # zero-weight monitoring, WandB helpers
│
├── eval/
│   ├── tasks.py                   # Task 1, 2, 3 implementations
│   └── corpora/                   # held-out eval strings (general + tech domain)
│
├── notebooks/
│   └── 00-environment-check.ipynb # Milestone 0 only — exploration
│
├── runs/                          # checkpoints + WandB artifacts, gitignored
│
├── train.py                       # entrypoint
├── evaluate.py                    # entrypoint
└── requirements.txt
```

---

## Dependencies

```
# requirements.txt
torch>=2.0
sentence-transformers
tokenizers
datasets
transformers
evaluate
bitlinear  # schneiderkamplab/bitlinear — verify against paper spec before using
wandb
scikit-learn
pyyaml
```

**Before installing `bitlinear`:** read the source. The prototype doc's hand-rolled BitLinear uses absmean scaling with a 0.5× threshold. Verify the schneiderkamplab implementation matches this. If it diverges from the paper spec, use the hand-rolled version instead.

---

## WandB Run Naming Convention

```
{tier}-{mode}-{short-note}

Examples:
  micro-fp32-baseline       # Milestone 2
  micro-qat-warmup5         # Milestone 3, 5-epoch warmup
  micro-qat-warmup10        # Milestone 3 retry with longer warmup
  micro-qat-full            # Milestone 4 full run
  small-qat-full            # Milestone 4 fallback tier
```

Consistent naming makes WandB run comparison useful — you can directly overlay `micro-fp32-baseline` vs `micro-qat-full` loss curves to see the quantization cost.
