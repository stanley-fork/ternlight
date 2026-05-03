
Let me read the prototype doc and both repos first.Good, I have a clear picture now. Let me give you a direct, opinionated assessment of each question.

---

## schneiderkamplab/bitlinear — Use It, With One Caveat

It aims to be a production-ready implementation of 1.58-bit layers for QAT and efficient inference, building directly on the BitNet b1.58 paper. Crucially it includes a `replace_modules()` utility that swaps all `nn.Linear` layers in any existing model in one call.

That `replace_modules()` function is the key reason to use it. Your prototype doc's `BitLinear` implementation is functionally correct but hand-rolled. The schneiderkamplab version has been tested against more edge cases and handles the STE correctly.

**The one caveat:** read the source before using it. The authors note that most other BitLinear implementations get at least some details wrong. That claim cuts both ways — verify their ternary projection matches the paper's formulation (absmean scaling, threshold at 0.5×scale) rather than trusting the claim blindly. Cross-check against your prototype doc's implementation — they should be functionally identical. If they match, use theirs and delete yours.

---

## kyegomez/BitNet — Reference Only, Don't Depend On It

This is a community implementation, not an official Microsoft release. It has known bugs including issues with the dequantization algorithm, and the BitLinear 1.5 implementation is still in progress. Looking at the actual weight quantization code, it subtracts the weight mean before applying sign — `u = (w - e).sign() * scale` — which diverges from the standard absmean formulation in the paper. That mean subtraction is a design choice that may or may not be beneficial, but it means this is not a clean reference implementation of BitNet b1.58.

**Use it as a reading reference** to understand how others have structured the training loop and attention integration. Don't pip install it as a dependency. The C implementation you mentioned doesn't exist in this repo — you're thinking of Microsoft's official `bitnet.cpp` which is a separate project for inference only, not training.

---

## HuggingFace — Leverage Heavily, But Selectively

This is where you get the most leverage for the least handrolling. Here's exactly what to take and what to skip:

**Take all of this:**

`sentence-transformers` handles the teacher entirely. Loading, encoding, normalisation — zero custom code needed:
```python
from sentence_transformers import SentenceTransformer
teacher = SentenceTransformer("all-MiniLM-L6-v2")
teacher_embeddings = teacher.encode(sentences, normalize_embeddings=True)
```

`tokenizers` for the student tokenizer — already in your prototype doc, correct call.

`datasets` for loading MS MARCO, Natural Questions, Stack Overflow. Saves you writing data pipeline code:
```python
from datasets import load_dataset
ds = load_dataset("ms_marco", "v2.1", split="train")
```

`evaluate` for STS-B eval at the end of each epoch — one function call gives you Spearman correlation:
```python
import evaluate
sts = evaluate.load("stsb_multi_mt", "en")
```

**Do not take:**

`transformers.BertModel` for the student. Using a HuggingFace transformer class as your student is tempting but wrong for two reasons: it builds in assumptions about float32 throughout, and the model graph becomes harder to export cleanly to your custom `.bin` format. Your student architecture needs to be handrolled specifically because of the ternary weight lifecycle — float32 shadow weights during training, hardened ternary at export. HuggingFace's model classes don't have clean hooks for that.

`transformers` BitNet quantization — HuggingFace transformers does have native BitNet support, but it is specifically designed for models that were pretrained with BitNet from scratch, not for distillation into a custom small encoder. It won't map cleanly onto your use case.

---

## Do You Need to Handroll Your PyTorch Model and Loss? 

**Model — yes, you need to write it yourself**, but it's not much code. The student is a straightforward encoder: embedding table, two transformer layers with BitLinear substituted in, mean pooling, output projection. Maybe 150 lines of clean PyTorch. The key reason it needs to be custom is the weight lifecycle: float32 shadow weights coexist with the ternary snapshot during QAT, and the export path needs to extract just the ternary integers. No existing framework manages that lifecycle exactly as you need it.

**Loss — mostly yes, but with HuggingFace scaffolding.** The three-term loss (distillation + contrastive + variance) is custom, but the `Trainer` class from HuggingFace can still wrap your training loop and handle checkpointing, logging, and device management. You pass it a custom `compute_loss` function:

```python
from transformers import Trainer, TrainingArguments

class TernTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        student_emb = model(inputs["input_ids"], inputs["attention_mask"])
        teacher_emb = inputs["teacher_embedding"]
        
        loss = (
            1.0 * distillation_loss(student_emb, teacher_emb)
          + 0.15 * contrastive_loss(student_emb)
          + 0.05 * variance_loss(student_emb)
        )
        return loss
```

This gives you free gradient accumulation, mixed precision, checkpointing, and early stopping without writing any of that yourself.

---

## Eval Dashboard — Weights & Biases Free Tier

For something local with minimal setup, the practical answer is **Weights & Biases free tier** plus a single `wandb.log()` call in your training loop. It's not fully local but setup is one `pip install wandb` and one `wandb login`, and the free tier has no meaningful limits for a project of this scale. You get live loss curves, embedding std tracking, and your go/no-go metric table all in one place with essentially zero integration code.

If you want genuinely local and offline, **TensorBoard** is the alternative:

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/tern_phase1")

# in training loop:
writer.add_scalar("loss/total", total_loss, step)
writer.add_scalar("embed/std_mean", emb_std, step)
writer.add_scalar("weights/zero_fraction", zero_frac, step)
```

Then `tensorboard --logdir runs/` in a terminal gives you the dashboard at localhost:6006. No account, no setup, no network. The downside is the UI is less polished and comparing runs is more cumbersome than W&B.

**Recommendation:** Use W&B for the actual training runs — the experiment comparison across hyperparameter sweeps is worth the one-time login. Use TensorBoard only if you have a strict offline/air-gapped constraint during development.

---

## Summary

| Component | What to do |
|---|---|
| BitLinear layer | Use `schneiderkamplab/bitlinear` after verifying it matches paper spec |
| kyegomez/BitNet | Read for reference, do not depend on |
| Teacher model | 100% `sentence-transformers`, zero custom code |
| Tokenizer | 100% HuggingFace `tokenizers` crate, already in prototype doc |
| Dataset loading | HuggingFace `datasets` |
| Student model | Handroll ~150 lines — required for ternary weight lifecycle |
| Training loop | HuggingFace `Trainer` with custom `compute_loss` |
| Loss function | Handroll the three-term loss, ~40 lines |
| Eval metrics | HuggingFace `evaluate` for STS-B Spearman |
| Dashboard | W&B free tier (or TensorBoard if offline required) |

Let me check the official Microsoft repo first.That confirms everything I needed to know. Clear answer incoming.

---

## Microsoft's BitNet Repo — Inference Only, Not Relevant to Phase 1

This is the critical thing to understand upfront: the Microsoft BitNet repo is an official **inference framework** for 1-bit LLMs. Look at the file structure — `CMakeLists.txt`, `src/`, `preset_kernels/`, `run_inference.py`. It's a C++ inference engine, not a training framework. It has no training code, no QAT implementation, no distillation pipeline.

It is **Phase 2 relevant**, not Phase 1. Specifically:

- The `preset_kernels/` and `src/` directories contain the optimised CPU kernels for ternary matrix multiplication — the exact thing you'll be reimplementing in Rust/Wasm for the @tern engine
- The bit-packing format used in their `.bin` files is worth studying closely so your export format is compatible with established conventions
- Their weight quantization math (the absmean scaling approach) is the canonical reference implementation

**For Phase 1 training, Microsoft's repo gives you nothing directly.** Your training stack is still: handrolled student model + `schneiderkamplab/bitlinear` + HuggingFace ecosystem. Come back to the Microsoft repo when you're designing the Wasm kernel in Phase 2.

---

## Python Files vs Notebook — Yes, Python Files, Definitively

For a prototype that answers go/no-go questions, a notebook is the wrong tool. Here's why this matters specifically for @tern rather than just general engineering preference:

**Your training loop will run for hours.** Notebooks are fragile across long runs — kernel restarts lose state, you can't easily resume from checkpoints, and the implicit execution order causes subtle bugs that are hard to reproduce. A Python script with proper checkpointing runs, crashes, and resumes cleanly.

**You have multiple distinct concerns that should be separated.** Mixing dataset prep, model definition, training loop, and eval in one notebook means you can't run just eval against a saved checkpoint without re-running everything above it. As separate files, you can run `python eval.py --checkpoint runs/epoch_10.pt` independently at any time.

**The go/no-go criteria require reproducibility.** If Task 2 STS proxy AUC comes in at 0.78 (marginal), you need to rerun with `d_model=384` and compare cleanly. With scripts and a config file, that's one parameter change. With a notebook, it's a risky manual edit.

---

## Recommended Project Structure

This maps directly to the outputs section of your prototype doc — `student_model.py`, `train.py`, `eval.py`, `results/` — but with the full supporting structure around them:

```
tern/
│
├── config/
│   ├── micro.yaml          # d_model=256, n_layers=2 — default prototype
│   └── small.yaml          # d_model=384 — fallback if micro hits marginal threshold
│
├── data/
│   ├── prepare.py          # download MS MARCO / NQ / SO titles, run teacher, cache embeddings
│   └── dataset.py          # TernDataset — loads (token_ids, teacher_embedding) pairs
│
├── model/
│   ├── bitlinear.py        # thin wrapper around schneiderkamplab/bitlinear
│   ├── student.py          # TernStudent — full encoder with BitLinear layers
│   └── export.py           # Phase 2 placeholder — harden weights, pack to .bin
│
├── training/
│   ├── loss.py             # distillation_loss, contrastive_loss, variance_loss
│   ├── trainer.py          # TernTrainer extending HuggingFace Trainer
│   └── health.py           # embedding std monitoring, tripwires, zero-weight fraction
│
├── eval/
│   ├── eval.py             # Task 1/2/3 eval pipeline — inference-mirroring
│   └── metrics.py          # cosine alignment, STS AUC, Recall@3
│
├── runs/                   # checkpoints, per-epoch metrics, W&B artifacts
│
├── train.py                # entrypoint — loads config, wires everything, starts training
├── evaluate.py             # entrypoint — loads checkpoint, runs full eval suite
├── requirements.txt
└── README.md
```

The key design principle here is that `train.py` and `evaluate.py` at the root are thin entrypoints — they parse a config file and call into the modules. All the logic lives in the submodules. This means:

```bash
# Run prototype at micro tier
python train.py --config config/micro.yaml

# Eval a specific checkpoint without retraining
python evaluate.py --config config/micro.yaml --checkpoint runs/epoch_15.pt

# Rerun at small tier if micro hits marginal threshold
python train.py --config config/small.yaml
```

---

## The Config File Is Worth Getting Right Early

Your two go/no-go scenarios from the prototype doc — `d_model=256` baseline and `d_model=384` fallback — should be entirely config-driven from day one. A minimal `micro.yaml`:

```yaml
model:
  d_model: 256
  n_layers: 2
  n_heads: 8
  ffn_dim: 1024
  vocab_size: 30522
  max_seq_len: 128
  output_dim: 384       # projection to match teacher

training:
  teacher: "all-MiniLM-L6-v2"
  dataset_size: 50000   # reduce to 20000 for first smoke test
  batch_size: 64
  epochs: 30
  lr: 1e-4
  warmup_epochs: 5      # float32 only before QAT activates
  weight_decay: 0.01
  grad_clip: 1.0
  loss_weights:
    distillation: 1.0
    contrastive: 0.15
    variance: 0.05

eval:
  holdout_size: 2000    # Task 1
  sts_pairs: 200        # Task 2
  retrieval_queries: 50 # Task 3
  retrieval_corpus: 500 # Task 3

thresholds:
  min_cosine_sim: 0.75
  min_sts_auc: 0.80
  min_recall_at_3: 0.70
  max_zero_weight_fraction: 0.60
```

Switching to the `small.yaml` fallback is then just changing four lines under `model:`. Everything else — training loop, eval pipeline, tripwires — reads from the same config and adapts automatically.