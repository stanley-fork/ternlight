# Phase 1 — Implementation Guide & Reference Map

> How the prototype design maps to code, which reference implementations to trust and how to read them, and the minimum theory needed to work with the architecture without getting lost.

---

## 1. The Minimum Theory You Need

You don't need to understand BitNet fully before starting. Here's the minimum to work confidently with the code.

### What a transformer encoder does

Takes a sequence of token IDs → produces a sequence of vectors (one per token) → we mean-pool those into one vector = the embedding. That's it. The embedding captures the meaning of the whole input.

### What BitLinear changes

In a normal transformer, every weight matrix is float32. A matrix multiply is: `output = input @ W.T`. In BitLinear, `W` is constrained to `{-1, 0, +1}`. The multiply becomes: add input rows where W=+1, subtract where W=-1, skip where W=0. No floating-point multiply anywhere.

**During training (QAT):** W is still stored as float32 shadow weights (so gradients can flow), but the *forward pass* snaps them to ternary. Gradients pretend the snap didn't happen (straight-through estimator).

**At inference:** Only the ternary values exist. The float32 shadows are deleted.

### What distillation means here

The teacher (`all-MiniLM-L6-v2`) already knows how to produce good embeddings. We show both teacher and student the same sentence. Teacher produces a 384-dim vector. Student produces a 256-dim vector, projected to 384. We minimise the distance between them. The student is learning to copy the teacher's output space.

---

## 2. The BitLinear Layer — Which Implementation to Use

### schneiderkamplab/bitlinear (pip install bitlinear)

This is the implementation to use for training. The key thing to verify before using it:

```bash
# Clone it locally (already done if you followed setup.md)
# Read: refs/bitlinear/bitlinear/bitlinear.py
```

**What to look for — the absmean scaling:**

The paper (BitNet b1.58) specifies:
```
scale = mean(|W|)
W_ternary = sign(W) * (|W| > 0.5 * scale)
```

The `schneiderkamplab` implementation should match this exactly. The `kyegomez/BitNet` community implementation *diverges* — it subtracts the weight mean before applying sign, which changes the quantization. Do not use that one.

**The `replace_modules()` utility:**

```python
from bitlinear import replace_modules

model = StudentEncoder(cfg)
# Replaces all nn.Linear layers with BitLinear, in-place
# Pass exclude= to protect layers that must stay float32
replace_modules(model, exclude=["projection"])
```

This is the reason to use this library over hand-rolling. It handles the module graph traversal cleanly.

### Fallback: hand-rolled BitLinear

If the library has issues, the hand-rolled version from the phase 1 prototype doc is correct:

```python
class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        scale = self.weight.abs().mean()
        w_ternary = torch.sign(self.weight) * (self.weight.abs() > 0.5 * scale).float()
        w = self.weight + (w_ternary - self.weight).detach()  # straight-through
        return F.linear(x, w)
```

The `.detach()` line is the straight-through estimator — gradients flow as if no quantization happened.

---

## 3. The Student Architecture — What to Write vs What to Borrow

### Write yourself (short, specific to @tern)

```
model/student.py      ~120 lines   TernStudent encoder
model/bitlinear.py    ~20 lines    thin wrapper around schneiderkamplab
training/loss.py      ~40 lines    three-term loss
training/health.py    ~30 lines    zero-weight fraction monitoring
```

### Borrow from HuggingFace (do not rewrite)

```
SentenceTransformer       teacher loading, encoding, normalization
Trainer / TrainingArguments  training loop, checkpointing, gradient accumulation
load_dataset              MS MARCO, Natural Questions loading
evaluate                  STS-B Spearman eval
```

### Do NOT use HuggingFace for (reasons below)

| Component | Why not |
|---|---|
| `BertModel` as student | Float32 assumptions throughout, hard to hook BitLinear into, hard to export to custom `.bin` |
| `transformers` BitNet quantization | Designed for pretraining from scratch, not distillation into a small custom encoder |
| `AutoModel` for the student | Same as BertModel — too opinionated about weight lifecycle |

The student model must be handrolled because of the **ternary weight lifecycle**: float32 shadow weights coexist with the ternary projection during QAT, and the export path needs to extract just the ternary integers in a specific layout. No existing framework manages this.

---

## 4. HuggingFace Trainer Pattern

The `Trainer` class handles: gradient accumulation, mixed precision, checkpointing, device management, logging hooks. You plug in your custom loss via `compute_loss`.

```python
# training/trainer.py
from transformers import Trainer

class TernTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        teacher_emb = inputs["teacher_embedding"]

        student_emb = model(input_ids, attention_mask)

        loss = full_loss(student_emb, teacher_emb, self.args.loss_weights)

        return (loss, student_emb) if return_outputs else loss
```

```python
# train.py
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="runs/micro-qat-full",
    num_train_epochs=30,
    per_device_train_batch_size=64,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    logging_steps=50,
    save_strategy="epoch",
    save_steps=5,
    report_to="wandb",              # free WandB integration
    run_name="micro-qat-full",
    fp16=torch.cuda.is_available(), # mixed precision on CUDA
)

trainer = TernTrainer(
    model=student,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

**One important addition — the QAT warmup hook:**

The Trainer doesn't know about ternary warmup. Add it via a callback:

```python
from transformers import TrainerCallback

class QATWarmupCallback(TrainerCallback):
    def __init__(self, warmup_epochs):
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        epoch = int(state.epoch)
        active = epoch >= self.warmup_epochs
        # Toggle ternary quantization on/off on all BitLinear layers
        for module in model.modules():
            if isinstance(module, BitLinear):
                module.quantize = active

trainer = TernTrainer(
    ...
    callbacks=[QATWarmupCallback(warmup_epochs=5)],
)
```

---

## 5. Reference Notebooks — What Each One Covers

The four notebooks in `01-ternary-transformer/notebooks/` are for learning, not for the training pipeline. They exist to build your mental model. Use them as reference when specific concepts feel unclear.

| Notebook | What it teaches | When to read it |
|---|---|---|
| `01-attention-from-scratch.ipynb` | How attention works — Q, K, V, scaled dot-product | Before Milestone 2 if the forward pass feels unclear |
| `02-bitlinear-layer.ipynb` | The ternary weight mechanics and STE | Before Milestone 3 — read alongside the BitLinear source |
| `03-full-model-architecture.ipynb` | How the layers compose into an encoder | When writing `model/student.py` |
| `04-distillation-training.ipynb` | The distillation training loop | Before Milestone 2/3 — overlap with `train.py` |

These notebooks use `AutoTokenizer.from_pretrained('bert-base-uncased')` in some places. For the training pipeline, use the `tokenizers` library directly instead (as specified in the phase 1 prototype doc and setup.md).

---

## 6. Microsoft BitNet Repo — What's Relevant for Phase 2

**Not relevant to Phase 1 training.** Come back to this when building the Wasm engine.

When you do:

```
refs/bitnet-cpp/
├── src/
│   └── ggml-bitnet.cpp    ← The ternary kernel math. This is what the Wasm engine reimplements.
├── preset_kernels/         ← Optimised per-shape kernels. Reference for SIMD optimisation.
└── utils/weight_processing_and_packing.py  ← Bit-packing logic for the .bin export format
```

The weight packing format in `weight_processing_and_packing.py` is worth studying before writing the Phase 1 export script — it's useful to know whether your `.bin` format is compatible with established conventions even if you're not using their tooling.

---

## 7. Key Papers (Read Selectively)

You don't need to read these end to end. Specific sections are noted.

| Paper | What to read | Why |
|---|---|---|
| **BitNet b1.58** (Ma et al. 2024) | Section 3 (method), Table 1 | The ternary weight formulation — absmean scaling, the zero state as sparsity |
| **DistilBERT** (Sanh et al. 2019) | Section 2 (distillation approach) | The original output distillation approach. Simple, well-validated. |
| **TinyBERT** (Jiao et al. 2020) | Section 3.2 (transformer distillation) | Intermediate layer alignment — the approach described in `tern-future-work.md` Section 2 |
| **all-MiniLM-L6-v2** (Wang et al.) | The model card on HuggingFace | What the teacher was trained on — important for understanding what you're distilling |

Find them on ArXiv or the HuggingFace model card.

---

## 8. Common Failure Modes and What They Look Like

Keep this section open during Milestone 3 training runs.

**Weight collapse (Risk 1 from phase 1 doc)**
```
Symptom: zero_fraction climbs above 0.6 and stays there, loss plateaus
WandB: weights/zero_fraction trending up after QAT activation
Fix: extend warmup_epochs (try 10 instead of 5), restart
```

**Embedding collapse (distinct from weight collapse)**
```
Symptom: embed/std_mean drops toward 0 — all embeddings becoming similar
WandB: embed/std_mean < 0.05 after epoch 10
Fix: increase variance_loss weight (0.05 → 0.15), check contrastive loss isn't zero
```

**Loss divergence**
```
Symptom: loss spikes to NaN or very large values
WandB: loss/total goes to inf
Fix: reduce learning rate (1e-4 → 5e-5), check grad_clip is active (max_grad_norm=1.0)
```

**Teacher-student dimension mismatch**
```
Symptom: RuntimeError on the cosine_similarity call
Fix: verify model output projection is 256 → 384, teacher output is 384
     print(student_emb.shape, teacher_emb.shape) before the loss call
```
