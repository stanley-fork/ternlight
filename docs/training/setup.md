# Phase 1 — Setup & Environment

> Covers everything needed before writing any training code: Python environment, dependencies, reference projects, remote training options, WandB, and MCP configuration for Claude to assist effectively.

---

## 1. Directory Structure

```
nano-semantics/
├── tern-core/                          # documentation + design docs
│   ├── tern-scoping.md
│   ├── tern-architecture.md
│   ├── tern-phase1-prototype.md
│   ├── tern-phase2-prototype.md
│   ├── tern-future-work.md
│   ├── tern-model-sizing.md
│   ├── 01-ternary-transformer/
│   │   └── notebooks/                 # learning / exploration notebooks
│   │       ├── 01-attention-from-scratch.ipynb
│   │       ├── 02-bitlinear-layer.ipynb
│   │       ├── 03-full-model-architecture.ipynb
│   │       └── 04-distillation-training.ipynb
│   └── tern-distill-prototype/        # Phase 1 design docs (here)
│       ├── design.md
│       ├── milestones.md
│       ├── setup.md                   ← this file
│       └── implementation-guide.md
│
├── tern-distill/                      # Phase 1 training code
│   ├── config/
│   │   ├── micro.yaml
│   │   ├── micro-fp32.yaml
│   │   └── small.yaml
│   ├── data/
│   │   ├── prepare.py
│   │   ├── dataset.py
│   │   └── cache/                     # gitignored
│   ├── model/
│   │   ├── bitlinear.py
│   │   └── student.py
│   ├── training/
│   │   ├── loss.py
│   │   ├── trainer.py
│   │   └── health.py
│   ├── eval/
│   │   ├── tasks.py
│   │   └── corpora/
│   ├── runs/                          # gitignored
│   ├── train.py
│   ├── evaluate.py
│   └── requirements.txt
│
└── refs/                              # reference implementations — read-only
    ├── bitlinear/                     # schneiderkamplab/bitlinear
    ├── bitnet-cpp/                    # microsoft/BitNet (inference kernels)
    └── tokenizers/                    # huggingface/tokenizers (Phase 2 ref)
```

**Key rule:** `tern-core/` is docs only. `tern-distill/` is code only. `refs/` is cloned upstream source — never modify, only read.

---

## 2. Python Environment

A `.venv` already exists at `tern-core/.venv`. For the training code in `tern-distill/`, use the same environment or create a sibling one — either works, just be consistent.

**Recommended: use `uv` for fast dependency management**

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv at the project root level (recommended — shared across tern-distill and notebooks)
cd /Users/wenshutang/Documents/Projects/nano-semantics
uv venv .venv --python 3.11

# Activate
source .venv/bin/activate
```

If you prefer to keep using the existing `tern-core/.venv`, activate it from anywhere:

```bash
source /Users/wenshutang/Documents/Projects/nano-semantics/tern-core/.venv/bin/activate
```

---

## 3. Dependencies

```bash
# Core ML stack — standard build includes MPS support on Apple Silicon automatically
uv pip install torch
# For CUDA (remote only — not needed for M4 Max):
# uv pip install torch --index-url https://download.pytorch.org/whl/cu121

# HuggingFace ecosystem
uv pip install sentence-transformers tokenizers datasets transformers evaluate

# Reference BitLinear implementation
uv pip install bitlinear  # schneiderkamplab/bitlinear

# Experiment tracking
uv pip install wandb

# Utilities
uv pip install scikit-learn pyyaml tqdm

# Notebook support (for the 01-ternary-transformer notebooks)
uv pip install jupyter ipykernel ipywidgets
```

Full `requirements.txt` for `tern-distill/`:

```
torch>=2.2
sentence-transformers>=2.7  # teacher model only — training-time, never ships

tokenizers>=0.19
datasets>=2.19
transformers>=4.40
evaluate>=0.4
bitlinear>=0.1
wandb>=0.17
scikit-learn>=1.4
pyyaml>=6.0
tqdm>=4.66
```

**Verify GPU (Mac MPS or remote CUDA):**

```python
import torch
print(torch.cuda.is_available())          # True on CUDA (remote)
print(torch.backends.mps.is_available())  # True on Apple Silicon (local)
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using: {device}")
```

> **M4 Max (this machine):** 40 GPU cores, 48GB unified memory, Metal 4. MPS handles float32 transformers well — Milestones 0–2 run comfortably locally. For Milestone 3+ (QAT with BitLinear), run 1 epoch locally first to check for MPS op compatibility with the STE autograd. If it errors, fall back to `device="cpu"` for short runs or go remote for full training. The 48GB unified memory means you won't hit OOM — the risk is op support, not memory.

---

## 4. Reference Projects

Clone into `refs/` once. Never pip-install from them — they're read-only source references.

```bash
mkdir -p /Users/wenshutang/Documents/Projects/nano-semantics/refs
cd /Users/wenshutang/Documents/Projects/nano-semantics/refs

# Primary training reference — BitLinear implementation
git clone https://github.com/schneiderkamplab/bitlinear

# Microsoft's official BitNet — inference kernels reference for Phase 2
# The C++ kernel implementations in src/ are the Wasm engine reference
git clone https://github.com/microsoft/BitNet bitnet-cpp

# HuggingFace tokenizers — Rust source for Phase 2 Wasm build
git clone https://github.com/huggingface/tokenizers
```

**What to read in each (don't read everything):**

| Repo | What to read | Why |
|---|---|---|
| `bitlinear/` | `bitlinear/bitlinear.py` | The actual BitLinear layer implementation — verify it matches the absmean spec before using |
| `bitnet-cpp/` | `src/ggml-bitnet.cpp`, `preset_kernels/` | Ternary kernel math — reference for the Phase 2 Wasm engine |
| `tokenizers/` | `tokenizers/src/models/wordpiece.rs` | WordPiece Rust implementation — Phase 2 reference |

---

## 5. Training Hardware

### Primary: Mac Studio M4 Max (this machine)

**Apple M4 Max — 40 GPU cores, 48GB unified memory, Metal 4.**

This is a serious training machine. The 48GB unified memory pool means the model, optimizer states, and dataset all fit comfortably without hitting OOM. Plan to do all milestones locally first before considering remote.

| Milestone | Local M4 Max | Notes |
|---|---|---|
| 1 — data pipeline | ✓ | Teacher encoding on MPS, fast |
| 2 — float32 baseline | ✓ | MPS handles standard transformers well |
| 3 — QAT / BitLinear | ✓ test first | Run 1 epoch — if BitLinear STE hits an MPS op error, fall back to `device="cpu"` for that run |
| 4 — full 30-epoch run | ✓ viable | Slower than an A100 but entirely doable. Estimate ~3–6 hrs at micro tier. |
| 5 — eval | ✓ | No GPU required |

**Device selection in code:**

```python
device = (
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)
```

Do not hardcode `"cuda"` — it will fail silently or error on this machine.

**If BitLinear STE hits an MPS error at Milestone 3:** fall back to CPU for that specific run. CPU on M4 Max is still fast for small batch sizes and short smoke-test runs. The risk is op support, not memory or compute.

### Optional: Remote GPU

Only worth considering if the full Milestone 4 run feels too slow locally, or if MPS proves incompatible with BitLinear and CPU is unacceptably slow.

| Provider | Notes |
|---|---|
| **RunPod** | On-demand, RTX 4090 ~$0.50/hr. Spin up, run, shut down. |
| **Lambda Labs** | Persistent storage, A10 ~$0.75/hr. Better for iterative runs. |
| **Google Colab Pro** | $10/month, convenient but session time limits require robust checkpointing. |

**If going remote:**
```bash
git clone https://github.com/YOUR_FORK/nano-semantics
cd nano-semantics/tern-distill
pip install -r requirements.txt
wandb login
python train.py --config config/micro-fp32.yaml
```

Always checkpoint every 5 epochs — remote instances can be preempted.

---

## 6. WandB Setup

```bash
pip install wandb
wandb login  # prompts for API key — get it from wandb.ai/authorize
```

**Project setup — do this once:**

```python
import wandb
wandb.init(
    project="tern-distill",
    entity="YOUR_WANDB_USERNAME",  # optional, defaults to personal
)
```

All training runs go into the `tern-distill` project. Follow the naming convention from `milestones.md`:

```
micro-fp32-baseline     # Milestone 2
micro-qat-warmup5       # Milestone 3
micro-qat-full          # Milestone 4
small-qat-full          # Milestone 4 fallback
```

**What to track (minimum):**

```python
wandb.log({
    "loss/total": total_loss,
    "loss/distillation": distill_loss,
    "train/cosine_sim": cosine_sim,
    "weights/zero_fraction": zero_frac,   # Milestone 3+
    "embed/std_mean": embed_std,
    "epoch": epoch,
})
```

---

## 7. MCP Configuration for Claude

MCPs give Claude persistent access to documentation and source code, which is particularly useful when working through unfamiliar architecture code. Recommended for this project:

### Context7 — Library Documentation

Resolves live documentation for HuggingFace, PyTorch, sentence-transformers, and other libraries. Prevents Claude from hallucinating deprecated APIs.

```json
// .claude/settings.json — add to mcpServers
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    }
  }
}
```

**Most useful for:** Looking up current `Trainer` API, `SentenceTransformer` encode parameters, `tokenizers` crate Rust API, `datasets` load_dataset syntax.

### GitHub MCP — Reference Repo Exploration

Lets Claude read files from any public GitHub repo directly without needing them cloned locally. Useful for exploring `schneiderkamplab/bitlinear` and `microsoft/BitNet` source on demand.

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_TOKEN"
      }
    }
  }
}
```

**Most useful for:** Reading the BitLinear source to verify the absmean spec, exploring the Microsoft BitNet C++ kernels for Phase 2 reference, checking HuggingFace tokenizers Rust source.

Get a token at: GitHub → Settings → Developer Settings → Personal Access Tokens (read-only public repo scope is sufficient).

### Combined settings.json

```json
{
  "mcpServers": {
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp@latest"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_TOKEN"
      }
    }
  }
}
```

Place this at either:
- `~/.claude/settings.json` — applies globally to all Claude Code sessions
- `nano-semantics/tern-core/.claude/settings.json` — already exists, applies only to this project (preferred)

---

## 8. Gitignore

Add to `nano-semantics/.gitignore`:

```
# Python
.venv/
__pycache__/
*.pyc
*.pyo

# Training artifacts
tern-distill/data/cache/
tern-distill/runs/

# Secrets
.env
wandb/

# Mac
.DS_Store

# Jupyter checkpoints
.ipynb_checkpoints/
```
