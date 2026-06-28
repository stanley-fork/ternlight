# Model Internals — Forward Pass, Backprop, and Distillation

> Math reference for what the student model computes, how gradients flow during
> QAT, and why the distillation design works the way it does. Read this when
> implementing a custom loader, modifying training, or porting to a new target.
>
> For the runtime engine, see [inference-engine.md](inference-engine.md). For
> the high-level system layout, see [architecture.md](architecture.md).

---

## 1. Forward Pass

The model takes a batch of token IDs and produces L2-normalized 384-dim
embeddings. Configuration: `d_model=256`, `n_layers=2`, `n_heads=4`,
`ffn_dim=1024`, `output_dim=384`, `max_len=128`.

```
input_ids               (batch, 128)                  ← token IDs
    ↓ embedding lookup
x                       (batch, 128, 256)             ← float32
    ↓ TransformerLayer × 2 (Pre-LN):
    │   x = x + dropout(attn(norm1(x), mask))
    │   x = x + dropout(ffn(norm2(x)))
    ↓
final norm              (batch, 128, 256)
    ↓ mean pool (mask-aware)
pooled                  (batch, 256)
    ↓ projection (Linear 256→384, float32)
projected               (batch, 384)
    ↓ L2 normalize
output                  (batch, 384)                  ← unit vectors
```

**Pre-LN, not Post-LN.** LayerNorm is applied *before* the sublayer in each
residual block — more stable for training than the Post-LN layout used in
original BERT.

### Attention sublayer

Standard multi-head self-attention with 4 heads, `d_head=64`:

```
Q, K, V = BitLinear projections of norm1(x)
scores  = (Q @ K^T) / sqrt(d_head),   mask padding columns to -inf
attn    = softmax(scores) @ V
out     = BitLinear projection of merge_heads(attn)
```

Q/K/V projections have no bias; the output projection does.

### Feed-forward sublayer

Standard `d → 4d → d` expansion with GELU:

```
hidden = GELU(BitLinear_up(norm2(x)))    # 256 → 1024
out    = BitLinear_down(hidden)          # 1024 → 256
```

### Mean pool over real tokens

```
pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
```

Padding positions contribute zero to the sum and don't count in the divisor.

### L2 normalize

```
output = projected / ||projected||₂
```

After normalization, cosine similarity is just a dot product:

```
cos(a, b) = a · b   when ||a|| = ||b|| = 1
```

---

## 2. BitLinear — What Replaces `nn.Linear` During QAT

Every `nn.Linear` in the attention and FFN sublayers becomes a `BitLinear`
during QAT (12 total: 6 per layer × 2 layers). The projection layer is
deliberately excluded.

The BitLinear forward is more involved than naive ternary quantization:

```python
def forward(self, x):
    # 1. Internal parameter-less LayerNorm
    x_norm = layer_norm(x, [in_features])               # eps=1e-5, no affine

    # 2. Activation int8 quantization (per-token AbsMax)
    x_scale = 128 / x_norm.abs().max(dim=-1, keepdim=True).clamp(min=1e-5)
    x_quant = round_clamp(x_norm * x_scale, [-128, 127])

    # 3. Weight ternary quantization (AbsMedian, scalar per matrix)
    w_scale = 1 / weight.abs().median().clamp(min=1e-5)
    w_quant = round_clamp(weight * w_scale, [-1, 1])    # {-1, 0, +1}

    # 4. Quantized matmul + bias
    y_quant = linear(x_quant, w_quant, bias)

    # 5. Rescale to the natural magnitude
    return y_quant / (w_scale * x_scale)
```

**Five facts that are easy to miss** - every implementation needs to handle
these to match training-time arithmetic exactly:

1. The internal LayerNorm exists. It's parameter-less.
2. Activations are int8-quantized via `round_clamp` before the matmul - not
   passed as raw float32.
3. Weights use AbsMedian (not AbsMean or AbsMax). Different cutoff for snapping
   to zero.
4. The output is divided by `(w_scale · x_scale)`. Without this, magnitudes are
   off by a per-matrix learned factor.
5. The bias is stored in *post-rescale* space - added before the division and
   divided too. Implementations that "store bias separately and add later"
   need to compensate.

---

## 3. Backprop and QAT

The challenge with ternary quantization (`round_clamp`) is that it has zero gradient almost everywhere. Without intervention, no gradient would reach the weights through BitLinear, and the model couldn't learn.

### Straight-through estimator (STE)

The standard fix: forward pass uses quantized values, backward pass treats the quantization as identity. Gradients flow through as if the operation were a no-op.

```python
def round_clamp(input, range, lambda_=1):
    return lambda_ * (input.round().clamp(range[0], range[1]) - input).detach() + input
```

When `lambda_ = 1`:
- Forward: behaves as `round_clamp(input)` (`.detach()` makes the residual a
  constant in value).
- Backward: gradient flows through the `+ input` term as identity. STE in
  effect.

When `lambda_ = 0`: forward is `input` (pass-through). Used during warmup.

### Two-stage training

Training starts with `lambda_ = 0` for the first 5 epochs — pure fp32. At epoch 6, `lambda_ = 1` activates QAT. A small loss spike usually appears as the model adjusts; by epoch ~10 the loss recovers.

Why: starting with random weights *and* aggressive quantization simultaneously is brittle. Warmup gets the fp32 weights into a basin where their *signs* are
well-determined (which is what ternary quantization actually preserves), then
activating quantization adds noise the model can absorb.

### Shadow weights

During QAT the optimizer updates the **fp32** weights; the forward pass
quantizes them on the fly. The fp32 values are the model's actual learned
state. At export, they're snapped to ternary one final time and written as
2-bit packed values to `.bin`. Then the fp32 shadows are discarded.

---

## 4. Distillation Loss

### The full loss

```python
loss = 1.0 * distillation_loss + 0.15 * contrastive_loss

distillation_loss = (1 - cos(student, teacher)).mean()
contrastive_loss  = student_sim_matrix.fill_diagonal_(0).pow(2).mean()
```

### Why cosine sim over MSE

Three reasons:

1. **Both vectors are unit vectors** (L2-normalized). MSE over-weights scale
   differences that don't exist here.
2. **Embedding spaces are direction-encoded.** Meaning lives in the angle, not
   the magnitude. Cosine sim is the natural metric.
3. **Bounded in [0, 2]** — well-behaved for optimization, no gradient scaling.

Loss equals 0 at perfect alignment, 1 at orthogonal, 2 at opposite.

### Why the contrastive guardrail (W=0.15)

Distillation alone has no penalty for mode collapse. If the student maps every
input to roughly the same vector (matching teacher poorly but uniformly), the average cosine distance is still finite. The model can settle into a
low-entropy equilibrium where embeddings cluster in a tiny region of the
hypersphere.

The contrastive term penalizes high pairwise similarity *within* a batch — a
"spreading force" pushing the embedding cloud toward filling the hypersphere
more uniformly.

Why 0.15 specifically: at W ≥ 0.5 the student prefers spread over teacher
matching, hurting the primary signal. At W < 0.05 the force is too weak to
prevent clustering on long runs. 0.15 was empirically chosen.

