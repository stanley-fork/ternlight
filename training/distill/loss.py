"""Loss functions for distillation training.

Phase 2 (fp32 baseline) uses `distillation_loss` only.
Phase 3 (QAT) adds `contrastive_loss` as a guardrail against quantization-
induced embedding collapse — see docs/tern-training-pipeline.md.
"""

import torch
import torch.nn.functional as F


def distillation_loss(
    student_emb: torch.Tensor,   # (B, output_dim) — L2 normalized
    teacher_emb: torch.Tensor,   # (B, output_dim) — L2 normalized
) -> torch.Tensor:
    """Cosine distillation loss.

    Both inputs are unit vectors, so cosine similarity ∈ [-1, 1] and
    `1 - cosine_similarity` ∈ [0, 2]. Perfect alignment → 0. Orthogonal → 1.

    The mean over the batch is what we backprop through. The teacher
    embedding is a fixed target; no gradient flows into it.
    """
    cos_sim = F.cosine_similarity(student_emb, teacher_emb, dim=-1)
    return (1.0 - cos_sim).mean()


def contrastive_loss(student_emb: torch.Tensor) -> torch.Tensor:
    """Within-batch repulsion — penalize high similarity between *different*
    samples in the same batch.

    Used in Phase 3 (QAT) as a guardrail. Under ternary quantization, the
    model has reduced expressive capacity and can collapse the embedding
    space — many inputs mapped to the same region, each near its teacher
    target but indistinguishable from siblings at retrieval time. This term
    penalizes that pattern.

    Not a true contrastive loss (no known positive/negative pairs). It's a
    constraint that says "different inputs should land at different points."
    Real contrastive learning using anchor/positive pairs is a v2 lever —
    see docs/tern-training-pipeline.md.

    Input `student_emb` is L2-normalized, so `student_emb @ student_emb.T`
    is the cosine similarity matrix.
    """
    sim_matrix = student_emb @ student_emb.T
    return sim_matrix.fill_diagonal_(0).pow(2).mean()
