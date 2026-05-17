"""Loss functions for distillation training.

Phase 2 (fp32 baseline) uses `distillation_loss` only.
Phase 3 (QAT) will add a contrastive guardrail — kept out of here until then
to avoid premature surface area.
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
