from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


def grl(x, alpha: float = 1.0):
    return GradientReverse.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """Domain discriminator matching the original CHIMERA/CDAN shape.

    The original repository builds a two-class discriminator on the flattened
    conditional representation ``softmax(prediction) ⊗ GRL(feature)``.  This
    module keeps that two-class output while using the shared engineering
    trainer.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 2)),
            nn.BatchNorm1d(max(hidden_dim // 2, 2)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 2), 2),
        )

    def reset_parameters(self):
        for module in self.modules():
            if hasattr(module, "reset_parameters") and module is not self:
                module.reset_parameters()

    def forward(self, x):
        return self.net(x)


def entropy_logits(linear_output: torch.Tensor) -> torch.Tensor:
    p = F.softmax(linear_output, dim=1)
    return -torch.sum(p * torch.log(p + 1e-5), dim=1)


def _weighted_domain_ce(linear_output: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    labels = labels.long().view(-1)
    if weights is None:
        return F.cross_entropy(linear_output, labels)
    losses = F.cross_entropy(linear_output, labels, reduction="none")
    return torch.sum(weights * losses) / torch.clamp(torch.sum(weights), min=1e-12)


def cdan_representation(
    rep: torch.Tensor,
    logits_or_pred: torch.Tensor,
    task_type: str | None = None,
    alpha: float = 1.0,
    reverse: bool = False,
) -> torch.Tensor:
    """Build the original CHIMERA CDAN conditional representation.

    The original code computes ``softmax(score).detach()`` and then forms
    ``torch.bmm(prob.unsqueeze(2), GRL(feature).unsqueeze(1))``.  For regression,
    ``score`` is the scalar output with shape ``[B, 1]``; softmax over one column
    equals one, so this reduces to feature-domain alignment, matching the
    original regression implementation.
    """
    if logits_or_pred.ndim == 1:
        logits_or_pred = logits_or_pred.view(-1, 1)
    probs = F.softmax(logits_or_pred.detach(), dim=1)
    rep_used = grl(rep, alpha) if reverse else rep
    feature = torch.bmm(probs.unsqueeze(2), rep_used.unsqueeze(1))
    return feature.view(feature.size(0), -1)


def domain_loss(
    rep_src: torch.Tensor,
    logits_src: torch.Tensor,
    rep_tgt: torch.Tensor,
    logits_tgt: torch.Tensor,
    task_type: str,
    discriminator: nn.Module,
    alpha: float = 1.0,
    entropy_weight: bool = True,
) -> torch.Tensor:
    """Original-style CDAN loss used by the GitHub implementation.

    Classification uses entropy-weighted source/target domain CE, following the
    original ``train_simada`` code. Regression keeps the original unweighted form;
    because the regression prediction has one column, the conditional product is
    equivalent to GRL(feature).
    """
    src_feat = cdan_representation(rep_src, logits_src, task_type, alpha=alpha, reverse=True)
    tgt_feat = cdan_representation(rep_tgt, logits_tgt, task_type, alpha=alpha, reverse=True)
    src_logits = discriminator(src_feat)
    tgt_logits = discriminator(tgt_feat)

    src_labels = torch.zeros(src_feat.size(0), dtype=torch.long, device=src_feat.device)
    tgt_labels = torch.ones(tgt_feat.size(0), dtype=torch.long, device=tgt_feat.device)

    src_weight = tgt_weight = None
    if task_type == "classification" and entropy_weight:
        entropy_src = grl(entropy_logits(logits_src), 1.0)
        entropy_tgt = grl(entropy_logits(logits_tgt), 1.0)
        src_weight = 1.0 + torch.exp(-entropy_src)
        tgt_weight = 1.0 + torch.exp(-entropy_tgt)
        src_weight = src_weight / torch.clamp(torch.sum(src_weight), min=1e-12)
        tgt_weight = tgt_weight / torch.clamp(torch.sum(tgt_weight), min=1e-12)

    return _weighted_domain_ce(src_logits, src_labels, src_weight) + _weighted_domain_ce(tgt_logits, tgt_labels, tgt_weight)
