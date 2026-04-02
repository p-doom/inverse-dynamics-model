from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 0.0,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Per-token focal loss (reduces to standard CE when gamma=0)."""
    ce = F.cross_entropy(
        logits, targets,
        reduction="none",
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    if gamma > 0.0:
        with torch.no_grad():
            p_t = torch.exp(-ce)
            focal_weight = (1.0 - p_t) ** gamma
        ce = focal_weight * ce
    return ce


def _weighted_causal_lm_loss(
    logits_BSV: torch.Tensor,
    labels_BS: torch.Tensor,
    label_weights_BS: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    focal_gamma: float = 0.0,
    class_balanced: bool = False,
) -> torch.Tensor:
    """Causal LM loss with optional per-token weighting and class-balanced normalisation.

    When ``class_balanced=True`` the loss is normalised *per weight-bucket* (each unique
    non-zero weight value forms one bucket) and then averaged across buckets.  This gives
    equal gradient contribution from every action class regardless of how often each class
    appears in the batch, which is critical when NO_OP tokens far outnumber MOUSE tokens.
    """
    shift_logits = logits_BSV[:, :-1, :].contiguous()
    shift_labels = labels_BS[:, 1:].contiguous()
    token_loss = _focal_cross_entropy(
        logits=shift_logits.view(-1, shift_logits.shape[-1]),
        targets=shift_labels.view(-1),
        gamma=focal_gamma,
        label_smoothing=label_smoothing,
        ignore_index=-100,
    ).view_as(shift_labels)

    valid_mask = shift_labels != -100  # (B, S-1)

    # ── No weights: plain mean over valid tokens ─────────────────────────────
    if label_weights_BS is None:
        denom = torch.clamp(valid_mask.sum(), min=1).to(token_loss.dtype)
        return (token_loss * valid_mask.to(token_loss.dtype)).sum() / denom

    shift_weights = label_weights_BS[:, 1:].to(token_loss.dtype).contiguous()

    # ── Standard weighted mean ────────────────────────────────────────────────
    if not class_balanced:
        valid_weights = shift_weights * valid_mask.to(token_loss.dtype)
        denom = torch.clamp(valid_weights.sum(), min=1.0)
        return (token_loss * valid_weights).sum() / denom

    # ── Class-balanced: mean per weight-bucket, then average buckets ──────────
    # Each unique non-zero weight value corresponds to one action class.
    # Tokens with weight == 0.0 (format/structural tokens or padding) are excluded.
    active_weights = shift_weights[valid_mask]
    if active_weights.numel() == 0:
        return token_loss.sum() * 0.0

    unique_ws = torch.unique(active_weights)
    unique_ws = unique_ws[unique_ws > 0.0]  # exclude zero-weighted tokens
    if unique_ws.numel() == 0:
        return token_loss.sum() * 0.0

    bucket_losses: list[torch.Tensor] = []
    for w in unique_ws:
        bucket_mask = valid_mask & (shift_weights == w)
        n = bucket_mask.sum()
        if n > 0:
            bucket_losses.append(
                (token_loss * bucket_mask.to(token_loss.dtype)).sum()
                / n.to(token_loss.dtype)
            )

    if not bucket_losses:
        return token_loss.sum() * 0.0

    return torch.stack(bucket_losses).mean()


# ---------------------------------------------------------------------------
# Distance-aware (Gaussian soft-label) auxiliary loss for mouse delta tokens
# ---------------------------------------------------------------------------

def _build_vocab_int_values_tensor(tokenizer: Any, vocab_size: int) -> torch.Tensor:
    """Return a float tensor of shape (vocab_size,) where entry i is the integer
    value of token i if the token decodes to a plain integer in [-256, 256],
    otherwise float('nan').

    Called once at training startup; result is cached and moved to device before
    each use in _mouse_soft_label_loss.
    """
    values = torch.full((vocab_size,), float("nan"))
    for tok_id in range(vocab_size):
        try:
            text = tokenizer.decode([tok_id], skip_special_tokens=True).strip()
        except Exception:
            continue
        if not text:
            continue
        try:
            val = int(text)
        except ValueError:
            continue
        # Reject tokens like " 5" (residual spaces) or "05" (leading zero)
        if str(val) == text and -256 <= val <= 256:
            values[tok_id] = float(val)
    return values


def _mouse_soft_label_loss(
    logits_BSV: torch.Tensor,
    labels_BS: torch.Tensor,
    label_weights_BS: torch.Tensor,
    vocab_int_values_V: torch.Tensor,
    mouse_loss_weight: float,
    sigma: float,
) -> torch.Tensor:
    """Gaussian soft-label CE loss for mouse delta numeric token positions.

    For every token position where:
      (a) label_weight == mouse_loss_weight  (it's part of a MOUSE action), AND
      (b) the ground-truth label token represents a plain integer

    the standard one-hot CE target is replaced with a Gaussian distribution
    over all integer-valued vocabulary tokens centred on the true value:

        q(k) ∝ exp( -(value(k) - value(gt))² / (2σ²) )

    This means the model is penalised less for predicting a nearby value than for
    predicting a far-away one, while still rewarding exact matches most.

    Returns the mean CE loss under the soft target, or zero if no eligible
    positions are found (e.g. sigma==0 or no MOUSE tokens in batch).
    """
    if sigma <= 0.0:
        return logits_BSV.new_zeros(())

    shift_logits = logits_BSV[:, :-1, :].contiguous()    # (B, S-1, V)
    shift_labels = labels_BS[:, 1:].contiguous()          # (B, S-1)
    shift_weights = label_weights_BS[:, 1:].contiguous()  # (B, S-1)

    # Positions belonging to a MOUSE action (non-padding)
    mouse_mask = (shift_weights == mouse_loss_weight) & (shift_labels != -100)
    if not mouse_mask.any():
        return logits_BSV.new_zeros(())

    # Move the vocab value lookup to the same device (cheap if already there)
    vocab_int_values_V = vocab_int_values_V.to(shift_logits.device)

    # Index of integer-valued vocab tokens and their values
    int_vocab_mask = ~torch.isnan(vocab_int_values_V)           # (V,)
    int_ids = int_vocab_mask.nonzero(as_tuple=True)[0]          # (K,)
    if int_ids.numel() == 0:
        return logits_BSV.new_zeros(())
    int_values_K = vocab_int_values_V[int_ids]                   # (K,)

    # For each sequence position, look up the integer value of the label token
    # (NaN for tokens not in the integer vocab)
    label_int_values = vocab_int_values_V[shift_labels.clamp(min=0, max=vocab_int_values_V.shape[0] - 1)]  # (B, S-1)
    label_int_values[~mouse_mask] = float("nan")

    # Keep only positions where the ground-truth token is an integer
    int_mouse_mask = mouse_mask & ~torch.isnan(label_int_values)
    if not int_mouse_mask.any():
        return logits_BSV.new_zeros(())

    logits_NV = shift_logits[int_mouse_mask]           # (N, V)
    true_vals_N = label_int_values[int_mouse_mask]      # (N,)

    # Gaussian weights over integer vocab tokens: q[n, k] ∝ exp(-(v_k - v_gt_n)²/2σ²)
    diff_NK = true_vals_N.unsqueeze(1) - int_values_K.unsqueeze(0)  # (N, K)
    gaussian_NK = torch.exp(-diff_NK.pow(2) / (2.0 * sigma * sigma))
    gaussian_NK = gaussian_NK / gaussian_NK.sum(dim=1, keepdim=True)  # row-normalise

    # log p(k | context) for integer tokens only, using full-vocab softmax denominator
    log_Z_N = torch.logsumexp(logits_NV, dim=-1)              # (N,)
    log_p_int_NK = logits_NV[:, int_ids] - log_Z_N.unsqueeze(1)  # (N, K)

    # CE with Gaussian soft target: -Σ_k q[n,k] · log p[n,k]
    loss_N = -(gaussian_NK * log_p_int_NK).sum(dim=1)          # (N,)
    return loss_N.mean()
