"""Turning an action mask into a distribution that can never sample an illegal action."""

from __future__ import annotations

import torch
from torch.distributions import Categorical

__all__ = ["masked_categorical"]


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Categorical over the legal actions only.

    Masked logits are driven to -inf so their probability is exactly zero, not merely small.

    Args:
        logits: Unmasked action logits.
        mask: Boolean legality mask, same shape as `logits`.

    Returns:
        A `Categorical` distribution that assigns zero probability to every illegal action.

    Raises:
        ValueError: A row of the mask has no legal action, which the caller should have
            detected as a terminal state instead of asking for a distribution.
    """
    legal = mask.bool()
    if not legal.any(dim=-1).all():
        msg = "action mask has a row with no legal action"
        raise ValueError(msg)
    return Categorical(logits=logits.masked_fill(~legal, float("-inf")))
