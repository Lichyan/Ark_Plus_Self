"""Utilities for applying LoRA adapters to classification backbones."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wrap an existing ``nn.Linear`` layer with a LoRA adapter."""

    def __init__(
        self,
        linear: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank must be positive")

        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.r = r
        self.scaling = alpha / r

        self.lora_A = nn.Linear(linear.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, linear.out_features, bias=False)

        # Recommended initialisation: A ~ N(0, 0.02), B = 0
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        result = self.linear(x)
        lora_update = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return result + lora_update


def _matches_any(name: str, targets: Sequence[str]) -> bool:
    return any(target in name for target in targets)


def inject_lora(
    model: nn.Module,
    targets: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> Iterable[str]:
    """Replace target linear layers with LoRA wrapped variants.

    Args:
        model: model to modify in-place.
        targets: list of substrings used to match module qualified names.
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: LoRA dropout probability.

    Returns:
        Iterable of module names that were wrapped.
    """

    replaced = []

    def _recursive_replace(module: nn.Module, parent_name: str = "") -> None:
        for child_name, child in list(module.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and _matches_any(full_name, targets):
                wrapped = LoRALinear(child, rank, alpha, dropout)
                setattr(module, child_name, wrapped)
                replaced.append(full_name)
            else:
                _recursive_replace(child, full_name)

    _recursive_replace(model)
    return replaced


def freeze_non_lora_parameters(model: nn.Module, keep_head: bool = True) -> None:
    """Freeze all parameters except LoRA adapters and optionally the classification head."""

    for param in model.parameters():
        param.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad = True
            module.lora_B.weight.requires_grad = True

    if keep_head and hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True


def get_lora_trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """Yield trainable LoRA parameters to help with logging/debugging."""

    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield from module.lora_A.parameters()
            yield from module.lora_B.parameters()

