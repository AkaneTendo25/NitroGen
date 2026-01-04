import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Low-rank adapter for a Linear layer: y = W x + scale * B(Ax).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.dropout is not None:
            x = self.dropout(x)
        lora = F.linear(x, self.lora_A)
        lora = F.linear(lora, self.lora_B) * self.scale
        return out + lora


def _replace_linear(module: nn.Module, rank: int, alpha: float, dropout: float) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
        else:
            _replace_linear(child, rank, alpha, dropout)


def apply_lora_to_modules(
    root: nn.Module,
    modules: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    for name in modules:
        target = getattr(root, name, None)
        if target is None:
            raise ValueError(f"Module '{name}' not found on root model")
        _replace_linear(target, rank, alpha, dropout)


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
