"""Frozen ViT-B/16 image towers, LoRA adapters and classification heads."""
from __future__ import annotations

import math

import timm
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .config import LORA_ALPHA, METHODS, RANK

class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, RANK, bias=False)
        self.up = nn.Linear(RANK, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + (LORA_ALPHA / RANK) * self.up(self.down(x))


class MultiHeadModel(nn.Module):
    def __init__(self, name, counts, tuple_count, method, pretrained=True):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown training method: {method}")
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        inserted = 0
        for block in self.backbone.blocks:
            for attr in ("qkv", "proj"):
                old = getattr(block.attn, attr)
                if not isinstance(old, nn.Linear):
                    raise TypeError(f"Unexpected attention module {name}: {attr}")
                setattr(block.attn, attr, LoRALinear(old))
                inserted += 1
        if inserted != 24:
            raise ValueError(f"Expected 24 LoRA targets, got {inserted}")
        features = self.backbone.num_features
        if features != 768:
            raise ValueError("The paper uses ViT-B image towers with 768-dimensional features")
        if method == "joint":
            self.heads = nn.ModuleList([nn.Linear(features, tuple_count)])
        else:
            self.heads = nn.ModuleList([nn.Linear(features, count) for count in counts])

    def forward(self, x):
        z = self.backbone(x)
        if z.ndim != 2:
            raise ValueError(f"Unexpected feature shape: {z.shape}")
        return [head(z) for head in self.heads]


def transforms_for(model):
    config = timm.data.resolve_model_data_config(model.backbone)
    mean, std = config["mean"], config["std"]
    normalize = transforms.Normalize(mean, std)
    train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    valid = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(), normalize,
    ])
    return train, valid, config
