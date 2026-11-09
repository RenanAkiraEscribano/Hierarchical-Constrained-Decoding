"""Training operations used by the main LoRA experiment."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import LAMBDA_KL, WEIGHT_DECAY

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def kl_targets(tuples, counts, device):
    output = []
    for depth in range(len(counts) - 1):
        matrix = torch.zeros((counts[depth], counts[depth + 1]), dtype=torch.float32)
        for values in tuples:
            matrix[values[depth], values[depth + 1]] = 1
        matrix /= matrix.sum(dim=1, keepdim=True).clamp_min(1)
        output.append(matrix.to(device))
    return output


def loss_for(logits, labels, tuple_ids, method, targets):
    if method == "joint":
        return F.cross_entropy(logits[0], tuple_ids)
    ce = torch.stack([F.cross_entropy(value, labels[:, depth]) for depth, value in enumerate(logits)]).mean()
    if method == "independent":
        return ce
    penalties = []
    for depth, target in enumerate(targets):
        q = target[labels[:, depth]]
        log_p = F.log_softmax(logits[depth + 1].float(), dim=1)
        log_q = torch.where(q > 0, q.clamp_min(1e-20).log(), 0)
        penalties.append((q * (log_q - log_p)).sum(dim=1).mean())
    return ce + LAMBDA_KL * torch.stack(penalties).mean()


def exact_count(logits, labels, tuple_ids, method):
    if method == "joint":
        return int((logits[0].argmax(dim=1) == tuple_ids).sum().item())
    predicted = torch.stack([value.argmax(dim=1) for value in logits], dim=1)
    return int((predicted == labels).all(dim=1).sum().item())


def one_epoch(model, loader, method, targets, optimizer=None, scaler=None, device='cuda'):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    for images, labels, tuple_ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        tuple_ids = tuple_ids.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=torch.device(device).type == "cuda"):
            logits = model(images)
            loss = loss_for(logits, labels, tuple_ids, method, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss: {loss.item()}")
        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer)
            scaler.update()
        size = len(images)
        total += size
        total_loss += loss.item() * size
        total_correct += exact_count(logits, labels, tuple_ids, method)
    return {"loss": total_loss / total, "exact": total_correct / total, "samples": total}


def make_optimizer(model, lr):
    lora = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("heads.")]
    heads = list(model.heads.parameters())
    if sum(p.numel() for p in lora) != 884736:
        raise ValueError(f"Unexpected LoRA parameter count: {sum(p.numel() for p in lora)}")
    return torch.optim.AdamW([
        {"params": lora, "lr": lr},
        {"params": heads, "lr": lr * 10},
    ], weight_decay=WEIGHT_DECAY)
