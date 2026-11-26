"""Learning-rate, dropout, and parameter-group schedules for CEM-Net (Table 2, Eq. 13-14)."""
import math
from torch import nn


def coslog4_lr(progress: float) -> float:
    """Multi-cycle coslog4 schedule h(s) over progress s in [0, 1]."""
    s = min(max(float(progress), 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(2.0 * math.pi * math.log2(1.0 + 15.0 * s)))


def cosine_decay_half(progress: float) -> float:
    """Second-half cosine decay g(s) for dropout and decoupled weight decay."""
    s = min(max(float(progress), 0.0), 1.0)
    if s <= 0.5:
        return 1.0
    return 0.5 * (1.0 + math.cos(math.pi * (2.0 * s - 1.0)))


def build_param_groups(model: nn.Module, base_lr: float = 0.04):
    """Partition CEM-Net parameters into the six optimization groups from Table 2b."""
    weights, biases, embeddings, pselu, scale, members = [], [], [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "input_members":
            members.append(p)
        elif name == "front_scale":
            scale.append(p)
        elif name.startswith("numerical."):
            embeddings.append(p)
        elif ".1.weight" in name and name.startswith("blocks."):
            pselu.append(p)
        elif name.endswith(".bias"):
            biases.append(p)
        else:
            weights.append(p)

    return [
        {"name": "shared_weights", "params": weights, "lr": base_lr * 1.0, "wd_base": 0.02, "betas": (0.9, 0.95)},
        {"name": "shared_biases", "params": biases, "lr": base_lr * 0.1, "wd_base": 0.0, "betas": (0.9, 0.95)},
        {"name": "embeddings", "params": embeddings, "lr": base_lr * 0.1, "wd_base": 0.002, "betas": (0.9, 0.95)},
        {"name": "pselu", "params": pselu, "lr": base_lr * 0.1, "wd_base": 0.002, "betas": (0.9, 0.95)},
        {"name": "input_scale", "params": scale, "lr": base_lr * 6.0, "wd_base": 0.12, "betas": (0.9, 0.95)},
        {"name": "member_factors", "params": members, "lr": base_lr * 0.0075, "wd_base": 0.001, "betas": (0.9, 0.999)},
    ]
