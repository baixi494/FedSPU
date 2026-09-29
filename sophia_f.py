from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def estimate_prototypes_sophia_f(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
    curvature_eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, int]:


    if curvature_eps <= 0:
        raise ValueError("curvature_eps must be positive")

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")

    g_sum = [torch.zeros_like(p, device=device) for p in params]
    h_sum = [torch.zeros_like(p, device=device) for p in params]
    criterion = nn.CrossEntropyLoss(reduction="mean")

    sample_count = 0
    for batch_idx, (xb, yb) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        batch_size = int(yb.shape[0])
        if batch_size == 0:
            continue

        model.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        with torch.no_grad():
            for idx, (p, grad) in enumerate(zip(params, grads)):
                g = grad if grad is not None else torch.zeros_like(p)
                g_sum[idx].add_(g, alpha=float(batch_size))
                h_sum[idx].addcmul_(g, g, value=float(batch_size))
        sample_count += batch_size

    if sample_count == 0:
        total = sum(p.numel() for p in params)
        return (
            torch.zeros(total, dtype=torch.float32),
            torch.ones(total, dtype=torch.float32),
            0,
        )

    inv_n = 1.0 / float(sample_count)
    with torch.no_grad():
        g_vec = torch.cat([(g * inv_n).reshape(-1) for g in g_sum], dim=0)
        h_vec = torch.cat([(h * inv_n).reshape(-1) for h in h_sum], dim=0)
        h_vec = torch.clamp(h_vec, min=curvature_eps)

    return g_vec.detach().cpu(), h_vec.detach().cpu(), sample_count
