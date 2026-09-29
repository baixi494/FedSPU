from __future__ import annotations

from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import FLNet
from sophia_f import estimate_prototypes_sophia_f


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_trigger_to_tensor(
    img: torch.Tensor,
    trigger_size: int = 3,
    intensity: float = 1.0,
) -> torch.Tensor:

    if trigger_size <= 0:
        return img.clone()
    if img.dim() == 3:
        im = img.squeeze(0).clone()
    elif img.dim() == 2:
        im = img.clone()
    else:
        raise ValueError(f"expected [1,H,W] or [H,W], got {tuple(img.shape)}")

    height, width = im.shape[-2:]
    ts = min(int(trigger_size), height, width)
    im[height - ts : height, width - ts : width] = float(intensity)
    return im.unsqueeze(0)


def get_prototypes_at_state(
    train_loader: DataLoader,
    reference_state: dict,
    *,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:


    model = FLNet().to(DEVICE)
    model.load_state_dict(reference_state, strict=True)
    g_vec, h_vec, sample_count = estimate_prototypes_sophia_f(
        model,
        train_loader,
        device=DEVICE,
        max_batches=max_batches,
    )
    return g_vec, h_vec, int(sample_count)


def local_train_and_get_prototypes(
    train_loader: DataLoader,
    client_id: int = 0,
    local_epochs: int = 10,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    init_state: Optional[dict] = None,
    *,
    prototype_state: Optional[dict] = None,
    compute_prototypes: bool = True,
) -> Tuple[dict, Optional[torch.Tensor], Optional[torch.Tensor], int]:


    del client_id

    if local_epochs < 0:
        raise ValueError("local_epochs must be non-negative")

    model = FLNet().to(DEVICE)
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    model.train()
    for _ in range(local_epochs):
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

    local_state = deepcopy(model.cpu().state_dict())
    sample_count = len(train_loader.dataset)

    if not compute_prototypes:
        return local_state, None, None, int(sample_count)

    ref_state = prototype_state if prototype_state is not None else local_state
    g_vec, h_vec, estimated_count = get_prototypes_at_state(
        train_loader,
        ref_state,
    )
    return local_state, g_vec, h_vec, int(estimated_count)
