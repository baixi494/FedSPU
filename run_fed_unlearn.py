from __future__ import annotations

import json
import os
import random
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

from client_proto import (
    add_trigger_to_tensor,
    get_prototypes_at_state,
    local_train_and_get_prototypes,
)
from model import FLNet
from server_unlearn import posttrain_first_order, unlearn_with_J
from utils import Utils, make_asr_strict_loader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset_by_count(train_set, num_clients: int = 5):
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2")
    indices = np.random.permutation(len(train_set))
    parts = np.array_split(indices, num_clients)
    return [Subset(train_set, part.tolist()) for part in parts]


def offline_inject_client0_raw_dataset_mnist(
    base_trainset,
    client0_indices,
    *,
    target_label: int = 9,
    poison_rate: float = 0.80,
    trigger_size: int = 3,
    position: str = "br",
) -> None:

    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError("poison_rate must be in [0, 1]")

    labels = base_trainset.targets
    labels_np = labels.cpu().numpy() if hasattr(labels, "cpu") else np.asarray(labels)
    candidates = np.asarray(client0_indices, dtype=np.int64)
    candidates = candidates[labels_np[candidates] != int(target_label)]
    selected_count = int(round(poison_rate * len(candidates)))
    if selected_count == 0:
        print("[Poison] no sample selected.")
        return

    selected = np.random.choice(candidates, size=selected_count, replace=False)
    images = base_trainset.data
    height, width = images.shape[-2:]
    ts = min(int(trigger_size), height, width)

    row_slice = slice(0, ts) if "t" in position else slice(height - ts, height)
    col_slice = slice(0, ts) if "l" in position else slice(width - ts, width)
    images[selected, row_slice, col_slice] = 255

    if isinstance(base_trainset.targets, list):
        for idx in selected.tolist():
            base_trainset.targets[idx] = int(target_label)
    else:
        base_trainset.targets[selected] = int(target_label)

    print(
        f"[Poison] injected client0 samples={selected_count}/{len(client0_indices)} "
        f"-> target={target_label}"
    )


def weighted_average_states(state_dicts, sample_counts):
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")
    total = float(sum(sample_counts))
    if total <= 0:
        raise ValueError("sample_counts must have a positive sum")

    output = {}
    for key in state_dicts[0].keys():
        value = None
        for state, count in zip(state_dicts, sample_counts):
            term = state[key] * (float(count) / total)
            value = term.clone() if value is None else value + term
        output[key] = value
    return output


def evaluate_state(state, clean_loader, asr_loader):
    model = FLNet()
    model.load_state_dict(state, strict=True)
    clean = Utils.evaluate(clean_loader, model)
    asr = Utils.evaluate(asr_loader, model)
    return clean, asr


def run_retrain_baseline(
    *,
    init_state,
    client_loaders,
    retain_ids,
    clean_loader,
    asr_loader,
    rounds: int,
    local_epochs: int,
    lr: float,
    weight_decay: float,
):

    state = {key: value.clone() for key, value in init_state.items()}
    print("\n=== Retrain(-client0): evaluation baseline only ===")

    for round_idx in range(1, rounds + 1):
        local_states = []
        sample_counts = []
        for client_id in retain_ids:
            local_state, _, _, sample_count = local_train_and_get_prototypes(
                client_loaders[client_id],
                client_id=client_id,
                local_epochs=local_epochs,
                lr=lr,
                weight_decay=weight_decay,
                init_state=state,
                compute_prototypes=False,
            )
            local_states.append(local_state)
            sample_counts.append(sample_count)

        state = weighted_average_states(local_states, sample_counts)
        clean, asr = evaluate_state(state, clean_loader, asr_loader)
        print(
            f"[Retrain {round_idx}/{rounds}] Clean={clean:.2f} | ASR={asr:.2f}"
        )

    return state


def main() -> None:
    set_seed(42)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_set = MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = MNIST(root="./data", train=False, download=True, transform=transform)

    num_clients = 5
    forget_id = 0
    target_label = 9
    subsets = split_dataset_by_count(train_set, num_clients=num_clients)

    offline_inject_client0_raw_dataset_mnist(
        train_set,
        list(subsets[forget_id].indices),
        target_label=target_label,
        poison_rate=0.80,
        trigger_size=3,
        position="br",
    )

    client_loaders = [
        DataLoader(
            subset,
            batch_size=128,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        for subset in subsets
    ]
    clean_test_loader = DataLoader(
        test_set,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
    )
    asr_strict_loader = make_asr_strict_loader(
        test_dataset=test_set,
        target_label=target_label,
        apply_trigger_fn=add_trigger_to_tensor,
        batch_size=1000,
        num_workers=0,
    )

    initial_model = FLNet()
    init_state = deepcopy(initial_model.state_dict())


    fed_rounds = 10
    local_epochs = 2
    local_lr = 0.03
    weight_decay = 5e-4

    global_state = deepcopy(init_state)
    last_local_states = None
    print("\n=== 1. Iterative FedAvg baseline ===")

    for round_idx in range(1, fed_rounds + 1):
        round_states = []
        round_counts = []
        for client_id in range(num_clients):
            local_state, _, _, sample_count = local_train_and_get_prototypes(
                client_loaders[client_id],
                client_id=client_id,
                local_epochs=local_epochs,
                lr=local_lr,
                weight_decay=weight_decay,
                init_state=global_state,
                compute_prototypes=False,
            )
            round_states.append(local_state)
            round_counts.append(sample_count)

        global_state = weighted_average_states(round_states, round_counts)
        last_local_states = round_states
        clean, asr = evaluate_state(
            global_state,
            clean_test_loader,
            asr_strict_loader,
        )
        print(
            f"[FedAvg {round_idx}/{fed_rounds}] Clean={clean:.2f} | ASR={asr:.2f}"
        )

    if last_local_states is None:
        raise RuntimeError("FedAvg did not produce local states")

    baseline_state = deepcopy(global_state)
    baseline_clean, baseline_asr = evaluate_state(
        baseline_state,
        clean_test_loader,
        asr_strict_loader,
    )
    print(
        f"[Baseline] Clean={baseline_clean:.2f} | ASR={baseline_asr:.2f}"
    )


    print("\n=== 2. Unified-reference prototype upload ===")
    client_states = []
    for client_id in range(num_clients):
        g_proto, h_proto, sample_count = get_prototypes_at_state(
            client_loaders[client_id],
            baseline_state,
        )
        client_states.append(
            {
                "model_state": last_local_states[client_id],
                "g_proto": g_proto,
                "h_proto": h_proto,
                "sample_count": sample_count,
                "eff_bs": sample_count,
            }
        )
        print(
            f"[Prototype client={client_id}] samples={sample_count} | "
            f"g_norm={g_proto.norm().item():.6e} | h_mean={h_proto.mean().item():.6e}"
        )


    print("\n=== 3. Bounded second-order unlearning ===")
    unlearned_state = unlearn_with_J(
        agg_state=baseline_state,
        client_states=client_states,
        forget_id=forget_id,
        alpha=1.0,
        beta=2.0,
        eta_f=0.08,
        eta_r=0.03,
        rho=1.0,
        steps_forget=30,
        steps_retain=5,
        damping=5e-3,
        target_curvature_weight=0.05,
        curvature_gate_strength=0.15,
        anchor_mu=0.10,
        retain_mu=0.05,
        proj_radius=None,
        max_step_norm=0.02,
        retain_forget_ratio=0.85,
        eval_clean_loader=clean_test_loader,
        eval_backdoor_loader=asr_strict_loader,
        log_every=1,
    )
    unlearn_clean, unlearn_asr = evaluate_state(
        unlearned_state,
        clean_test_loader,
        asr_strict_loader,
    )
    print(f"[Unlearn] Clean={unlearn_clean:.2f} | ASR={unlearn_asr:.2f}")


    print("\n=== 4. Retain-proxy post-training (no Retrain anchor) ===")
    final_state = posttrain_first_order(
        unlearned_state=unlearned_state,
        reference_state=baseline_state,
        client_states=client_states,
        rounds=5,
        lr=0.01,
        exclude_id=forget_id,
        anchor_mu=0.10,
        damping=5e-3,
        max_step_norm=0.02,
        eval_clean_loader=clean_test_loader,
        eval_backdoor_loader=asr_strict_loader,
    )
    final_clean, final_asr = evaluate_state(
        final_state,
        clean_test_loader,
        asr_strict_loader,
    )
    print(f"[Final] Clean={final_clean:.2f} | ASR={final_asr:.2f}")


    retrain_state = run_retrain_baseline(
        init_state=init_state,
        client_loaders=client_loaders,
        retain_ids=[i for i in range(num_clients) if i != forget_id],
        clean_loader=clean_test_loader,
        asr_loader=asr_strict_loader,
        rounds=fed_rounds,
        local_epochs=local_epochs,
        lr=local_lr,
        weight_decay=weight_decay,
    )
    retrain_clean, retrain_asr = evaluate_state(
        retrain_state,
        clean_test_loader,
        asr_strict_loader,
    )
    print(
        f"[Retrain baseline] Clean={retrain_clean:.2f} | ASR={retrain_asr:.2f}"
    )

    os.makedirs("runs", exist_ok=True)
    torch.save(baseline_state, "runs/baseline.pth")
    torch.save(unlearned_state, "runs/unlearned.pth")
    torch.save(final_state, "runs/final_post.pth")
    torch.save(retrain_state, "runs/retrain_minus_client0.pth")

    metrics = {
        "baseline": {"clean": baseline_clean, "asr": baseline_asr},
        "unlearn": {"clean": unlearn_clean, "asr": unlearn_asr},
        "final": {"clean": final_clean, "asr": final_asr},
        "retrain_baseline": {"clean": retrain_clean, "asr": retrain_asr},
    }
    with open("runs/metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    print("Saved models and metrics to runs/.")


if __name__ == "__main__":
    cst = timezone(timedelta(hours=8), name="CST")
    started = time.perf_counter()
    print(datetime.now(cst).strftime("[Start] %Y-%m-%d %H:%M:%S %Z"))
    main()
    elapsed = time.perf_counter() - started
    print(datetime.now(cst).strftime("[End]   %Y-%m-%d %H:%M:%S %Z"))
    print(f"[Duration] {elapsed / 60.0:.2f} minutes")
