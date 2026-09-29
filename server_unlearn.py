from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from model import FLNet
from utils import Utils


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _vec_model_helpers():

    template = FLNet()

    def to_vec(state_dict: dict) -> torch.Tensor:
        flat = [state_dict[name].reshape(-1) for name in template.state_dict().keys()]
        return torch.cat(flat, dim=0).to(DEVICE)

    def to_sd(vector: torch.Tensor) -> dict:
        return Utils.vec_to_sd(vector.detach().cpu(), template)

    return template, to_vec, to_sd


def _aggregate_clients_states(
    sd_list: List[dict],
    poison_weight: float = 1.0,
) -> dict:


    if not sd_list:
        raise ValueError("sd_list must not be empty")
    if poison_weight <= 0:
        raise ValueError("poison_weight must be positive")

    weights = [float(poison_weight)] + [1.0] * (len(sd_list) - 1)
    total = sum(weights)
    output = deepcopy(sd_list[0])
    for key in output:
        output[key] = sum(w * sd[key] for w, sd in zip(weights, sd_list)) / total
    return output


def _sample_count(client_state: Dict) -> float:
    value = client_state.get("sample_count", client_state.get("eff_bs", 1))
    value = float(value)
    if value <= 0:
        raise ValueError("client sample_count/eff_bs must be positive")
    return value


def _weighted_average_states(client_states: Sequence[Dict], ids: Sequence[int]) -> dict:
    if not ids:
        raise ValueError("ids must not be empty")
    weights = [_sample_count(client_states[i]) for i in ids]
    total = float(sum(weights))
    output = deepcopy(client_states[ids[0]]["model_state"])
    for key in output:
        output[key] = sum(
            weight * client_states[i]["model_state"][key]
            for i, weight in zip(ids, weights)
        ) / total
    return output


def _weighted_mean_proto(
    client_states: Sequence[Dict],
    ids: Sequence[int],
    key: str,
) -> torch.Tensor:
    if key not in {"g_proto", "h_proto"}:
        raise ValueError(f"unsupported prototype key: {key}")
    if not ids:
        raise ValueError("ids must not be empty")

    weights = [_sample_count(client_states[i]) for i in ids]
    total = float(sum(weights))
    result = None
    for i, weight in zip(ids, weights):
        value = client_states[i][key]
        if value is None:
            raise ValueError(f"client {i} has no {key}")
        value = value.to(DEVICE)
        term = value * (weight / total)
        result = term.clone() if result is None else result + term
    return result


def _aggregate_unlearning_inputs(
    client_states: Sequence[Dict],
    forget_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, dict]:

    num_clients = len(client_states)
    if num_clients < 2:
        raise ValueError("at least two clients are required")
    if not 0 <= forget_id < num_clients:
        raise IndexError("forget_id is out of range")

    all_ids = list(range(num_clients))
    retain_ids = [i for i in all_ids if i != forget_id]
    total_count = sum(_sample_count(client_states[i]) for i in all_ids)
    forget_count = _sample_count(client_states[forget_id])
    p_forget = forget_count / total_count
    if not 0.0 < p_forget < 1.0:
        raise ValueError("invalid target-client aggregation weight")

    g_forget = client_states[forget_id]["g_proto"].to(DEVICE).clone()
    h_forget = client_states[forget_id]["h_proto"].to(DEVICE).clone()
    g_retain = _weighted_mean_proto(client_states, retain_ids, "g_proto")
    h_retain = _weighted_mean_proto(client_states, retain_ids, "h_proto")
    retain_state = _weighted_average_states(client_states, retain_ids)

    h_forget = torch.clamp(h_forget, min=1e-12)
    h_retain = torch.clamp(h_retain, min=1e-12)
    return g_forget, h_forget, g_retain, h_retain, p_forget, retain_state


def _clip_step(step: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = torch.norm(step)
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    if norm > max_norm:
        step = step * (max_norm / (norm + 1e-12))
    return step


def _evaluate_vector(
    vector: torch.Tensor,
    to_sd,
    clean_loader=None,
    backdoor_loader=None,
):
    if clean_loader is None and backdoor_loader is None:
        return None, None
    model = FLNet().to(DEVICE)
    model.load_state_dict(to_sd(vector), strict=True)
    clean = Utils.evaluate(clean_loader, model) if clean_loader is not None else None
    backdoor = Utils.evaluate(backdoor_loader, model) if backdoor_loader is not None else None
    return clean, backdoor


def unlearn_with_J(
    agg_state: dict,
    client_states: List[Dict],
    forget_id: int = 0,
    *,
    alpha: float = 1.0,
    beta: float = 2.0,
    eta_f: float = 0.08,
    eta_r: float = 0.03,
    rho: float = 1.0,
    steps_forget: int = 30,
    steps_retain: int = 5,
    damping: float = 5e-3,
    target_curvature_weight: float = 0.05,
    curvature_gate_strength: float = 0.15,
    anchor_mu: float = 0.10,
    retain_mu: float = 0.05,
    proj_radius: Optional[float] = None,
    max_step_norm: float = 0.02,
    retain_forget_ratio: float = 0.85,
    tolerance: float = 1e-7,
    eval_clean_loader=None,
    eval_backdoor_loader=None,
    log_every: int = 1,
) -> dict:


    if alpha < 0 or beta <= 0:
        raise ValueError("alpha must be non-negative and beta must be positive")
    if eta_f <= 0 or eta_r <= 0:
        raise ValueError("eta_f and eta_r must be positive")
    if damping <= 0:
        raise ValueError("damping must be positive")
    if not 0.0 <= curvature_gate_strength <= 1.0:
        raise ValueError("curvature_gate_strength must be in [0, 1]")
    if not 0.0 < retain_forget_ratio <= 1.0:
        raise ValueError("retain_forget_ratio must be in (0, 1]")

    _, to_vec, to_sd = _vec_model_helpers()
    w_global = to_vec(agg_state).clone()
    w = w_global.clone()

    g_f, h_f, g_r, h_r, p_f, retain_state = _aggregate_unlearning_inputs(
        client_states,
        forget_id,
    )
    w_retain = to_vec(retain_state).clone()

    if any(v.numel() != w.numel() for v in (g_f, h_f, g_r, h_r)):
        raise ValueError("prototype dimension does not match model parameter dimension")


    delete_scale = p_f / (1.0 - p_f)


    sensitivity = h_f / (h_f + h_r + 1e-12)
    sensitivity = torch.clamp(sensitivity, min=0.05, max=1.0)
    gate = (1.0 - curvature_gate_strength) + curvature_gate_strength * sensitivity
    g_f_effective = gate * g_f

    g_f_effective = gate * g_f

    h_effective = (
        beta * h_r
        + target_curvature_weight * h_f
        + float(damping)
    )
    h_effective = torch.clamp(h_effective, min=damping, max=1e4)

    base_distance = torch.norm(w_global - w_retain).item()
    rms_global = float(w_global.pow(2).mean().sqrt().item() + 1e-12)
    radius_floor = 1e-3 * rms_global * (w_global.numel() ** 0.5)
    if proj_radius is None:
        proj_radius = max(1.10 * base_distance, radius_floor)
    if proj_radius <= 0:
        raise ValueError("proj_radius must be positive")

    def objective(vector: torch.Tensor):
        delta = vector - w_global
        q_forget_linear = delete_scale * Utils.dot(g_f_effective, delta)
        q_retain = Utils.dot(g_r, delta) + 0.5 * Utils.dot(h_r * delta, delta)
        target_damping = 0.5 * target_curvature_weight * Utils.dot(h_f * delta, delta)
        anchor_penalty = 0.5 * anchor_mu * torch.sum((vector - w_retain) ** 2)
        value = alpha * q_forget_linear - beta * q_retain - target_damping - anchor_penalty
        return value, q_forget_linear, q_retain

    j_current, forget_score, retain_proxy = objective(w)
    clean, backdoor = _evaluate_vector(
        w,
        to_sd,
        eval_clean_loader,
        eval_backdoor_loader,
    )

    if log_every:
        print("[Server] Second-order unlearning starts (Retrain is not used).")
        print(
            f"[Server] p_forget={p_f:.6f} | delete_scale={delete_scale:.6f} | "
            f"||w_global-w_retain||={base_distance:.6e} | radius={proj_radius:.6e}"
        )


    objective_history = []
    objective_patience = 5
    objective_min_gain = 1e-3

    for step_idx in range(steps_forget):
        delta = w - w_global
        retain_grad = g_r + h_r * delta


        grad_j = (
            alpha * delete_scale * g_f_effective
            - beta * retain_grad
            - target_curvature_weight * h_f * delta
            - anchor_mu * (w - w_retain)
        )

        grad_norm = float(torch.norm(grad_j).item())


        denominator = (
            rho * h_effective
            + damping
        )
        base_step = (
            eta_f
            * grad_j
            /
            denominator
        )
        base_step = _clip_step(
            base_step,
            max_step_norm
        )

        accepted = False
        used_step = 0.0
        for factor in (1.0, 0.8, 0.6, 0.5, 0.33, 0.25, 0.125, 0.0625):
            candidate = Utils.l2_project(
                w + factor * base_step,
                w_retain,
                proj_radius,
            )
            j_trial, forget_trial, retain_trial = objective(candidate)
            if j_trial >= j_current - 1e-10:
                used_step = float(torch.norm(candidate - w).item())
                w = candidate
                j_current = j_trial
                forget_score = forget_trial
                retain_proxy = retain_trial
                accepted = True
                break

        if not accepted:
            eta_f *= 0.5
            if eta_f < 1e-6:
                if log_every:
                    print(f"[Forget] stop at step {step_idx + 1}: line search failed.")
                break

        objective_history.append(float(j_current))

        if len(objective_history) > objective_patience:
            recent_gain = (
                objective_history[-1]
                -
                objective_history[-objective_patience]
            )
            if recent_gain < objective_min_gain:
                if log_every:
                    print("[Forget] objective saturated, stop early.")
                break

        if log_every and (step_idx % log_every == 0 or step_idx == steps_forget - 1):
            clean, backdoor = _evaluate_vector(
                w,
                to_sd,
                eval_clean_loader,
                eval_backdoor_loader,
            )
            metric_text = ""
            if clean is not None:
                metric_text += f" | Clean={clean:.2f}"
            if backdoor is not None:
                metric_text += f" | ASR={backdoor:.2f}"
            print(
                f"[Forget {step_idx + 1}/{steps_forget}] "
                f"J={float(j_current):.6e} | Fscore={float(forget_score):.6e} | "
                f"Rproxy={float(retain_proxy):.6e} | grad={grad_norm:.6e} | "
                f"step={used_step:.6e}{metric_text}"
            )

        if used_step <= tolerance or grad_norm <= tolerance:
            if log_every:
                print(f"[Forget] converged at step {step_idx + 1}.")
            break

    forget_score_after = float(forget_score.detach().item())
    if forget_score_after > 0:
        minimum_forget_score = retain_forget_ratio * forget_score_after
    else:
        minimum_forget_score = forget_score_after


    def retain_objective(vector: torch.Tensor):
        delta = vector - w_global
        q_r = Utils.dot(g_r, delta) + 0.5 * Utils.dot(h_r * delta, delta)
        anchor = 0.5 * retain_mu * torch.sum((vector - w_retain) ** 2)
        return q_r + anchor

    retain_current = retain_objective(w)
    retain_radius = max(proj_radius, 1.25 * torch.norm(w - w_retain).item())

    for step_idx in range(steps_retain):
        delta = w - w_global
        grad_r = g_r + h_r * delta + retain_mu * (w - w_retain)
        denominator = rho * (h_r + retain_mu + damping)
        base_step = -eta_r * grad_r / denominator
        base_step = _clip_step(base_step, 0.5 * max_step_norm)

        accepted = False
        used_step = 0.0
        for factor in (1.0, 0.8, 0.6, 0.5, 0.33, 0.25, 0.125, 0.0625):
            candidate = Utils.l2_project(
                w + factor * base_step,
                w_retain,
                retain_radius,
            )
            retain_trial = retain_objective(candidate)
            delta_trial = candidate - w_global
            forget_trial = delete_scale * Utils.dot(g_f_effective, delta_trial)
            preserve_forgetting = float(forget_trial) >= minimum_forget_score - 1e-10
            if retain_trial <= retain_current + 1e-10 and preserve_forgetting:
                used_step = float(torch.norm(candidate - w).item())
                w = candidate
                retain_current = retain_trial
                accepted = True
                break

        if not accepted:
            eta_r *= 0.5
            if eta_r < 1e-6:
                break

        if log_every and (step_idx % log_every == 0 or step_idx == steps_retain - 1):
            clean, backdoor = _evaluate_vector(
                w,
                to_sd,
                eval_clean_loader,
                eval_backdoor_loader,
            )
            metric_text = ""
            if clean is not None:
                metric_text += f" | Clean={clean:.2f}"
            if backdoor is not None:
                metric_text += f" | ASR={backdoor:.2f}"
            print(
                f"[Retain {step_idx + 1}/{steps_retain}] "
                f"Robj={float(retain_current):.6e} | step={used_step:.6e}{metric_text}"
            )

        if used_step <= tolerance:
            break

    return to_sd(w)


def posttrain_first_order(
    unlearned_state: dict,
    reference_state: dict,
    client_states: List[Dict],
    *,
    rounds: int = 5,
    lr: float = 0.05,
    exclude_id: int = 0,
    anchor_mu: float = 0.05,
    damping: float = 1e-3,
    max_step_norm: float = 0.02,
    forget_score_floor: float = 0.9,
    eval_clean_loader=None,
    eval_backdoor_loader=None,
) -> dict:


    if rounds <= 0:
        return deepcopy(unlearned_state)

    _, to_vec, to_sd = _vec_model_helpers()
    w_reference = to_vec(reference_state)
    w = to_vec(unlearned_state)

    g_f, _, g_r, h_r, p_f, retain_state = _aggregate_unlearning_inputs(
        client_states,
        exclude_id,
    )
    w_retain = to_vec(retain_state)

    radius = max(
        1.25 * torch.norm(w - w_retain).item(),
        1.05 * torch.norm(w_reference - w_retain).item(),
        1e-8,
    )

    def proxy_loss(vector: torch.Tensor):
        delta = vector - w_reference
        return (
            Utils.dot(g_r, delta)
            + 0.5 * Utils.dot(h_r * delta, delta)
            + 0.5 * anchor_mu * torch.sum((vector - w_retain) ** 2)
        )

    current_loss = proxy_loss(w)


    delete_scale = p_f / (1.0 - p_f)
    initial_forget_score = delete_scale * Utils.dot(g_f, w - w_reference)
    minimum_forget_score = forget_score_floor * initial_forget_score

    for round_idx in range(rounds):
        delta = w - w_reference
        aligned_grad = g_r + h_r * delta
        grad = aligned_grad + anchor_mu * (w - w_retain)
        step = -lr * grad / (h_r + anchor_mu + damping)
        step = _clip_step(step, max_step_norm)

        accepted = False
        used_step = 0.0
        for factor in (1.0, 0.8, 0.6, 0.5, 0.33, 0.25, 0.125, 0.0625):
            candidate = Utils.l2_project(w + factor * step, w_retain, radius)
            candidate_loss = proxy_loss(candidate)
            candidate_forget_score = delete_scale * Utils.dot(g_f, candidate - w_reference)
            if (
                candidate_loss <= current_loss + 1e-10
                and candidate_forget_score >= minimum_forget_score
            ):
                used_step = float(torch.norm(candidate - w).item())
                w = candidate
                current_loss = candidate_loss
                accepted = True
                break

        if not accepted:
            lr *= 0.5

        clean, backdoor = _evaluate_vector(
            w,
            to_sd,
            eval_clean_loader,
            eval_backdoor_loader,
        )
        metric_text = ""
        if clean is not None:
            metric_text += f" | Clean={clean:.2f}"
        if backdoor is not None:
            metric_text += f" | ASR={backdoor:.2f}"
        print(
            f"[Posttrain {round_idx + 1}/{rounds}] "
            f"Rproxy={float(current_loss):.6e} | step={used_step:.6e}{metric_text}"
        )

        if used_step <= 1e-8:
            break

    return to_sd(w)
