import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.train import Hparams
from training.utils import build_reference_model, get_completion_token_logprobs


logger = logging.getLogger(__name__)


@dataclass
class FIPOHparams(Hparams):
    num_rollouts: int = 8
    max_prompt_length: int = 2048
    max_response_length: int = 8192
    temperature: float = 1.0
    clip_epsilon: float = 0.2
    kl_coef: float = 0.01
    advantage_epsilon: float = 1e-6
    # FIPO-specific hyperparameters
    tau: float = 32.0               # half-life for future-KL decay
    future_kl_clip_low: float = 0.0  # lower clip for influence weight (paper: 1-eps_flow)
    future_kl_clip_high: float = 1.2  # upper clip for influence weight
    safety_threshold: float = 10.0   # dual-clip safety threshold c
    normalize_future_kl: bool = False  # normalize influence weights to mean 1 per sequence
    brevity_scaling: bool = False       # scale correct-sequence advantages by mean_len / seq_len


def compute_future_kl(
    new_token_logprobs: torch.Tensor,
    old_token_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    tau: float,
    safety_threshold: float,
) -> torch.Tensor:
    """Compute discounted future-KL divergence at each position.

    FutureKL_t = sum_{k=t}^{T} M_k * gamma^{k-t} * delta_log_p_k

    where delta_log_p_k = log pi_theta(o_k) - log pi_old(o_k),
    M_k = 1 if ratio <= c (safety mask),
    gamma = 2^{-1/tau} (decay factor).
    """
    mask = token_mask.to(new_token_logprobs.dtype)
    delta_logp = (new_token_logprobs - old_token_logprobs) * mask  # (B, T)

    # Safety mask: mask out positions where the ratio is too large
    ratios = torch.exp(delta_logp)
    safety_mask = (ratios.abs() <= safety_threshold).to(delta_logp.dtype) * mask

    masked_delta = delta_logp * safety_mask  # (B, T)

    gamma = 2.0 ** (-1.0 / tau) if tau > 0 else 0.0
    B, T = masked_delta.shape

    # Compute future-KL via reverse cumulative sum with exponential decay.
    # FutureKL_t = delta_t + gamma * FutureKL_{t+1}
    future_kl = torch.zeros_like(masked_delta)
    future_kl[:, T - 1] = masked_delta[:, T - 1]
    for t in range(T - 2, -1, -1):
        future_kl[:, t] = masked_delta[:, t] + gamma * future_kl[:, t + 1]

    return future_kl


def compute_influence_weights(
    future_kl: torch.Tensor,
    token_mask: torch.Tensor,
    clip_low: float,
    clip_high: float,
    normalize: bool,
) -> torch.Tensor:
    """Compute per-token influence weights from future-KL values.

    f_t = clip(exp(FutureKL_t), clip_low, clip_high)

    If normalize=True, applies position-level normalization:
    w_t = f_t / sum_{k=1}^{T} f_k
    scaled by T to preserve magnitude.
    """
    mask = token_mask.to(future_kl.dtype)
    f = torch.exp(future_kl)
    f = torch.clamp(f, clip_low, clip_high)

    if normalize:
        # Normalize to mean 1 per sequence: redistributes credit within
        # each sequence without changing its overall gradient magnitude
        f_mean = (f * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        f = f / f_mean.clamp(min=1e-8)

    return f * mask


def compute_fipo_loss(
    new_token_logprobs: torch.Tensor,
    old_token_logprobs: torch.Tensor,
    ref_token_logprobs: torch.Tensor,
    token_mask: torch.Tensor,
    advantages: torch.Tensor,
    rewards: torch.Tensor,
    influence_weights: torch.Tensor,
    clip_epsilon: float,
    kl_coef: float,
    brevity_scaling: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """FIPO loss: PPO-clip with dense future-KL-reweighted advantages.

    Uses per-sequence normalization so each sequence contributes equally
    to the gradient. Optionally scales correct-sequence advantages by
    mean_seqlen / seq_len to prefer concise correct solutions.
    """
    mask = token_mask.to(new_token_logprobs.dtype)
    B = mask.shape[0]
    seq_lengths = mask.sum(dim=1)  # (B,)
    num_sequences = max(B, 1)

    ratios = torch.exp(new_token_logprobs - old_token_logprobs)

    # Brevity scaling: for correct sequences, scale advantage by
    # mean_seqlen / seq_len. Incorrect sequences are untouched.
    scaled_advantages = advantages.clone()
    if brevity_scaling:
        mean_seqlen = seq_lengths.mean()
        brevity_scale = mean_seqlen / seq_lengths.clamp(min=1.0)  # (B,)
        correct_mask = (rewards > 0).to(scaled_advantages.dtype)
        # Apply scaling only to correct sequences
        scaled_advantages = scaled_advantages * (
            correct_mask * brevity_scale + (1.0 - correct_mask)
        )

    # Dense advantage: per-sequence advantage * per-token influence weight
    dense_advantages = scaled_advantages.unsqueeze(1) * influence_weights  # (B, T)

    unclipped = ratios * dense_advantages
    clipped = torch.clamp(ratios, 1.0 - clip_epsilon,
                          1.0 + clip_epsilon) * dense_advantages

    # Per-sequence normalization: compute loss per sequence, then average
    # across sequences so each contributes equally regardless of length
    per_token_loss = -torch.minimum(unclipped, clipped) * mask
    per_seq_loss = per_token_loss.sum(dim=1) / seq_lengths.clamp(min=1.0)  # (B,)
    policy_loss = per_seq_loss.mean()

    per_token_kl = (new_token_logprobs - ref_token_logprobs) * mask
    per_seq_kl = per_token_kl.sum(dim=1) / seq_lengths.clamp(min=1.0)
    kl = per_seq_kl.mean()

    loss = policy_loss + (kl_coef * kl)

    clip_fraction = ((ratios > (1.0 + clip_epsilon)) |
                     (ratios < (1.0 - clip_epsilon))).to(mask.dtype)
    total_tokens = mask.sum().clamp(min=1.0)
    clip_fraction = (clip_fraction * mask).sum() / total_tokens

    metrics = {
        "policy_loss": policy_loss.item(),
        "kl": kl.item(),
        "clip_fraction": clip_fraction.item(),
        "ratio_mean": ((ratios * mask).sum() / total_tokens).item(),
        "influence_weight_mean": ((influence_weights * mask).sum() / total_tokens).item(),
        "influence_weight_std": (influence_weights[token_mask].std().item()
                                 if token_mask.any() else 0.0),
    }
    if brevity_scaling:
        metrics["brevity_scale_mean"] = (
            (mean_seqlen / seq_lengths.clamp(min=1.0))[rewards > 0].mean().item()
            if (rewards > 0).any() else 1.0
        )
    return loss, metrics


def forward(
    accelerator,
    model: AutoModelForCausalLM,
    batch: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    hparams: FIPOHparams,
    reference_model: AutoModelForCausalLM,
    rollout_fn: Callable,
    get_feedback_fn: Callable,
    build_student_messages_fn: Callable,
) -> Optional[Dict[str, Any]]:
    batch_data: List[Dict[str, Any]] = []
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()

    for example in batch:
        rollouts = rollout_fn(
            unwrapped_model,
            tokenizer,
            example,
            num_rollouts=hparams.num_rollouts,
            temperature=hparams.temperature,
            max_new_tokens=hparams.max_response_length,
        )

        rewards: List[float] = []
        for rollout in rollouts:
            feedback = get_feedback_fn(rollout["completion"], example)
            all_passed = bool(feedback.metadata.get(
                "all_passed", feedback.success)) if feedback.metadata else bool(feedback.success)
            rewards.append(1.0 if all_passed else 0.0)

        reward_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=accelerator.device)
        reward_mean = reward_tensor.mean()
        reward_std = reward_tensor.std(unbiased=False)
        advantages = (reward_tensor - reward_mean) / (
            reward_std + hparams.advantage_epsilon
        )

        for rollout, reward, advantage in zip(
            rollouts, rewards, advantages.tolist()
        ):
            batch_data.append({
                "example": example,
                "completion": rollout["completion"],
                "reward": float(reward),
                "advantage": float(advantage),
            })

    if not batch_data:
        return None

    messages = [
        build_student_messages_fn(data["example"], data["completion"])
        for data in batch_data
    ]
    max_seq_length = hparams.max_prompt_length + hparams.max_response_length

    model.train()
    new_token_logprobs, token_mask = get_completion_token_logprobs(
        model, tokenizer, messages,
        max_seq_length=max_seq_length, requires_grad=True,
    )
    old_token_logprobs = new_token_logprobs.detach()
    ref_token_logprobs, _ = get_completion_token_logprobs(
        reference_model, tokenizer, messages,
        max_seq_length=max_seq_length, requires_grad=False,
    )

    # Compute future-KL and influence weights
    future_kl = compute_future_kl(
        new_token_logprobs=new_token_logprobs.detach(),
        old_token_logprobs=old_token_logprobs,
        token_mask=token_mask,
        tau=hparams.tau,
        safety_threshold=hparams.safety_threshold,
    )
    influence_weights = compute_influence_weights(
        future_kl=future_kl,
        token_mask=token_mask,
        clip_low=hparams.future_kl_clip_low,
        clip_high=hparams.future_kl_clip_high,
        normalize=hparams.normalize_future_kl,
    )

    advantages = torch.tensor(
        [data["advantage"] for data in batch_data],
        dtype=new_token_logprobs.dtype,
        device=new_token_logprobs.device,
    )
    rewards = torch.tensor(
        [data["reward"] for data in batch_data],
        dtype=new_token_logprobs.dtype,
        device=new_token_logprobs.device,
    )

    return {
        "new_token_logprobs": new_token_logprobs,
        "old_token_logprobs": old_token_logprobs,
        "ref_token_logprobs": ref_token_logprobs,
        "token_mask": token_mask,
        "advantages": advantages,
        "rewards": rewards,
        "influence_weights": influence_weights,
        "future_kl": future_kl,
    }


def make_forward_backward_fn(
    rollout_fn: Callable,
    get_feedback_fn: Callable,
    build_student_messages_fn: Callable,
) -> Callable:
    def forward_backward(
        accelerator,
        model: AutoModelForCausalLM,
        batch: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        hparams: FIPOHparams,
        auxiliary_model: AutoModelForCausalLM,
    ):
        forward_outputs = forward(
            accelerator, model, batch, tokenizer, hparams,
            auxiliary_model, rollout_fn, get_feedback_fn,
            build_student_messages_fn,
        )
        if forward_outputs is None:
            return None, {}

        loss, metrics = compute_fipo_loss(
            new_token_logprobs=forward_outputs["new_token_logprobs"],
            old_token_logprobs=forward_outputs["old_token_logprobs"],
            ref_token_logprobs=forward_outputs["ref_token_logprobs"],
            token_mask=forward_outputs["token_mask"],
            advantages=forward_outputs["advantages"],
            rewards=forward_outputs["rewards"],
            influence_weights=forward_outputs["influence_weights"],
            clip_epsilon=hparams.clip_epsilon,
            kl_coef=hparams.kl_coef,
            brevity_scaling=hparams.brevity_scaling,
        )
        mask = forward_outputs["token_mask"].to(forward_outputs["future_kl"].dtype)
        token_count = mask.sum().clamp(min=1.0)
        metrics.update({
            "reward_mean": forward_outputs["rewards"].mean().item(),
            "reward_std": forward_outputs["rewards"].std(unbiased=False).item(),
            "advantage_mean": forward_outputs["advantages"].mean().item(),
            "advantage_std": forward_outputs["advantages"].std(unbiased=False).item(),
            "completion_tokens": forward_outputs["token_mask"].sum().item(),
            "future_kl_mean": ((forward_outputs["future_kl"] * mask).sum() / token_count).item(),
        })
        return loss, metrics

    return forward_backward
