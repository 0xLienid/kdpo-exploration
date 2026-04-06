import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import torch
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable, Optional
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from validators import Validator
from training.utils import get_world_size, get_grad_norm


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


@dataclass
class Hparams:
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    num_epochs: int = 1
    max_steps_per_epoch: Optional[int] = None
    minibatch_size: int = 1
    gradient_accumulation_steps: int = 32
    num_rollouts: int = 8

    log_interval: int = 10
    validation_interval: int = 10


@dataclass
class StepContext:
    accelerator: Accelerator
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    optimizer: torch.optim.Optimizer
    hparams: Hparams
    batch: Any
    epoch: int
    global_step: int
    auxiliary_model: Optional[Any] = None


@dataclass
class ValidatorRunConfig:
    batch_size: int = 4
    max_new_tokens: int = 2048
    max_seq_length: int = 2048


def validate(
    accelerator: Accelerator,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    validators: List[Tuple[Validator, ValidatorRunConfig]],
):
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.eval()

    original_padding_side = tokenizer.padding_side
    original_pad_token = tokenizer.pad_token
    original_pad_token_id = tokenizer.pad_token_id

    validator_results = {}

    for validator, val_config in validators:
        try:
            if accelerator.is_main_process:
                logger.info(f"Validating with {validator.name}...")

            score = validator.validate(
                model=unwrapped_model,
                tokenizer=tokenizer,
                batch_size=val_config.batch_size,
                max_new_tokens=val_config.max_new_tokens,
                max_seq_length=val_config.max_seq_length,
                accelerator=accelerator,
            )
            validator_results[validator.name] = score
            if accelerator.is_main_process:
                logger.info(f"Validation score for {validator.name}: {score:.4f}")
        except Exception as e:
            logger.error(f"Error validating with {validator.name}: {e}")
            validator_results[validator.name] = 0.0
        finally:
            tokenizer.padding_side = original_padding_side
            tokenizer.pad_token = original_pad_token
            tokenizer.pad_token_id = original_pad_token_id
            unwrapped_model.train()

    return validator_results


def train(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    hparams: Hparams,
    collate_fn: Callable,
    forward_backward_fn: Callable,
    validators: List[Tuple[Validator, ValidatorRunConfig]],
    on_optimizer_step_fn: Optional[Callable] = None,
    auxiliary_model: Optional[Any] = None,
    output_dir: str = "outputs/train",
    wandb_project: str = "kdpo",
    wandb_run_name: str = None,
):
    output_dir = f"{output_dir}/{wandb_run_name}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    world_size = get_world_size()
    adjusted_gradient_accumulation_steps = max(
        1, hparams.gradient_accumulation_steps // world_size)

    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=adjusted_gradient_accumulation_steps,
        adjust_scheduler=True
    )
    accelerator = Accelerator(
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        log_with="wandb" if wandb_project is not None else None,
    )

    if accelerator.is_main_process and wandb_project:
        accelerator.init_trackers(
            project_name=wandb_project,
            config=hparams.__dict__,
            init_kwargs={"wandb": {"name": wandb_run_name}
                         } if wandb_run_name else None,
        )

    dataloader = DataLoader(
        dataset, batch_size=hparams.minibatch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        weight_decay=hparams.weight_decay
    )

    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader)

    if auxiliary_model is not None:
        auxiliary_model.to(accelerator.device)

    model.train()

    global_step = 0
    validation_history = defaultdict(list)

    if accelerator.is_main_process:
        logger.info("Beginning training...")
        logger.info(f"World size: {world_size}")

        for key, value in hparams.__dict__.items():
            logger.info(f"{key}: {value}")


    if accelerator.is_main_process:
        logger.info(f"Running initial validation...")
    validator_results = validate(accelerator, model, tokenizer, validators)
    if accelerator.is_main_process:
        for validator, score in validator_results.items():
            validation_history[validator].append({
                "step": global_step,
                "score": score
            })
            accelerator.log(
                {f"val/{validator}": score}, step=global_step)

    for epoch in range(hparams.num_epochs):
        for step, batch in enumerate(dataloader):
            if hparams.max_steps_per_epoch is not None and global_step >= hparams.max_steps_per_epoch:
                break

            if accelerator.is_main_process:
                logger.info(
                    f"Processing batch {step + 1} of {len(dataloader)}...")

            with accelerator.accumulate(model):

                loss, metrics = forward_backward_fn(
                    accelerator, model, batch, tokenizer, hparams, auxiliary_model)
                if loss is None:
                    continue

                accelerator.backward(loss)

                grad_norm = None
                performed_optimizer_step = False
                if accelerator.sync_gradients:
                    if hparams.max_grad_norm > 0:
                        grad_norm = accelerator.clip_grad_norm_(
                            model.parameters(), hparams.max_grad_norm)
                        if isinstance(grad_norm, torch.Tensor):
                            grad_norm = grad_norm.item()
                    else:
                        grad_norm = get_grad_norm(model)

                    optimizer.step()

                    if on_optimizer_step_fn:
                        on_optimizer_step_fn(StepContext(
                            accelerator=accelerator,
                            model=model,
                            tokenizer=tokenizer,
                            optimizer=optimizer,
                            hparams=hparams,
                            batch=batch,
                            epoch=epoch,
                            global_step=global_step,
                            auxiliary_model=auxiliary_model
                        ))

                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    performed_optimizer_step = True

                    accelerator.wait_for_everyone()

                if accelerator.is_main_process and (performed_optimizer_step and global_step % hparams.log_interval == 0):
                    logger.info(
                        f"Step {global_step} - Loss: {loss.item():.4f}")

                    log_dict = {
                        "train/loss": loss.item(),
                        "train/global_step": global_step,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    }
                    if grad_norm is not None:
                        log_dict["train/grad_norm"] = grad_norm

                    for key, value in metrics.items():
                        log_dict[f"train/{key}"] = value

                    accelerator.log(log_dict, step=global_step)

                if performed_optimizer_step and global_step % hparams.validation_interval == 0:
                    validator_results = validate(accelerator, model, tokenizer, validators)
                    if accelerator.is_main_process:
                        for validator, score in validator_results.items():
                            validation_history[validator].append({
                                "step": global_step,
                                "score": score
                            })
                            accelerator.log(
                                {f"val/{validator}": score}, step=global_step)

    if accelerator.is_main_process:
        logger.info("Running final validation...")
    validator_results = validate(accelerator, model, tokenizer, validators)
    if accelerator.is_main_process:
        for validator, score in validator_results.items():
            validation_history[validator].append({
                "step": global_step,
                "score": score
            })
            accelerator.log(
                {f"val/{validator}": score}, step=global_step)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")
        logger.info("Training complete!")

        if wandb_project:
            accelerator.end_training()

    return {
        "validation_history": validation_history,
    }
