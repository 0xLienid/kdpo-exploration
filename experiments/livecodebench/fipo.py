import argparse

from data.livecodebench import (
    LiveCodeBenchDataset,
    collate_fn as lcb_collate_fn,
    get_environment_feedback as lcb_get_feedback,
    rollout as lcb_rollout,
)
from experiments.common import (
    add_common_training_args,
    configure_logging,
    load_model_and_tokenizer,
    seed_everything,
)
from experiments.livecodebench.common import build_student_messages, make_validators
from training.fipo import FIPOHparams, make_forward_backward_fn
from training.train import train as run_train
from training.utils import build_reference_model


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_training_args(parser, output_dir_default="outputs/livecodebench/fipo")
    parser.add_argument("--num-rollouts", type=int, default=8)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--advantage-epsilon", type=float, default=1e-6)
    parser.add_argument("--tau", type=float, default=32.0)
    parser.add_argument("--future-kl-clip-low", type=float, default=0.0)
    parser.add_argument("--future-kl-clip-high", type=float, default=1.2)
    parser.add_argument("--safety-threshold", type=float, default=10.0)
    parser.add_argument("--normalize-future-kl", action="store_true")
    parser.add_argument("--brevity-scaling", action="store_true")
    args = parser.parse_args()

    configure_logging()
    seed_everything(args.seed)

    hparams = FIPOHparams(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        max_steps_per_epoch=args.max_steps_per_epoch,
        minibatch_size=args.minibatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_rollouts=args.num_rollouts,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        temperature=args.temperature,
        clip_epsilon=args.clip_epsilon,
        kl_coef=args.kl_coef,
        advantage_epsilon=args.advantage_epsilon,
        tau=args.tau,
        future_kl_clip_low=args.future_kl_clip_low,
        future_kl_clip_high=args.future_kl_clip_high,
        safety_threshold=args.safety_threshold,
        normalize_future_kl=args.normalize_future_kl,
        brevity_scaling=args.brevity_scaling,
        log_interval=args.log_interval,
        validation_interval=args.validation_interval,
    )

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    reference_model = build_reference_model(model)
    forward_backward_fn = make_forward_backward_fn(
        lcb_rollout,
        lcb_get_feedback,
        build_student_messages,
    )

    run_train(
        model=model,
        tokenizer=tokenizer,
        dataset=LiveCodeBenchDataset(),
        hparams=hparams,
        collate_fn=lcb_collate_fn,
        forward_backward_fn=forward_backward_fn,
        validators=make_validators(),
        auxiliary_model=reference_model,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
