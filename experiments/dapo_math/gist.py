import argparse

from data.dapo_math import (
    DAPOMathDataset,
    collate_fn as dapo_collate_fn,
    get_environment_feedback as dapo_get_feedback,
    rollout as dapo_rollout,
)
from experiments.common import (
    add_common_training_args,
    configure_logging,
    load_model_and_tokenizer,
    seed_everything,
)
from experiments.dapo_math.common import (
    build_insight_prompt,
    build_insight_teacher_messages,
    build_student_messages,
    make_validators,
)
from training.gist import GISTHparams, make_forward_backward_fn, on_optimizer_step
from training.opsd.EMATeacher import EMATeacher
from training.train import train as run_train


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_training_args(parser, output_dir_default="outputs/dapo_math/gist")
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--teacher-alpha", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-train-examples", type=int, default=10_000)
    args = parser.parse_args()

    configure_logging()
    seed_everything(args.seed)

    hparams = GISTHparams(
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
        top_k=args.top_k,
        temperature=args.temperature,
        teacher_alpha=args.teacher_alpha,
        beta=args.beta,
        log_interval=args.log_interval,
        validation_interval=args.validation_interval,
    )

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    teacher = EMATeacher(model, alpha=hparams.teacher_alpha)
    forward_backward_fn = make_forward_backward_fn(
        dapo_rollout,
        dapo_get_feedback,
        build_student_messages,
        build_insight_prompt,
        build_insight_teacher_messages,
    )

    run_train(
        model=model,
        tokenizer=tokenizer,
        dataset=DAPOMathDataset(split="train", num_examples=args.num_train_examples),
        hparams=hparams,
        collate_fn=dapo_collate_fn,
        forward_backward_fn=forward_backward_fn,
        validators=make_validators(),
        on_optimizer_step_fn=on_optimizer_step,
        auxiliary_model=teacher,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
