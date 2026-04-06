import copy
from typing import Any, Dict, List

from data.feedback import noop_feedback
from data.dapo_math import get_ground_truth
from training.train import ValidatorRunConfig
from validators import DAPOMathValidator, AIME2025Validator, FineWebValidator


def _render_problem(example: Dict[str, Any]) -> str:
    rendered_messages = []
    for message in example["prompt"]:
        role = message["role"].upper()
        rendered_messages.append(f"## {role}\n{message['content']}")
    return "\n\n".join(rendered_messages)


def build_student_messages(
    example: Dict[str, Any],
    completion: str,
) -> List[Dict[str, Any]]:
    messages = copy.deepcopy(example["prompt"])
    messages.append({"role": "assistant", "content": completion})
    return messages


def build_teacher_messages(
    example: Dict[str, Any],
    student_attempt: str,
    feedback_text: str,
    completion: str,
) -> List[Dict[str, Any]]:
    problem = _render_problem(example)
    correct_answer = get_ground_truth(example["reward_model"])
    return [
        {
            "role": "user",
            "content": (
                "You are helping correct a student's solution to a math "
                "problem.\n\n"
                f"## Original Problem\n{problem}\n\n"
                f"## Previous Attempt\n{student_attempt}\n\n"
                f"## Verification Feedback\n{feedback_text}\n\n"
                f"## Correct Answer\n{correct_answer}\n\n"
                "Correctly solve the original problem. Keep the reasoning "
                "concise and end with a clear final answer."
            ),
        },
        {"role": "assistant", "content": completion},
    ]


def build_gold_teacher_messages(
    example: Dict[str, Any],
    _student_attempt: str,
    _feedback_text: str,
    completion: str,
) -> List[Dict[str, Any]]:
    problem = _render_problem(example)
    return [
        {
            "role": "user",
            "content": (
                "You are solving a math problem with access to a gold "
                "standard reference solution.\n\n"
                f"## Original Problem\n{problem}\n\n"
                "## Gold Standard Solution\n"
                f"{example['gold_standard']}\n\n"
                "Use the gold standard solution as reference context while "
                "solving the original problem. Keep the reasoning concise and "
                "end with a clear final answer."
            ),
        },
        {"role": "assistant", "content": completion},
    ]


def build_insight_prompt(
    example: Dict[str, Any],
    student_attempt: str,
    feedback: Any,
) -> List[Dict[str, Any]]:
    problem = _render_problem(example)
    correct_answer = get_ground_truth(example["reward_model"])
    return [
        {
            "role": "user",
            "content": (
                "You are a tutor reviewing a student's attempt at a math "
                "problem. You know the correct answer, but your job is to "
                "help the student reason better without writing out the full "
                "correct solution.\n\n"
                f"## Original Problem\n{problem}\n\n"
                f"## Student's Attempt\n{student_attempt}\n\n"
                "## Verification Feedback\n"
                f"{feedback.feedback_text}\n\n"
                f"## Correct Answer\n{correct_answer}\n\n"
                "Deeply reflect on the student's attempt and the correct "
                "answer to identify where the student likely went wrong and "
                "what would help them reason better. Produce insights in the "
                "following categories:\n"
                "- STRATEGY: What general approach or framework applies to "
                "this problem? What are helpful reasoning approaches to "
                "follow?\n"
                "- PITFALLS: What does the attempt reveal about common "
                "mistakes or the student's mistakes in problems like this?\n"
                "- DIAGNOSIS: What conceptual error likely produced the "
                "observed failure? Do not state the full fix -- identify the "
                "flawed reasoning pattern.\n"
                "- STRENGTHS: What did the student do well that should be "
                "reinforced?\n\n"
                "Do not write the full corrected derivation or final "
                "solution. Your insights should help the student develop "
                "better reasoning habits, not simply copy the right answer."
            ),
        }
    ]


def build_insight_teacher_messages(
    example: Dict[str, Any],
    insight: str,
    completion: str,
) -> List[Dict[str, Any]]:
    problem = _render_problem(example)
    return [
        {
            "role": "user",
            "content": (
                "Solve the following math problem. Keep the reasoning concise "
                "and end with a clear final answer.\n\n"
                f"## Original Problem\n{problem}\n\n"
                "Here are some insights about how to reason through this "
                f"problem:\n{insight}"
            ),
        },
        {"role": "assistant", "content": completion},
    ]


def make_validators() -> list[tuple[Any, ValidatorRunConfig]]:
    return [
        (
            DAPOMathValidator(),
            ValidatorRunConfig(
                batch_size=8,
                max_new_tokens=2048,
                max_seq_length=2048,
            ),
        ),
        (
            AIME2025Validator(),
            ValidatorRunConfig(
                batch_size=8,
                max_new_tokens=2048,
                max_seq_length=2048,
            ),
        ),
        (
            FineWebValidator(),
            ValidatorRunConfig(
                batch_size=8,
                max_new_tokens=0,
                max_seq_length=1024,
            ),
        ),
    ]


__all__ = [
    "build_student_messages",
    "build_teacher_messages",
    "build_gold_teacher_messages",
    "build_insight_prompt",
    "build_insight_teacher_messages",
    "make_validators",
    "noop_feedback",
]
