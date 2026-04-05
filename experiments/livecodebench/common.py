from typing import Any, Dict, List

from data.feedback import noop_feedback
from data.livecodebench import format_question
from training.train import ValidatorRunConfig
from validators import FineWebValidator, LiveCodeBenchValidator


def build_student_messages(
    example: Dict[str, Any],
    completion: str,
) -> List[Dict[str, Any]]:
    question = format_question(example)
    return [
        {
            "role": "user",
            "content": (
                "Answer the following question, please keep your reasoning "
                "concise, and put your code in a ```python{code}``` block:\n\n"
                f"{question}"
            ),
        },
        {"role": "assistant", "content": completion},
    ]


def build_teacher_messages(
    example: Dict[str, Any],
    student_attempt: str,
    feedback_text: str,
    completion: str,
) -> List[Dict[str, Any]]:
    question = format_question(example)
    return [
        {
            "role": "user",
            "content": (
                f"## Question\n{question}\n\n"
                f"## Previous Attempt\n{student_attempt}\n\n"
                "## Feedback (from environment) for the previous attempt\n"
                f"{feedback_text}\n\n"
                "Correctly solve the original question."
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
    question = format_question(example)
    return [
        {
            "role": "user",
            "content": (
                f"## Question\n{question}\n\n"
                "## Gold Standard Solution\n"
                f"{example['gold_standard']}\n\n"
                "Use the gold standard solution as reference context while "
                "solving the original question."
            ),
        },
        {"role": "assistant", "content": completion},
    ]


def build_insight_prompt(
    example: Dict[str, Any],
    student_attempt: str,
    feedback: Any,
) -> List[Dict[str, Any]]:
    question = format_question(example)
    return [
        {
            "role": "user",
            "content": (
                "You are a tutor reviewing a student's attempt at a problem "
                "alongside execution feedback from the environment. Your job "
                "is to help the student reason better without telling them "
                "the answer.\n\n"
                f"## Question\n{question}\n\n"
                f"## Student's Attempt\n{student_attempt}\n\n"
                "## Feedback (from environment) for the student's attempt\n"
                f"{feedback.feedback_text}\n\n"
                "Deeply reflect on the student's attempt and the feedback to "
                "identify if the student went wrong, and if so how. You are "
                "trying to extract as much information as possible from the "
                "attempt and feedback. Produce insights in the following "
                "categories:\n"
                "- STRATEGY: What general approach or framework applies to "
                "this problem? What are helpful reasoning approaches to "
                "follow?\n"
                "- PITFALLS: What does the feedback reveal about common "
                "mistakes or the student's mistakes in problems like this?\n"
                "- DIAGNOSIS: What conceptual error likely produced the "
                "observed failure (if any)? Do not state the fix -- identify "
                "the flawed reasoning pattern.\n"
                "- STRENGTHS: What did the student do well that should be "
                "reinforced?\n\n"
                "Do not state the corrected code or solution. Your insights "
                "should help the student develop better reasoning habits, not "
                "solve this specific instance."
            ),
        }
    ]


def build_insight_teacher_messages(
    example: Dict[str, Any],
    insight: str,
    completion: str,
) -> List[Dict[str, Any]]:
    question = format_question(example)
    return [
        {
            "role": "user",
            "content": (
                "Answer the following question, please keep your reasoning "
                "concise, and put your code in a ```python{code}``` block.\n\n"
                "Here are some insights about how to reason through this "
                f"problem:\n{insight}\n\n"
                f"Question:\n{question}"
            ),
        },
        {"role": "assistant", "content": completion},
    ]


def make_validators() -> list[tuple[Any, ValidatorRunConfig]]:
    return [
        (
            LiveCodeBenchValidator(),
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
