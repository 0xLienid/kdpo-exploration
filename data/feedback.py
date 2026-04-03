from dataclasses import dataclass
from typing import Any


@dataclass
class FeedbackResult:
    feedback_text: str
    success: bool
    metadata: dict[str, Any] | None = None
