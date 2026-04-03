import re
from fractions import Fraction
from typing import Any, Dict

ANSWER_LINE_RE = re.compile(r"(?im)^\s*Answer:\s*(.+?)\s*$")
INTEGER_RE = re.compile(r"^[+-]?\d[\d,]*$")
FRACTION_RE = re.compile(r"^[+-]?\d[\d,]*/[+-]?\d[\d,]*$")
DECIMAL_RE = re.compile(r"^[+-]?(?:\d[\d,]*\.\d+|\.\d+)$")


def _find_matching_brace(text: str, start: int) -> int:
    depth = 1
    index = start

    while index < len(text) and depth > 0:
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
        index += 1

    return index - 1 if depth == 0 else -1


def _extract_boxed_answer(text: str) -> str | None:
    boxed_start = text.rfind("\\boxed{")
    if boxed_start == -1:
        return None

    content_start = boxed_start + len("\\boxed{")
    closing_brace = _find_matching_brace(text, content_start)
    if closing_brace == -1:
        return None

    return text[content_start:closing_brace]


def _strip_wrappers(text: str) -> str:
    candidate = text.strip()

    while candidate:
        updated = candidate
        boxed = _extract_boxed_answer(updated)
        if boxed is not None:
            updated = boxed.strip()

        if updated.startswith("$") and updated.endswith("$") and len(updated) >= 2:
            updated = updated[1:-1].strip()
        elif updated.startswith("\\(") and updated.endswith("\\)") and len(updated) >= 4:
            updated = updated[2:-2].strip()
        elif updated.startswith("{") and updated.endswith("}") and len(updated) >= 2:
            updated = updated[1:-1].strip()

        if updated == candidate:
            break
        candidate = updated

    return candidate


def extract_math_answer(text: str) -> str:
    matches = list(ANSWER_LINE_RE.finditer(text))
    if matches:
        return matches[-1].group(1).strip()

    boxed = _extract_boxed_answer(text)
    if boxed is not None:
        return boxed.strip()

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def normalize_math_answer(text: str) -> str:
    candidate = _strip_wrappers(text)
    compact = re.sub(r"\s+", "", candidate).rstrip(".")

    if not compact:
        return ""

    if INTEGER_RE.fullmatch(compact):
        return str(int(compact.replace(",", "")))

    if FRACTION_RE.fullmatch(compact):
        numerator, denominator = compact.split("/", 1)
        denominator_int = int(denominator.replace(",", ""))
        if denominator_int == 0:
            return compact

        fraction = Fraction(int(numerator.replace(",", "")), denominator_int)
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"

    if DECIMAL_RE.fullmatch(compact):
        fraction = Fraction(compact.replace(",", ""))
        if fraction.denominator == 1:
            return str(fraction.numerator)
        return f"{fraction.numerator}/{fraction.denominator}"

    return re.sub(r"\s+", " ", candidate).strip().rstrip(".")


def verify_answer(attempt: str, ground_truth_dict: Dict[str, Any] | str) -> bool:
    extracted_attempt = normalize_math_answer(extract_math_answer(attempt))
    if isinstance(ground_truth_dict, dict):
        ground_truth = ground_truth_dict["ground_truth"]
    else:
        ground_truth = ground_truth_dict

    return extracted_attempt == normalize_math_answer(str(ground_truth))
    
