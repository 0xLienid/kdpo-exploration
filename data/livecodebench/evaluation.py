import os
import re
import subprocess
import tempfile
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


BASE_IMPORTS = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\nimport heapq\nimport bisect\n"
    "import copy\nimport math\nimport random\nimport statistics\nimport itertools\nimport functools\n"
    "import operator\nimport io\nimport sys\nimport json\n"
    "sys.setrecursionlimit(50000)\n"
)


CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool
    error_message: Optional[str] = None


@dataclass
class TestCaseResult:
    test_input: str
    expected_output: str
    actual_output: str
    passed: bool
    error_message: Optional[str] = None
    timed_out: bool = False


@dataclass
class FeedbackResult:
    feedback_text: str
    success: bool
    metadata: Optional[Dict[str, Any]] = None


def extract_python_code(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)

    if match:
        return match.group(1).strip()

    return text.strip()


def get_stripped_lines(val: str) -> List[str]:
    val = val.strip()
    return [line.strip() for line in val.split("\n")]


def compare_outputs(actual: str, expected: str) -> bool:
    actual_lines = get_stripped_lines(actual)
    expected_lines = get_stripped_lines(expected)

    if len(actual_lines) != len(expected_lines):
        return False

    for actual_line, expected_line in zip(actual_lines, expected_lines):
        if actual_line != expected_line:
            return False

    return True


def format_test_results(test_results: List[TestCaseResult], truncate_length: int = 500) -> str:
    if not test_results:
        return "No test cases were executed."

    lines = []
    passed_count = sum(1 for r in test_results if r.passed)
    total_count = len(test_results)

    lines.append(f"Test Results: {passed_count}/{total_count} passed")
    lines.append("")

    for i, result in enumerate(test_results):
        lines.append(f"Test {i + 1}:")
        lines.append(
            f"\tInput: {result.test_input[:truncate_length]}{'...' if len(result.test_input) > truncate_length else ''}")
        lines.append(
            f"\tExpected: {result.expected_output[:truncate_length]}{'...' if len(result.expected_output) > truncate_length else ''}")
        lines.append(
            f"\tActual: {result.actual_output[:truncate_length]}{'...' if len(result.actual_output) > truncate_length else ''}")

        if result.passed:
            lines.append("\tStatus: PASSED")
        else:
            lines.append("\tStatus: FAILED")

            if result.error_message:
                lines.append(
                    f"\tError: {result.error_message[:truncate_length]}{'...' if len(result.error_message) > truncate_length else ''}")

        lines.append("")

    return "\n".join(lines)


def run_code(
    code: str,
    stdin_input: str = "",
    timeout_seconds: int = 10,
    include_base_imports: bool = True
) -> ExecutionResult:
    full_code = f"{BASE_IMPORTS}\n{code}" if include_base_imports else code

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as temp_file:
        temp_file.write(full_code)
        temp_file_path = temp_file.name

    try:
        result = subprocess.run(
            ["python", temp_file_path],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )

        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.returncode,
            timed_out=False,
            error_message=result.stderr if result.returncode != 0 else None,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            stdout="",
            stderr="",
            return_code=-1,
            timed_out=True,
            error_message=f"Execution timed out after {timeout_seconds} seconds"
        )
    except Exception as e:
        return ExecutionResult(
            stdout="",
            stderr=str(e),
            return_code=-1,
            timed_out=False,
            error_message=str(e)
        )
    finally:
        try:
            os.remove(temp_file_path)
        except OSError:
            pass


def run_test_case(
    code: str,
    test_case: Dict[str, str],
    timeout_seconds: int = 10
) -> TestCaseResult:
    test_type = test_case["testtype"]

    if test_type != "stdin":
        return TestCaseResult(
            test_input="",
            expected_output="",
            actual_output="",
            passed=False,
            error_message=f"Unsupported test type: {test_type}",
        )

    test_input = test_case["input"]
    expected_output = str(test_case["output"])

    execution_result = run_code(
        code, stdin_input=test_input, timeout_seconds=timeout_seconds)

    if execution_result.timed_out:
        return TestCaseResult(
            test_input=test_input,
            expected_output=expected_output,
            actual_output="",
            passed=False,
            error_message=execution_result.error_message,
            timed_out=True,
        )

    if execution_result.return_code != 0:
        return TestCaseResult(
            test_input=test_input,
            expected_output=expected_output,
            actual_output=execution_result.stdout,
            passed=False,
            error_message=execution_result.stderr
        )

    passed = compare_outputs(execution_result.stdout, expected_output)

    return TestCaseResult(
        test_input=test_input,
        expected_output=expected_output,
        actual_output=execution_result.stdout,
        passed=passed,
        error_message=execution_result.stderr if not passed else None,
    )


def run_test_cases(
    code: str,
    test_cases: List[Dict[str, str]],
    timeout_seconds: int = 10,
    stop_on_first_failure: bool = False
) -> Tuple[bool, List[TestCaseResult]]:
    results = []
    all_passed = True

    for test_case in test_cases:
        result = run_test_case(code, test_case, timeout_seconds)
        results.append(result)

        if not result.passed:
            all_passed = False

            if stop_on_first_failure:
                break

    return all_passed, results


def get_environment_feedback(
    completion: str,
    example: Dict[str, Any],
    timeout_seconds: int = 10
) -> FeedbackResult:
    code = extract_python_code(completion)
    test_cases = example["public_test_cases"]

    if not test_cases:
        return FeedbackResult(
            feedback_text="No test cases available for this problem.",
            success=False,
            metadata={"test_results": [], "all_passed": False}
        )

    all_passed, test_results = run_test_cases(
        code=code,
        test_cases=test_cases,
        timeout_seconds=timeout_seconds,
        stop_on_first_failure=False
    )
    feedback_text = format_test_results(test_results)

    return FeedbackResult(
        feedback_text=feedback_text,
        success=all_passed,
        metadata={
            "test_results": [
                {
                    "input": r.test_input,
                    "expected": r.expected_output,
                    "actual": r.actual_output,
                    "passed": r.passed,
                    "error": r.error_message,
                }
                for r in test_results
            ],
            "all_passed": all_passed,
            "passed_count": sum(1 for r in test_results if r.passed),
            "total_count": len(test_results),
        }
    )
