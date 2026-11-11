"""Fail-closed executable verifier, mirroring glm53_flash/evaluator.py.

Reads a batch of requests as JSON on stdin and writes a batch of results as
JSON on stdout, so one interpreter start serves many candidates.

Request:  {"family": str, "entry_point": str, "prompt": str,
           "completion": str, "run_tests": bool}
Response: {"status": "passed"|"failed"|"invalid"|"skipped",
           "passed": bool, "tests_passed": int, "tests_total": int,
           "pass_fraction": float, "message": str, "parseable": bool}

The tests deliberately match the reference: same DENIED node list, same
SAFE_CALLS allowlist, same restricted builtins, same 2048-byte cap, and the
same strict `type(actual) is type(expected)` comparison.
"""
from __future__ import annotations

import ast
import json
import sys

MAX_SOURCE_BYTES = 2048

DENIED = (
    ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom, ast.With,
    ast.AsyncWith, ast.Lambda, ast.Global, ast.Nonlocal, ast.While, ast.For,
    ast.AsyncFor, ast.Try, ast.Raise, ast.Delete,
)
SAFE_CALLS = {"sum", "len", "abs", "min", "max", "bool", "int", "str"}

# Test cases per family, copied from the reference's tasks.py.
CASES = {
    "increment":   [((-3,), -2), ((0,), 1), ((7,), 8)],
    "double":      [((-4,), -8), ((0,), 0), ((6,), 12)],
    "square":      [((-4,), 16), ((0,), 0), ((5,), 25)],
    "absolute":    [((-7,), 7), ((0,), 0), ((9,), 9)],
    "nonnegative": [((-5,), 0), ((0,), 0), ((8,), 8)],
    "even":        [((-3,), False), ((0,), True), ((8,), True)],
    "reverse":     [(("",), ""), (("abc",), "cba"), (("level",), "level")],
    "list_sum":    [(([],), 0), (([1, 2, 3],), 6), (([-2, 5],), 3)],
}


def _safe_builtins():
    import builtins

    return {name: getattr(builtins, name) for name in SAFE_CALLS}


def validate_source(source: str, entry_point: str):
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("source too large")
    tree = ast.parse(source)
    functions = []
    for top in tree.body:
        if (
            isinstance(top, ast.Expr)
            and isinstance(top.value, ast.Constant)
            and isinstance(top.value.value, str)
        ):
            continue
        if isinstance(top, ast.FunctionDef):
            functions.append(top)
            continue
        raise ValueError("top level may contain only the requested function and comments")
    if len(functions) != 1 or functions[0].name != entry_point:
        raise ValueError("required function missing or extra function present")
    for node in ast.walk(tree):
        if isinstance(node, DENIED):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder name disallowed")
        if isinstance(node, ast.Attribute):
            raise ValueError("attribute access disallowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_CALLS:
                raise ValueError("unsafe call")
            if node.func.id == entry_point:
                raise ValueError("recursion disallowed")
    return tree


def evaluate(family: str, entry_point: str, source: str) -> dict:
    cases = CASES[family]
    total = len(cases)
    try:
        tree = validate_source(source, entry_point)
        namespace = {"__builtins__": _safe_builtins()}
        exec(compile(tree, "candidate.py", "exec"), namespace)
        function = namespace[entry_point]
    except (SyntaxError, ValueError, TypeError, KeyError) as error:
        return {
            "status": "invalid",
            "passed": False,
            "tests_passed": 0,
            "tests_total": total,
            "pass_fraction": 0.0,
            "message": str(error),
        }

    passed = 0
    for arguments, expected in cases:
        try:
            actual = function(*arguments)
            passed += int(type(actual) is type(expected) and actual == expected)
        except Exception:
            pass

    return {
        "status": "passed" if passed == total else "failed",
        "passed": passed == total,
        "tests_passed": passed,
        "tests_total": total,
        "pass_fraction": passed / total,
        "message": "",
    }


def is_parseable(prompt: str, completion: str, entry_point: str) -> bool:
    try:
        validate_source(prompt + completion, entry_point)
        return completion.endswith("\n")
    except (SyntaxError, ValueError):
        return False


def handle(request: dict) -> dict:
    family = request["family"]
    entry_point = request["entry_point"]
    prompt = request["prompt"]
    completion = request["completion"]

    parseable = is_parseable(prompt, completion, entry_point)

    if not request.get("run_tests", True):
        total = len(CASES.get(family, ()))
        result = {
            "status": "skipped",
            "passed": False,
            "tests_passed": 0,
            "tests_total": total,
            "pass_fraction": 0.0,
            "message": "",
        }
    elif family not in CASES:
        result = {
            "status": "invalid",
            "passed": False,
            "tests_passed": 0,
            "tests_total": 0,
            "pass_fraction": 0.0,
            "message": f"unknown family: {family}",
        }
    else:
        result = evaluate(family, entry_point, prompt + completion)

    result["parseable"] = parseable
    return result


def main() -> int:
    payload = json.load(sys.stdin)
    results = [handle(request) for request in payload["requests"]]
    json.dump({"results": results}, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
