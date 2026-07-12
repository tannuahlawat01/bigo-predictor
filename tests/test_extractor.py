"""
test_extractor.py

Unit tests for the AST feature extractor, verifying it correctly
identifies known complexity patterns.

Run with: pytest tests/test_extractor.py -v
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from features.ast_extractor import extract_features


def test_constant_time():
    code = "def f(arr):\n    return arr[0]"
    features = extract_features(code)
    assert features["max_loop_depth"] == 0
    assert features["has_recursion"] is False


def test_linear_time_range():
    code = "def f(n):\n    for i in range(n):\n        print(i)"
    features = extract_features(code)
    assert features["max_loop_depth"] == 1
    assert features["has_input_dependent_loop_bound"] is True


def test_linear_time_direct_iteration():
    code = "def f(arr):\n    for x in arr:\n        print(x)"
    features = extract_features(code)
    assert features["max_loop_depth"] == 1
    assert features["has_input_dependent_loop_bound"] is True


def test_constant_bound_loop_not_input_dependent():
    code = "def f():\n    for i in range(10):\n        print(i)"
    features = extract_features(code)
    assert features["has_input_dependent_loop_bound"] is False


def test_quadratic_nested_loops():
    code = "def f(arr):\n    for i in arr:\n        for j in arr:\n            print(i, j)"
    features = extract_features(code)
    assert features["max_loop_depth"] == 2
    assert features["num_loops"] == 2


def test_recursion_detected():
    code = "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)"
    features = extract_features(code)
    assert features["has_recursion"] is True
    assert features["recursion_branch_factor"] == 2


def test_sort_call_detected():
    code = "def f(arr):\n    return sorted(arr)"
    features = extract_features(code)
    assert features["has_sort_call"] is True


def test_hash_structure_detected():
    code = "def f(arr):\n    seen = set()\n    for x in arr:\n        seen.add(x)\n    return seen"
    features = extract_features(code)
    assert features["has_hash_structure"] is True


def test_invalid_syntax_returns_none():
    code = "def f(:\n    this is not valid python"
    features = extract_features(code)
    assert features is None