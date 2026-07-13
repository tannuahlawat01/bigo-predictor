"""
ast_extractor.py
Extracts structural features from Python source code using the built-in
`ast` module, for use as input to a time-complexity classifier.
Core idea: we don't treat code as text. We parse it into a tree and look
at structural patterns (loop nesting, recursion, etc.) that correlate
with time complexity.
"""
import ast
class ComplexityFeatureExtractor(ast.NodeVisitor):
    """
    Walks a Python AST and collects features relevant to time complexity.

    Usage:
        extractor = ComplexityFeatureExtractor()
        extractor.visit(tree)
        features = extractor.get_features()
    """

    def __init__(self):
        self.max_loop_depth = 0
        self.current_loop_depth = 0
        self.num_loops = 0

        self.function_names = set()      # names of all functions defined
        self.recursive_functions = set() # functions that call themselves
        self.current_function_stack = [] # tracks which function we're inside

        self.has_sort_call = False
        self.has_hash_structure = False
        self.has_input_dependent_loop_bound = False
        self.has_halving_pattern = False
        self.recursive_call_counts = []  # calls-per-invocation, for branch factor

    # ---- Loops ----

    def visit_For(self, node):
        self._enter_loop(node)
        self.generic_visit(node)
        self._exit_loop()

    def visit_While(self, node):
        self._enter_loop(node)
        self._check_while_loop_bound(node)
        self.generic_visit(node)
        self._exit_loop()

    def _check_while_loop_bound(self, node):
        """
        Heuristic for while-loop bound detection: only trust an explicit
        len() call in the condition (e.g. `while i < len(arr)`), since
        this is an unambiguous signal the loop scales with input size.

        Earlier version also flagged any two-variable comparison (e.g.
        `while lo < hi`) as input-dependent, but this fired too broadly
        in practice (e.g. unrelated counter comparisons) and diluted the
        feature's signal, measurably hurting model accuracy. Deliberately
        narrowed to len()-only after that regression was caught via
        testing. Two-pointer patterns without an explicit len() call
        (e.g. `while lo < hi` alone) are a known remaining gap.
        """
        for subnode in ast.walk(node.test):
            if isinstance(subnode, ast.Call):
                func_name = self._get_call_name(subnode)
                if func_name == "len":
                    self.has_input_dependent_loop_bound = True
                    return
                    
    def _enter_loop(self, node):
        self.num_loops += 1
        self.current_loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.current_loop_depth)

        if isinstance(node, ast.For):
            self._check_for_loop_bound(node)
        # Note: While loops are not analyzed for bound type here - their
        # termination condition can be arbitrarily complex (e.g. binary
        # search halving a range), which is a known limitation documented
        # in the README rather than guessed at here.

    def _check_for_loop_bound(self, node):
        """
        Determine whether a `for` loop's bound depends on input size.

        Two patterns are checked:
          1. range(...) calls - e.g. range(len(arr)), range(n) -> input-dependent
             vs. range(10) -> constant
          2. Direct iteration over a collection - e.g. `for x in arr` - this
             is ALWAYS input-dependent, since the loop runs once per element
             of whatever `arr` is, and its length isn't known statically.
        """
        if isinstance(node.iter, ast.Call) and getattr(node.iter.func, "id", "") == "range":
            for arg in node.iter.args:
                # a constant number means fixed bound; anything else
                # (a variable, a len() call, etc.) we treat as input-dependent
                if not isinstance(arg, ast.Constant):
                    self.has_input_dependent_loop_bound = True

        elif isinstance(node.iter, (ast.Name, ast.Attribute, ast.Subscript)):
            # `for x in arr`, `for x in self.data`, `for x in arr[1:]`
            # all iterate over a collection whose size isn't known at
            # parse time -> input-dependent.
            self.has_input_dependent_loop_bound = True

        elif isinstance(node.iter, ast.Call):
            # Common collection-producing calls: enumerate(arr), zip(a, b),
            # sorted(arr), reversed(arr), arr.items(), etc.
            func_name = ComplexityFeatureExtractor._get_call_name(node.iter)
            if func_name in ("enumerate", "zip", "sorted", "reversed", "items", "keys", "values"):
                self.has_input_dependent_loop_bound = True

    def _exit_loop(self):
        self.current_loop_depth -= 1

    # ---- Function definitions (to detect recursion) ----

    def visit_FunctionDef(self, node):
        self.function_names.add(node.name)
        self.current_function_stack.append(node.name)
        self.generic_visit(node)
        self.current_function_stack.pop()

    # ---- Function calls (sort, hash structures, recursion) ----

    def visit_Call(self, node):
        func_name = self._get_call_name(node)

        if func_name in ("sort", "sorted"):
            self.has_sort_call = True

        if func_name in ("dict", "set"):
            self.has_hash_structure = True

        # Recursion check: are we inside a function, and does this call
        # match the name of the function we're currently inside?
        if self.current_function_stack and func_name == self.current_function_stack[-1]:
            self.recursive_functions.add(func_name)

        self.generic_visit(node)

    def visit_Assign(self, node):
        self._check_halving_pattern(node.targets[0] if node.targets else None, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        # Catches `x //= 2` directly
        if isinstance(node.op, ast.FloorDiv) and self._is_constant_two(node.value):
            self.has_halving_pattern = True
        self.generic_visit(node)

    def _check_halving_pattern(self, target, value):
        """
        Detects patterns like:
            mid = (lo + hi) // 2
            x = x // 2
        which are strong signals of binary-search-style O(log n) behavior.
        """
        if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.FloorDiv):
            return
        if self._is_constant_two(value.right):
            self.has_halving_pattern = True

    @staticmethod
    def _is_constant_two(node):
        return isinstance(node, ast.Constant) and node.value == 2

    def visit_Dict(self, node):
        self.has_hash_structure = True
        self.generic_visit(node)

    def visit_Set(self, node):
        self.has_hash_structure = True
        self.generic_visit(node)

    @staticmethod
    def _get_call_name(node):
        """Extract the function name from a Call node, handling both
        direct calls (sorted(x)) and method calls (x.sort())."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    # ---- Recursion branch factor: count recursive calls per function ----

    def _count_recursive_calls(self, tree):
        """
        For each recursive function, count how many times it calls itself
        within its own body. 1 call = linear recursion, 2+ = branching
        (e.g. naive Fibonacci calls itself twice -> exponential-like).
        """
        branch_factors = {}

        class CallCounter(ast.NodeVisitor):
            def __init__(self, target_name):
                self.target_name = target_name
                self.count = 0

            def visit_Call(self, node):
                name = ComplexityFeatureExtractor._get_call_name(node)
                if name == self.target_name:
                    self.count += 1
                self.generic_visit(node)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in self.recursive_functions:
                counter = CallCounter(node.name)
                counter.visit(node)
                branch_factors[node.name] = counter.count

        return branch_factors

    # ---- Final feature output ----

    def get_features(self, tree) -> dict:
        branch_factors = self._count_recursive_calls(tree)
        max_branch_factor = max(branch_factors.values()) if branch_factors else 0

        return {
            "max_loop_depth": self.max_loop_depth,
            "num_loops": self.num_loops,
            "has_recursion": len(self.recursive_functions) > 0,
            "recursion_branch_factor": max_branch_factor,
            "has_sort_call": self.has_sort_call,
            "has_hash_structure": self.has_hash_structure,
            "has_input_dependent_loop_bound": self.has_input_dependent_loop_bound,
            "has_halving_pattern": self.has_halving_pattern,
        }


def extract_features(code: str) -> dict | None:
    """
    Main entry point. Parses source code and returns a feature dict.
    Returns None if the code doesn't parse (handles messy real-world
    submissions gracefully instead of crashing the whole pipeline).
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None

    extractor = ComplexityFeatureExtractor()
    extractor.visit(tree)
    return extractor.get_features(tree)


if __name__ == "__main__":
    # Quick manual sanity checks while building this file
    examples = {
        "O(1)": "def f(arr):\n    return arr[0]",
        "O(n)": "def f(arr):\n    for x in arr:\n        print(x)",
        "O(n^2)": "def f(arr):\n    for i in arr:\n        for j in arr:\n            print(i, j)",
        "O(log n) binary search": (
            "def f(arr, target):\n"
            "    lo, hi = 0, len(arr)\n"
            "    while lo < hi:\n"
            "        mid = (lo + hi) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            lo = mid + 1\n"
            "        else:\n"
            "            hi = mid\n"
            "    return -1"
        ),
        "O(2^n) naive fibonacci": (
            "def fib(n):\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    return fib(n - 1) + fib(n - 2)"
        ),
        "O(n log n) sort": "def f(arr):\n    return sorted(arr)",
    }

    for label, code in examples.items():
        features = extract_features(code)
        print(f"{label}:")
        print(f"  {features}")
        print()