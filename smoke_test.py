"""Smoke test: verify the full pipeline works end-to-end."""

import argparse
from aicomputing import VirtualRuntime, ExecuteRequest, Constraint

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3:4b")
parser.add_argument("--host", default=None, help="Ollama URL (default: local server). E.g. https://abc123.ngrok-free.app")
args = parser.parse_args()

vr = VirtualRuntime(backend="ollama", model=args.model, host=args.host)

# --- Test 1: Simple arithmetic ---
r = vr.execute(ExecuteRequest(
    signature="add(x: int, y: int) -> int",
    docstring="Return the sum of x and y.",
    inputs={"x": 2, "y": 3},
    return_type=int,
))
print(f"add(2, 3) = {r.result}  [expected: 5]  {'✓' if r.result == 5 else '✗'}")

# --- Test 2: String operation ---
r = vr.execute(ExecuteRequest(
    signature="reverse(s: str) -> str",
    docstring="Return the reversed string.",
    inputs={"s": "hello"},
    return_type=str,
))
print(f"reverse('hello') = {r.result!r}  [expected: 'olleh']  {'✓' if r.result == 'olleh' else '✗'}")

# --- Test 3: List sorting ---
r = vr.execute(ExecuteRequest(
    signature="sort(arr: list[int]) -> list[int]",
    docstring="Return a sorted copy of the array.",
    inputs={"arr": [5, 3, 1, 4, 2]},
    return_type=list,
))
print(f"sort([5,3,1,4,2]) = {r.result}  [expected: [1,2,3,4,5]]  {'✓' if r.result == [1,2,3,4,5] else '✗'}")

# --- Test 4: Algorithmic (Dijkstra) ---
r = vr.execute(ExecuteRequest(
    signature="dijkstra(graph, start_node: str) -> dict[str, int]",
    docstring="Return the shortest path distances from start_node to all other nodes. Graph is {node: [(neighbor, weight), ...]}.",
    inputs={
        "graph": {"A": [("B", 4), ("C", 2)], "B": [("A", 4), ("C", 5), ("D", 10)], "C": [("A", 2), ("B", 5), ("D", 3)], "D": [("B", 10), ("C", 3)]},
        "start_node": "A",
    },
    return_type=dict,
))
expected = {"A": 0, "B": 4, "C": 2, "D": 5}
print(f"dijkstra = {r.result}  [expected: {expected}]  {'✓' if r.result == expected else '✗'}")

# --- Test 5: NLP task (sentiment) ---
r = vr.execute(ExecuteRequest(
    signature="score_review(text: str) -> int",
    docstring="Return an integer sentiment score from 0 to 5 inclusive. 0 = terrible, 5 = excellent.",
    inputs={"text": "Absolutely loved this product, works perfectly!"},
    return_type=int,
    constraints=[Constraint(name="range", predicate=lambda x: 0 <= x <= 5, message="score must be 0-5")],
))
print(f"sentiment = {r.result}  [expected: 4 or 5]  {'✓' if r.result in (4, 5) else '~'}")

# --- Test 6: CoT strategy ---
r = vr.execute(ExecuteRequest(
    signature="gcd(a: int, b: int) -> int",
    docstring="Return the greatest common divisor of a and b.",
    inputs={"a": 48, "b": 18},
    return_type=int,
    strategy="cot",
))
print(f"gcd(48, 18) = {r.result}  [expected: 6]  [strategy: {r.strategy}]  {'✓' if r.result == 6 else '✗'}")

# --- Test 7: Hybrid strategy routing ---
r = vr.execute(ExecuteRequest(
    signature="fibonacci(n: int) -> int",
    docstring="Return the nth Fibonacci number (0-indexed).",
    inputs={"n": 10},
    return_type=int,
    strategy="hybrid",
))
print(f"fibonacci(10) = {r.result}  [expected: 55]  [strategy: {r.strategy}]  {'✓' if r.result == 55 else '✗'}")

# --- Summary ---
print(f"\nLatency on last call: {r.latency_ms:.0f}ms")
print(f"Cache: {vr.cache.hits} hits, {vr.cache.misses} misses")
