"""Debug: show raw model output for each test case."""

import argparse
from aicomputing import VirtualRuntime, ExecuteRequest

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3:4b")
parser.add_argument("--host", default=None, help="Ollama URL (default: local server). E.g. https://abc123.ngrok-free.app")
args = parser.parse_args()

vr = VirtualRuntime(backend="ollama", model=args.model, host=args.host)

cases = [
    ("add(x: int, y: int) -> int", "Return the sum of x and y.", {"x": 2, "y": 3}, int),
    ("reverse(s: str) -> str", "Return the reversed string.", {"s": "hello"}, str),
    ("sort(arr: list[int]) -> list[int]", "Return a sorted copy of the array.", {"arr": [5, 3, 1, 4, 2]}, list),
    (
        "dijkstra(graph, start_node: str) -> dict[str, int]",
        "Return the shortest path distances from start_node to all other nodes. Graph is {node: [(neighbor, weight), ...]}.",
        {
            "graph": {"A": [("B", 4), ("C", 2)], "B": [("A", 4), ("C", 5), ("D", 10)], "C": [("A", 2), ("B", 5), ("D", 3)], "D": [("B", 10), ("C", 3)]},
            "start_node": "A",
        },
        dict,
    ),
    ("score_review(text: str) -> int", "Return an integer sentiment score from 0 to 5 inclusive. 0 = terrible, 5 = excellent.", {"text": "Absolutely loved this product, works perfectly!"}, int),
]

for sig, doc, inputs, ret in cases:
    r = vr.execute(ExecuteRequest(
        signature=sig, docstring=doc, inputs=inputs,
        return_type=ret, strict_types=False,
    ))
    print(f"=== {sig.split('(')[0]} ===")
    print(f"  raw_text:  {r.raw_text!r:.200}")
    print(f"  parsed:    {r.result!r}")
    print(f"  type:      {type(r.result).__name__}")
    print(f"  warnings:  {r.type_warnings}")
    print(f"  strategy:  {r.strategy}")
    print(f"  latency:   {r.latency_ms:.0f}ms")
    print()
