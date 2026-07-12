"""
Raw Ollama diagnostic — bypass aicomputing entirely.
Shows exactly what fields Ollama returns so we can fix the backend.
"""
import ollama

prompt = (
    'You are a VIRTUAL RUNTIME. Return ONLY a JSON object with key "result".\n\n'
    'Signature: reverse(s: str) -> str\n'
    'Docstring: Return the reversed string.\n'
    'Inputs: {"s": "hello"}\n'
    'Output:'
)

def dump(label, obj):
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Type: {type(obj).__name__}")
    if hasattr(obj, 'model_dump'):
        d = obj.model_dump()
    elif isinstance(obj, dict):
        d = obj
    else:
        d = vars(obj) if hasattr(obj, '__dict__') else {}
    for k, v in d.items():
        s = repr(v)
        print(f"  {k}: {s[:300]}{'...' if len(s)>300 else ''}")

def get_response(r):
    if hasattr(r, 'response'): return r.response
    if isinstance(r, dict): return r.get('response')
    return None

def get_content(r):
    msg = getattr(r, 'message', None) or (r.get('message') if isinstance(r, dict) else None)
    if msg is None: return None
    return getattr(msg, 'content', None) or (msg.get('content') if isinstance(msg, dict) else None)

# TEST 1
r = ollama.generate(model="qwen3:4b", prompt=prompt, options={"num_predict": 2048})
dump("TEST 1: generate() — default", r)
print(f"\n>>> response: {get_response(r)!r:.500}")

# TEST 2
r = ollama.generate(model="qwen3:4b", prompt=prompt + "\n/no_think", options={"num_predict": 512})
dump("TEST 2: generate() — /no_think", r)
print(f"\n>>> response: {get_response(r)!r:.500}")

# TEST 3
r = ollama.generate(model="qwen3:4b", prompt=prompt + "\n/no_think", options={"num_predict": 512}, format="json")
dump("TEST 3: generate() — /no_think + format=json", r)
print(f"\n>>> response: {get_response(r)!r:.500}")

# TEST 4
r = ollama.chat(model="qwen3:4b", messages=[{"role":"user","content":prompt}], options={"num_predict": 2048})
dump("TEST 4: chat() — default", r)
print(f"\n>>> content: {get_content(r)!r:.500}")

# TEST 5
r = ollama.chat(model="qwen3:4b", messages=[{"role":"user","content":prompt+"\n/no_think"}], options={"num_predict": 512})
dump("TEST 5: chat() — /no_think", r)
print(f"\n>>> content: {get_content(r)!r:.500}")

# TEST 6
r = ollama.chat(model="qwen3:4b", messages=[{"role":"user","content":prompt+"\n/no_think"}], options={"num_predict": 512}, format="json")
dump("TEST 6: chat() — /no_think + format=json", r)
print(f"\n>>> content: {get_content(r)!r:.500}")

print(f"\nOllama pkg version: {getattr(ollama, '__version__', 'unknown')}")