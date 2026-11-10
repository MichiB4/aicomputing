"""Run generated Python in a disposable subprocess using the caller's environment."""
from __future__ import annotations

import ast
from io import StringIO
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import tokenize
from types import ModuleType


MAX_ERROR_MESSAGE_CHARS = 500
_CLEANUP_TIMEOUT_SECONDS = 5.0
logger = logging.getLogger("aicomputing")
_GENERATED_FILENAME = "<generated>"
_WORKER_START_SIGNAL = b"\x01"
# -c keeps the working directory on Python's normal module search path.
_WORKER_BOOTSTRAP = "import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name='__main__')"


def _error_details(exc):
    """Describe a failure without exposing a traceback or inventing a source line."""
    try:
        message = str(exc.msg) if isinstance(exc, SyntaxError) and exc.msg is not None else str(exc)
    except Exception:
        message = "Exception message could not be formatted"
    message = " ".join(message.split())
    if len(message) > MAX_ERROR_MESSAGE_CHARS:
        message = message[:MAX_ERROR_MESSAGE_CHARS - 3] + "..."

    line = None
    if isinstance(exc, SyntaxError) and exc.filename == _GENERATED_FILENAME:
        line = exc.lineno
    traceback_entry = exc.__traceback__
    while traceback_entry is not None:
        if traceback_entry.tb_frame.f_code.co_filename == _GENERATED_FILENAME:
            line = traceback_entry.tb_lineno
        traceback_entry = traceback_entry.tb_next
    return {"error_type": type(exc).__name__, "error_message": message, "error_line": line}


def _contains_extra_fence(source: str) -> bool:
    """Find bare fence lines outside Python strings without validating syntax."""
    fence_lines = {
        number for number, line in enumerate(source.split("\n"), 1)
        if re.fullmatch(r"[ \t]*```[ \t]*\r?", line)
    }
    if not fence_lines:
        return False
    try:
        for token in tokenize.generate_tokens(StringIO(source).readline):
            if (token.start[0] in fence_lines and token.type in (tokenize.OP, tokenize.ERRORTOKEN)
                    and "`" in token.string):
                return True
    except (tokenize.TokenError, SyntaxError, UnicodeError):
        # Incomplete strings and other broken source are diagnosed by the worker.
        pass
    return False


def extract_code(raw_text: str) -> str:
    """Extract raw source or one Python fence; leave syntax checks to the worker."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("Expected nonempty Python source")
    text = raw_text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n```", text,
                             flags=re.DOTALL | re.IGNORECASE)
        if match is None or _contains_extra_fence(match.group(1)):
            raise ValueError("Expected exactly one Python code block")
        text = match.group(1).strip()
    if not text:
        raise ValueError("Expected nonempty Python source")
    return text


def _encode_value(value, active=None):
    """Encode Python values for the standalone worker's typed JSON protocol."""
    active = set() if active is None else active
    kind = type(value)
    if value is None:
        return ["none"]
    if kind is bool:
        return ["bool", value]
    if kind is int:
        return ["int", hex(value)]
    if kind is float:
        return ["float", value.hex()]
    if kind is str:
        return ["str", value]
    if kind is bytes:
        return ["bytes", value.hex()]
    if kind not in (list, tuple, set, frozenset, dict):
        raise TypeError(f"Unsupported value type: {kind.__module__}.{kind.__qualname__}")
    identity = id(value)
    if identity in active:
        raise ValueError("Cyclic value")
    active.add(identity)
    try:
        if kind is dict:
            return ["dict", [[_encode_value(k, active), _encode_value(v, active)] for k, v in value.items()]]
        values = [_encode_value(item, active) for item in value]
        if kind in (set, frozenset):
            values.sort(key=json.dumps)
        return [kind.__name__, values]
    finally:
        active.remove(identity)


def _decode_value(node):
    """Reconstruct a Python value from a valid typed-protocol node."""
    tag = node[0]
    if tag == "none":
        return None
    payload = node[1]
    if tag in ("bool", "str"):
        return payload
    if tag == "int":
        return int(payload, 16)
    if tag == "float":
        return float.fromhex(payload)
    if tag == "bytes":
        return bytes.fromhex(payload)
    if tag == "dict":
        return {_decode_value(k): _decode_value(v) for k, v in payload}
    kinds = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}
    return kinds[tag](_decode_value(item) for item in payload)


def _signature_without_annotations(function):
    """Inspect binding metadata using annotation-free views of Python callables."""
    from functools import partial
    from inspect import isfunction, ismethod, signature, unwrap
    from types import FunctionType, MethodType

    def view(target):
        if ismethod(target):
            return MethodType(view(target.__func__), target.__self__)
        target = unwrap(target, stop=lambda candidate: getattr(candidate, "__signature__", None) is not None)
        if getattr(target, "__signature__", None) is not None:
            return target
        if isinstance(target, partial):
            return partial(view(target.func), *target.args, **(target.keywords or {}))
        if isfunction(target):
            # Preserve actual default/closure objects; do not copy annotation metadata.
            clone = FunctionType(target.__code__, target.__globals__, target.__name__,
                                 target.__defaults__, target.__closure__)
            clone.__kwdefaults__ = target.__kwdefaults__
            return clone
        if not isinstance(target, type) and callable(target):
            call_method = target.__call__
            if ismethod(call_method):
                return view(call_method)
        return target

    return signature(view(function), eval_str=False)


def _call_with_inputs(function, declaration, inputs):
    """Translate named inputs into the declared positional/keyword call.

    Fill only gaps before a supplied positional-only argument. Defaults come
    from the implementation; specification expressions are never evaluated.
    """
    names = [parameter.arg for parameter in declaration.args.posonlyargs]
    count = max((index + 1 for index, name in enumerate(names) if name in inputs), default=0)
    if count == 0:
        return function(**inputs)

    positional, keywords = [], dict(inputs)
    defaults = None
    for index, name in enumerate(names[:count]):
        if name in keywords:
            positional.append(keywords.pop(name))
        else:
            if defaults is None:
                from inspect import Parameter
                defaults = [parameter.default for parameter in _signature_without_annotations(function).parameters.values()
                            if parameter.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)]
            if index >= len(defaults) or defaults[index] is Parameter.empty:
                raise TypeError(f"Missing required positional-only argument: {name!r}")
            positional.append(defaults[index])
    return function(*positional, **keywords)


def _worker_main(request_path, result_path):
    ready_path = Path(result_path).with_suffix(".ready")
    ready_temporary = ready_path.with_suffix(".ready.tmp")
    ready_temporary.write_text(json.dumps({"status": "ready", "pid": os.getpid()}), encoding="utf-8")
    ready_temporary.replace(ready_path)
    # EOF means the parent failed or disconnected before cleanup setup completed.
    if sys.stdin.buffer.read(1) != _WORKER_START_SIGNAL:
        return
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    result = {"status": "runtime_error", "value": None, "error_type": None,
              "error_message": None, "error_line": None}
    compiled_code = None
    try:
        declaration = ast.parse("def " + request["signature"].rstrip(":") + ":\n    pass\n").body[0]
        module = ModuleType("__aicomp_generated__")
        module.__file__ = _GENERATED_FILENAME
        sys.modules[module.__name__] = module
        namespace = vars(module)
        compiled_code = compile(request["code"], _GENERATED_FILENAME, "exec", dont_inherit=True)
        exec(compiled_code, namespace, namespace)
        function = namespace[declaration.name]
        value = _call_with_inputs(function, declaration, _decode_value(request["inputs"]))
    except SyntaxError as exc:
        result.update(status="syntax_error", **_error_details(exc))
    except UnicodeError as exc:
        status = "syntax_error" if compiled_code is None else "runtime_error"
        result.update(status=status, **_error_details(exc))
    except BaseException as exc:
        result.update(status="runtime_error", **_error_details(exc))
    else:
        try:
            result.update(status="ok", value=_encode_value(value))
        except BaseException as exc:
            result.update(status="serialization_error", value=None, **_error_details(exc))
    Path(result_path).write_text(json.dumps(result, ensure_ascii=True), encoding="utf-8")


def _windows_job(process, worker_pid=None):
    """Group the worker and its children for cleanup without resource limits."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", wintypes.DWORD), ("min_working", ctypes.c_size_t),
                    ("max_working", ctypes.c_size_t), ("processes", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t),
                    ("peak_job", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL

    def job_error(operation):
        error_code = ctypes.get_last_error()
        message = f"{operation} failed: {ctypes.FormatError(error_code).strip()}"
        return ctypes.WinError(error_code, message)

    handle = kernel.CreateJobObjectW(None, None)
    if not handle:
        raise job_error("CreateJobObjectW")
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    limits = Extended()
    limits.basic.flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                          ctypes.byref(limits), ctypes.sizeof(limits)):
        error = job_error("SetInformationJobObject")
        kernel.CloseHandle(handle)
        raise error
    worker_handle = None
    try:
        target_handle = wintypes.HANDLE(int(process._handle))
        # A Windows venv launcher can be a different process from its Python worker.
        if worker_pid is not None and worker_pid != process.pid:
            PROCESS_SET_QUOTA = 0x0100
            PROCESS_TERMINATE = 0x0001
            worker_handle = kernel.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, worker_pid)
            if not worker_handle:
                raise job_error("OpenProcess")
            target_handle = worker_handle
        if not kernel.AssignProcessToJobObject(handle, target_handle):
            raise job_error("AssignProcessToJobObject")
        if worker_handle:
            if not kernel.CloseHandle(worker_handle):
                raise job_error("CloseHandle")
            worker_handle = None
    except BaseException:
        if worker_handle:
            kernel.CloseHandle(worker_handle)
        kernel.CloseHandle(handle)
        raise

    def close_job():
        if not kernel.CloseHandle(handle):
            raise job_error("CloseHandle")

    return close_job


def _wait_for_worker_ready(process, result_path, deadline, timeout_seconds):
    """Read the worker PID while generated source remains behind the startup gate."""
    ready_path = result_path.with_suffix(".ready")
    while True:
        try:
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            break
        except (FileNotFoundError, PermissionError) as exc:
            if process.poll() is not None:
                raise OSError(f"Python worker exited before startup completed (code {process.returncode}).") from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if isinstance(exc, PermissionError):
                    raise
                raise subprocess.TimeoutExpired(process.args, timeout_seconds) from exc
            time.sleep(min(0.01, remaining))
    if (not isinstance(ready, dict) or ready.get("status") != "ready"
            or type(ready.get("pid")) is not int or ready["pid"] <= 0):
        raise OSError("Python worker returned an invalid startup response.")
    return ready["pid"]


def _cleanup_execution(process, close_job, temporary, original_error=None):
    """Attempt all cleanup steps, retaining any error already in flight."""
    failures = []

    def attempt(action, *, ignore_missing=False):
        try:
            action()
        except ProcessLookupError as exc:
            if not ignore_missing:
                failures.append(exc)
        except BaseException as exc:
            failures.append(exc)

    if process is not None and process.stdin is not None:
        attempt(process.stdin.close)
    if close_job is not None:
        attempt(close_job)
    if process is not None:
        if os.name != "nt":
            attempt(lambda: os.killpg(process.pid, signal.SIGKILL), ignore_missing=True)
        # Popen.kill() is a no-op when it knows the process has already exited.
        attempt(process.kill)
        attempt(lambda: process.wait(timeout=_CLEANUP_TIMEOUT_SECONDS))
    if temporary is not None:
        attempt(temporary.cleanup)

    if not failures:
        return
    if original_error is not None:
        logger.warning("Worker cleanup also failed with %s; preserving the original error.",
                       ", ".join(type(exc).__name__ for exc in failures))
        return
    # A cancellation during cleanup must not be hidden by an ordinary cleanup error.
    primary = next((exc for exc in failures if not isinstance(exc, Exception)), failures[0])
    additional = [type(exc).__name__ for exc in failures if exc is not primary]
    if additional:
        logger.warning("Additional worker cleanup failures: %s.", ", ".join(additional))
    raise primary


def execute_code(code: str, signature: str, inputs: dict, timeout_seconds=120.0) -> dict:
    """Run one implementation/input with normal Python imports and permissions."""
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("timeout_seconds must be finite and positive")
    started = time.perf_counter()
    result = {"status": "harness_error", "value": None, "error_type": None,
              "error_message": None, "error_line": None, "elapsed_seconds": 0.0}
    process, close_job, temporary, original_error = None, None, None, None
    try:
        try:
            temporary = tempfile.TemporaryDirectory(prefix="aicomp-code-")
            directory = Path(temporary.name)
            request_path, result_path = directory / "request.json", directory / "result.json"
            request_path.write_text(json.dumps({"code": code, "signature": signature,
                                               "inputs": _encode_value(inputs)}), encoding="utf-8")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            deadline = time.monotonic() + timeout_seconds
            process = subprocess.Popen([sys.executable, "-c", _WORKER_BOOTSTRAP, str(Path(__file__).resolve()),
                                        "--worker", str(request_path), str(result_path)],
                                       stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, creationflags=flags,
                                       start_new_session=os.name != "nt")
            try:
                worker_pid = _wait_for_worker_ready(process, result_path, deadline, timeout_seconds)
                close_job = _windows_job(process, worker_pid=worker_pid)
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(process.args, timeout_seconds)
                process.stdin.write(_WORKER_START_SIGNAL)
                process.stdin.close()
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                result.update(status="timeout", error_type="TimeoutExpired",
                              error_message=f"Worker exceeded the {timeout_seconds:g}-second timeout.")
            else:
                if process.returncode != 0 or not result_path.is_file():
                    result.update(status="runtime_error", error_type="WorkerProcessError",
                                  error_message=f"Worker exited with code {process.returncode} without a valid result.")
                else:
                    payload = json.loads(result_path.read_text(encoding="utf-8"))
                    result.update(payload)
                    if payload["status"] == "ok":
                        result["value"] = _decode_value(payload["value"])
        except BaseException as exc:
            original_error = exc
            raise
        finally:
            _cleanup_execution(process, close_job, temporary, original_error)
    except Exception as exc:
        result.update(status="harness_error", value=None, **_error_details(exc))
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def execute_code_with_retries(code: str, signature: str, inputs: dict,
                              timeout_seconds=120.0, timeout_retries=2) -> dict:
    """Retry only worker timeouts, using identical source and input values."""
    if type(timeout_retries) is not int or timeout_retries < 0:
        raise ValueError("timeout_retries must be a nonnegative integer")
    import hashlib

    # Retain a fingerprint even for malformed source; compilation remains in the worker.
    source_hash = hashlib.sha256(code.encode("utf-8", errors="surrogatepass")).hexdigest()
    attempts = []
    for index in range(timeout_retries + 1):
        outcome = execute_code(code, signature, inputs, timeout_seconds=timeout_seconds)
        attempts.append({"attempt": index + 1, "status": outcome["status"],
                         "error_type": outcome["error_type"],
                         "error_message": outcome["error_message"], "error_line": outcome["error_line"],
                         "elapsed_seconds": outcome["elapsed_seconds"],
                         "code_sha256": source_hash})
        if outcome["status"] != "timeout":
            break
    return {**outcome, "execution_attempts": attempts,
            "elapsed_seconds": sum(attempt["elapsed_seconds"] for attempt in attempts)}


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--worker":
    _worker_main(sys.argv[2], sys.argv[3])
