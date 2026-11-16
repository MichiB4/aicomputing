"""Resource ownership and failure precedence, without inference services."""

import traceback
import unittest
from unittest.mock import patch

from aicomputing import Backend, ExecuteRequest, VirtualRuntime
from aicomputing.answer_protocol import frame_answer
from aicomputing.backends.base import BackendCommunicationError, GenerateResponse


class LifecycleBackend(Backend):
    name = "offline"
    default_model = "offline"

    def __init__(self, *, generation_error=None, cleanup_error=None):
        self.generation_error = generation_error
        self.cleanup_error = cleanup_error
        self.close_calls = 0
        self.generation_calls = 0

    def generate(self, prompt, **settings):
        self.generation_calls += 1
        if self.generation_error is not None:
            raise self.generation_error
        return GenerateResponse(text=frame_answer('{"result": 5}'), model="offline")

    def cache_state(self):
        return {}

    def close(self):
        self.close_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


def request():
    return ExecuteRequest(signature="f(x: int) -> int", docstring="Return x.", inputs={"x": 5})


class RuntimeLifecycleTests(unittest.TestCase):
    def test_invalid_trace_paths_are_rejected_before_backend_creation(self):
        cases = [(value, TypeError) for value in (False, True, 0, 1.5, [], {}, (), b"trace.jsonl")]
        cases += [(value, ValueError) for value in ("", "  ", "\t\r\n", "\u2003")]
        for path, error in cases:
            with self.subTest(path=path), patch("aicomputing.runtime.create_backend") as factory:
                with self.assertRaisesRegex(error, "trace_file"):
                    VirtualRuntime(backend="openai", trace_file=path)
                factory.assert_not_called()

    def test_inference_error_identity_cause_and_traceback_survive_cleanup_failure(self):
        primary = BackendCommunicationError("PRIMARY_BODY_SENTINEL")
        cause = ValueError("Original cause")
        primary.__cause__ = cause
        backend = LifecycleBackend(generation_error=primary, cleanup_error=OSError("CLEANUP_BODY_SENTINEL"))
        with patch("aicomputing.runtime.create_backend", return_value=backend):
            with self.assertLogs("aicomputing", level="WARNING") as logs:
                with self.assertRaises(BackendCommunicationError) as caught:
                    with VirtualRuntime(backend="openai") as runtime:
                        runtime.execute(request())
        self.assertIs(caught.exception, primary)
        self.assertIs(caught.exception.__cause__, cause)
        self.assertEqual(backend.close_calls, 1)
        self.assertIn("OSError", logs.output[0])
        self.assertNotIn("PRIMARY_BODY_SENTINEL", "\n".join(logs.output))
        self.assertNotIn("CLEANUP_BODY_SENTINEL", "\n".join(logs.output))

    def test_inference_error_keeps_its_original_frames(self):
        primary = BackendCommunicationError("Offline inference failure")
        backend = LifecycleBackend(generation_error=primary, cleanup_error=OSError("Offline cleanup failure"))
        with patch("aicomputing.runtime.create_backend", return_value=backend):
            with self.assertLogs("aicomputing", level="WARNING"):
                with self.assertRaises(BackendCommunicationError):
                    try:
                        with VirtualRuntime(backend="openai") as runtime:
                            runtime.execute(request())
                    except BackendCommunicationError as caught:
                        self.assertIs(caught, primary)
                        names = [frame.name for frame in traceback.extract_tb(caught.__traceback__)]
                        self.assertIn("generate", names)
                        self.assertNotIn("close", names)
                        raise

    def test_setup_error_remains_primary_when_owned_backend_cleanup_fails(self):
        backend = LifecycleBackend(cleanup_error=OSError("Cleanup failure"))
        with patch("aicomputing.runtime.create_backend", return_value=backend):
            with self.assertLogs("aicomputing", level="WARNING"):
                with self.assertRaisesRegex(ValueError, "cache maxsize must be non-negative"):
                    VirtualRuntime(backend="openai", cache_size=-1)
        self.assertEqual(backend.close_calls, 1)

    def test_setup_failure_releases_owned_resources_without_a_warning_when_cleanup_succeeds(self):
        backend = LifecycleBackend()
        with patch("aicomputing.runtime.create_backend", return_value=backend):
            with self.assertNoLogs("aicomputing", level="WARNING"):
                with self.assertRaises(ValueError):
                    VirtualRuntime(backend="openai", cache_size=-1)
        self.assertEqual(backend.close_calls, 1)

    def test_cleanup_only_failure_propagates_and_can_be_retried(self):
        error = OSError("Cleanup failure")
        backend = LifecycleBackend(cleanup_error=error)
        with patch("aicomputing.runtime.create_backend", return_value=backend):
            with self.assertNoLogs("aicomputing", level="WARNING"):
                with self.assertRaises(OSError) as caught:
                    with VirtualRuntime(cache_size=8, backend="openai") as runtime:
                        self.assertEqual(runtime.execute(request()).result, 5)
        self.assertIs(caught.exception, error)
        self.assertIs(runtime.__enter__(), runtime)
        self.assertEqual(len(runtime.cache), 1)
        backend.cleanup_error = None
        runtime.close()
        self.assertEqual(len(runtime.cache), 0)
        runtime.close()
        self.assertEqual(backend.close_calls, 2)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            runtime.execute(request())

    def test_direct_close_failure_propagates_without_becoming_a_warning(self):
        error = OSError("Cleanup failure")
        backend = LifecycleBackend(cleanup_error=error)
        with patch("aicomputing.runtime.create_backend", return_value=backend):
            runtime = VirtualRuntime(backend="openai")
        with self.assertNoLogs("aicomputing", level="WARNING"), self.assertRaises(OSError) as caught:
            runtime.close()
        self.assertIs(caught.exception, error)
        backend.cleanup_error = None
        runtime.close()

    def test_borrowed_backends_are_not_closed_on_body_or_setup_failure(self):
        backend = LifecycleBackend(cleanup_error=OSError("Borrowed backend must stay open"))
        primary = RuntimeError("Body failure")
        with self.assertNoLogs("aicomputing", level="WARNING"):
            with self.assertRaises(RuntimeError) as caught:
                with VirtualRuntime(backend=backend):
                    raise primary
            self.assertIs(caught.exception, primary)
            with self.assertRaises(ValueError):
                VirtualRuntime(backend=backend, cache_size=-1)
        self.assertEqual(backend.close_calls, 0)

    def test_active_interrupts_are_preserved_when_cleanup_raises_an_ordinary_error(self):
        for error in (KeyboardInterrupt(), SystemExit(17)):
            with self.subTest(error=type(error).__name__):
                backend = LifecycleBackend(cleanup_error=OSError("Cleanup failure"))
                with patch("aicomputing.runtime.create_backend", return_value=backend):
                    with self.assertLogs("aicomputing", level="WARNING"):
                        with self.assertRaises(type(error)) as caught:
                            with VirtualRuntime(backend="openai"):
                                raise error
                self.assertIs(caught.exception, error)

    def test_interrupt_during_cleanup_is_not_suppressed(self):
        backend = LifecycleBackend(cleanup_error=KeyboardInterrupt())
        with patch("aicomputing.runtime.create_backend", return_value=backend):
            with self.assertNoLogs("aicomputing", level="WARNING"):
                with self.assertRaises(KeyboardInterrupt):
                    with VirtualRuntime(backend="openai"):
                        raise ValueError("Body failure")

    def test_backend_context_manager_uses_the_same_failure_precedence(self):
        primary = BackendCommunicationError("Inference failure")
        backend = LifecycleBackend(generation_error=primary, cleanup_error=OSError("Cleanup failure"))
        with self.assertLogs("aicomputing", level="WARNING"):
            with self.assertRaises(BackendCommunicationError) as caught:
                with backend:
                    backend.generate("prompt")
        self.assertIs(caught.exception, primary)
        self.assertEqual(backend.close_calls, 1)
        with self.assertRaises(OSError):
            with backend:
                pass


if __name__ == "__main__":
    unittest.main()
