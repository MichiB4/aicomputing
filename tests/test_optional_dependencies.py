"""The core runtime remains usable without provider-specific dependencies."""
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


class OptionalDependencyTests(unittest.TestCase):
    def test_core_import_execution_and_backend_construction_without_provider_sdks(self):
        script = textwrap.dedent('''
            import contextlib
            import importlib.abc
            import io
            import sys

            optional = {"ollama", "openai", "httpx", "pydantic", "torch", "transformers",
                        "accelerate", "sentencepiece", "truststore"}
            class MissingProviders(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".")[0] in optional:
                        raise ModuleNotFoundError("Optional dependency unavailable", name=fullname)
            sys.meta_path.insert(0, MissingProviders())

            from aicomputing import Backend, ExecuteRequest, VirtualRuntime, create_backend
            from aicomputing.answer_protocol import frame_answer
            from aicomputing.backends.base import GenerateResponse
            from aicomputing.cli import main

            class OfflineBackend(Backend):
                name = "offline"
                default_model = "offline"
                def generate(self, prompt, **options):
                    return GenerateResponse(text=frame_answer('{"result": 5}'), model="offline")

            with VirtualRuntime(backend=OfflineBackend()) as runtime:
                result = runtime.execute(ExecuteRequest(signature="add(x: int, y: int) -> int",
                    docstring="Return the sum.", inputs={"x": 2, "y": 3}))
            assert result.result == 5 and result.validation_status == "valid"
            for name in ("ollama", "openai", "hf"):
                backend = create_backend(name, model="offline")
                backend.close()
            try:
                create_backend("ollama", model="offline").generate("prompt")
            except ImportError as error:
                assert "aicomputing[ollama]" in str(error)
            else:
                raise AssertionError("Missing Ollama dependency was not reported")
            for name in ("ollama", "openai", "hf"):
                output, errors = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
                    code = main(["f(x: int) -> int", "Return x.", "--inputs", '{"x": 1}',
                                 "--backend", name, "--model", "offline"])
                assert code == 1
                assert output.getvalue() == ""
                assert "aicomputing[" + name + "]" in errors.getvalue()
                assert "Traceback" not in errors.getvalue()
            assert optional.isdisjoint(sys.modules)
        ''')
        completed = subprocess.run([sys.executable, "-B", "-c", script],
                                   cwd=Path(__file__).resolve().parents[1],
                                   capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
