"""Verify actual command exit codes using a loopback-only provider fixture."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
from threading import Thread
import unittest

from aicomputing.answer_protocol import frame_answer


class CliEntrypointTests(unittest.TestCase):
    def setUp(self):
        fixture = self
        self.response_text = ""
        self.requests = 0

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                fixture.requests += 1
                data = json.dumps(dict(id="offline", object="chat.completion", created=0, model="offline",
                    choices=[dict(index=0, finish_reason="stop",
                                  message=dict(role="assistant", content=fixture.response_text))])).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def test_console_and_module_distinguish_failure_valid_none_and_warn_output(self):
        launcher = Path(sysconfig.get_path("scripts")) / ("aicomp.exe" if os.name == "nt" else "aicomp")
        self.assertTrue(launcher.is_file(), "Install this checkout in editable mode before testing the CLI")
        commands = ([str(launcher)], [sys.executable, "-B", "-m", "aicomputing.cli"])
        environment = dict(os.environ, NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
        cases = (
            ("parse_failure", "int", "unframed response", "strict", 1, "", "unparsed"),
            ("strict_rejection", "int", frame_answer('{"result": "wrong"}'), "strict", 1, "", "invalid_type"),
            ("valid_none", "None", frame_answer('{"result": null}'), "strict", 0, "null\n", ""),
            ("warn_output", "int", frame_answer('{"result": "wrong"}'), "warn", 0, '"wrong"\n', "warning"),
        )
        for command in commands:
            for label, return_type, content, mode, code, output, diagnostic in cases:
                with self.subTest(command=command, scenario=label):
                    self.response_text = content
                    before = self.requests
                    completed = subprocess.run(command + [f"f(x: int) -> {return_type}", "Return x.",
                        "--inputs", '{"x": 2}', "--backend", "openai", "--model", "offline",
                        "--base-url", f"http://127.0.0.1:{self.server.server_port}/v1",
                        "--api-key", "offline-fixture", "--validation-mode", mode],
                        cwd=Path(__file__).resolve().parents[1], env=environment,
                        capture_output=True, text=True, timeout=20)
                    self.assertEqual(completed.returncode, code, completed.stderr)
                    self.assertEqual(completed.stdout, output)
                    self.assertNotIn("Traceback", completed.stderr)
                    if diagnostic:
                        self.assertIn(diagnostic, completed.stderr)
                    else:
                        self.assertEqual(completed.stderr, "")
                    self.assertGreater(self.requests, before)


if __name__ == "__main__":
    unittest.main()
