"""Exercise result locking across independent processes without model calls."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from evaluation import runner


PROBE = """
import os
import sys
from evaluation import runner

runner.tempfile.tempdir = sys.argv[2]
try:
    with runner.output_lock(sys.argv[1]):
        if sys.argv[3] == 'crash':
            os._exit(23)
        print('acquired')
except ValueError:
    print('blocked')
"""


class OutputLockTests(unittest.TestCase):
    def setUp(self):
        # Relative path aliases require the same drive as the checkout on Windows.
        directory = tempfile.TemporaryDirectory(dir=runner.ROOT)
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name).resolve()
        self.base = self.directory / "results" / "run"
        self.system_temp = self.directory / "system-temp"
        self.system_temp.mkdir()
        temp_patch = patch("evaluation.runner.tempfile.gettempdir", return_value=str(self.system_temp))
        temp_patch.start()
        self.addCleanup(temp_patch.stop)

    def probe(self, base=None, *, crash=False):
        return subprocess.run(
            [sys.executable, "-B", "-c", PROBE, str(base or self.base), str(self.system_temp),
             "crash" if crash else "normal"],
            cwd=runner.ROOT, capture_output=True, text=True, timeout=15,
        )

    def test_other_process_is_blocked_until_release_and_the_same_file_is_reused(self):
        with runner.output_lock(self.base):
            locked = self.probe()
            self.assertEqual((locked.returncode, locked.stdout.strip()), (0, "blocked"), locked.stderr)
            path, = self.system_temp.rglob("*.lock")
            identity = path.stat().st_ino
        self.assertTrue(path.is_file())
        released = self.probe()
        self.assertEqual((released.returncode, released.stdout.strip()), (0, "acquired"), released.stderr)
        self.assertEqual(path.stat().st_ino, identity)
        self.assertEqual(path.read_bytes(), b"0")
        self.assertFalse(self.base.parent.exists())

    def test_different_output_can_be_locked_in_another_process_at_the_same_time(self):
        with runner.output_lock(self.base):
            other = self.probe(self.base.with_name("other"))
            self.assertEqual((other.returncode, other.stdout.strip()), (0, "acquired"), other.stderr)
        self.assertEqual(len(list(self.system_temp.rglob("*.lock"))), 2)

    def test_equivalent_paths_use_the_same_lock(self):
        equivalent = self.base.parent / "unused" / ".." / self.base.name
        relative = Path(os.path.relpath(self.base, runner.ROOT))
        with runner.output_lock(self.base):
            for base in (equivalent, relative):
                with self.subTest(base=str(base)):
                    other = self.probe(base)
                    self.assertEqual((other.returncode, other.stdout.strip()), (0, "blocked"), other.stderr)
        self.assertEqual(len(list(self.system_temp.rglob("*.lock"))), 1)

    @unittest.skipUnless(os.name == "nt", "Windows paths are case-insensitive")
    def test_windows_case_variants_use_the_same_lock(self):
        with runner.output_lock(self.base):
            other = self.probe(Path(str(self.base).upper()))
            self.assertEqual((other.returncode, other.stdout.strip()), (0, "blocked"), other.stderr)

    def test_exception_and_interrupt_release_the_lock(self):
        for exception in (RuntimeError, KeyboardInterrupt):
            with self.subTest(exception=exception.__name__):
                with self.assertRaises(exception), runner.output_lock(self.base):
                    raise exception
                other = self.probe()
                self.assertEqual((other.returncode, other.stdout.strip()), (0, "acquired"), other.stderr)

    def test_abrupt_process_exit_releases_the_lock_without_removing_its_file(self):
        crashed = self.probe(crash=True)
        self.assertEqual(crashed.returncode, 23, crashed.stderr)
        path, = self.system_temp.rglob("*.lock")
        self.assertTrue(path.is_file())
        other = self.probe()
        self.assertEqual((other.returncode, other.stdout.strip()), (0, "acquired"), other.stderr)


if __name__ == "__main__":
    unittest.main()
