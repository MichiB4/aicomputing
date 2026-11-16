"""Checkpoint replacement preserves complete files through temporary conflicts."""
import errno
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import common


class CheckpointIOTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.target = Path(directory.name) / "checkpoint.json"
        self.target.write_text('{"version": 1}', encoding="utf-8")

    def test_transient_permission_conflicts_preserve_the_old_file_until_replacement(self):
        original_replace = common.os.replace
        attempts = []

        def replace(source, target):
            attempts.append(target)
            if len(attempts) <= 2:
                self.assertEqual(json.loads(target.read_text()), {"version": 1})
                self.assertEqual(json.loads(source.read_text()), {"version": 2})
                raise PermissionError("temporary sharing conflict")
            original_replace(source, target)

        with patch.object(common.os, "replace", side_effect=replace), patch.object(common.time, "sleep") as sleep:
            common.atomic_json(self.target, {"version": 2})
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(json.loads(self.target.read_text()), {"version": 2})
        self.assertFalse(self.target.with_suffix(".json.tmp").exists())

    def test_persistent_permission_failure_is_bounded_and_preserves_the_old_file(self):
        error = PermissionError("persistent permission failure")
        with patch.object(common.os, "replace", side_effect=error) as replace, \
                patch.object(common.time, "sleep") as sleep:
            with self.assertRaises(PermissionError) as raised:
                common.atomic_json(self.target, {"version": 2})
        self.assertIs(raised.exception, error)
        self.assertEqual(replace.call_count, 5)
        self.assertEqual(sleep.call_count, 4)
        self.assertLessEqual(sum(call.args[0] for call in sleep.call_args_list), 1)
        self.assertEqual(json.loads(self.target.read_text()), {"version": 1})

    def test_other_io_failures_propagate_without_retry(self):
        error = OSError(errno.ENOSPC, "disk full")
        with patch.object(common.os, "replace", side_effect=error) as replace, \
                patch.object(common.time, "sleep") as sleep:
            with self.assertRaises(OSError) as raised:
                common.atomic_json(self.target, {"version": 2})
        self.assertIs(raised.exception, error)
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(json.loads(self.target.read_text()), {"version": 1})
