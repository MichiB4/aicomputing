"""Portable, frozen inputs and software records for future evaluations."""
from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import dataclass
from importlib.metadata import distribution
from pathlib import Path

from packaging.requirements import Requirement

RUNTIME_PACKAGES = ("openai", "httpx", "ollama", "pydantic", "truststore", "packaging")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def capture_environment():
    """Record runtime dependencies, including active transitive requirements."""
    packages, seen = {}, set()
    pending = [(name, ()) for name in RUNTIME_PACKAGES]
    while pending:
        name, extras = pending.pop()
        name = re.sub(r"[-_.]+", "-", name).lower()
        identity = (name, tuple(sorted(extras)))
        if identity in seen:
            continue
        seen.add(identity)
        installed = distribution(name)
        packages[name] = installed.version
        for text in installed.requires or ():
            requirement = Requirement(text)
            if requirement.marker and not any(
                requirement.marker.evaluate({"extra": extra}) for extra in ("", *extras)
            ):
                continue
            dependency = distribution(requirement.name)
            if not requirement.specifier.contains(dependency.version, prereleases=True):
                raise ValueError("Installed dependency conflicts with " + requirement.name)
            pending.append((requirement.name, tuple(requirement.extras)))
    return {
        "python": {"implementation": platform.python_implementation(), "version": platform.python_version()},
        "packages": dict(sorted(packages.items())),
        "platform": {"system": platform.system(), "machine": platform.machine()},
    }


def requirements_bytes(environment):
    """A version-pinned installation list; credentials and local paths are absent."""
    lines = ["# Python " + environment["python"]["version"]]
    lines.extend(name + "==" + version for name, version in sorted(environment["packages"].items()))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _unique_object(pairs):
    value = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("Duplicate manifest field")
        value[name] = item
    return value


@dataclass(frozen=True)
class LoadedInputs:
    manifest: dict
    manifest_bytes: bytes
    dataset_bytes: bytes
    requirements_bytes: bytes
    environment: dict

    def snapshot(self, directory):
        for name, raw in (
            ("manifest.executed.json", self.manifest_bytes),
            ("dataset-release.json", self.dataset_bytes),
            ("requirements.txt", self.requirements_bytes),
        ):
            with (directory / name).open("xb") as stream:
                stream.write(raw)


def load_inputs(path):
    """Resolve bundle-relative data in memory; never rewrite the frozen manifest."""
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    manifest = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be an object")
    reference = manifest.get("dataset_release")
    if not isinstance(reference, dict) or reference.get("path") != "dataset-release.json":
        raise ValueError("Prepared manifest must use the adjacent dataset-release.json")
    record = manifest.get("reproducibility")
    if not isinstance(record, dict) or type(record.get("format_version")) is not int or record["format_version"] != 1:
        raise ValueError("Unsupported reproducibility format")
    files = {}
    for name in ("dataset-release.json", "requirements.txt"):
        resolved = (source.parent / name).resolve(strict=True)
        if resolved.parent != source.parent:
            raise ValueError("Prepared input escapes its bundle directory")
        files[name] = resolved.read_bytes()
    if digest(files["dataset-release.json"]) != reference.get("sha256"):
        raise ValueError("Frozen dataset SHA256 changed")
    requirements = files["requirements.txt"]
    if digest(requirements) != record.get("requirements_sha256"):
        raise ValueError("Frozen requirements SHA256 changed")
    expected = record.get("environment")
    if not isinstance(expected, dict) or not isinstance(expected.get("packages"), dict):
        raise ValueError("Missing frozen software environment")
    if requirements != requirements_bytes(expected):
        raise ValueError("Requirements disagree with the frozen software environment")
    actual = capture_environment()
    installed_pins = {name: distribution(name).version for name in expected["packages"]}
    if (expected.get("python") != actual["python"] or expected["packages"] != installed_pins
            or any(expected["packages"].get(name) != version for name, version in actual["packages"].items())):
        raise ValueError("Software versions changed; restore the frozen Python and requirements")
    manifest["dataset_release"] = {**reference, "path": str((source.parent / "dataset-release.json").resolve())}
    from experiment import check_compatible
    check_compatible(manifest)
    return LoadedInputs(manifest, raw, files["dataset-release.json"], requirements, actual)
