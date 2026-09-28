"""Regression tests for the LAS VOCES canonical projection boundary."""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("axioma_sync", REPO / "scripts" / "axioma_sync.py")
assert SPEC and SPEC.loader
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    project = tmp_path / "projects" / "las-voces"
    shutil.copytree(REPO / "projects" / "las-voces", project, ignore=shutil.ignore_patterns("AGENTS.md", "CLAUDE.md", "QWEN.md", ".qwen", "manifest.json"))
    return tmp_path


def generated(root: Path) -> Path:
    return root / "projects" / "las-voces"


def test_canonical_projects_each_harness_and_is_deterministic(root: Path) -> None:
    assert sync.generate(root) == 0
    project = generated(root)
    for relative in ("AGENTS.md", "CLAUDE.md", "QWEN.md", ".qwen/skills/las-voces-governance/SKILL.md", ".qwen/agents/primary-builder.md"):
        data = (project / relative).read_bytes()
        assert b"GENERATED FROM AXIOMA CANONICAL SOURCE" in data
    first = (project / "sync/manifest.json").read_bytes()
    assert sync.generate(root) == 0
    assert (project / "sync/manifest.json").read_bytes() == first


def test_source_change_requires_sync_and_check_does_not_mutate(root: Path) -> None:
    assert sync.generate(root) == 0
    project = generated(root)
    before = (project / "AGENTS.md").read_bytes()
    canonical = project / "skills/las-voces-governance.json"
    value = json.loads(canonical.read_text())
    value["purpose"] = "Changed canonical purpose."
    canonical.write_text(json.dumps(value), encoding="utf-8")
    assert sync.check(root) == 1
    assert (project / "AGENTS.md").read_bytes() == before


def test_manual_projection_edit_is_drift_and_explicit_generate_reconciles(root: Path) -> None:
    assert sync.generate(root) == 0
    target = generated(root) / "CLAUDE.md"
    target.write_text("manual edit", encoding="utf-8")
    assert sync.check(root) == 1
    assert sync.generate(root) == 0
    assert "manual edit" not in target.read_text(encoding="utf-8")


def test_atomic_failure_restores_previous_projection_set(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert sync.generate(root) == 0
    project = generated(root)
    old = {path: (project / path).read_bytes() for path in ("AGENTS.md", "CLAUDE.md", "QWEN.md")}
    canonical = project / "skills/las-voces-governance.json"
    value = json.loads(canonical.read_text())
    value["purpose"] = "Different, not partially written."
    canonical.write_text(json.dumps(value), encoding="utf-8")
    real_replace = sync.os.replace
    calls = 0

    def fail_once(source: Path | str, destination: Path | str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(sync.os, "replace", fail_once)
    with pytest.raises(OSError):
        sync.generate(root)
    assert {path: (project / path).read_bytes() for path in old} == old


@pytest.mark.parametrize("relative, mutation", [
    ("agents/ariadna.json", lambda value: value.pop("authority")),
    ("project.json", lambda value: value["project"].pop("id")),
])
def test_invalid_canonical_or_missing_identity_fails_closed(root: Path, relative: str, mutation) -> None:
    assert sync.generate(root) == 0
    target = generated(root) / relative
    value = json.loads(target.read_text())
    mutation(value)
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(sync.SyncError):
        sync.check(root)


def test_preserves_unrelated_harness_configuration(root: Path) -> None:
    unrelated = root / ".claude/settings.json"
    unrelated.parent.mkdir(parents=True)
    unrelated.write_text('{"enabledPlugins":{"existing":true}}\n', encoding="utf-8")
    assert sync.generate(root) == 0
    assert unrelated.read_text(encoding="utf-8") == '{"enabledPlugins":{"existing":true}}\n'
