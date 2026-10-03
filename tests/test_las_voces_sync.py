"""Regression tests for the LAS VOCES canonical projection boundary."""
from __future__ import annotations

import importlib.util
import inspect
import json
import shutil
from pathlib import Path

import pytest
import yaml

import _las_voces_process as local_process


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("axioma_sync", REPO / "scripts" / "axioma_sync.py")
assert SPEC and SPEC.loader
sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    project = tmp_path / "projects" / "las-voces"
    shutil.copytree(REPO / "projects" / "las-voces", project, ignore=shutil.ignore_patterns("AGENTS.md", "CLAUDE.md", "QWEN.md", ".qwen", "manifest.json"))
    local_process.git_init_fixture(tmp_path)
    local_process.git_empty_commit(tmp_path)
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


def _frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, metadata, body = text.split("---\n", 2)
    return yaml.safe_load(metadata), body


def test_qwen_skill_projection_has_native_frontmatter_from_canonical_source(root: Path) -> None:
    assert sync.generate(root) == 0
    project = generated(root)
    canonical = json.loads((project / "skills/las-voces-governance.json").read_text())
    path = project / ".qwen/skills/las-voces-governance/SKILL.md"
    metadata, body = _frontmatter(path)
    assert metadata == {"name": canonical["id"], "description": canonical["purpose"]}
    assert body.startswith("<!-- GENERATED FROM AXIOMA CANONICAL SOURCE. DO NOT EDIT DIRECTLY. -->\n")
    assert "Human Authority is Fernando" in body
    assert "CANONICAL → GENERATED PROJECTIONS" in body


def test_qwen_primary_builder_projection_uses_qwen_canonical_identity(root: Path) -> None:
    assert sync.generate(root) == 0
    project = generated(root)
    qwen = next(item for item in json.loads((project / "project.json").read_text())["agents"] if item["name"] == "Qwen")
    path = project / ".qwen/agents/primary-builder.md"
    metadata, body = _frontmatter(path)
    assert metadata == {
        "name": "primary-builder",
        "description": f"{qwen['name']} — {qwen['role']}. Authority: {qwen['authority']}.",
        "tools": ["*"],
        "disallowedTools": [],
        "approvalMode": qwen["approvalMode"],
    }
    assert body.startswith("<!-- GENERATED FROM AXIOMA CANONICAL SOURCE. DO NOT EDIT DIRECTLY. -->\n")
    assert "Qwen is the PRIMARY BUILDER" in " ".join(body.split())
    assert "may not merge, deploy" in body
    assert "Canonical agent: ariadna-project-manager" not in body


@pytest.mark.parametrize("field", ["tools", "disallowedTools"])
def test_missing_canonical_qwen_tools_fails_closed(root: Path, field: str) -> None:
    path = generated(root) / "project.json"
    value = json.loads(path.read_text())
    next(item for item in value["agents"] if item["name"] == "Qwen").pop(field, None)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(sync.SyncError, match="tools"):
        sync.check(root)


def test_missing_canonical_qwen_approval_mode_fails_closed(root: Path) -> None:
    path = generated(root) / "project.json"
    value = json.loads(path.read_text())
    next(item for item in value["agents"] if item["name"] == "Qwen").pop("approvalMode", None)
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(sync.SyncError, match="approvalMode"):
        sync.check(root)


@pytest.mark.parametrize("relative", [".qwen/agents/rogue.md", ".qwen/skills/rogue/SKILL.md"])
def test_unlisted_qwen_projection_fails_check(root: Path, relative: str, capsys) -> None:
    assert sync.generate(root) == 0
    rogue = generated(root) / relative
    rogue.parent.mkdir(parents=True, exist_ok=True)
    rogue.write_text("---\nname: rogue\ntools: [run_shell_command, write_file]\n---\n", encoding="utf-8")
    assert sync.check(root) == 1
    assert "UNLISTED PROJECTION" in capsys.readouterr().err


@pytest.mark.parametrize("field,value", [
    ("generator_template_version", "old"),
    ("target_harness", "wrong"),
    ("source_commit", "UNAVAILABLE"),
])
def test_manifest_entry_metadata_drift_fails_check(root: Path, field: str, value: str) -> None:
    assert sync.generate(root) == 0
    path = generated(root) / "sync/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["projections"][0][field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert sync.check(root) == 1


def test_skill_directory_follows_canonical_id(root: Path) -> None:
    assert sync.generate(root) == 0
    old = generated(root) / ".qwen/skills/las-voces-governance/SKILL.md"
    assert old.is_file()
    path = generated(root) / "skills/las-voces-governance.json"
    value = json.loads(path.read_text())
    value["id"] = "governance-renamed"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert sync.generate(root) == 0
    assert (generated(root) / ".qwen/skills/governance-renamed/SKILL.md").is_file()
    assert not old.exists()


def test_skill_rename_keeps_unlisted_file_and_fails_closed(root: Path) -> None:
    assert sync.generate(root) == 0
    rogue = generated(root) / ".qwen/skills/rogue/SKILL.md"
    rogue.parent.mkdir(parents=True)
    rogue.write_text("manually added", encoding="utf-8")
    path = generated(root) / "skills/las-voces-governance.json"
    value = json.loads(path.read_text())
    value["id"] = "governance-renamed"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert sync.generate(root) == 1
    assert rogue.read_text() == "manually added"


def test_skill_rename_refuses_manual_edits_to_old_projection(root: Path) -> None:
    assert sync.generate(root) == 0
    old = generated(root) / ".qwen/skills/las-voces-governance/SKILL.md"
    old.write_text("manual change", encoding="utf-8")
    path = generated(root) / "skills/las-voces-governance.json"
    value = json.loads(path.read_text())
    value["id"] = "governance-renamed"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(sync.SyncError, match="manual edits"):
        sync.generate(root)
    assert old.read_text(encoding="utf-8") == "manual change"


def test_skill_rename_refuses_symlinked_old_directory(root: Path) -> None:
    assert sync.generate(root) == 0
    project = generated(root)
    old_directory = project / ".qwen/skills/las-voces-governance"
    external_directory = root / "external-skill"
    old_directory.rename(external_directory)
    old_directory.symlink_to(external_directory, target_is_directory=True)
    path = project / "skills/las-voces-governance.json"
    value = json.loads(path.read_text())
    value["id"] = "governance-renamed"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(sync.SyncError, match="symlink"):
        sync.generate(root)
    assert (external_directory / "SKILL.md").is_file()


def test_generate_refuses_symlinked_expected_directory_without_external_write(root: Path) -> None:
    project = generated(root)
    outside = root / "outside-skill"
    outside.mkdir()
    victim = outside / "SKILL.md"
    victim.write_bytes(b"DO NOT OVERWRITE")
    skills = project / ".qwen/skills"
    skills.mkdir(parents=True)
    (skills / "las-voces-governance").symlink_to(outside, target_is_directory=True)
    with pytest.raises(sync.SyncError, match="symlink"):
        sync.generate(root)
    assert victim.read_bytes() == b"DO NOT OVERWRITE"
    assert not (project / "sync/manifest.json").exists()


@pytest.mark.parametrize("relative", [
    ".qwen/agents/primary-builder.md",
    ".qwen/skills/las-voces-governance/SKILL.md",
    "sync/manifest.json",
])
def test_check_rejects_symlinked_projection_even_with_identical_bytes(root: Path, relative: str) -> None:
    assert sync.generate(root) == 0
    target = generated(root) / relative
    alternate = root / "alternate-projection"
    target.rename(alternate)
    target.symlink_to(alternate)
    assert sync.check(root) == 1


@pytest.mark.parametrize("relative", [
    "project.json", "PROJECT_CHARTER.md", "SYNC_CONTRACT.md",
    "agents/ariadna.json", "skills/las-voces-governance.json",
    "sync/message-envelope.schema.json",
])
def test_check_rejects_symlinked_canonical_source_even_with_identical_bytes(root: Path, relative: str) -> None:
    assert sync.generate(root) == 0
    target = generated(root) / relative
    alternate = root / "alternate-source"
    target.rename(alternate)
    target.symlink_to(alternate)
    with pytest.raises(sync.SyncError, match="symlink"):
        sync.check(root)


def test_skill_rename_rolls_back_if_old_projection_cannot_be_removed(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert sync.generate(root) == 0
    project = generated(root)
    old = project / ".qwen/skills/las-voces-governance/SKILL.md"
    manifest = project / "sync/manifest.json"
    before_old, before_manifest = old.read_bytes(), manifest.read_bytes()
    path = project / "skills/las-voces-governance.json"
    value = json.loads(path.read_text())
    value["id"] = "governance-renamed"
    path.write_text(json.dumps(value), encoding="utf-8")
    real_unlink = Path.unlink

    def fail_old_unlink(self: Path, *args, **kwargs) -> None:
        if self == old:
            raise OSError("simulated old projection removal failure")
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_old_unlink)
    with pytest.raises(OSError, match="removal failure"):
        sync.generate(root)
    assert old.read_bytes() == before_old
    assert manifest.read_bytes() == before_manifest
    assert not (project / ".qwen/skills/governance-renamed/SKILL.md").exists()


@pytest.mark.parametrize("bad", ["\u2028", "\u2029", "\u0085", "\u007f", "\ud800"])
def test_adversarial_frontmatter_text_is_safe_or_sync_error(root: Path, bad: str) -> None:
    path = generated(root) / "skills/las-voces-governance.json"
    value = json.loads(path.read_text())
    value["purpose"] += bad
    path.write_text(json.dumps(value), encoding="utf-8")
    try:
        assert sync.generate(root) == 0
    except sync.SyncError:
        return
    frontmatter = (generated(root) / ".qwen/skills/las-voces-governance/SKILL.md").read_text().split("---\n", 2)[1]
    assert bad not in frontmatter


def test_lone_surrogate_cli_exits_two_without_traceback(root: Path) -> None:
    path = generated(root) / "skills/las-voces-governance.json"
    value = json.loads(path.read_text())
    value["purpose"] += "\ud800"
    path.write_text(json.dumps(value), encoding="utf-8")
    script = root / "scripts/axioma_sync.py"
    script.parent.mkdir()
    shutil.copyfile(REPO / "scripts/axioma_sync.py", script)
    result = local_process.run_sync_check(script)
    assert result.returncode == 2
    assert result.stderr.startswith("SYNC FAILED CLOSED:")
    assert "Traceback" not in result.stderr


def test_local_process_helper_exposes_only_fixed_purpose_calls() -> None:
    expected = {
        "git_init_fixture": ["path"],
        "git_empty_commit": ["path"],
        "run_sync_check": ["script"],
    }
    public = {
        name: value for name, value in vars(local_process).items()
        if inspect.isfunction(value) and value.__module__ == local_process.__name__
        and not name.startswith("_")
    }
    assert set(public) == set(expected)
    for name, function in public.items():
        assert list(inspect.signature(function).parameters) == expected[name]


def test_missing_canonical_qwen_builder_identity_fails_closed(root: Path) -> None:
    project = generated(root)
    value = json.loads((project / "project.json").read_text())
    next(item for item in value["agents"] if item["name"] == "Qwen").pop("authority")
    (project / "project.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(sync.SyncError, match="canonical Qwen builder identity"):
        sync.check(root)


def test_invalid_qwen_skill_name_fails_closed(root: Path) -> None:
    project = generated(root)
    path = project / "skills/las-voces-governance.json"
    value = json.loads(path.read_text())
    value["id"] = "las/voces"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(sync.SyncError, match="canonical skill name"):
        sync.check(root)


@pytest.mark.parametrize("name", [".", "..", ":hidden", "governance..old"])
def test_qwen_skill_name_cannot_escape_or_hide_directory(root: Path, name: str) -> None:
    path = generated(root) / "skills/las-voces-governance.json"
    value = json.loads(path.read_text())
    value["id"] = name
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(sync.SyncError, match="canonical skill name"):
        sync.check(root)


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
