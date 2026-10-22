"""Process-scoped Git trust for the jaxqwen capability."""
from __future__ import annotations

import os
import importlib.util
import pwd
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "projects/las-voces/authority/jaxqwen_capability.py"
_SPEC = importlib.util.spec_from_file_location("jaxqwen_git_trust_capability", _MODULE_PATH)
assert _SPEC and _SPEC.loader
capability = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = capability
_SPEC.loader.exec_module(capability)


def _repo(path: Path) -> None:
    subprocess.run(["/usr/bin/git", "init", "-q", str(path)], check=True)
    subprocess.run(["/usr/bin/git", "-C", str(path), "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "--allow-empty", "-qm", "fixture"], check=True)


def _as_foreign_identity(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run Git as nobody with the capability's exact hermetic process env."""
    nobody = pwd.getpwnam("nobody")
    if nobody.pw_uid == os.geteuid():
        pytest.skip("foreign identity is not distinct from test runner")
    sudo = shutil.which("sudo")
    if not sudo:
        pytest.skip("sudo is required for real cross-UID Git ownership regression")
    assignments = [f"{key}={value}" for key, value in env.items()]
    return subprocess.run([sudo, "-n", "-u", "nobody", "/usr/bin/env", "-i", *assignments,
                           "/usr/bin/git", "-C", str(root), "rev-parse", "--show-toplevel"],
                          text=True, capture_output=True)


@pytest.fixture
def foreign_owned_repo():
    # A world-searchable temporary parent lets nobody inspect files while the
    # repository itself remains owned by the test runner, a different UID.
    base = Path(tempfile.mkdtemp(prefix="jaxqwen-git-trust-", dir="/tmp"))
    base.chmod(0o755)
    repo = base / "authorized"
    repo.mkdir(mode=0o755)
    _repo(repo)
    try:
        yield repo, base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_real_dubious_ownership_failure_without_exception(foreign_owned_repo):
    repo, _base = foreign_owned_repo
    result = _as_foreign_identity(repo, capability._git_env())
    assert result.returncode != 0
    assert "detected dubious ownership" in result.stderr


def test_exact_authorized_checkout_is_accepted_process_locally(foreign_owned_repo):
    repo, _base = foreign_owned_repo
    env = capability._git_env(safe_directories=(repo,))
    result = _as_foreign_identity(repo, env)
    assert result.returncode == 0
    assert result.stdout.strip() == str(repo)
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert env["GIT_CONFIG_KEY_0"] == "safe.directory"
    assert env["GIT_CONFIG_VALUE_0"] == str(repo)


def test_exact_dispatcher_worktree_can_be_trusted_without_trusting_sibling(foreign_owned_repo):
    _repo_path, base = foreign_owned_repo
    dispatcher_worktree = base / "dispatcher-worktree"
    dispatcher_worktree.mkdir()
    _repo(dispatcher_worktree)
    sibling = base / "unrelated-repository"
    sibling.mkdir()
    _repo(sibling)
    trusted_env = capability._git_env(safe_directories=(dispatcher_worktree,))
    assert _as_foreign_identity(dispatcher_worktree, trusted_env).returncode == 0
    unrelated = _as_foreign_identity(sibling, trusted_env)
    assert unrelated.returncode != 0
    assert "detected dubious ownership" in unrelated.stderr


@pytest.mark.parametrize("path", [Path("relative/repo"), Path("/tmp/../tmp"), Path("*")])
def test_noncanonical_relative_traversal_and_wildcard_trust_roots_rejected(path):
    with pytest.raises(capability.CapabilityError, match="unsafe Git trust root"):
        capability._git_env(safe_directories=(path,))


def test_symlink_alias_is_rejected_and_request_cannot_supply_git_config():
    with tempfile.TemporaryDirectory(prefix="jaxqwen-git-link-") as raw:
        base = Path(raw)
        repo = base / "repo"; repo.mkdir()
        alias = base / "alias"; alias.symlink_to(repo, target_is_directory=True)
        with pytest.raises(capability.CapabilityError, match="unsafe Git trust root"):
            capability._git_env(safe_directories=(alias,))
    env = capability._git_env()
    assert not any("safe.directory" in value or value == "*" for key, value in env.items()
                   if key.startswith("GIT_CONFIG_VALUE_"))
    assert not any(key.startswith("GIT_CONFIG_KEY_") for key in env)
    assert env["GIT_CONFIG_COUNT"] == "0"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
