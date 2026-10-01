"""Host IPC admission tests use real AF_UNIX peer credentials."""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
AUTHORITY = REPO / "projects/las-voces/authority"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


host_module = load("jaxqwen_host_test", AUTHORITY / "jaxqwen_host.py")


async def request(host, parent: Path, payload: dict):
    socket_path = parent / "dispatch.sock"
    server = await asyncio.start_unix_server(host._serve_client, path=str(socket_path))
    os.chmod(socket_path, 0o660)
    try:
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        writer.write(json.dumps(payload).encode() + b"\n")
        await writer.drain()
        response = json.loads(await asyncio.wait_for(reader.readline(), 2))
        writer.close()
        await writer.wait_closed()
        return response
    finally:
        server.close()
        await server.wait_closed()


def test_host_dispatch_requires_peer_uid_then_passes_only_ack_key(tmp_path):
    async def scenario():
        host = object.__new__(host_module.JaxQwenHost)
        host.dispatcher_uid = os.getuid()
        host.dispatch_enabled = True
        host.model_socket = tmp_path / "unused.sock"
        host.model_socket_uid, host.model_socket_gid = 10001, 10002
        host.model, host.max_output_tokens = "fixed-qwen", 32
        host.transport_module = SimpleNamespace(JaxQwenUnixModelTransport=lambda *a, **k: object())
        seen = []
        host._holder = {"capability": SimpleNamespace(start=lambda **kwargs: (seen.append(kwargs) or SimpleNamespace(decision="REJECTED", reason="synthetic fail closed", mission_id=None)))}
        response = await request(host, tmp_path, {"action": "dispatch", "idempotency_key": "a" * 64})
        assert response["decision"] == "REJECTED"
        assert seen == [{"idempotency_key": "a" * 64}]

    asyncio.run(scenario())


def test_host_does_not_accept_body_identity_or_uid_without_configured_dispatcher(tmp_path):
    async def scenario():
        host = object.__new__(host_module.JaxQwenHost)
        host.dispatcher_uid = os.getuid() + 1
        host.dispatch_enabled = True
        called = []
        host._holder = {"capability": SimpleNamespace(start=lambda **kwargs: called.append(kwargs))}
        response = await request(host, tmp_path, {"action": "dispatch", "idempotency_key": "b" * 64,
                                                  "service_identity": "jaxqwen", "actor": "jaxqwen"})
        assert response["decision"] == "REJECTED"
        assert called == []

    asyncio.run(scenario())


def test_disabled_first_dispatch_gate_never_calls_capability(tmp_path):
    async def scenario():
        host = object.__new__(host_module.JaxQwenHost)
        host.dispatcher_uid = os.getuid()
        host.dispatch_enabled = False
        called = []
        host._holder = {"capability": SimpleNamespace(start=lambda **kwargs: called.append(kwargs))}
        response = await request(host, tmp_path, {"action": "dispatch", "idempotency_key": "c" * 64})
        assert response["decision"] == "HUMAN_REQUIRED"
        assert called == []

    asyncio.run(scenario())


def _credential_directory_fd(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)


def _fake_systemd_file_owner(monkeypatch):
    """Model systemd's observed root:root metadata on an unprivileged test runner."""
    real_fstat = host_module._fstat_trust_credential

    def fstat(fd):
        st = real_fstat(fd)
        if stat.S_ISREG(st.st_mode):
            return SimpleNamespace(st_mode=st.st_mode, st_uid=0, st_gid=0,
                                   st_nlink=st.st_nlink, st_size=st.st_size,
                                   st_dev=st.st_dev, st_ino=st.st_ino)
        return st

    monkeypatch.setattr(host_module, "_fstat_trust_credential", fstat)


def _fake_systemd_service_file_owner(monkeypatch, *, uid=None, gid=None):
    """Model systemd's per-service 0400 credential metadata in an unprivileged test."""
    real_fstat = host_module._fstat_trust_credential

    def fstat(fd):
        st = real_fstat(fd)
        if stat.S_ISREG(st.st_mode):
            return SimpleNamespace(st_mode=st.st_mode, st_uid=os.getuid() if uid is None else uid,
                                   st_gid=os.getgid() if gid is None else gid,
                                   st_nlink=st.st_nlink, st_size=st.st_size,
                                   st_dev=st.st_dev, st_ino=st.st_ino)
        return st

    monkeypatch.setattr(host_module, "_fstat_trust_credential", fstat)


def _fake_read_only_credential_mount(monkeypatch):
    real_fstatvfs = os.fstatvfs
    monkeypatch.setattr(host_module.os, "fstatvfs", lambda fd: SimpleNamespace(
        f_flag=real_fstatvfs(fd).f_flag | os.ST_RDONLY))


def test_loadcredential_root_owned_0440_regular_file_is_accepted(tmp_path, monkeypatch):
    secret = b"s" * 32
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir(mode=0o750)
    credential = credential_dir / host_module._TRUST_CREDENTIAL_NAME
    credential.write_bytes(secret)
    credential.chmod(0o440)

    if os.geteuid() != 0:
        _fake_systemd_file_owner(monkeypatch)
    else:
        os.chown(credential, 0, 0)

    fd = _credential_directory_fd(credential_dir)
    try:
        assert host_module._read_trust_credential(fd) == secret
    finally:
        os.close(fd)


def test_systemd_per_service_owned_0400_credential_is_accepted(tmp_path, monkeypatch):
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir(mode=0o750)
    credential = credential_dir / host_module._TRUST_CREDENTIAL_NAME
    credential.write_bytes(b"s" * 32)
    credential.chmod(0o400)
    _fake_systemd_service_file_owner(monkeypatch)
    _fake_read_only_credential_mount(monkeypatch)
    fd = _credential_directory_fd(credential_dir)
    try:
        assert host_module._read_trust_credential(fd) == b"s" * 32
    finally:
        os.close(fd)


def test_service_uid_does_not_trust_arbitrary_owner_or_insecure_mode(tmp_path, monkeypatch):
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir(mode=0o750)
    credential = credential_dir / host_module._TRUST_CREDENTIAL_NAME
    credential.write_bytes(b"s" * 32)
    credential.chmod(0o400)
    _fake_systemd_service_file_owner(monkeypatch, uid=os.getuid() + 10000)
    _fake_read_only_credential_mount(monkeypatch)
    fd = _credential_directory_fd(credential_dir)
    try:
        with pytest.raises(ValueError, match="untrusted systemd trust credential"):
            host_module._read_trust_credential(fd)
    finally:
        os.close(fd)

    # A service UID in the right place still cannot authorize group-readable data.
    st = SimpleNamespace(st_mode=stat.S_IFREG | 0o440, st_uid=10000, st_gid=0, st_nlink=1)
    with pytest.raises(ValueError, match="untrusted systemd trust credential"):
        host_module._validate_trust_credential(
            st, service_uid=10000, credential_mount_read_only=True)


def test_service_owned_credential_on_writable_filesystem_is_rejected(tmp_path, monkeypatch):
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir(mode=0o750)
    credential = credential_dir / host_module._TRUST_CREDENTIAL_NAME
    credential.write_bytes(b"s" * 32)
    credential.chmod(0o400)
    _fake_systemd_service_file_owner(monkeypatch)
    fd = _credential_directory_fd(credential_dir)
    try:
        with pytest.raises(ValueError, match="untrusted systemd trust credential"):
            host_module._read_trust_credential(fd)
    finally:
        os.close(fd)


def test_jaxqwen_owned_arbitrary_credential_is_not_trusted(tmp_path, monkeypatch):
    """UID authentication remains insufficient even when the runner is root."""
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir(mode=0o750)
    credential = credential_dir / host_module._TRUST_CREDENTIAL_NAME
    credential.write_bytes(b"s" * 32)
    credential.chmod(0o440)
    real_fstat = host_module._fstat_trust_credential

    def untrusted_owner(fd):
        st = real_fstat(fd)
        if stat.S_ISREG(st.st_mode):
            return SimpleNamespace(st_mode=st.st_mode, st_uid=10001, st_gid=10001,
                                   st_nlink=st.st_nlink, st_size=st.st_size,
                                   st_dev=st.st_dev, st_ino=st.st_ino)
        return st

    monkeypatch.setattr(host_module, "_fstat_trust_credential", untrusted_owner)
    fd = _credential_directory_fd(credential_dir)
    try:
        with pytest.raises(ValueError, match="untrusted systemd trust credential"):
            host_module._read_trust_credential(fd)
    finally:
        os.close(fd)


def test_credential_directory_rejects_group_or_world_write():
    for mode, uid in ((0o770, 0), (0o757, 0), (0o750, os.getuid())):
        st = SimpleNamespace(st_mode=stat.S_IFDIR | mode, st_uid=uid)
        with pytest.raises(ValueError, match="untrusted systemd credential directory"):
            host_module._validate_credential_directory(st)


def test_credential_file_rejects_unsafe_modes(tmp_path, monkeypatch):
    for mode in (0o466, 0o460, 0o660, 0o444, 0o400):
        credential_dir = tmp_path / f"credentials-{mode:o}"
        credential_dir.mkdir(mode=0o750)
        credential = credential_dir / host_module._TRUST_CREDENTIAL_NAME
        credential.write_bytes(b"s" * 32)
        credential.chmod(mode)
        _fake_systemd_file_owner(monkeypatch)
        fd = _credential_directory_fd(credential_dir)
        try:
            with pytest.raises(ValueError, match="untrusted systemd trust credential"):
                host_module._read_trust_credential(fd)
        finally:
            os.close(fd)


def test_credential_file_rejects_untrusted_group(tmp_path, monkeypatch):
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir(mode=0o750)
    credential = credential_dir / host_module._TRUST_CREDENTIAL_NAME
    credential.write_bytes(b"s" * 32)
    credential.chmod(0o440)
    real_fstat = os.fstat

    def wrong_group(fd):
        st = real_fstat(fd)
        if stat.S_ISREG(st.st_mode):
            return SimpleNamespace(st_mode=st.st_mode, st_uid=0, st_gid=os.getgid() or 1,
                                   st_nlink=st.st_nlink, st_size=st.st_size,
                                   st_dev=st.st_dev, st_ino=st.st_ino)
        return st

    monkeypatch.setattr(host_module, "_fstat_trust_credential", wrong_group)
    fd = _credential_directory_fd(credential_dir)
    try:
        with pytest.raises(ValueError, match="untrusted systemd trust credential"):
            host_module._read_trust_credential(fd)
    finally:
        os.close(fd)


def test_credential_file_rejects_hardlinks_and_metadata_change_during_read(tmp_path, monkeypatch):
    _fake_systemd_file_owner(monkeypatch)
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir(mode=0o750)
    credential = credential_dir / host_module._TRUST_CREDENTIAL_NAME
    credential.write_bytes(b"s" * 32)
    credential.chmod(0o440)
    os.link(credential, tmp_path / "second-link")
    fd = _credential_directory_fd(credential_dir)
    try:
        with pytest.raises(ValueError, match="untrusted systemd trust credential"):
            host_module._read_trust_credential(fd)
    finally:
        os.close(fd)

    (tmp_path / "second-link").unlink()
    real_fstat = host_module._fstat_trust_credential
    calls = 0

    def changed_metadata(file_fd):
        nonlocal calls
        calls += 1
        st = real_fstat(file_fd)
        return SimpleNamespace(st_mode=st.st_mode, st_uid=0, st_gid=0,
                               st_nlink=st.st_nlink, st_size=st.st_size + (calls == 2),
                               st_dev=st.st_dev, st_ino=st.st_ino)

    monkeypatch.setattr(host_module, "_fstat_trust_credential", changed_metadata)
    fd = _credential_directory_fd(credential_dir)
    try:
        with pytest.raises(ValueError, match="changed while reading"):
            host_module._read_trust_credential(fd)
    finally:
        os.close(fd)


def test_credential_file_rejects_symlink_directory_fifo_and_missing(tmp_path, monkeypatch):
    _fake_systemd_file_owner(monkeypatch)
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir(mode=0o750)
    target = tmp_path / "secret-target"
    target.write_bytes(b"s" * 32)
    target.chmod(0o440)
    link = credential_dir / host_module._TRUST_CREDENTIAL_NAME
    link.symlink_to(target)
    fd = _credential_directory_fd(credential_dir)
    try:
        with pytest.raises(OSError):
            host_module._read_trust_credential(fd)
    finally:
        os.close(fd)

    link.unlink()
    link.mkdir()
    fd = _credential_directory_fd(credential_dir)
    try:
        with pytest.raises(ValueError, match="untrusted systemd trust credential"):
            host_module._read_trust_credential(fd)
    finally:
        os.close(fd)

    link.rmdir()
    os.mkfifo(link)
    fd = _credential_directory_fd(credential_dir)
    try:
        with pytest.raises(ValueError, match="untrusted systemd trust credential"):
            host_module._read_trust_credential(fd)
    finally:
        os.close(fd)

    link.unlink()
    fd = _credential_directory_fd(credential_dir)
    try:
        with pytest.raises(FileNotFoundError):
            host_module._read_trust_credential(fd)
    finally:
        os.close(fd)


def test_credential_contents_must_be_nonempty_well_sized_and_are_never_logged(tmp_path, monkeypatch, caplog):
    _fake_systemd_file_owner(monkeypatch)
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir(mode=0o750)
    credential = credential_dir / host_module._TRUST_CREDENTIAL_NAME
    for secret in (b"", b"s" * 31, b"s" * (host_module._MAX_TRUST_KEY_BYTES + 1)):
        if credential.exists():
            credential.chmod(0o600)
        credential.write_bytes(secret)
        credential.chmod(0o440)
        fd = _credential_directory_fd(credential_dir)
        try:
            with pytest.raises(ValueError, match="invalid systemd trust credential length"):
                host_module._read_trust_credential(fd)
        finally:
            os.close(fd)
    assert "s" * 32 not in caplog.text


def test_accepted_credential_contents_are_never_logged(tmp_path, monkeypatch, caplog):
    secret = b"accepted-secret-must-not-appear-123"
    _fake_systemd_file_owner(monkeypatch)
    credential_dir = tmp_path / "credentials"
    credential_dir.mkdir(mode=0o750)
    credential = credential_dir / host_module._TRUST_CREDENTIAL_NAME
    credential.write_bytes(secret)
    credential.chmod(0o440)
    fd = _credential_directory_fd(credential_dir)
    try:
        assert host_module._read_trust_credential(fd) == secret
    finally:
        os.close(fd)
    assert secret.decode() not in caplog.text


def test_real_filesystem_writable_credential_directory_and_symlink_are_rejected(tmp_path, monkeypatch):
    """Walk real path components and reject writable or symlink parents."""
    root = tmp_path / "root"
    credentials = root / "run/credentials"
    credentials.mkdir(parents=True)
    service_dir = credentials / "jaxqwen.service"
    service_dir.mkdir()
    target = root / "target"
    target.mkdir()
    root.chmod(0o755)
    (root / "run").chmod(0o755)
    credentials.chmod(0o755)
    service_dir.chmod(0o700)

    # Preserve the real mode bits while modeling root ownership on a non-root runner.
    validate_directory = host_module._validate_credential_directory

    def root_owned_directory(st):
        if st.st_uid == 0:
            return validate_directory(st)
        return validate_directory(SimpleNamespace(st_mode=st.st_mode, st_uid=0))

    monkeypatch.setattr(host_module, "_validate_credential_directory", root_owned_directory)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        (root / "run").chmod(0o770)
        with pytest.raises(ValueError, match="untrusted systemd credential directory"):
            host_module._open_systemd_credential_directory(root_fd)

        (root / "run").chmod(0o755)
        (credentials / "jaxqwen.service").rmdir()
        (credentials / "jaxqwen.service").symlink_to(target)
        with pytest.raises(OSError):
            host_module._open_systemd_credential_directory(root_fd)

        (credentials / "jaxqwen.service").unlink()
        (credentials / "other.service").mkdir()
        with pytest.raises(FileNotFoundError):
            host_module._open_systemd_credential_directory(root_fd)
    finally:
        os.close(root_fd)


@pytest.mark.parametrize("directory", [
    "/tmp/attacker-credentials",
    "/run/credentials/jaxqwen.service/../other.service",
    "/run/credentials/other.service",
    "/run/credentials/jaxqwen.service/",
])
def test_production_accessor_rejects_caller_selected_credential_directory(directory, monkeypatch):
    monkeypatch.setenv("CREDENTIALS_DIRECTORY", directory)
    with pytest.raises(ValueError, match="systemd trust credential directory is unavailable"):
        host_module._systemd_trust_key()


def test_environment_secret_fallback_is_not_used(monkeypatch):
    monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
    monkeypatch.setenv("JAXQWEN_TRUST_KEY", "would-be-secret")
    with pytest.raises(ValueError, match="systemd trust credential directory is unavailable"):
        host_module._systemd_trust_key()


def test_systemd_credential_config_cannot_be_overridden_by_host_json(tmp_path):
    script = load("jaxqwen_host_entry_test", REPO / "scripts/jaxqwen_host.py")
    value = {name: "/tmp/x" for name in script._FIELDS if name in {
        "root", "canonical_root", "handoff_state_dir", "source_worktree_root", "workspace_root",
        "mission_state_dir", "trust_state_dir", "dispatch_socket", "model_socket"}}
    value.update({"dispatcher_uid": 11, "dispatch_gid": 12, "model_socket_uid": 13,
                  "model_socket_gid": 14, "model": "fixed", "max_output_tokens": 32,
                  "dispatch_enabled": False, "trust_key_file": str(tmp_path / "attacker.key")})
    config = tmp_path / "host.json"
    config.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="schema mismatch"):
        script._config(config)
