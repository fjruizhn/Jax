"""Contrato del publicador root-owned del paquete permanente de LAS VOCES."""
from __future__ import annotations

import importlib.util
import os
import stat
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops/las-voces/faro_paquete_permanente.py"


def modulo():
    spec = importlib.util.spec_from_file_location("faro_paquete_permanente", SCRIPT)
    assert spec and spec.loader
    subject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subject)
    return subject


def test_production_locations_and_environment_are_closed_and_exact():
    subject = modulo()
    assert subject.MIRROR == Path("/srv/faro/claude-skills.git")
    assert subject.DESTINO == Path("/srv/faro/ecosistema")
    assert subject.ENV_FILE == Path("/etc/jax/las-voces-faro.env")
    assert subject.REF_FRESCURA == "refs/heads/main"
    assert subject._env_bytes("a" * 40) == (
        b"JAX_FARO_REPO=/srv/faro/claude-skills.git\n"
        b"JAX_FARO_SHA=" + b"a" * 40 + b"\n"
        b"JAX_FARO_ECOSISTEMA_DIR=/srv/faro/ecosistema\n"
        b"JAX_FARO_REF_FRESCURA=refs/heads/main\n"
    )


def test_official_oid_rejects_a_non_sha_response(monkeypatch):
    subject = modulo()
    monkeypatch.setattr(subject, "_run", lambda argv: "main\n" if argv[-1] == ".default_branch" else "wrong\n")
    with pytest.raises(subject.PublicacionRechazada, match="SHA oficial"):
        subject._oid_oficial()


def test_command_environment_uses_root_home_not_cwd_or_inherited_git(monkeypatch):
    subject = modulo()
    monkeypatch.setenv("HOME", "/tmp/usuario")
    monkeypatch.setenv("GIT_DIR", "/tmp/repo-hostil")
    vistos = []
    def fake_run(argv, **kwargs):
        vistos.append(kwargs["env"])
        return SimpleNamespace(stdout="main\n")
    monkeypatch.setattr(subject.subprocess, "run", fake_run)
    assert subject._run(["gh", "api", "repos/fjruizhn/claude-skills"]) == "main\n"
    assert vistos[0]["HOME"] == "/root"
    assert vistos[0]["XDG_CONFIG_HOME"] == "/root/.config"
    assert vistos[0]["PATH"] == "/usr/bin:/bin"
    assert "GIT_DIR" not in vistos[0]


def test_no_replace_refs_or_unsafe_mirror_are_accepted(monkeypatch, tmp_path):
    subject = modulo()
    mirror = tmp_path / "claude-skills.git"
    mirror.mkdir()
    monkeypatch.setattr(subject, "MIRROR", mirror)
    monkeypatch.setattr(subject, "_ruta_root_segura", lambda *_, **__: None)
    monkeypatch.setattr(subject, "_run", lambda argv, **_: "deadbeef\n" if argv[-2:] == ["replace", "-l"] else "true\n")
    with pytest.raises(subject.PublicacionRechazada, match="replace"):
        subject._validar_mirror()


def test_atomic_environment_keeps_old_file_when_preparation_fails(monkeypatch, tmp_path):
    subject = modulo()
    env_file = tmp_path / "las-voces-faro.env"
    env_file.write_text("old\n")
    monkeypatch.setattr(subject, "ENV_FILE", env_file)
    monkeypatch.setattr(subject, "_crear_directorio_root", lambda _: None)
    monkeypatch.setattr(subject, "_ruta_root_segura", lambda *_, **__: None)
    monkeypatch.setattr(subject.os, "fchown", lambda *_: (_ for _ in ()).throw(OSError("no chown")))
    with pytest.raises(OSError, match="no chown"):
        subject._publicar_env("a" * 40)
    assert env_file.read_text() == "old\n"
    assert list(tmp_path.glob(".las-voces-faro.env.*")) == []


def test_check_requires_exact_env_and_a_root_owned_verified_package(monkeypatch, tmp_path):
    subject = modulo()
    env_file = tmp_path / "las-voces-faro.env"
    env_file.write_bytes(subject._env_bytes("a" * 40))
    env_file.chmod(0o644)
    monkeypatch.setattr(subject, "ENV_FILE", env_file)
    monkeypatch.setattr(subject, "_ruta_root_segura", lambda *_, **__: None)
    monkeypatch.setattr(subject, "ConfigFaro", lambda **kwargs: type("Cfg", (), kwargs)())
    calls: list[object] = []
    monkeypatch.setattr(subject, "cargar_paquete", lambda cfg: calls.append(cfg))
    assert subject.comprobar_publicacion() is None
    assert calls and calls[0].uid_duenio == 0
    env_file.write_text("JAX_FARO_REPO=/srv/faro/claude-skills.git\n")
    with pytest.raises(subject.PublicacionRechazada, match="exactamente"):
        subject.comprobar_publicacion()


def test_publish_keeps_previous_env_when_tree_verification_fails(monkeypatch):
    subject = modulo()
    calls: list[str] = []
    monkeypatch.setattr(subject, "_exigir_root_y_codigo", lambda: None)
    monkeypatch.setattr(subject, "_bloqueo_publicacion", nullcontext)
    monkeypatch.setattr(subject, "_oid_oficial", lambda: "a" * 40)
    monkeypatch.setattr(subject, "_preparar_mirror", lambda _: calls.append("mirror"))
    monkeypatch.setattr(subject, "ConfigFaro", lambda **_: object())
    monkeypatch.setattr(subject, "construir_paquete", lambda _: calls.append("construir"))
    monkeypatch.setattr(subject, "verificar_contra_arbol", lambda _: (type("F", (), {"codigo": "alterado"})(),))
    monkeypatch.setattr(subject, "_publicar_env", lambda _: calls.append("env"))
    with pytest.raises(subject.PublicacionRechazada, match="verificar_contra_arbol"):
        subject.publicar()
    assert calls == ["mirror", "construir"]


def test_publish_success_orders_construct_verify_load_then_atomic_env(monkeypatch):
    subject = modulo()
    calls: list[str] = []
    monkeypatch.setattr(subject, "_exigir_root_y_codigo", lambda: None)
    monkeypatch.setattr(subject, "_bloqueo_publicacion", nullcontext)
    monkeypatch.setattr(subject, "_oid_oficial", lambda: "b" * 40)
    monkeypatch.setattr(subject, "_preparar_mirror", lambda _: calls.append("mirror"))
    monkeypatch.setattr(subject, "ConfigFaro", lambda **_: object())
    monkeypatch.setattr(subject, "construir_paquete", lambda _: calls.append("construir"))
    monkeypatch.setattr(subject, "verificar_contra_arbol", lambda _: calls.append("verificar") or ())
    monkeypatch.setattr(subject, "cargar_paquete", lambda _: calls.append("cargar"))
    monkeypatch.setattr(subject, "_cargar_como_fruiz", lambda _: calls.append("cargar_fruiz"))
    monkeypatch.setattr(subject, "_publicar_env", lambda _: calls.append("rename"))
    monkeypatch.setattr(subject, "comprobar_publicacion", lambda: calls.append("check"))
    assert subject.publicar() == "b" * 40
    assert calls == ["mirror", "construir", "verificar", "cargar", "cargar_fruiz", "rename", "check"]


def test_publish_sanitizes_path_before_imported_faro_git(monkeypatch):
    subject = modulo()
    monkeypatch.setenv("PATH", "/tmp/controlado:/usr/bin")
    monkeypatch.setattr(subject, "_exigir_root_y_codigo", lambda: None)
    monkeypatch.setattr(subject, "_bloqueo_publicacion", nullcontext)
    monkeypatch.setattr(subject, "_oid_oficial", lambda: "a" * 40)
    monkeypatch.setattr(subject, "_preparar_mirror", lambda _: None)
    monkeypatch.setattr(subject, "ConfigFaro", lambda **_: object())
    monkeypatch.setattr(subject, "construir_paquete", lambda _: os.environ["PATH"] == "/usr/bin:/bin" or pytest.fail("PATH heredado"))
    monkeypatch.setattr(subject, "verificar_contra_arbol", lambda _: ())
    monkeypatch.setattr(subject, "cargar_paquete", lambda _: None)
    monkeypatch.setattr(subject, "_cargar_como_fruiz", lambda _: None)
    monkeypatch.setattr(subject, "_publicar_env", lambda _: None)
    monkeypatch.setattr(subject, "comprobar_publicacion", lambda: None)
    assert subject.publicar() == "a" * 40


def test_directory_mode_is_0755_even_with_restrictive_umask(monkeypatch, tmp_path):
    subject = modulo()
    monkeypatch.setattr(subject, "_ruta_root_segura", lambda *_, **__: None)
    anterior = os.umask(0o077)
    try:
        subject._crear_directorio_root(tmp_path / "faro")
    finally:
        os.umask(anterior)
    assert (tmp_path / "faro").stat().st_mode & 0o777 == 0o755


def test_lock_rejects_concurrent_publisher(monkeypatch, tmp_path):
    subject = modulo()
    monkeypatch.setattr(subject, "LOCK_FILE", tmp_path / "lock")
    monkeypatch.setattr(subject, "_crear_directorio_root", lambda _: None)
    monkeypatch.setattr(subject.os, "fstat", lambda _: SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=0, st_gid=0))
    with subject._bloqueo_publicacion():
        with pytest.raises(subject.PublicacionRechazada, match="otra publicación"):
            with subject._bloqueo_publicacion():
                pytest.fail("entró una segunda renovación")


def test_post_rename_failure_reports_indeterminate_without_false_rollback(monkeypatch, tmp_path):
    subject = modulo()
    env_file = tmp_path / "las-voces-faro.env"
    env_file.write_text("old\n")
    monkeypatch.setattr(subject, "ENV_FILE", env_file)
    monkeypatch.setattr(subject, "_crear_directorio_root", lambda _: None)
    monkeypatch.setattr(subject, "_ruta_root_segura", lambda *_, **__: None)
    monkeypatch.setattr(subject.os, "fchown", lambda *_: None)
    llamadas: list[int] = []
    def fsync_segundo(fd: int) -> None:
        llamadas.append(fd)
        if len(llamadas) == 2:
            raise OSError("fsync dir failed")
    monkeypatch.setattr(subject.os, "fsync", fsync_segundo)
    with pytest.raises(subject.PublicacionIndeterminada, match="reemplazado"):
        subject._publicar_env("a" * 40)
    assert env_file.read_bytes() == subject._env_bytes("a" * 40)


def test_mirror_oid_mismatch_fails_before_package_construction(monkeypatch, tmp_path):
    subject = modulo()
    mirror = tmp_path / "claude-skills.git"
    mirror.mkdir()
    monkeypatch.setattr(subject, "MIRROR", mirror)
    monkeypatch.setattr(subject, "_crear_directorio_root", lambda _: None)
    monkeypatch.setattr(subject, "_validar_mirror", lambda: None)
    monkeypatch.setattr(subject, "_run", lambda argv, **_: "c" * 40 + "\n" if argv[-2:] == ["rev-parse", subject.REF_FRESCURA] else "")
    with pytest.raises(subject.PublicacionRechazada, match="no coincide"):
        subject._preparar_mirror("d" * 40)


def test_check_cli_also_requires_root_and_trusted_code(monkeypatch):
    subject = modulo()
    monkeypatch.setattr(subject, "_exigir_root_y_codigo",
                        lambda: (_ for _ in ()).throw(subject.PublicacionRechazada("root requerido")))
    monkeypatch.setattr(subject, "comprobar_publicacion", lambda: pytest.fail("no debe comprobar sin root"))
    assert subject.main(["--check"]) == 2
