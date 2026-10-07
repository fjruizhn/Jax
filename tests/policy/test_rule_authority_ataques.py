"""Ataques A–J del auditor adversarial (jax#370 ronda 2) como regresiones.

Cada ataque se nego en la auditoria de la ronda 1 (`5e9b4b8a`) y quedo aqui
para que ninguna vuelta atras lo reabra en silencio.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import subprocess
import zlib
from pathlib import Path

import pytest

from policy.canonicalization.strict_yaml import load_strict_yaml
from policy.rule_authority.errors import RuleSchemaError, RuleSnapshotError
from policy.rule_authority.schema import validar_regla
from policy.rule_authority.snapshot import (
    TrustedPolicyPin,
    load_trusted_policy_snapshot,
)

FIXTURES = Path(__file__).parent / "fixtures" / "faro_rules"
REGLA = (FIXTURES / "regla-ejemplo.yaml").read_bytes()
TOPE = (FIXTURES / "regla-ejemplo-tope.yaml").read_bytes()


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout.decode().strip()


def _repo(tmp: Path, files: dict[str, bytes], nombre: str = "repo") -> tuple[Path, str, str]:
    repo = tmp / nombre
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    for ruta, contenido in files.items():
        destino = repo / ruta
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(contenido)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    return repo, _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD:policy")


def _cargar(repo: Path, commit: str, arbol: str):
    return load_trusted_policy_snapshot(
        repo, TrustedPolicyPin("jax", commit, arbol, "prueba:pin"))


# A: un nombre de rama con forma de hex de 64 no es un commit de un repo SHA-1
def test_ataque_a_pin_de_64_hex_resuelto_por_rama_movil_niega(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    falso = "ab" * 32
    _git(repo, "update-ref", f"refs/heads/{falso}", commit)
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, falso, arbol)


# B: el oid de un tag anotado no es el commit
def test_ataque_b_oid_de_tag_anotado_como_pin_niega(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "tag", "-a", "-m", "m", "v1", commit)
    tag = _git(repo, "rev-parse", "v1")
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, tag, arbol)


# C: dataclasses.replace no fabrica objetos sellados
def test_ataque_c_replace_no_conserva_el_testigo(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA,
                                           "policy/faro/ejemplo-tope.yaml": TOPE})
    snap = _cargar(repo, commit, arbol)
    r0, r1 = snap.reglas
    with pytest.raises(TypeError):
        dataclasses.replace(r1, regla=r0.regla)          # type: ignore[arg-type]
    with pytest.raises(TypeError):
        dataclasses.replace(snap, reglas=(r0,))          # type: ignore[arg-type]


# D: digitos arabigo-indicos en el timestamp
def test_ataque_d_digitos_unicode_en_timestamp_niega() -> None:
    datos = dict(load_strict_yaml(REGLA))
    datos["validity"]["not_before_utc"] = "٢٠٢٦-10-05T00:00:00Z"
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


# E/E2: topes de infraestructura, por clase y por nombre (decision de Fernando)
@pytest.mark.parametrize("clase,recurso", [
    ("connections", "puerto.connections"),
    ("concurrencia", "llm.paralelo"),
    ("workers", "pool.workers"),
    ("hilos", "hilos.maximo"),
    ("procesos_hijos", "procesos.hijos"),
])
def test_ataque_e_topes_de_infraestructura_niegan(clase: str, recurso: str) -> None:
    datos = dict(load_strict_yaml(TOPE))
    datos["tope"] = {"resource_class": clase, "resource": recurso, "maximum": 5, "period": "dia"}
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_ataque_e2_clase_concurrencia_con_recurso_inocente_niega() -> None:
    datos = dict(load_strict_yaml(TOPE))
    datos["tope"] = {"resource_class": "concurrencia", "resource": "mensajes.externos",
                     "maximum": 2, "period": "dia"}
    with pytest.raises(RuleSchemaError):                 # la clase no existe y punto
        validar_regla(datos)


# F: extensiones que aparentan regla no se ignoran en silencio
def test_ataque_f_yaml_mayusculas_y_cola_bak_niegan(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.YAML": REGLA,
                                           "policy/faro/x.yaml.bak": REGLA})
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, commit, arbol)


# F2: policy/faro como symlink
def test_ataque_f2_policy_faro_symlink_niega(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "policy").mkdir()
    (repo / "otra").mkdir()
    (repo / "otra" / "ejemplo.yaml").write_bytes(REGLA)
    os.symlink("../otra", repo / "policy" / "faro")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD:policy"))


# G: nombre de archivo no UTF-8 — error tipado, no UnicodeDecodeError
def test_ataque_g_nombre_no_utf8_da_error_tipado(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "policy" / "faro").mkdir(parents=True)
    (repo / "policy" / "faro" / "ejemplo.yaml").write_bytes(REGLA)
    with open(os.path.join(bytes(repo / "policy" / "faro"), b"\xff.txt"), "wb") as fh:
        fh.write(b"x")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD:policy"))


# H: el python tan estricto como el espejo JSON
def test_ataque_h_claves_ausentes_niegan() -> None:
    datos = dict(load_strict_yaml(REGLA))
    datos["obligation_limits"] = {}
    datos["validity"] = {"not_before_utc": "2026-10-05T00:00:00Z"}
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


# I: symlink cuyo destino SI es una regla valida
def test_ataque_i_symlink_a_regla_valida_niega(tmp_path: Path) -> None:
    destino = tmp_path / "regla.yaml"
    destino.write_bytes(REGLA)
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "policy" / "faro").mkdir(parents=True)
    os.symlink(str(destino), repo / "policy" / "faro" / "enlace.yaml")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD:policy"))


# J: objeto suelto adulterado — el OID se recalcula desde los bytes
def test_ataque_j_objeto_suelto_adulterado_niega(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    oid = _git(repo, "rev-parse", "HEAD:policy/faro/ejemplo.yaml")
    otro = REGLA.replace(b"ttl_seconds: 60", b"ttl_seconds: 61")
    p = repo / ".git" / "objects" / oid[:2] / oid[2:]
    os.chmod(p, 0o644)
    p.write_bytes(zlib.compress(b"blob %d\0" % len(otro) + otro))
    with pytest.raises(RuleSnapshotError) as excinfo:
        _cargar(repo, commit, arbol)
    assert "OID" in str(excinfo.value)
    # y el sha1 del contenido adulterado NO es el oid bajo el que se coló
    assert hashlib.sha1(b"blob %d\0" % len(otro) + otro).hexdigest() != oid
