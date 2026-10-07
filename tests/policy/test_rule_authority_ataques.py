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


# ------------------------------------------------- ronda 3: ataques r2 del auditor

RECURSOS_PROHIBIDOS_R2 = [
    "subprocess.spawn", "parallel.calls", "concurrent.requests", "multithread.jobs",
    "sockets.abiertos", "conns.db", "tasks.systemd", "subagentes",
    "multiagente.lanzados", "pids", "jobs.cola", "fork.hijos",
    "sesiones.mcp", "subprocesos",
]


@pytest.mark.parametrize("clase,recurso", [
    ("actos_externos", "subprocess.spawn"), ("actos_externos", "parallel.calls"),
    ("frecuencia", "concurrent.requests"), ("actos_externos", "multithread.jobs"),
    ("actos_externos", "sockets.abiertos"), ("actos_externos", "conns.db"),
    ("frecuencia", "tasks.systemd"), ("actos_externos", "subagentes"),
    ("actos_externos", "multiagente.lanzados"), ("duracion", "pids"),
    ("actos_externos", "jobs.cola"), ("tokens_costo", "fork.hijos"),
    ("actos_externos", "sesiones.mcp"), ("actos_externos", "subprocesos"),
])
def test_r2_recurso_fabricado_o_trasladado_niega_en_schema(clase: str, recurso: str) -> None:
    datos = dict(load_strict_yaml(TOPE))
    datos["tope"] = {"resource_class": clase, "resource": recurso, "maximum": 2, "period": "hora"}
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


@pytest.mark.parametrize("recurso", RECURSOS_PROHIBIDOS_R2)
def test_r2_recurso_fabricado_niega_en_runtime(recurso: str) -> None:
    from jax.faro.catalogo_topes import es_recurso_de_catalogo
    from jax.faro.topes import es_recurso_sin_tope
    assert not es_recurso_de_catalogo(recurso)
    assert es_recurso_sin_tope(recurso)          # el runtime los deja sin tope (TopeProhibido)


def test_r2_el_catalogo_es_identico_en_json_python_y_runtime() -> None:
    import json
    from jax.faro.catalogo_topes import CATALOGO_TOPES, es_recurso_de_catalogo, recursos_del_catalogo
    espejo = json.loads((Path(__file__).resolve().parents[2] / "policy" / "faro"
                         / "schemas" / "rule-v1.schema.json").read_text())
    enum_json = espejo["properties"]["tope"]["properties"]["resource"]["enum"]
    assert sorted(enum_json) == sorted(recursos_del_catalogo())
    for recurso in recursos_del_catalogo():
        assert es_recurso_de_catalogo(recurso)
    for clase in CATALOGO_TOPES:
        for subid in CATALOGO_TOPES[clase]:
            assert es_recurso_de_catalogo(f"{clase}.{subid}")


def test_r2_copias_y_pickle_no_fabrican(tmp_path: Path) -> None:
    import copy
    import pickle
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    snap = _cargar(repo, commit, arbol)
    for operacion in (lambda: copy.copy(snap), lambda: copy.deepcopy(snap),
                      lambda: pickle.loads(pickle.dumps(snap)),
                      lambda: copy.copy(snap.reglas[0]),
                      lambda: pickle.loads(pickle.dumps(snap.reglas[0]))):
        with pytest.raises(RuleSnapshotError):
            operacion()


def test_r2_el_testigo_no_se_puede_leer_de_la_instancia(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    snap = _cargar(repo, commit, arbol)
    with pytest.raises(AttributeError):
        _ = snap.reglas[0]._testigo
    with pytest.raises(AttributeError):
        _ = snap._testigo


def test_r2_igualdad_por_valor_entre_recargas(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    a = _cargar(repo, commit, arbol)
    b = _cargar(repo, commit, arbol)
    assert a == b and hash(a) == hash(b)
    assert a.reglas[0] == b.reglas[0] and hash(a.reglas[0]) == hash(b.reglas[0])


def test_r2_pin_en_mayusculas_niega(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    with pytest.raises(RuleSnapshotError):
        TrustedPolicyPin("jax", commit.upper(), arbol, "p")


def test_r2_commit_distinto_con_mismo_arbol_policy_carga_ese_commit(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    (repo / "otro.txt").write_text("x")          # fuera de policy/: el arbol policy no cambia
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "y")
    c2 = _git(repo, "rev-parse", "HEAD")
    snap = _cargar(repo, c2, arbol)
    assert snap.commit == c2 and len(snap.reglas) == 1


@pytest.mark.parametrize("nombre", ["ejemplo．yaml", "ejemplo.yaml ", "ejemplo", "ejemplo.yаml"])
def test_r2_nombres_que_esquivan_la_vista_niegan(tmp_path: Path, nombre: str) -> None:
    repo = tmp_path / "r"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "policy" / "faro").mkdir(parents=True)
    (repo / "policy" / "faro" / nombre).write_bytes(REGLA)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD:policy"))


def test_r3_blob_gigante_con_nombre_de_regla_niega(tmp_path: Path) -> None:
    gigante = b"# relleno\n" * (1024 * 1024 // 10 + 1)     # > 1 MiB, nombre canonico
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/grande.yaml": gigante})
    with pytest.raises(RuleSnapshotError) as excinfo:
        _cargar(repo, commit, arbol)
    assert "excede" in str(excinfo.value)


def test_r3_las_dos_listas_de_identity_foundation_son_iguales() -> None:
    """La lista vive dos veces en policy.yml (paso principal y paso del piso):
    si divergen, el piso mide otra cosa que la que corre. Un solo archivo."""
    import re as _re
    import yaml as _yaml
    flujo = _yaml.safe_load((Path(__file__).resolve().parents[2] / ".github" / "workflows"
                             / "policy.yml").read_text())
    listas = []
    for job in flujo["jobs"].values():
        for paso in job.get("steps", []):
            run = paso.get("run", "")
            if "identity-foundation-shadow-pytest" in str(run) or \
               ("identity-foundation-shadow/policy" in str(run) and "tee" in str(run)):
                listas.append(sorted(_re.findall(r"(?:tests/policy|policy/enforcement_evidence)/[a-z_0-9]+\.py", run)))
    assert len(listas) == 2, f"se esperaban exactamente 2 pasos con la lista, hay {len(listas)}"
    assert listas[0] == listas[1], "las dos listas de Identity Foundation divergen"
