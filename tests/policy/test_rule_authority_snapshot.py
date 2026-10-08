"""Snapshot confiable de policy/faro (diseno F1.1 §6, pruebas §13 «Bytes y snapshot»).

Todas negativas de construccion: cualquier byte, cualquier desviacion, cualquier
fuente no confiable niega. Los repos se construyen efimeros con git real en
``tmp_path``; las reglas vienen SOLO de los fixtures de tests/policy/fixtures/faro_rules/.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import unicodedata
from pathlib import Path

import pytest

from jax.faro.git_objetos import FuenteInvalida
from policy.rule_authority.errors import RuleSnapshotError
from policy.rule_authority.snapshot import (
    ReglaSellada,
    TrustedPolicyPin,
    TrustedPolicySnapshot,
    load_trusted_policy_snapshot,
)

FIXTURES = Path(__file__).parent / "fixtures" / "faro_rules"
REGLA = (FIXTURES / "regla-ejemplo.yaml").read_bytes()
REGLA_TOPE = (FIXTURES / "regla-ejemplo-tope.yaml").read_bytes()

# El catalogo de la decision de Fernando (B-3: en el arbol del pin, no en disco)
import json as _json
CATALOGO = _json.dumps({"version": 1,
                        "decision": "Fernando 2026-10-06: solo actos y dinero",
                        "clases": {"monto_dinero": ["hnl", "usd"],
                                   "actos_externos": ["mensajes", "correos", "publicaciones",
                                                      "compras", "pagos"],
                                   "frecuencia": ["por_hora", "por_dia"],
                                   "duracion": ["segundos"],
                                   "tokens_costo": ["tokens", "usd"]}}).encode()
RUTA_CATALOGO = "policy/faro/catalogo-topes.json"

_BASE = (b"kind: JAX_FARO_RULE\neffect: PERMIT\naction_class: REVERSIBLE\n"
         b"scope:\n  subjects: [actor:ejemplo]\n  capabilities: [CAPABILITY_ID]\n"
         b"  objectives: [objetivo-ejemplo]\nobligation_limits:\n  quantity: null\n"
         b"  amount: null\n  frequency: null\nvalidity:\n"
         b'  not_before_utc: "2026-10-05T00:00:00Z"\n  not_after_utc: null\n'
         b"permit:\n  ttl_seconds: 60\n")


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout.decode()


def _repo(tmp: Path, archivos: dict[str, bytes], nombre: str = "repo",
          con_catalogo: bool = True) -> tuple[Path, str, str]:
    """Repo efimero con ``archivos`` cometidos: devuelve (repo, commit, policy_tree_oid).
    El catalogo de topes va SIEMPRE (B-3: es parte del pin); quien quiera probar
    su ausencia pasa ``con_catalogo=False``."""
    repo = tmp / nombre
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    if con_catalogo and "policy/faro" not in " ".join(archivos) :
        archivos = dict(archivos)
    if con_catalogo and RUTA_CATALOGO not in archivos:
        archivos = {RUTA_CATALOGO: CATALOGO, **archivos}
    for ruta, contenido in archivos.items():
        destino = repo / ruta
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(contenido)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "reglas")
    commit = _git(repo, "rev-parse", "HEAD").strip()
    arbol = _git(repo, "rev-parse", "HEAD:policy").strip()
    return repo, commit, arbol


def _cargar(repo: Path, commit: str, arbol: str, *, procedencia: str = "prueba:pin") -> TrustedPolicySnapshot:
    pin = TrustedPolicyPin(repositorio="jax", commit=commit, policy_tree_oid=arbol,
                           procedencia=procedencia)
    return load_trusted_policy_snapshot(repo, pin)


# ------------------------------------------------------------------- carga sana

def test_carga_las_reglas_con_hash_de_bytes_crudos(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA,
                                           "policy/faro/ejemplo-tope.yaml": REGLA_TOPE})
    snap = _cargar(repo, commit, arbol)
    assert [r.ruta for r in snap.reglas] == ["policy/faro/ejemplo-tope.yaml",
                                             "policy/faro/ejemplo.yaml"]      # ordenadas por ruta
    esperado = "sha256:" + hashlib.sha256(REGLA).hexdigest()
    assert snap.reglas[1].content_hash == esperado                            # hash de BYTES, no de YAML
    assert snap.reglas[1].regla.rule_id == "ejemplo-regla"


def test_policy_faro_solo_con_catalogo_es_snapshot_valido_vacio(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {})                 # solo el catalogo
    snap = _cargar(repo, commit, arbol)
    assert snap.reglas == ()
    assert snap.catalogo_hash.startswith("sha256:")


def test_r5_falta_el_catalogo_en_el_pin_niega(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA},
                                con_catalogo=False)
    with pytest.raises(RuleSnapshotError) as excinfo:
        _cargar(repo, commit, arbol)
    assert "catalogo" in str(excinfo.value)


def test_readme_y_esquema_json_no_son_reglas_y_no_estorban(tmp_path: Path) -> None:
    archivos = {"policy/faro/ejemplo.yaml": REGLA,
                "policy/faro/README.md": b"# doc\n",
                "policy/faro/schemas/rule-v1.schema.json": b'{"x": 1}'}
    repo, commit, arbol = _repo(tmp_path, archivos)
    snap = _cargar(repo, commit, arbol)
    assert [r.ruta for r in snap.reglas] == ["policy/faro/ejemplo.yaml"]


# ------------------------------------------------------------ bytes exactos §13

@pytest.mark.parametrize("mutacion", [
    lambda b: b + b" ",                                  # espacio al final
    lambda b: b + b"# comentario\n",                     # comentario
    lambda b: b + b"\n",                                 # newline extra
    lambda b: b.replace(b"\n", b"\r\n"),                 # LF -> CRLF
    lambda b: b.replace(b"ttl_seconds: 60", b"ttl_seconds: 60 "),  # espacio tras el escalar
])
def test_cualquier_cambio_de_byte_cambia_el_hash(tmp_path: Path, mutacion) -> None:
    original = _repo(tmp_path / "a", {"policy/faro/ejemplo.yaml": REGLA})
    mutado = _repo(tmp_path / "b", {"policy/faro/ejemplo.yaml": mutacion(REGLA)})
    hash_original = _cargar(*original).reglas[0].content_hash
    hash_mutado = _cargar(*mutado).reglas[0].content_hash
    assert hash_original != hash_mutado


def test_un_identificador_en_nfd_niega_el_snapshot_nunca_se_confunde(tmp_path: Path) -> None:
    # §7 exige identificadores NFC y ASCII por patron: los bytes NFD no cargan,
    # asi que jamas pueden colisionar con la forma NFC de la misma regla.
    nfd = REGLA.replace(b"ejemplo-regla", unicodedata.normalize("NFD", "ejemplo-reglá").encode())
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": nfd})
    with pytest.raises(RuleSnapshotError) as excinfo:
        _cargar(repo, commit, arbol)
    assert "regla invalida" in str(excinfo.value)     # muere por identificador, no por otra cosa


def test_nfd_en_un_comentario_cambia_el_hash_igual_que_cualquier_byte(tmp_path: Path) -> None:
    nfc = REGLA + b"# comentario con a\n"
    nfd = REGLA + unicodedata.normalize("NFD", "# comentario con á\n").encode()
    a = _repo(tmp_path / "a", {"policy/faro/ejemplo.yaml": nfc})
    b = _repo(tmp_path / "b", {"policy/faro/ejemplo.yaml": nfd})
    assert _cargar(*a).reglas[0].content_hash != _cargar(*b).reglas[0].content_hash


def test_el_yaml_canonizado_nunca_produce_el_hash_de_los_bytes(tmp_path: Path) -> None:
    # Misma semantica minima, otra serializacion: NUNCA igual al hash de los bytes crudos.
    regla_minima = _BASE + b'rule_id: a-regla\nschema_version: "1.0"\n'
    reordenada = _BASE + b'schema_version: "1.0"\nrule_id: a-regla\n'
    a = _repo(tmp_path / "a", {"policy/faro/a.yaml": regla_minima})
    b = _repo(tmp_path / "b", {"policy/faro/a.yaml": reordenada})
    hash_a = _cargar(*a).reglas[0].content_hash
    hash_b = _cargar(*b).reglas[0].content_hash
    assert hash_a != hash_b
    assert hash_a == "sha256:" + hashlib.sha256(regla_minima).hexdigest()


def test_reordenar_claves_yaml_cambia_el_hash(tmp_path: Path) -> None:
    textos = [
        _BASE + b'rule_id: a-regla\nschema_version: "1.0"\n',
        _BASE + b'schema_version: "1.0"\nrule_id: a-regla\n',
    ]
    hashes = set()
    for i, texto in enumerate(textos):
        repo, commit, arbol = _repo(tmp_path / f"r{i}", {f"policy/faro/orden.yaml": texto})
        hashes.add(_cargar(repo, commit, arbol).reglas[0].content_hash)
    assert len(hashes) == 2


# ------------------------------------------------------------------- el pin

def test_pin_con_arbol_que_no_coincide_rechaza(tmp_path: Path) -> None:
    repo, commit, _ = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    (repo / "policy" / "nuevo.yaml").write_bytes(b"x: 1\n")     # cambia el arbol policy/
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "otro")
    otro_arbol = _git(repo, "rev-parse", "HEAD:policy").strip()
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, commit, otro_arbol)


def test_pin_con_commit_inexistente_rechaza(tmp_path: Path) -> None:
    repo, _, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    with pytest.raises((RuleSnapshotError, FuenteInvalida)):
        _cargar(repo, "0" * 40, arbol)


def test_el_commit_del_pin_no_cambia_aunque_avance_la_historia(tmp_path: Path) -> None:
    # La raiz es el SHA, no la rama: avanzar la historia con OTRO contenido en la
    # misma ruta no altera lo que el pin exacto devuelve.
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    antes = _cargar(repo, commit, arbol)
    (repo / "policy" / "faro" / "ejemplo.yaml").write_bytes(REGLA_TOPE)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "cambio")
    despues = _cargar(repo, commit, arbol)
    assert despues.snapshot_hash == antes.snapshot_hash
    assert despues.reglas[0].content_hash == "sha256:" + hashlib.sha256(REGLA).hexdigest()


@pytest.mark.parametrize("pin_kwargs", [
    {"commit": "XYZ"},                        # no es hex
    {"commit": "0" * 41},                     # largo raro
    {"policy_tree_oid": ""},
    {"procedencia": ""},
    {"repositorio": ""},
])
def test_pin_mal_formado_rechaza(pin_kwargs: dict) -> None:
    campos = {"repositorio": "jax", "commit": "0" * 40, "policy_tree_oid": "0" * 40,
              "procedencia": "p"}
    campos.update(pin_kwargs)
    with pytest.raises(RuleSnapshotError):
        TrustedPolicyPin(**campos)


def test_mover_la_rama_no_cambia_un_pin_ya_cargado(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    antes = _cargar(repo, commit, arbol)
    (repo / "policy" / "faro" / "ejemplo.yaml").write_bytes(REGLA + b"\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "cambio")
    _git(repo, "checkout", "-q", "-b", "rama-nueva")        # HEAD y ramas se movieron
    despues = _cargar(repo, commit, arbol)
    assert despues.snapshot_hash == antes.snapshot_hash


def test_mutar_el_checkout_no_cambia_el_snapshot(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    antes = _cargar(repo, commit, arbol)
    (repo / "policy" / "faro" / "ejemplo.yaml").write_bytes(b"rule_id: mutado\n")
    (repo / "policy" / "faro" / "intruso.yaml").write_bytes(b"rule_id: intruso\n")
    despues = _cargar(repo, commit, arbol)
    assert despues.snapshot_hash == antes.snapshot_hash
    assert len(despues.reglas) == 1


# ------------------------------------------------ git endurecido (§6.3, §13)

def test_replace_objects_no_desvia_la_lectura(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    (repo / "policy" / "faro" / "ejemplo.yaml").write_bytes(b"rule_id: suplantado\n")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "suplantacion")
    reemplazo = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "replace", commit, reemplazo)   # refs/replace: ese SHA mostraria OTRO contenido
    snap = _cargar(repo, commit, arbol)
    assert snap.reglas[0].content_hash == "sha256:" + hashlib.sha256(REGLA).hexdigest()


def test_las_defensas_de_git_van_en_cada_invocacion(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin este chequeo, un mutante que quite hooksPath/fsmonitor/replace/env
    limpio sobrevive: el plumbing de git no corre hooks de todos modos. Se
    captura la invocacion REAL y se exigen las banderas y el entorno."""
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    capturadas = []
    real_run = subprocess.run

    def espia_run(*args, **kwargs):
        argv = args[0] if args else kwargs.get("args")
        capturadas.append((list(argv), dict(kwargs.get("env") or {})))
        return real_run(*args, **kwargs)

    import jax.faro.git_objetos as go
    monkeypatch.setattr(go.subprocess, "run", espia_run)
    hooks = repo / ".git" / "hooks-propios"
    hooks.mkdir()
    (hooks / "pre-commit").write_text("#!/bin/sh\nexit 1\n")
    os.chmod(hooks / "pre-commit", 0o755)
    _git(repo, "config", "core.hooksPath", str(hooks))
    _git(repo, "config", "core.fsmonitor", "true")
    snap = _cargar(repo, commit, arbol)
    assert len(snap.reglas) == 1
    # Solo las invocaciones del loader (vanane con --no-replace-objects); las del
    # helper de prueba (_git: "git -C ...") no llevan defensas ni deben.
    deloader = [(a, e) for a, e in capturadas if len(a) > 1 and a[1] == "--no-replace-objects"]
    assert deloader, "el loader no invoco git con sus defensas"
    for argv, entorno in deloader:
        assert "--no-replace-objects" in argv
        assert "core.hooksPath=/dev/null" in argv
        assert "core.fsmonitor=false" in argv
        assert entorno.get("GIT_CONFIG_NOSYSTEM") == "1"
        assert entorno.get("GIT_CONFIG_GLOBAL") == "/dev/null"
        assert entorno.get("GIT_NO_REPLACE_OBJECTS") == "1"
        assert not any(k.startswith("GIT_") and k not in
                       ("GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "GIT_NO_REPLACE_OBJECTS",
                        "GIT_NO_LAZY_FETCH", "GIT_TERMINAL_PROMPT") for k in entorno)


def test_entorno_git_envenenado_no_desvia_la_lectura(monkeypatch: pytest.MonkeyPatch,
                                                      tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    monkeypatch.setenv("GIT_DIR", "/tmp/no-existe")
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/tmp/tampoco")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "core.fsmonitor=true")
    snap = _cargar(repo, commit, arbol)
    assert len(snap.reglas) == 1


# ------------------------------------------------------- entradas invalidas

def _repo_entrada(tmp: Path, accion) -> tuple[Path, str, str]:
    repo = tmp / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "policy" / "faro").mkdir(parents=True)
    (repo / "policy" / "faro" / "ejemplo.yaml").write_bytes(REGLA)
    (repo / "policy" / "faro" / "catalogo-topes.json").write_bytes(CATALOGO)
    accion(repo)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    return repo, _git(repo, "rev-parse", "HEAD").strip(), _git(repo, "rev-parse", "HEAD:policy").strip()


def test_symlink_a_una_regla_valida_rechaza(tmp_path: Path) -> None:
    destino = tmp_path / "regla-de-verdad.yaml"
    destino.write_bytes(REGLA)                       # si se quita el chequeo de modo, CARGA
    def accion(repo: Path) -> None:
        os.symlink(str(destino), repo / "policy" / "faro" / "enlace.yaml")
    repo, commit, arbol = _repo_entrada(tmp_path, accion)
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, commit, arbol)


def test_gitlink_de_submodulo_rechaza(tmp_path: Path) -> None:
    repo, commit, _ = _repo_entrada(tmp_path, lambda repo: None)
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{commit},policy/faro/submodulo")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "gitlink")
    commit = _git(repo, "rev-parse", "HEAD").strip()
    arbol = _git(repo, "rev-parse", "HEAD:policy").strip()
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, commit, arbol)


def test_bit_de_ejecucion_rechaza(tmp_path: Path) -> None:
    def accion(repo: Path) -> None:
        os.chmod(repo / "policy" / "faro" / "ejemplo.yaml", 0o755)
    repo, commit, arbol = _repo_entrada(tmp_path, accion)
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, commit, arbol)


def test_yaml_anidado_bajo_subdirectorio_rechaza(tmp_path: Path) -> None:
    def accion(repo: Path) -> None:
        (repo / "policy" / "faro" / "sub").mkdir()
        (repo / "policy" / "faro" / "sub" / "oculta.yaml").write_bytes(REGLA)
    repo, commit, arbol = _repo_entrada(tmp_path, accion)
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, commit, arbol)


@pytest.mark.parametrize("nombre", ["Ejemplo.yaml", "ejemplo.yml", "ejemplo regla.yaml",
                                    "a" * 65 + ".yaml"])
def test_nombre_no_canonico_rechaza_por_el_nombre(tmp_path: Path, nombre: str) -> None:
    # Contenido DISTINTO y valido (otro rule_id): la unica razon de rechazo es el nombre.
    unica = REGLA.replace(b"ejemplo-regla", b"regla-unica") + b"# propia\n"
    def accion(repo: Path) -> None:
        (repo / "policy" / "faro" / nombre).write_bytes(unica)
    repo, commit, arbol = _repo_entrada(tmp_path, accion)
    with pytest.raises(RuleSnapshotError) as excinfo:
        _cargar(repo, commit, arbol)
    assert "no canonica" in str(excinfo.value)


# ------------------------------------- atomicidad y contenido invalido §13

def test_un_yaml_invalido_rechaza_el_snapshot_completo(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/buena.yaml": REGLA,
                                           "policy/faro/rota.yaml": b"\t{invalido: [si"})
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, commit, arbol)


def test_rule_id_duplicado_rechaza_el_snapshot_completo(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/a.yaml": REGLA,
                                           "policy/faro/b.yaml": REGLA})
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, commit, arbol)


def test_una_regla_invalida_deja_al_snapshot_sin_partes(tmp_path: Path) -> None:
    invalida = REGLA.replace(b"effect: PERMIT", b"effect: DENY")     # fuera de vocabulario
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/valida.yaml": REGLA,
                                           "policy/faro/invalida.yaml": invalida})
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, commit, arbol)


# --------------------------------------------- procedencia del clon §13

def test_un_clone_limpio_conserva_la_procedencia(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path / "origen", {"policy/faro/ejemplo.yaml": REGLA})
    clon = tmp_path / "clon"
    subprocess.run(["git", "clone", "-q", str(repo), str(clon)], check=True, capture_output=True)
    snap = _cargar(clon, commit, arbol)
    assert snap.reglas[0].content_hash == "sha256:" + hashlib.sha256(REGLA).hexdigest()


def test_un_clone_con_historia_reescrita_no_conserva_procedencia(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path / "origen", {"policy/faro/ejemplo.yaml": REGLA})
    clon = tmp_path / "clon-rewritten"
    subprocess.run(["git", "clone", "-q", str(repo), str(clon)], check=True, capture_output=True)
    (clon / "policy" / "faro" / "ejemplo.yaml").write_bytes(REGLA + b"\n")
    _git(clon, "add", "-A")
    _git(clon, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--amend", "-qm", "reescrito")
    # el objeto huérfano hay que podarlo de verdad: el ref remoto origin/main
    # seguía apuntando al commit original del clone, así que también se suelta.
    _git(clon, "update-ref", "-d", "refs/remotes/origin/main")
    _git(clon, "reflog", "expire", "--expire=now", "--all")
    _git(clon, "gc", "--prune=now", "-q")
    with pytest.raises((RuleSnapshotError, FuenteInvalida)):
        _cargar(clon, commit, arbol)                      # el commit ratificado ya no existe ahi


# --------------------------------------------------------- sello y API publica

def test_el_snapshot_no_se_fabrica_por_la_api_publica() -> None:
    with pytest.raises(RuleSnapshotError):
        TrustedPolicySnapshot(commit="0" * 40, policy_tree_oid="0" * 40, reglas=(),
                              repositorio="jax", procedencia="p")


def test_la_regla_sellada_no_se_fabrica_por_la_api_publica() -> None:
    with pytest.raises(RuleSnapshotError):
        ReglaSellada(ruta="policy/faro/x.yaml", modo="100644", blob_oid="0" * 40,
                     content_hash="sha256:x", regla=None)          # type: ignore[arg-type]


def test_dataclasses_replace_no_fabrica_objetos_sellados(tmp_path: Path) -> None:
    import dataclasses
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA,
                                           "policy/faro/ejemplo-tope.yaml": REGLA_TOPE})
    snap = _cargar(repo, commit, arbol)
    r0, r1 = snap.reglas
    with pytest.raises(TypeError):                      # no son dataclasses: no hay replace
        dataclasses.replace(r1, regla=r0.regla)         # type: ignore[arg-type]
    with pytest.raises(TypeError):
        dataclasses.replace(snap, reglas=(r0,))         # type: ignore[arg-type]


def test_el_snapshot_es_profundamente_inmutable(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    snap = _cargar(repo, commit, arbol)
    with pytest.raises(Exception):
        snap.reglas = ()                                   # type: ignore[misc]
    with pytest.raises(Exception):
        snap.reglas[0].ruta = "otra"                       # type: ignore[misc]


def test_el_hash_del_snapshot_es_determinista_y_distingue_contenido(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    a = _cargar(repo, commit, arbol)
    b = _cargar(repo, commit, arbol)
    assert a.snapshot_hash == b.snapshot_hash
    otro = _repo(tmp_path / "otro", {"policy/faro/ejemplo.yaml": REGLA_TOPE})
    c = _cargar(*otro)
    assert a.snapshot_hash != c.snapshot_hash


# --------------------------- ronda 7: vector dorado del hash del snapshot (MINOR-1)

_VECTOR_REGLA = b'''schema_version: "1.0"
kind: JAX_FARO_RULE
rule_id: vector-dorado
effect: PERMIT
action_class: REVERSIBLE

scope:
  subjects: [actor:dorado]
  capabilities: [CAP_DORADA]
  objectives: [objetivo-dorado]

obligation_limits:
  quantity: null
  amount: null
  frequency: null

validity:
  not_before_utc: "2026-10-05T00:00:00Z"
  not_after_utc: null

permit:
  ttl_seconds: 60
'''
_VECTOR_CATALOGO = (b'{"version":1,"decision":"vector dorado r7","clases":{"monto_dinero":["x"],'
                    b'"actos_externos":["x"],"frecuencia":["x"],"duracion":["x"],"tokens_costo":["x"]}}')


def test_r7_vector_dorado_del_hash_con_dominio_v2(tmp_path: Path) -> None:
    """MINOR-1: el hash EXACTO de este snapshot, fijado como vector dorado. La
    carga no lleva commit ni timestamps — solo blobs verificados — asi que este
    valor solo cambia si cambia lo que entra al hash o su dominio (v2). Con el
    mutante «-v1», el hash cambia y esta prueba rompe."""
    repo = tmp_path / "vector"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "policy" / "faro").mkdir(parents=True)
    (repo / "policy" / "faro" / "catalogo-topes.json").write_bytes(_VECTOR_CATALOGO)
    (repo / "policy" / "faro" / "vector.yaml").write_bytes(_VECTOR_REGLA)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "vector")
    snap = _cargar(repo, _git(repo, "rev-parse", "HEAD").strip(),
                   _git(repo, "rev-parse", "HEAD:policy").strip())
    assert snap.snapshot_hash == (
        "sha256:65f864b35a376428d15489fc0ed629f163dfab23b61a7247500d64bbecd990b4")
