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

import json

from jax.faro.catalogo_topes import CatalogoTopesInvalido, cargar_catalogo_bytes

FIXTURES = Path(__file__).parent / "fixtures" / "faro_rules"
REGLA = (FIXTURES / "regla-ejemplo.yaml").read_bytes()
TOPE = (FIXTURES / "regla-ejemplo-tope.yaml").read_bytes()

RAIZ = Path(__file__).resolve().parents[2]
RUTA_CATALOGO = "policy/faro/catalogo-topes.json"
CATALOGO_BYTES_REPO = (RAIZ / RUTA_CATALOGO).read_bytes()
CATALOGO = cargar_catalogo_bytes(CATALOGO_BYTES_REPO)


def validar_regla(datos, **kwargs):
    from policy.rule_authority.schema import validar_regla as _validar
    kwargs.setdefault("catalogo", CATALOGO)
    return _validar(datos, **kwargs)


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    assert r.returncode == 0, r.stderr.decode()
    return r.stdout.decode().strip()


def _repo(tmp: Path, files: dict[str, bytes], nombre: str = "repo",
          con_catalogo: bool = True) -> tuple[Path, str, str]:
    repo = tmp / nombre
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    if con_catalogo:
        files = {RUTA_CATALOGO: CATALOGO_BYTES_REPO, **files}
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
    (repo / "policy" / "faro" / "catalogo-topes.json").write_bytes(
        (RAIZ / "policy" / "faro" / "catalogo-topes.json").read_bytes())
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
    from jax.faro.catalogo_topes import es_de_catalogo
    from jax.faro.topes import es_recurso_sin_tope
    assert not es_de_catalogo(recurso, CATALOGO)
    assert es_recurso_sin_tope(recurso, CATALOGO)   # el runtime los deja sin tope (TopeProhibido)


# --------------------------------------- ronda 4: la decision vive en policy/** (M-7)

LITERAL_DECISION_FERNANDO = {
    "version": 1,
    "decision": "Fernando 2026-10-06: solo actos y dinero; nunca conexiones, concurrencia, workers, hilos, procesos ni agentes (D-4)",
    "clases": {
        "monto_dinero": ["hnl", "usd"],
        "actos_externos": ["mensajes", "correos", "publicaciones", "compras", "pagos"],
        "frecuencia": ["por_hora", "por_dia"],
        "duracion": ["segundos"],
        "tokens_costo": ["tokens", "usd"],
    },
}


def test_r4_el_catalogo_es_el_literal_exacto_de_la_decision_de_fernando() -> None:
    """C2 + B-3(6): los BYTES del arbol git de HEAD (no del disco suelto) contra
    el LITERAL copiado aqui; y el catalogo cargado de esos bytes, igual."""
    import subprocess
    crudo = subprocess.run(["git", "show", "HEAD:policy/faro/catalogo-topes.json"],
                           capture_output=True, check=True, cwd=RAIZ).stdout
    assert json.loads(crudo) == LITERAL_DECISION_FERNANDO
    clases = cargar_catalogo_bytes(crudo)
    assert {c: list(s) for c, s in clases.items()} == LITERAL_DECISION_FERNANDO["clases"]


def test_r4_el_espejo_json_ata_cada_recurso_a_su_clase() -> None:
    """monto_dinero + tokens_costo.usd tambien falla en el JSON: cada variante
    oneOf fija la clase (const) y solo SUS recursos."""
    import json
    espejo = json.loads((RAIZ / "policy" / "faro" / "schemas" / "rule-v1.schema.json").read_text())
    variantes = espejo["properties"]["tope"]["oneOf"]
    por_clase = {v["properties"]["resource_class"]["const"]:
                 v["properties"]["resource"]["enum"] for v in variantes}
    assert sorted(por_clase) == sorted(CATALOGO)
    for clase, subids in CATALOGO.items():
        assert sorted(por_clase[clase]) == sorted(f"{clase}.{s}" for s in subids)
    assert "monto_dinero.usd" in por_clase["monto_dinero"]
    assert "tokens_costo.usd" not in por_clase["monto_dinero"]     # clase ajena: atado


def test_r4_subid_valido_de_otra_clase_niega(C=None) -> None:
    """C3: el subid existe en el catalogo pero en OTRA clase."""
    datos = dict(load_strict_yaml(TOPE))
    datos["tope"] = {"resource_class": "monto_dinero", "resource": "tokens_costo.usd",
                     "maximum": 2, "period": "hora"}
    with pytest.raises(RuleSchemaError) as excinfo:
        validar_regla(datos)
    assert "SU clase" in str(excinfo.value)
    datos["tope"]["resource"] = "actos_externos.tokens"
    datos["tope"]["resource_class"] = "actos_externos"
    with pytest.raises(RuleSchemaError):
        validar_regla(datos)


def test_r4_el_catalogo_es_inmutable_en_el_proceso() -> None:
    with pytest.raises(TypeError):
        CATALOGO["actos_externos"] = CATALOGO["actos_externos"] + ("agentes",)  # type: ignore[index]
    with pytest.raises(TypeError):
        CATALOGO["nueva"] = ("x",)                                              # type: ignore[index]
    from jax.faro.catalogo_topes import es_de_catalogo
    assert not es_de_catalogo("actos_externos.agentes", CATALOGO)


def test_r4_catalogo_invalido_falla_cerrado() -> None:
    import json as J
    from jax.faro.catalogo_topes import CatalogoTopesInvalido
    # r6: los dos ultimos casos llevan las CINCO clases de D-4 a proposito:
    # con el piso de clases en codigo, un catalogo de relleno ({"a": ["b"]})
    # negaria por D-4 y la prueba pasaria por la razon equivocada — la clave
    # extra (N5) y la lista vacia (N9) tienen que ser la UNICA razon de negacion.
    cinco = LITERAL_DECISION_FERNANDO["clases"]
    for malo in [b"no es json",
                 J.dumps({"version": 2, "decision": "x", "clases": {"a": ["b"]}}).encode(),
                 J.dumps({"version": 1, "decision": "x", "clases": {"a": ["b", "b"]}}).encode(),
                 J.dumps({"version": 1, "decision": "x", "clases": {"a": ["B"]}}).encode(),
                 J.dumps({"version": 1, "decision": "x", "clases": {}}).encode(),
                 J.dumps({"version": 1, "decision": "x"}).encode(),
                 J.dumps({"version": 1, "decision": "", "clases": {"a": ["b"]}}).encode(),
                 J.dumps({"version": True, "decision": "x", "clases": {"a": ["b"]}}).encode(),
                 J.dumps({"version": 1, "decision": "x", "clases": {"a": []}}).encode(),
                 J.dumps({"version": 1, "decision": "x", "clases": {"a": ["b"]},
                          "extra": 1}).encode(),
                 b'{"version": 1, "version": 1, "decision": "x", "clases": {"a": ["b"]}}',
                 J.dumps({"version": 1, "decision": "x",
                          "clases": {**cinco, "monto_dinero": []}}).encode(),      # N9
                 J.dumps({"version": 1, "decision": "x", "clases": cinco,
                          "extra": 1}).encode(),                                    # N5
                 ('{"version": 1, "version": 1, "decision": "x", "clases": '
                  + J.dumps(cinco) + '}').encode()]:                                # P7/M-8
        with pytest.raises(CatalogoTopesInvalido):
            cargar_catalogo_bytes(malo)


def test_r4_snapshot_con_mismo_arbol_pero_otro_commit_no_es_igual(tmp_path: Path) -> None:
    """C10: __eq__ que ignore el commit haria pasar por «el mismo» un snapshot
    de otra revision. El commit es parte de la identidad."""
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    a = _cargar(repo, commit, arbol)
    (repo / "otro.txt").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "y")
    c2 = _git(repo, "rev-parse", "HEAD")
    b = _cargar(repo, c2, arbol)                       # mismo arbol policy, otro commit
    assert b.commit == c2 and a.commit == commit
    assert a != b and not (a == b)                      # C10: ignorar el commit rompe esto


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
    (repo / "policy" / "faro" / "catalogo-topes.json").write_bytes(
        (RAIZ / "policy" / "faro" / "catalogo-topes.json").read_bytes())
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


# ------------------------------------- ronda 5: el catalogo sale del PIN (B-3)

import json as _json5


def _catalogo_bytes(clases: dict) -> bytes:
    """r6 (D-4 como piso): las CLASES son codigo, no dato del pin — este helper
    solo sustituye SUBIDS de las cinco clases; un catalogo con otra clase no se
    puede armar por aqui, tiene que negar en el cargador."""
    fusion = {**LITERAL_DECISION_FERNANDO["clases"], **clases}
    assert set(fusion) == set(LITERAL_DECISION_FERNANDO["clases"])
    return _json5.dumps({"version": 1, "decision": "prueba", "clases": fusion}).encode()


def test_r5_ataque_k_catalogo_estrechado_en_el_pin_la_regla_con_tope_niega(tmp_path: Path) -> None:
    """K: el commit fijado estrecho el catalogo POR SUBID (actos_externos sin
    mensajes); la regla con tope actos_externos.mensajes NO puede validar
    contra el catalogo del working tree — niega contra el del PIN. (Estrechar
    por CLASE ya no es expresable desde la r6: las cinco clases de D-4 son piso
    de codigo y un catalogo sin alguna de ellas niega al cargarse.)"""
    estrecho = _catalogo_bytes({"actos_externos": ["correos"]})
    repo, commit, arbol = _repo(tmp_path, {
        RUTA_CATALOGO: estrecho,
        "policy/faro/regla.yaml": TOPE,            # tope actos_externos.mensajes
    })
    with pytest.raises(RuleSnapshotError) as excinfo:
        _cargar(repo, commit, arbol)
    assert "regla invalida" in str(excinfo.value)


def test_r5_catalogo_invalido_en_el_pin_niega_el_snapshot(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {RUTA_CATALOGO: b"no es json"})
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, commit, arbol)


def test_r5_catalogo_con_clave_duplicada_en_el_pin_niega(tmp_path: Path) -> None:
    """M-8: la clave duplicada no gana. Con las CINCO clases de D-4 en el
    catálogo: sin el object_pairs_hook la duplicada pasaría y el snapshot
    CARGARÍA (r6 — antes llevaba clases de relleno y negaba por D-4)."""
    duplicado = ('{"version": 1, "version": 1, "decision": "x", "clases": '
                 + _json5.dumps(LITERAL_DECISION_FERNANDO["clases"]) + '}').encode()
    repo, commit, arbol = _repo(tmp_path, {RUTA_CATALOGO: duplicado})
    with pytest.raises(RuleSnapshotError) as excinfo:
        _cargar(repo, commit, arbol)
    assert "clave duplicada" in str(excinfo.value)


def test_r5_catalogo_con_bit_de_ejecucion_en_el_pin_niega(tmp_path: Path) -> None:
    repo = tmp_path / "r"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / "policy" / "faro").mkdir(parents=True)
    (repo / "policy" / "faro" / "catalogo-topes.json").write_bytes(CATALOGO_BYTES_REPO)
    os.chmod(repo / "policy" / "faro" / "catalogo-topes.json", 0o755)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x")
    with pytest.raises(RuleSnapshotError):
        _cargar(repo, _git(repo, "rev-parse", "HEAD"), _git(repo, "rev-parse", "HEAD:policy"))


def test_r5_mutar_el_catalogo_del_disco_no_cambia_el_snapshot(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    antes = _cargar(repo, commit, arbol)
    (repo / "policy" / "faro" / "catalogo-topes.json").write_bytes(b"basura")
    os.symlink("/etc/passwd", tmp_path / "senuelo")
    (repo / "policy" / "faro" / "catalogo-topes.json").unlink()
    os.symlink("/etc/passwd", repo / "policy" / "faro" / "catalogo-topes.json")
    despues = _cargar(repo, commit, arbol)
    assert despues.snapshot_hash == antes.snapshot_hash
    assert despues.catalogo_hash == antes.catalogo_hash


def test_r5_el_hash_del_snapshot_cambia_si_cambia_el_catalogo(tmp_path: Path) -> None:
    catalogo_a = (RAIZ / "policy" / "faro" / "catalogo-topes.json").read_bytes()
    catalogo_b = _catalogo_bytes({"monto_dinero": ["eur"]})    # mismas cinco clases, otro subid
    a = _repo(tmp_path / "a", {"policy/faro/ejemplo.yaml": REGLA, RUTA_CATALOGO: catalogo_a})
    b = _repo(tmp_path / "b", {"policy/faro/ejemplo.yaml": REGLA, RUTA_CATALOGO: catalogo_b})
    assert _cargar(*a).snapshot_hash != _cargar(*b).snapshot_hash


def test_r5_el_snapshot_expone_el_catalogo_inmutable_del_pin(tmp_path: Path) -> None:
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo-tope.yaml": TOPE})
    snap = _cargar(repo, commit, arbol)
    assert snap.catalogo is not None
    assert "actos_externos" in snap.catalogo
    with pytest.raises(TypeError):
        snap.catalogo["nueva_clase"] = ("x",)                      # type: ignore[index]


def test_r5_json_suelto_en_faro_que_no_es_el_catalogo_niega(tmp_path: Path) -> None:
    """N6 (r6, con diente): cualquier *.json en faro/ que no sea
    catalogo-topes.json niega. El repo se arma CON el catalogo real presente y
    el suelto es un catalogo VALIDO con nombre ajeno (K14): asi la unica razon
    posible de negacion es el clasificador — con el mutante N6 (cualquier .json
    tolerado), el snapshot CARGARIA y esta prueba muere."""
    otro_valido = _catalogo_bytes({})            # cinco clases: valido de verdad
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/otro.json": otro_valido})
    with pytest.raises(RuleSnapshotError) as excinfo:
        _cargar(repo, commit, arbol)
    assert "no regla y no infraestructura" in str(excinfo.value)


def test_r5_regla_con_tope_sin_catalogo_en_validacion_niega() -> None:
    from policy.rule_authority.schema import validar_regla as _validar
    datos = dict(load_strict_yaml(TOPE))
    with pytest.raises(RuleSchemaError) as excinfo:
        _validar(datos)                                           # sin catalogo: sin tope
    assert "sin catalogo" in str(excinfo.value)


# ------------------------- ronda 6: D-4 como piso de codigo, UTF-8 y catalogo sellado


def test_r6_k2_sexta_clase_ratificada_niega_en_el_cargador() -> None:
    """K2 (D-4 como piso): «conexiones» ratificado en el pin no abre el catalogo.
    Las clases son CODIGO (las cinco de la decision de Fernando); los subid son
    dato. Que falte una clase, tambien niega: ni mas ni menos."""
    from jax.faro.catalogo_topes import CLASES_TOPEABLES
    seis = {**LITERAL_DECISION_FERNANDO["clases"], "conexiones": ["abiertas"]}
    with pytest.raises(CatalogoTopesInvalido) as excinfo:
        cargar_catalogo_bytes(_json5.dumps(
            {"version": 1, "decision": "ratificado igual niega", "clases": seis}).encode())
    assert "D-4" in str(excinfo.value)
    cuatro = {c: s for c, s in LITERAL_DECISION_FERNANDO["clases"].items() if c != "duracion"}
    with pytest.raises(CatalogoTopesInvalido):
        cargar_catalogo_bytes(_json5.dumps(
            {"version": 1, "decision": "x", "clases": cuatro}).encode())
    assert CLASES_TOPEABLES == frozenset(LITERAL_DECISION_FERNANDO["clases"])  # el piso ES la decision


def test_r6_k2_en_el_pin_la_regla_con_conexiones_niega_al_cargar(tmp_path: Path) -> None:
    """El ataque K2 completo del auditor: pin con «conexiones» y una regla que la
    usa bien — el snapshot niega AL CARGAR EL CATALOGO, antes de mirar la regla:
    no hay ratificacion que valga contra el piso de codigo."""
    k2 = _json5.dumps({"version": 1, "decision": "ampliado a espaldas del codigo",
                       "clases": {**LITERAL_DECISION_FERNANDO["clases"],
                                  "conexiones": ["abiertas"]}}).encode()
    regla = (TOPE.replace(b"actos_externos.mensajes", b"conexiones.abiertas")
                 .replace(b"resource_class: actos_externos", b"resource_class: conexiones"))
    repo, commit, arbol = _repo(tmp_path, {RUTA_CATALOGO: k2, "policy/faro/r.yaml": regla})
    with pytest.raises(RuleSnapshotError) as excinfo:
        _cargar(repo, commit, arbol)
    assert "catalogo-topes.json del pin invalido" in str(excinfo.value)
    assert "D-4" in str(excinfo.value)


def test_r6_objeto_suelto_del_catalogo_adulterado_niega(tmp_path: Path) -> None:
    """P3: el OID del catalogo se RECALCULA de los bytes leidos. Un objeto suelto
    de .git/objects reescrito con un zlib VALIDO de otro contenido (un catalogo
    valido, con otro subid) no pasa: git no valida el hash al leer sueltos — el
    snapshot si. Con el chequeo OID mutilado, este catalogo ajeno CARGARIA."""
    repo, commit, arbol = _repo(tmp_path, {"policy/faro/ejemplo.yaml": REGLA})
    oid = _git(repo, "rev-parse", "HEAD:policy/faro/catalogo-topes.json")
    suelto = repo / ".git" / "objects" / oid[:2] / oid[2:]
    assert suelto.is_file()                              # objeto suelto, no en pack
    otro = _catalogo_bytes({"monto_dinero": ["eur"]})    # valido, pero NO es el del OID
    suelto.chmod(0o644)                                  # git lo deja en solo-lectura
    suelto.write_bytes(zlib.compress(b"blob %d\0" % len(otro) + otro))
    with pytest.raises(RuleSnapshotError) as excinfo:
        _cargar(repo, commit, arbol)
    assert "catalogo-topes.json: el blob no corresponde a su OID" in str(excinfo.value)


def test_r6_el_cargador_acepta_solo_bytes() -> None:
    """P12: str, bytearray o memoryview no son el canal del catalogo: solo bytes."""
    for malo in (CATALOGO_BYTES_REPO.decode(), bytearray(CATALOGO_BYTES_REPO),
                 memoryview(CATALOGO_BYTES_REPO)):
        with pytest.raises(CatalogoTopesInvalido):
            cargar_catalogo_bytes(malo)


def test_r6_catalogo_anidado_en_subdirectorio_niega_igual(tmp_path: Path) -> None:
    """P13: faro/sub/catalogo-topes.json no es el catalogo aunque el nombre
    TERMINE igual — el clasificador exige la ruta exacta y el anidado niega el
    snapshot completo (K10). Con el mutante (endswith), el anidado pasaria por
    catalogo y el snapshot cargaria."""
    repo, commit, arbol = _repo(
        tmp_path, {"policy/faro/sub/catalogo-topes.json": CATALOGO_BYTES_REPO})
    with pytest.raises(RuleSnapshotError) as excinfo:
        _cargar(repo, commit, arbol)
    assert "no regla y no infraestructura" in str(excinfo.value)


def test_r6_el_catalogo_debe_ser_utf8_estrito_sin_bom() -> None:
    """K8 + r6: UTF-16/32 (con y sin BOM) y UTF-8 con BOM niegan. El catalogo se
    decodifica como UTF-8 estricto ANTES de json.loads: la autodeteccion de
    codificacion de json sobre bytes no vale como contrato."""
    for codificacion in ("utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le",
                         "utf-32-be"):
        with pytest.raises(CatalogoTopesInvalido):
            cargar_catalogo_bytes(CATALOGO_BYTES_REPO.decode().encode(codificacion))
    with pytest.raises(CatalogoTopesInvalido):
        cargar_catalogo_bytes(b"\xef\xbb\xbf" + CATALOGO_BYTES_REPO)   # BOM UTF-8


def test_r6_el_catalogo_sellado_no_se_fabrica_ni_se_muta() -> None:
    """El CatalogoTopes del snapshot es un tipo propio: no se construye por la
    API publica (testigo), no se muta, y es lo UNICO que es_de_catalogo acepta —
    un Mapping suelto con la misma forma niega."""
    from jax.faro.catalogo_topes import CatalogoTopes, es_de_catalogo
    misma_forma = {c: list(s) for c, s in CATALOGO.items()}
    with pytest.raises(CatalogoTopesInvalido):
        CatalogoTopes(misma_forma)                       # fabricacion directa: no
    with pytest.raises(CatalogoTopesInvalido):
        es_de_catalogo("actos_externos.mensajes", misma_forma)      # dict: no
    assert es_de_catalogo("actos_externos.mensajes", CATALOGO) is True
    with pytest.raises(TypeError):
        CATALOGO._clases["conexiones"] = ("x",)          # type: ignore[index]
