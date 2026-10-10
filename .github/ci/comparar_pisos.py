#!/usr/bin/env python3
"""Compara pisos del head con la base; el workflow ejecuta el checker extraído desde la base.

Vive en `.github/` (ruta reservada a Fernando) por la misma razón que `piso.py`: quien
pudiera debilitar este comparador podría debilitar todos los pisos sin pasar por la
integración reservada. Los DATOS están en `ci/pisos.json`; el archivo temporal que lee cada
piso está en la llamada de `.github/workflows/policy.yml`, así que se compara de ahí.

Uso: python3 -I comparar_pisos.py --head-root ROOT --base-ref REF --head-ref REF

Corre en `pisos-no-bajan`: usa historia completa y ejecuta el checker extraído de la ref base, nunca
el checker del head evaluado. Solo biblioteca estándar.

Reglas, para cada clave de la base:
  * la clave tiene que seguir en el head;
  * el archivo temporal que lee no cambia;
  * el patrón conserva EXACTAMENTE su forma (salvo la excepción de más abajo) y solo puede
    cambiar N (passed) hacia arriba;
    en la forma `^N passed, M skipped`, M no puede subir;
  * un mínimo numérico no baja ni deja de ser un entero.
Subir N y agregar claves nuevas está permitido. El mensaje puede cambiar.
Única excepción a «solo un N mayor»: bajar M (skipped) se permite a propósito, porque menos
saltadas endurece el piso (hay menos pruebas que pueden dejar de correr sin que nadie lo vea).
Única excepción a «la forma no cambia»: pasar de `^N passed` a `^N' passed, M skipped` con
N' >= N. Es un endurecimiento, porque agrega una exigencia sobre las saltadas (hasta entonces
cualquier número de saltadas pasaba). Al revés, de `^N passed, M skipped` a `^N passed`, sigue
prohibido: afloja, porque vuelve a aceptar cualquier número de saltadas.
Una clave repetida en `ci/pisos.json` es un error (falla cerrado), aunque hoy no rebaje nada.
Un patrón con una forma que este parser no reconoce, base o head, es un error: no se adivina.

La base se lee de `git show <ref>:ci/pisos.json` + `git show <ref>:.github/workflows/policy.yml`.
Arranque (la punta aún no tiene `ci/pisos.json`, o sea el PR que lo introduce): los pisos de la
base se derivan del `policy.yml` de esa MISMA punta, con `extraer_de_workflow_viejo`. Si de ahí
no sale ningún piso, falla cerrado. Si la base ya tiene `ci/pisos.json` y el head no, es un error
(clave desaparecida), nunca un arranque.

Códigos de salida: 0 sin violaciones; 1 hay violaciones; 2 no se pudo leer o interpretar algo
(falla cerrado: sin base legible no hay comparación, y sin comparación el paso falla).
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
REF_BASE = "refs/pisos-base/master"
RUTA_DATOS = "ci/pisos.json"
RUTA_WORKFLOW = ".github/workflows/policy.yml"
RUTA_GRANTS = ".github/workflows/floor-retirements.json"
_HEX_OID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GRANT_FIELDS = {
    "repository", "pull_request", "base_branch", "floor_key", "floor_definition",
    "floor_definition_sha256", "introducing_merge", "introducing_parent1",
    "introducing_parent2", "expected_diff_sha256",
}

_N = r"(0|[1-9][0-9]*)"
# Las únicas formas de patrón que existen. Cada una dice qué es N y, si lo hay, M.
FORMAS = {
    "passed": re.compile(rf"\^{_N} passed"),
    "passed-in": re.compile(rf"\^{_N} passed in "),
    "passed-in-sin-espacio": re.compile(rf"\^{_N} passed in"),
    "passed-skipped": re.compile(rf"\^{_N} passed, {_N} skipped"),
}

_JOB = re.compile(r"^  ([a-z0-9-]+):[ ]*$", re.M)
_GREP_VIEJO = re.compile(
    r'^[ ]*grep -qE "(?P<patron>[^"]*)" (?P<archivo>\S+) \|\| \{[ ]*\n?[ ]*echo "(?:[^"\\]|\\.)*"; exit 1; \}[ ]*$',
    re.M)
_MINIMO_VIEJO = re.compile(r"len\(cases\) < (\d+)")
_LLAMADA = re.compile(r"python3 \.github/ci/piso\.py verificar ([^\s)]+) (\S+)")


def _sin_duplicados(pares):
    d: dict = {}
    for k, v in pares:
        if k in d:
            raise ValueError(f"clave duplicada: {k!r}")
        d[k] = v
    return d


class PisosError(Exception):
    """No se pudo leer o interpretar algo: el paso tiene que fallar, nunca seguir."""


def parsear_patron(patron: object) -> tuple[str, int, int | None]:
    """(forma, N, M). M es None salvo en la forma con skipped."""
    if not isinstance(patron, str):
        raise PisosError(f"patrón que no es texto: {patron!r}")
    for forma, rx in FORMAS.items():
        m = rx.fullmatch(patron)
        if m:
            return forma, int(m.group(1)), int(m.group(2)) if forma == "passed-skipped" else None
    raise PisosError(f"forma de patrón no reconocida: {patron!r}")


def _job_de(texto: str, posicion: int) -> str:
    jobs = list(_JOB.finditer(texto, 0, posicion))
    if not jobs:
        raise PisosError("no se encontró el job de un piso en el workflow")
    return jobs[-1].group(1)


def extraer_de_workflow_viejo(texto: str) -> dict:
    """Pisos de un policy.yml con los `grep -qE "^N passed" ... || { echo ...; exit 1; }` literales.

    Clave `<job>/<archivo sin /tmp/>`, igual que en `ci/pisos.json`. Sin ningún piso: error.
    """
    pisos: dict = {}
    for m in _GREP_VIEJO.finditer(texto):
        archivo = m.group("archivo")
        clave = f"{_job_de(texto, m.start())}/{archivo.removeprefix('/tmp/')}"
        if clave in pisos:
            raise PisosError(f"clave repetida al extraer pisos del workflow base: {clave}")
        pisos[clave] = {"patron": m.group("patron"), "archivo": archivo}
    minimos: dict = {}
    mm = list(_MINIMO_VIEJO.finditer(texto))
    if len(mm) > 1:
        raise PisosError("más de un mínimo `len(cases) < N` en el workflow base")
    if mm:
        minimos[f"{_job_de(texto, mm[0].start())}/casos"] = int(mm[0].group(1))
    if not pisos:
        raise PisosError("no se pudo extraer ningún piso del policy.yml base y la base no trae ci/pisos.json")
    return {"pisos": pisos, "minimos": minimos}


def armar_nuevo(datos_json: str, texto_workflow: str, origen: str) -> dict:
    """Pisos de un estado nuevo: `ci/pisos.json` + el archivo que cada llamada del workflow lee."""
    try:
        datos = json.loads(datos_json, object_pairs_hook=_sin_duplicados)
    except ValueError as e:
        raise PisosError(f"{origen}: {RUTA_DATOS} no es JSON válido: {e}") from e
    if not isinstance(datos, dict) or not isinstance(datos.get("pisos"), dict) \
            or not isinstance(datos.get("minimos"), dict):
        raise PisosError(f"{origen}: {RUTA_DATOS} no tiene las secciones 'pisos' y 'minimos'")
    archivos: dict = {}
    for clave, archivo in _LLAMADA.findall(texto_workflow):
        if clave in archivos:
            raise PisosError(f"{origen}: la clave {clave} se llama dos veces en el workflow")
        archivos[clave] = archivo
    pisos: dict = {}
    for clave, entrada in datos["pisos"].items():
        if not isinstance(entrada, dict):
            raise PisosError(f"{origen}: el piso {clave} no es un objeto")
        pisos[clave] = {"patron": entrada.get("patron"), "archivo": archivos.get(clave)}
    return {"pisos": pisos, "minimos": dict(datos["minimos"])}


def comparar(base: dict, head: dict) -> list[str]:
    """Violaciones (lista vacía = ok). Lanza PisosError si algo no se puede interpretar."""
    errores: list[str] = []
    for clave, b in base["pisos"].items():
        if clave not in head["pisos"]:
            errores.append(f"{clave}: el piso desaparece")
            continue
        h = head["pisos"][clave]
        if h["archivo"] != b["archivo"]:
            errores.append(f"{clave}: cambia el archivo de salida ({b['archivo']!r} -> {h['archivo']!r})")
        fb, nb, mb = parsear_patron(b["patron"])
        fh, nh, mh = parsear_patron(h["patron"])
        if fb == "passed" and fh == "passed-skipped":
            pass  # endurecimiento permitido: solo se exige además que N no baje (abajo)
        elif fb != fh:
            errores.append(f"{clave}: el patrón cambia de forma ({b['patron']!r} -> {h['patron']!r})")
            continue
        if nh < nb:
            errores.append(f"{clave}: N baja de {nb} a {nh}")
        if mb is not None and mh > mb:
            errores.append(f"{clave}: M (skipped) sube de {mb} a {mh}")
    for clave, h in head["pisos"].items():
        if clave not in base["pisos"]:
            parsear_patron(h["patron"])  # una clave nueva también tiene que tener una forma conocida
    for clave, vb in base["minimos"].items():
        if _no_es_entero(vb):
            raise PisosError(f"mínimo de la base {clave} no es un entero: {vb!r}")
        if clave not in head["minimos"]:
            errores.append(f"{clave}: el mínimo desaparece")
            continue
        vh = head["minimos"][clave]
        if _no_es_entero(vh):
            errores.append(f"{clave}: el mínimo cambia de forma ({vb!r} -> {vh!r})")
        elif vh < vb:
            errores.append(f"{clave}: el mínimo baja de {vb} a {vh}")
    for clave, vh in head["minimos"].items():
        if clave not in base["minimos"] and _no_es_entero(vh):
            errores.append(f"{clave}: mínimo nuevo que no es un entero: {vh!r}")
    return errores


def _no_es_entero(v: object) -> bool:
    return isinstance(v, bool) or not isinstance(v, int)


def _git(raiz: Path, *args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(raiz), *args], capture_output=True, text=True)
    except OSError as e:
        raise PisosError(f"no se pudo ejecutar git: {e}") from e


def _mostrar(raiz: Path, ref: str, ruta: str) -> str:
    r = _git(raiz, "show", f"{ref}:{ruta}")
    if r.returncode != 0:
        raise PisosError(f"no se pudo leer {ref}:{ruta}: {r.stderr.strip()}")
    return r.stdout


def _hash_canonico(valor: object) -> str:
    bytes_json = json.dumps(valor, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(bytes_json).hexdigest()


def _parsear_json(texto: str, origen: str):
    try:
        return json.loads(texto, object_pairs_hook=_sin_duplicados)
    except (ValueError, TypeError) as e:
        raise PisosError(f"{origen}: JSON inválido: {e}") from e


def _validar_grant(grant: object, origen: str) -> dict:
    if not isinstance(grant, dict) or set(grant) != _GRANT_FIELDS:
        raise PisosError(f"{origen}: grant con campos ausentes o inesperados")
    if not isinstance(grant["repository"], str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", grant["repository"]):
        raise PisosError(f"{origen}: repository inválido")
    if isinstance(grant["pull_request"], bool) or not isinstance(grant["pull_request"], int) \
            or grant["pull_request"] < 1:
        raise PisosError(f"{origen}: pull_request inválido")
    if not isinstance(grant["base_branch"], str) or not re.fullmatch(
            r"[A-Za-z0-9._/-]{1,200}", grant["base_branch"]):
        raise PisosError(f"{origen}: base_branch inválida")
    if not isinstance(grant["floor_key"], str) or not grant["floor_key"] \
            or len(grant["floor_key"]) > 200 or any(c in grant["floor_key"] for c in "\r\n\0"):
        raise PisosError(f"{origen}: floor_key inválida")
    definicion = grant["floor_definition"]
    if not isinstance(definicion, dict) or set(definicion) != {"entry", "output_file"}:
        raise PisosError(f"{origen}: floor_definition inválida")
    entrada = definicion["entry"]
    if not isinstance(entrada, dict) or set(entrada) != {"patron", "mensaje"} \
            or not isinstance(entrada["patron"], str) or not isinstance(entrada["mensaje"], str) \
            or not isinstance(definicion["output_file"], str) or not definicion["output_file"]:
        raise PisosError(f"{origen}: floor_definition no tiene la forma cerrada")
    for campo in ("introducing_merge", "introducing_parent1", "introducing_parent2"):
        if not isinstance(grant[campo], str) or not _HEX_OID.fullmatch(grant[campo]):
            raise PisosError(f"{origen}: {campo} inválido")
    for campo in ("floor_definition_sha256", "expected_diff_sha256"):
        if not isinstance(grant[campo], str) or not _HEX_SHA256.fullmatch(grant[campo]):
            raise PisosError(f"{origen}: {campo} inválido")
    if _hash_canonico(definicion) != grant["floor_definition_sha256"]:
        raise PisosError(f"{origen}: hash de floor_definition no coincide")
    return grant


def _leer_registry(raiz: Path, revision: str) -> list[dict]:
    datos = _parsear_json(_mostrar(raiz, revision, RUTA_GRANTS), f"{revision}:{RUTA_GRANTS}")
    if not isinstance(datos, dict) or set(datos) != {"version", "grants"} \
            or datos.get("version") != 1 or isinstance(datos.get("version"), bool) \
            or not isinstance(datos.get("grants"), list):
        raise PisosError(f"{revision}:{RUTA_GRANTS}: esquema cerrado inválido")
    grants = [_validar_grant(g, f"{revision}:{RUTA_GRANTS}") for g in datos["grants"]]
    identidades = [(g["repository"], g["pull_request"], g["floor_key"]) for g in grants]
    if len(set(identidades)) != len(identidades):
        raise PisosError(f"{revision}:{RUTA_GRANTS}: grants ambiguos/duplicados")
    return grants


def _diff_filas(raiz: Path, izquierda: str, derecha: str) -> list[dict]:
    r = _git(raiz, "diff", "--raw", "--no-abbrev", "--no-renames", "-z",
             f"{izquierda}..{derecha}")
    if r.returncode != 0:
        raise PisosError(f"no se pudo leer el diff {izquierda}..{derecha}: {r.stderr.strip()}")
    partes = r.stdout.split("\0")
    if partes and partes[-1] == "":
        partes.pop()
    filas = []
    i = 0
    while i < len(partes):
        campos = partes[i].split()
        if len(campos) != 5 or not campos[0].startswith(":") or i + 1 >= len(partes):
            raise PisosError("salida git diff --raw inválida")
        old_mode, new_mode, old_oid, new_oid, estado = campos[0][1:], *campos[1:]
        if not _HEX_OID.fullmatch(old_oid) or not _HEX_OID.fullmatch(new_oid):
            raise PisosError("diff contiene OID truncado o inválido")
        filas.append({"path": partes[i + 1], "old_mode": old_mode, "new_mode": new_mode,
                      "old_oid": old_oid, "new_oid": new_oid, "status": estado})
        i += 2
    filas.sort(key=lambda x: x["path"])
    return filas


def _huella_diff_oids(raiz: Path, izquierda: str, derecha: str) -> str:
    return _hash_canonico(_diff_filas(raiz, izquierda, derecha))


def _padres(raiz: Path, revision: str) -> tuple[str, ...]:
    r = _git(raiz, "rev-list", "--parents", "-n", "1", revision)
    if r.returncode != 0:
        raise PisosError(f"historia incompleta: no se pudieron leer padres de {revision}: {r.stderr.strip()}")
    campos = r.stdout.split()
    oid = _git(raiz, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if oid.returncode != 0 or len(campos) < 1 or campos[0] != oid.stdout.strip():
        raise PisosError(f"salida de padres inválida para {revision}")
    padres = tuple(campos[1:])
    if any(not _HEX_OID.fullmatch(p) for p in padres):
        raise PisosError(f"padre inválido en {revision}")
    return padres


def _historia_completa(raiz: Path) -> None:
    r = _git(raiz, "rev-parse", "--is-shallow-repository")
    if r.returncode != 0 or r.stdout.strip() != "false":
        raise PisosError("historia Git incompleta o shallow; retiro denegado")


def _es_ancestro(raiz: Path, ancestro: str, revision: str) -> None:
    r = _git(raiz, "merge-base", "--is-ancestor", ancestro, revision)
    if r.returncode != 0:
        raise PisosError(f"historia incompleta o relación no demostrada: {ancestro} no es ancestro de {revision}")


def _definicion_piso(raiz: Path, revision: str, clave: str) -> dict | None:
    datos = _parsear_json(_mostrar(raiz, revision, RUTA_DATOS), f"{revision}:{RUTA_DATOS}")
    workflow = _mostrar(raiz, revision, RUTA_WORKFLOW)
    if not isinstance(datos, dict) or not isinstance(datos.get("pisos"), dict):
        raise PisosError(f"{revision}:{RUTA_DATOS}: pisos inválido")
    entrada = datos["pisos"].get(clave)
    if entrada is None:
        return None
    archivos = {}
    for k, archivo in _LLAMADA.findall(workflow):
        if k in archivos:
            raise PisosError(f"{revision}: la clave {k} aparece dos veces en policy.yml")
        archivos[k] = archivo
    archivo = archivos.get(clave)
    if archivo is None:
        raise PisosError(f"{revision}: el piso {clave} no tiene llamada de workflow")
    definicion = {"entry": entrada, "output_file": archivo}
    if not isinstance(entrada, dict) or set(entrada) != {"patron", "mensaje"}:
        raise PisosError(f"{revision}: la definición del piso {clave} no tiene forma cerrada")
    return definicion


def _oid_arbol(raiz: Path, revision: str, ruta: str) -> str | None:
    r = _git(raiz, "ls-tree", "--object-only", revision, "--", ruta)
    if r.returncode != 0:
        raise PisosError(f"no se pudo leer {revision}:{ruta}: {r.stderr.strip()}")
    lineas = r.stdout.splitlines()
    if not lineas:
        return None
    if len(lineas) != 1 or not _HEX_OID.fullmatch(lineas[0]):
        raise PisosError(f"árbol ambiguo o inválido para {revision}:{ruta}")
    return lineas[0]


def _evento_objetivo(grant: dict, base_sha: str) -> None:
    ruta = os.environ.get("GITHUB_EVENT_PATH")
    if not ruta:
        raise PisosError("falta GITHUB_EVENT_PATH; no se acredita el PR objetivo")
    try:
        evento = _parsear_json(Path(ruta).read_text(encoding="utf-8"), "evento GitHub")
    except OSError as e:
        raise PisosError(f"evento GitHub ilegible: {e}") from e
    pr = evento.get("pull_request") if isinstance(evento, dict) else None
    if not isinstance(pr, dict) or pr.get("number", evento.get("number")) != grant["pull_request"]:
        raise PisosError("el evento no es el PR autorizado por el grant")
    base = pr.get("base")
    if not isinstance(base, dict) or base.get("ref") != grant["base_branch"] \
            or base.get("sha") != base_sha or not isinstance(base.get("repo"), dict) \
            or base["repo"].get("full_name") != grant["repository"]:
        raise PisosError("repo, rama o SHA base del evento no coincide con el grant")


def _validar_merge_grant(raiz: Path, base_ref: str, grant: dict) -> tuple[str, str]:
    _historia_completa(raiz)
    r = _git(raiz, "rev-parse", "--verify", f"{base_ref}^{{commit}}")
    if r.returncode != 0:
        raise PisosError(f"no se pudo resolver la base {base_ref}: {r.stderr.strip()}")
    base_sha = r.stdout.strip()
    padres = _padres(raiz, base_sha)
    if len(padres) != 2:
        raise PisosError("la base candidata no es el merge dedicado del grant")
    baseline, grant_head = padres
    _es_ancestro(raiz, baseline, grant_head)
    grants_b = _leer_registry(raiz, baseline)
    grants_head = _leer_registry(raiz, grant_head)
    grants_merge = _leer_registry(raiz, base_sha)
    if grants_head != grants_b + [grant] or grants_merge != grants_head:
        raise PisosError("el merge base no añade exactamente un grant al registro previo")
    if [f["path"] for f in _diff_filas(raiz, baseline, grant_head)] != [RUTA_GRANTS]:
        raise PisosError("el head dedicado del grant modifica rutas ajenas al registro")
    if [f["path"] for f in _diff_filas(raiz, baseline, base_sha)] != [RUTA_GRANTS]:
        raise PisosError("la base candidata modifica rutas ajenas al registro del grant")
    if _oid_arbol(raiz, grant_head, RUTA_GRANTS) != _oid_arbol(raiz, base_sha, RUTA_GRANTS):
        raise PisosError("el merge del grant alteró su archivo respecto del grant-head")
    return base_sha, baseline


def retiro_exacto_autorizado(raiz: Path, base_ref: str, head_ref: str,
                             base: dict, head: dict, desaparecidas: set[str]) -> bool:
    if len(desaparecidas) != 1:
        return False
    _historia_completa(raiz)
    grants = _leer_registry(raiz, base_ref)
    candidatos = [g for g in grants if g["floor_key"] in desaparecidas]
    if len(candidatos) != 1:
        return False
    grant = candidatos[0]
    base_sha, baseline = _validar_merge_grant(raiz, base_ref, grant)
    _evento_objetivo(grant, base_sha)
    clave = grant["floor_key"]
    definicion = _definicion_piso(raiz, base_ref, clave)
    if definicion is None or definicion != grant["floor_definition"] \
            or _hash_canonico(definicion) != grant["floor_definition_sha256"]:
        return False
    intro = grant["introducing_merge"]
    padres_intro = _padres(raiz, intro)
    esperados = (grant["introducing_parent1"], grant["introducing_parent2"])
    if len(padres_intro) != 2 or padres_intro != esperados:
        raise PisosError("merge de introducción y padres no verificables o no coincidentes")
    _es_ancestro(raiz, intro, baseline)
    if _definicion_piso(raiz, esperados[0], clave) is not None:
        raise PisosError("el piso ya existía antes del merge de introducción")
    if _definicion_piso(raiz, intro, clave) != grant["floor_definition"]:
        raise PisosError("el merge declarado no introdujo la definición exacta del piso")
    if clave in head["pisos"]:
        return False
    return _huella_diff_oids(raiz, base_sha, head_ref) == grant["expected_diff_sha256"]


def _filtrar_retiro_autorizado(raiz: Path, base_ref: str, head_ref: str,
                               base: dict, head: dict, errores: list[str]) -> list[str]:
    desaparecidas = set(base["pisos"]) - set(head["pisos"])
    if not desaparecidas or not retiro_exacto_autorizado(
            raiz, base_ref, head_ref, base, head, desaparecidas):
        return errores
    clave = next(iter(desaparecidas))
    return [e for e in errores if e != f"{clave}: el piso desaparece"]

def cargar_head(raiz: Path) -> dict:
    try:
        datos = (raiz / RUTA_DATOS).read_text(encoding="utf-8")
        workflow = (raiz / RUTA_WORKFLOW).read_text(encoding="utf-8")
    except OSError as e:
        raise PisosError(f"head: no se puede leer {e.filename}: {e.strerror}") from e
    return armar_nuevo(datos, workflow, "head")


def cargar_base(raiz: Path, ref: str) -> tuple[dict, str]:
    """(pisos de la base, modo). Modo: 'datos' o 'arranque'."""
    r = _git(raiz, "ls-tree", ref, "--", RUTA_DATOS)
    if r.returncode != 0:
        raise PisosError(f"no se pudo leer la base {ref}: {r.stderr.strip()}")
    workflow = _mostrar(raiz, ref, RUTA_WORKFLOW)
    if r.stdout.strip():
        return armar_nuevo(_mostrar(raiz, ref, RUTA_DATOS), workflow, f"base {ref}"), "datos"
    return extraer_de_workflow_viejo(workflow), "arranque"


def main(argv: list[str], raiz: Path = RAIZ) -> int:
    base_ref = REF_BASE
    head_ref = "HEAD"
    args = list(argv)
    try:
        while args and args[0].startswith("--"):
            opcion = args.pop(0)
            if not args:
                raise ValueError(f"falta valor para {opcion}")
            valor = args.pop(0)
            if opcion == "--head-root":
                raiz = Path(valor)
            elif opcion == "--base-ref":
                base_ref = valor
            elif opcion == "--head-ref":
                head_ref = valor
            else:
                raise ValueError(f"opción desconocida: {opcion}")
        if args:
            if len(args) != 1 or argv and argv[0].startswith("--"):
                raise ValueError("argumentos posicionales inesperados")
            base_ref = args[0]
    except ValueError as e:
        print(f"uso: comparar_pisos.py [--head-root ROOT --base-ref REF --head-ref REF] [<ref-base>]: {e}",
              file=sys.stderr)
        return 2
    try:
        base, modo = cargar_base(raiz, base_ref)
        head = cargar_head(raiz)
        errores = comparar(base, head)
        if errores:
            errores = _filtrar_retiro_autorizado(raiz, base_ref, head_ref, base, head, errores)
    except PisosError as e:
        print(f"comparar_pisos: ERROR (falla cerrado): {e}", file=sys.stderr)
        return 2
    if errores:
        print(f"comparar_pisos: los pisos bajaron respecto de {base_ref} (modo {modo}):")
        for e in errores:
            print(f"  - {e}")
        return 1
    print(f"comparar_pisos: OK contra {base_ref} (modo {modo}): {len(base['pisos'])} pisos y "
          f"{len(base['minimos'])} mínimos de la base sin bajar; {len(head['pisos'])} pisos en el head")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
