#!/usr/bin/env python3
"""Compara los pisos de CI del head contra los de la PUNTA de refs/pisos-base/master: un piso nunca baja.

Vive en `.github/` (ruta reservada a Fernando) por la misma razón que `piso.py`: quien
pudiera debilitar este comparador podría debilitar todos los pisos sin pasar por la
integración reservada. Los DATOS están en `ci/pisos.json`; el archivo temporal que lee cada
piso está en la llamada de `.github/workflows/policy.yml`, así que se compara de ahí.

Uso:  python3 -I .github/ci/comparar_pisos.py [<ref-base>]      (por defecto `refs/pisos-base/master`)

Corre en el job aislado `pisos-no-bajan` de policy.yml: checkout, fetch de la base desde la URL fija
del repositorio a `refs/pisos-base/master` y este script, sin ejecutar código del PR antes (un
conftest.py o un sitecustomize podrían reescribir .git/config o las refs). Solo biblioteca estándar.

Reglas, para cada clave de la base:
  * la clave tiene que seguir en el head;
  * el archivo temporal que lee no cambia;
  * el patrón conserva EXACTAMENTE su forma y solo puede cambiar N (passed) hacia arriba;
    en la forma `^N passed, M skipped`, M no puede subir;
  * un mínimo numérico no baja ni deja de ser un entero.
Subir N y agregar claves nuevas está permitido. El mensaje puede cambiar.
Única excepción a «solo un N mayor»: bajar M (skipped) se permite a propósito, porque menos
saltadas endurece el piso (hay menos pruebas que pueden dejar de correr sin que nadie lo vea).
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
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
REF_BASE = "refs/pisos-base/master"
RUTA_DATOS = "ci/pisos.json"
RUTA_WORKFLOW = ".github/workflows/policy.yml"

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
        if fb != fh:
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
    ref = argv[0] if argv else REF_BASE
    if len(argv) > 1:
        print("uso: comparar_pisos.py [<ref-base>]", file=sys.stderr)
        return 2
    try:
        base, modo = cargar_base(raiz, ref)
        head = cargar_head(raiz)
        errores = comparar(base, head)
    except PisosError as e:
        print(f"comparar_pisos: ERROR (falla cerrado): {e}", file=sys.stderr)
        return 2
    if errores:
        print(f"comparar_pisos: los pisos bajaron respecto de {ref} (modo {modo}):")
        for e in errores:
            print(f"  - {e}")
        return 1
    print(f"comparar_pisos: OK contra {ref} (modo {modo}): {len(base['pisos'])} pisos y "
          f"{len(base['minimos'])} mínimos de la base sin bajar; {len(head['pisos'])} pisos en el head")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
