#!/usr/bin/env python3
"""Detector de estructura sobre la entrega de una rama (p. ej. de Qwen Code).

    verificar_entrega.py --repo <repo> --base <ref> --informe <archivo>

Contrasta lo que el INFORME de la entrega afirma (archivos, tablas, columnas,
símbolos) con la whitelist derivada del repo en su HEAD y de `information_schema`
de la base de la conexión (JAX_DB_*). Lo que la propia rama escribió (archivos
cambiados entre `base` y HEAD, y lo que su diff define) es propuesta, no
afirmación. No hay corpus confiable ni pasos previos: el informe es salida de un
modelo y solo lo respalda lo que existe.

Código de salida:
    0  limpio
    1  marcado: el informe afirma estructura sin respaldo (se imprime cuál)
    2  no pudo correr (informe, repo, base o base de datos inaccesibles), o el
       veredicto es no_verificable. Un 2 NO es un 0: si el detector no pudo
       mirar, nadie debe leer «limpio».

Solo lectura: no escribe en el repo ni en la base.

Uso desde pruebas: `main(argv, fuentes_db=(huella, esquema))` inyecta las dos
corrutinas del esquema; sin eso se consulta la base real.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from jax.anti_fabricacion import estructura, whitelist  # noqa: E402

LIMPIO, MARCADO, NO_PUDO = 0, 1, 2


def _analizar_argumentos(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", required=True, type=Path, help="repositorio con la rama entregada (HEAD)")
    ap.add_argument("--base", required=True, help="ref contra la que se mide lo que escribió la rama")
    ap.add_argument("--informe", required=True, type=Path, help="informe de la entrega (texto)")
    return ap.parse_args(argv)


def _diff_de_la_rama(repo: Path, base: str) -> tuple[list[str], str]:
    """(archivos cambiados, diff completo) entre `base` y HEAD."""
    whitelist._git(repo, "rev-parse", "--verify", f"{base}^{{commit}}")
    nombres = whitelist._git(repo, "diff", "--name-only", "-z", base, "HEAD").decode("utf-8", "replace")
    diff = whitelist._git(repo, "diff", "--no-color", "--no-ext-diff", base, "HEAD").decode("utf-8", "replace")
    return [n for n in nombres.split("\0") if n], diff


async def _veredicto(repo: Path, base: str, informe: str, fuentes_db) -> estructura.Veredicto:
    cambiados, diff = await asyncio.to_thread(_diff_de_la_rama, repo, base)
    _afirmados, definidos = estructura._diff(diff)
    huella, esquema = fuentes_db if fuentes_db else (whitelist._huella_db, whitelist._esquema_db)

    async def git():
        return await asyncio.to_thread(whitelist.leer_git, repo)

    async def simbolos(estado):
        return await asyncio.to_thread(whitelist.simbolos_de_archivos, repo, estado.entradas)

    wl = await whitelist.derivar(whitelist.Fuentes(huella=huella, esquema=esquema, git=git, simbolos=simbolos))
    return estructura.verificar(informe, wl, [], cambiados + definidos, previos=[])


def main(argv=None, *, fuentes_db=None) -> int:
    args = _analizar_argumentos(argv)
    try:
        informe = args.informe.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"no pudo correr: informe ilegible ({args.informe}): {exc}", file=sys.stderr)
        return NO_PUDO
    try:
        v = asyncio.run(_veredicto(args.repo, args.base, informe, fuentes_db))
    except Exception as exc:  # fail-closed: cualquier fallo del detector es «no pudo correr», nunca «limpio»
        print(f"no pudo correr: {type(exc).__name__}: {exc}", file=sys.stderr)
        return NO_PUDO

    if v.estado == estructura.LIMPIO:
        print("limpio: el informe no afirma estructura sin respaldo")
        if v.fuera_de_alcance:
            print("fuera de alcance (no revisado): " + ", ".join(v.fuera_de_alcance))
        return LIMPIO
    if v.estado == estructura.MARCADO:
        print("marcado: el informe afirma estructura sin respaldo")
        for i in v.items:
            print(f"  - {i.candidato} [{i.tipo}, {i.clase}]: {i.motivo}")
        return MARCADO
    print(f"no pudo correr: veredicto {v.estado} (categorías sin verificar: {', '.join(v.no_verificables) or v.motivo})",
          file=sys.stderr)
    return NO_PUDO


if __name__ == "__main__":
    sys.exit(main())
