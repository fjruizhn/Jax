#!/usr/bin/env python3
"""Mide si los extractos SIRVEN -- spec §7.B, corregido 2026-09-21.

El criterio de la ronda anterior ("30% menos tokens contra el pipeline del
ERP") estaba MAL PLANTEADO: los tres archivos de ese pipeline son texto
plano, la compuerta responde `sin_extractor` correctamente, y medir ahí da
cero ahorro por la razón equivocada. Este sistema no reduce tokens sobre
texto que ya era texto -- hace usables documentos binarios que antes no se
podían leer, y evita reextraerlos. Este medidor evalúa ESO.

Criterios evaluados acá (docs/superpowers/specs/2026-09-20-procesamiento-
archivos-design.md §7.B):

  B.1 (mata el proyecto si falla) -- 10 cifras elegidas a mano por Fernando
      de un estado financiero escaneado real salen correctas. NO se puede
      medir con código: lo verifica una persona leyendo. Este script lo
      deja explícitamente afuera del veredicto automático.

  B.2 (utilidad, binario, atado a un límite real) -- el extracto de un
      documento cuyo ORIGINAL no cabía en `tool_authority.MAX_READ_BYTES`
      tiene que caber. Si el original ya cabía, el extractor no resolvió
      nada y el criterio no aplica a ESE documento -- pero tiene que
      cumplirse en TODOS los documentos donde sí aplica, o la muestra
      dice que el sistema no sirve para el caso que importa.

  B.3 (el caché) -- un segundo pase sobre los MISMOS archivos hace CERO
      trabajo: cero llamadas reales a `compuerta.extraer`. Se cuenta
      envolviendo la función, no se asume.

  B.4 (que "parcial" signifique algo) -- sobre la muestra medida, el
      estado `parcial` no puede ser más del 60% de los documentos. Si lo
      es, el umbral de confianza del OCR está mal calibrado y no hay señal.

`MAX_READ_BYTES` se lee en vivo desde `tool_authority` en cada llamada
(nunca se copia a una constante de módulo al importar) -- así un test puede
parchearlo sin reescribir la lógica de la que depende B.2.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from motor_registry import tool_authority

from procesamiento import compuerta, ingesta
from procesamiento.ficha import Ficha

# B.4: no negociable en tiempo de ejecución por CLI a propósito -- es el
# umbral que el spec fijó ANTES de medir (§7.B.4), no un dial que se pueda
# correr para que el semáforo dé verde.
UMBRAL_PARCIAL_PORCENTUAL = 60.0

_NOMBRE_FICHA = "ficha.json"


def _tamano_extracto(carpeta: Path) -> int:
    """Suma el tamaño de las salidas reales en `procesado/<huella>/`,
    excluyendo `ficha.json` (que no es contenido, es metadato). `0` si la
    carpeta no existe -- estado `error`/`sin_extractor` no deja nada ahí."""
    if not carpeta.is_dir():
        return 0
    return sum(
        p.stat().st_size
        for p in carpeta.iterdir()
        if p.is_file() and p.name != _NOMBRE_FICHA
    )


def _medir_documento(archivo: Path, ficha: Ficha, carpeta: Path, tiempo_s: float) -> dict:
    """Un documento medido: tamaños, la razón, el criterio B.2 (binario),
    estado/extractor/tiempo, y tiempo por página cuando la ficha lo trae
    (OCR y PDF escaneado -- `ficha.detalle["paginas"]`)."""
    archivo = Path(archivo)
    original_bytes = archivo.stat().st_size

    hay_extracto = ficha.estado in {"ok", "parcial"}
    extracto_bytes = _tamano_extracto(carpeta) if hay_extracto else 0

    # Leído EN VIVO, no congelado en un global al importar -- ver docstring.
    max_bytes = tool_authority.MAX_READ_BYTES
    original_no_cabe = original_bytes > max_bytes
    extracto_cabe = hay_extracto and extracto_bytes <= max_bytes

    paginas = ficha.detalle.get("paginas")
    if not isinstance(paginas, int) or paginas <= 0:
        paginas = None

    return {
        "archivo": archivo.name,
        "original_bytes": original_bytes,
        "extracto_bytes": extracto_bytes,
        "razon": round(original_bytes / extracto_bytes, 2) if extracto_bytes else None,
        "extracto_cabe": extracto_cabe,
        "original_no_cabe": original_no_cabe,
        # B.2: binario -- las DOS cosas a la vez, no un promedio.
        "criterio_b2": extracto_cabe and original_no_cabe,
        "estado": ficha.estado,
        "extractor": ficha.extractor,
        "tiempo_s": round(tiempo_s, 3),
        "paginas": paginas,
        "tiempo_por_pagina_s": round(tiempo_s / paginas, 3) if paginas else None,
    }


def _totales(documentos: list[dict]) -> dict:
    """Agregados puros sobre una lista de documentos YA medidos -- sin
    tocar el filesystem ni la compuerta. Separado de `medir()` a propósito:
    así se puede probar la lógica de B.2/B.4 con datos fabricados, sin
    depender de una extracción real."""
    total = len(documentos)
    por_estado = dict(Counter(d["estado"] for d in documentos))
    parciales = por_estado.get("parcial", 0)
    porcentaje_parcial = round(100 * parciales / total, 2) if total else 0.0

    extractos_que_caben = sum(1 for d in documentos if d["extracto_cabe"])

    aplica_b2 = [d for d in documentos if d["original_no_cabe"]]
    aplica_b2_ok = [d for d in aplica_b2 if d["criterio_b2"]]
    # B.2 es binario POR documento, pero el veredicto de la MUESTRA exige
    # que TODOS los documentos donde el criterio aplica lo cumplan -- un
    # solo documento cuyo extracto tampoco cabe dice que el sistema no
    # resuelve el caso que importa. Sin ningún documento aplicable, no hay
    # nada que probar: tampoco pasa (spec: "sin número medido no hay GO").
    b2_pass = bool(aplica_b2) and len(aplica_b2_ok) == len(aplica_b2)

    b4_pass = porcentaje_parcial <= UMBRAL_PARCIAL_PORCENTUAL

    return {
        "total_documentos": total,
        "por_estado": por_estado,
        "porcentaje_parcial": porcentaje_parcial,
        "extractos_que_caben": extractos_que_caben,
        "documentos_donde_aplica_b2": len(aplica_b2),
        "documentos_donde_aplica_b2_ok": len(aplica_b2_ok),
        "b2_extracto_util_pass": b2_pass,
        "b4_parcial_bajo_umbral_pass": b4_pass,
    }


def _contar_extracciones(archivos: list[Path], trabajo: Path) -> int:
    """Envuelve `compuerta.extraer` y cuenta cuántas veces se invocó DE
    VERDAD mientras se ingieren `archivos` contra `trabajo`. Es la única
    forma honesta de medir B.3 -- contar, no asumir."""
    contador = {"n": 0}
    extraer_real = compuerta.extraer

    def _contando(origen):
        contador["n"] += 1
        return extraer_real(origen)

    compuerta.extraer = _contando  # type: ignore[assignment]
    try:
        for archivo in archivos:
            ingesta.ingerir(Path(archivo), Path(trabajo))
    finally:
        compuerta.extraer = extraer_real  # type: ignore[assignment]
    return contador["n"]


def medir(archivos: list[Path], trabajo: Path) -> dict:
    """Mide la utilidad real de los extractos sobre `archivos`, ingeridos
    contra `trabajo` (tiene que vivir DENTRO de `WORKSPACE_ROOT`, igual que
    cualquier llamador de `ingesta.ingerir` -- ver §9-bis del spec).

    Primer pase: ingiere cada archivo, cronometra, arma la ficha por
    documento (incluye B.2). Segundo pase: reingiere los MISMOS archivos
    contando extracciones reales -- tiene que dar cero (B.3)."""
    archivos = [Path(a) for a in archivos]
    trabajo = Path(trabajo)

    documentos: list[dict] = []
    contador = {"n": 0}
    extraer_real = compuerta.extraer

    def _contando(origen):
        contador["n"] += 1
        return extraer_real(origen)

    compuerta.extraer = _contando  # type: ignore[assignment]
    try:
        for archivo in archivos:
            t0 = time.perf_counter()
            ficha = ingesta.ingerir(archivo, trabajo)
            tiempo_s = time.perf_counter() - t0
            carpeta = ingesta.ruta_procesado(trabajo, ficha.sha256)
            documentos.append(_medir_documento(archivo, ficha, carpeta, tiempo_s))
    finally:
        compuerta.extraer = extraer_real  # type: ignore[assignment]
    extracciones_primera_pasada = contador["n"]

    extracciones_segunda_pasada = _contar_extracciones(archivos, trabajo)

    totales = _totales(documentos)
    b3_pass = extracciones_segunda_pasada == 0

    veredicto = {
        "b2_utilidad_binaria": totales["b2_extracto_util_pass"],
        "b3_cache_cero_extracciones": b3_pass,
        "b4_parcial_bajo_60pc": totales["b4_parcial_bajo_umbral_pass"],
    }

    return {
        "documentos": documentos,
        "totales": totales,
        "extracciones_primera_pasada": extracciones_primera_pasada,
        "extracciones_segunda_pasada": extracciones_segunda_pasada,
        "veredicto": veredicto,
        # B.1 queda explícitamente FUERA: no es medible por código, lo
        # verifica Fernando a mano contra el estado financiero escaneado.
        "go": all(veredicto.values()),
    }


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _imprimir_informe(r: dict) -> None:
    print("=" * 104)
    print("MEDICIÓN DE UTILIDAD DE EXTRACTOS -- spec §7.B (corregido 2026-09-21)")
    print("=" * 104)
    encabezado = (
        f"{'archivo':<42} {'original':>12} {'extracto':>12} {'razón':>9} "
        f"{'tiempo':>9} {'estado':<13} {'B.2':>4}"
    )
    print(encabezado)
    print("-" * len(encabezado))
    for d in r["documentos"]:
        razon = f"{d['razon']}x" if d["razon"] else "-"
        print(
            f"{d['archivo'][:42]:<42} {_miles(d['original_bytes']):>12} "
            f"{_miles(d['extracto_bytes']):>12} {razon:>9} {d['tiempo_s']:>8.2f}s "
            f"{d['estado']:<13} {'sí' if d['criterio_b2'] else 'no':>4}"
        )
        if d["paginas"]:
            print(
                f"    -> {d['paginas']} páginas, {d['tiempo_por_pagina_s']:.2f}s/página"
            )
    print("-" * len(encabezado))

    t = r["totales"]
    print(f"Total documentos:          {t['total_documentos']}")
    print(f"Por estado:                {t['por_estado']}")
    print(
        f"% parcial:                 {t['porcentaje_parcial']}% "
        f"(B.4: no puede pasar de {UMBRAL_PARCIAL_PORCENTUAL}%)"
    )
    print(f"Extractos que caben:       {t['extractos_que_caben']} / {t['total_documentos']}")
    print(
        f"Docs donde aplica B.2:     {t['documentos_donde_aplica_b2']} "
        f"(extracto cabe en {t['documentos_donde_aplica_b2_ok']} de ellos)"
    )
    print(f"Extracciones 1ra pasada:   {r['extracciones_primera_pasada']}")
    print(
        f"Extracciones 2da pasada:   {r['extracciones_segunda_pasada']} "
        "(B.3: tiene que ser 0)"
    )
    print()
    print("Veredicto (B.1 es de Fernando, no se mide acá):")
    for clave, valor in r["veredicto"].items():
        print(f"  {clave}: {'PASA' if valor else 'NO PASA'}")
    print(f"  GO (B.2 y B.3 y B.4): {'SÍ' if r['go'] else 'NO'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("archivos", nargs="+", type=Path, help="documentos reales a medir")
    p.add_argument(
        "--trabajo", type=Path, required=True,
        help="carpeta de trabajo DENTRO de JAX_WORKSPACE_DIR (ver §9-bis del spec)",
    )
    p.add_argument(
        "--json-out", type=Path, default=None,
        help="si se da, además de stdout escribe el JSON completo en esta ruta",
    )
    a = p.parse_args()

    r = medir(a.archivos, a.trabajo)
    _imprimir_informe(r)

    payload = json.dumps(r, indent=2, ensure_ascii=False, sort_keys=True)
    if a.json_out:
        a.json_out.write_text(payload, encoding="utf8")
        print(f"\nJSON escrito en {a.json_out}")
    else:
        print("\n--- JSON ---")
        print(payload)

    if not r["go"]:
        print(
            "\nNO PASA: al menos un criterio de utilidad (B.2/B.3/B.4) no se "
            "cumplió. Según el spec §7.B, esto se reporta -- no se supone que pase."
        )
        return 1
    print("\nPASA: B.2, B.3 y B.4 cumplidos sobre esta muestra.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
