"""Lectura del manifiesto de arranque de systemd (ops/versionar-drop-ins,
2026-09-25) y cómputo de la configuración EFECTIVA de una unidad -- unidad
base + sus drop-ins, en el mismo orden en que systemd los aplica de verdad.

Módulo compartido por tests/test_arranque_instalado.py y
tests/test_ejecutor_cuenta_de_servicio.py: el manifiesto es UNA sola lista
(ops/manifiesto-arranque-instalado.tsv) y el cómputo de la configuración
efectiva es UNA sola implementación -- dos copias que pudieran divergir
serían exactamente el tipo de "control que no controla nada" que
ops/verificar-arranque-instalado.sh existe para evitar del lado del
guion de shell.
"""
from __future__ import annotations

import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFIESTO = ROOT / "ops" / "manifiesto-arranque-instalado.tsv"


def leer_manifiesto() -> list[tuple[Path, Path]]:
    """Los pares (ruta absoluta en el repo, ruta absoluta instalada) que
    también lee ops/verificar-arranque-instalado.sh, del mismo archivo."""
    pares = []
    for linea in MANIFIESTO.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        repo_rel, instalada = linea.split("\t")
        pares.append((ROOT / repo_rel, Path(instalada)))
    return pares


def _archivos_de_unidad(nombre_unidad: str) -> list[Path]:
    """La unidad base MÁS sus drop-ins, en orden alfabético de nombre de
    archivo (systemd.unit(5): drop-ins de <unidad>.d/ se aplican en ese
    orden -- es por eso que z-pythonpath.conf usa el prefijo `z-`, para
    aplicarse último a propósito). Todo derivado del manifiesto: ni el
    directorio del repo ni el orden se asumen, salen de la columna
    "instalada"."""
    base = None
    dropins: list[Path] = []
    prefijo_dropins = f"/etc/systemd/system/{nombre_unidad}.d/"
    ruta_base = f"/etc/systemd/system/{nombre_unidad}"
    for repo_abs, instalada in leer_manifiesto():
        instalada_str = str(instalada)
        if instalada_str == ruta_base:
            base = repo_abs
        elif instalada_str.startswith(prefijo_dropins):
            dropins.append(repo_abs)
    if base is None:
        raise AssertionError(f"{nombre_unidad}: no hay unidad base en el manifiesto")
    dropins.sort(key=lambda p: p.name)
    return [base] + dropins


def _fusionar_environment(entorno: dict[str, str], valor: str) -> None:
    """Semántica real de systemd.exec(5) para `Environment=`: una lista de
    asignaciones VAR=valor separadas por espacio (con las reglas de citado
    de un shell POSIX -- de ahí shlex.split, no un split(' ') ingenuo, que
    rompería con un valor citado que contenga espacios). Un `Environment=`
    con el valor vacío BORRA todas las asignaciones previas de este mismo
    fragmento en adelante (no de las de otro Environment= con contenido,
    que siguen agregando/pisando por nombre de variable)."""
    valor = valor.strip()
    if valor == "":
        entorno.clear()
        return
    for asignacion in shlex.split(valor):
        var, sep, val = asignacion.partition("=")
        if sep:
            entorno[var] = val


def configuracion_efectiva(nombre_unidad: str) -> tuple[dict[str, str], dict[str, str]]:
    """La configuración EFECTIVA de una unidad -- lo que systemd corre de
    verdad, no sólo lo que dice el archivo base.

    Sólo mira la sección [Service] (una unidad `.timer` tiene [Timer], no
    [Service]; una futura unidad con [Service] Y otra sección no debería
    mezclar valores de la sección equivocada). Devuelve (simples, entorno):
    `simples` son las claves de asignación única (`User=`, `Group=`,
    `WorkingDirectory=`, ...) con el último valor visto en TODO el orden de
    aplicación; `entorno` es el resultado de fusionar TODAS las líneas
    `Environment=` con la semántica de arriba."""
    archivos = _archivos_de_unidad(nombre_unidad)
    simples: dict[str, str] = {}
    entorno: dict[str, str] = {}
    for archivo in archivos:
        seccion = None
        for linea in archivo.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if not linea or linea.startswith("#") or linea.startswith(";"):
                continue
            if linea.startswith("[") and linea.endswith("]"):
                seccion = linea[1:-1]
                continue
            if seccion != "Service":
                continue
            if "=" not in linea:
                continue
            clave, _, valor = linea.partition("=")
            clave = clave.strip()
            valor = valor.strip()
            if clave == "Environment":
                _fusionar_environment(entorno, valor)
            else:
                simples[clave] = valor
    return simples, entorno
