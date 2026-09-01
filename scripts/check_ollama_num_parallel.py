#!/usr/bin/env python3
"""Tripwire de OLLAMA_NUM_PARALLEL -- lee el ENTORNO DEL PROCESO VIVO.

Por que existe
--------------
El item de DEUDA.md "GPU_SEMAPHORE no cubre a Jacobs" se cerro como DECISION
TOMADA, no como pendiente, y el argumento entero cuelga de UN invariante
medido el 2026-08-28 (scripts/gpu_concurrency_probe.py):

    no hace falta exclusion mutua cross-proceso entre el REPL y Jacobs
    PORQUE Ollama serializa, y Ollama serializa porque OLLAMA_NUM_PARALLEL=1.

Si ese valor deja de ser 1, la decision no se degrada: se INVIERTE. Los tres
caminos a Ollama que no pasan por GPU_SEMAPHORE (jacobs/plan.py::_llm_plan,
jacobs/executor.py::_invoke_ollama, las_manos/motor_registry/worker.py con
transporte 'ollama') pasan a correr en paralelo de verdad contra la misma GPU.
Una decision que depende de un invariante sin tripwire es una suposicion con
fecha de vencimiento desconocida.

Por que /proc/<pid>/environ y no el archivo de unidad
-----------------------------------------------------
`systemctl cat ollama` y `systemctl show -p Environment` dicen lo que la
unidad DECLARA. El proceso puede estar corriendo con otra cosa: un drop-in
posterior, un `systemctl set-environment`, un arranque manual fuera de
systemd, o simplemente una unidad editada y nunca recargada. El unico lugar
donde vive el valor con el que Ollama esta corriendo AHORA es el entorno del
proceso. Leer el archivo de unidad seria medir la intencion, no el hecho --
exactamente el error que la primera leccion de metodo describe ("el CI en
verde no es evidencia de que el CI funcione").

Restriccion real, medida y no disimulada
----------------------------------------
Ollama corre como usuario `ollama`; /proc/<pid>/environ es 0400 del duenio,
asi que la lectura del proceso REAL exige root (sudo) o correr como `ollama`.
Por eso el chequeo en vivo es de hall9000 y no de un runner de GitHub: en
`ubuntu-latest` no hay ningun Ollama que leer. Lo que CI si verifica es que
ESTE tripwire funciona, apuntandolo a procesos vivos de verdad lanzados con
el valor puesto a mano (ver tests/test_ollama_num_parallel.py).

Fail-closed en todos los caminos
--------------------------------
Verde EXCLUSIVAMENTE si se pudo leer el entorno de un proceso vivo y ahi
OLLAMA_NUM_PARALLEL vale exactamente 1. Cualquier otra cosa -- variable
ausente, valor distinto, archivo ilegible, proceso inexistente, environ
vacio -- es rojo. Un tripwire que se pone en verde cuando no pudo medir es
el patron fail-open que este repo ya cerro en 19 archivos.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

VAR = "OLLAMA_NUM_PARALLEL"
EXPECTED = "1"


class TripwireError(Exception):
    """El invariante no se pudo confirmar. Siempre rojo, nunca verde."""


def parse_environ(blob: bytes) -> dict[str, str]:
    """Parsea el formato de /proc/<pid>/environ: pares NUL-separados.

    No usa split('=', 1) sobre el blob entero: un valor puede contener '='.
    Entradas sin '=' se descartan (existen en environs reales).
    """
    env: dict[str, str] = {}
    for chunk in blob.split(b"\0"):
        if not chunk:
            continue
        # errors="replace" no lanza: no hace falta un except aca, y uno de mas
        # seria un fail-open silencioso sobre bytes que no se supieron leer.
        text = chunk.decode("utf-8", errors="replace")
        if "=" not in text:
            continue
        key, _, value = text.partition("=")
        env[key] = value
    return env


def read_environ(path: str | os.PathLike[str]) -> dict[str, str]:
    """Lee y parsea un environ. Cualquier fallo de lectura es TripwireError."""
    try:
        blob = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise TripwireError(f"no existe {path} -- el proceso murio o nunca estuvo") from exc
    except PermissionError as exc:
        raise TripwireError(
            f"sin permiso para leer {path}. Ollama corre como usuario 'ollama' y "
            f"/proc/<pid>/environ es 0400 del duenio: hace falta sudo"
        ) from exc
    except OSError as exc:
        raise TripwireError(f"no se pudo leer {path}: {exc}") from exc
    if not blob:
        raise TripwireError(
            f"{path} vino vacio -- pasa con procesos zombie; no es evidencia de nada"
        )
    return parse_environ(blob)


def check_environ(env: dict[str, str]) -> None:
    """Aplica el invariante. Silencio = verde; TripwireError = rojo."""
    if VAR not in env:
        raise TripwireError(
            f"{VAR} no esta en el entorno del proceso. Sin ese valor Ollama usa su "
            f"default, que NO es 1 y cambia entre versiones -- la exclusion mutua "
            f"de la que depende la decision de GPU_SEMAPHORE deja de estar garantizada"
        )
    actual = env[VAR]
    if actual != EXPECTED:
        raise TripwireError(
            f"{VAR}={actual!r}, se esperaba {EXPECTED!r}. Con num_parallel > 1 Ollama "
            f"deja de serializar y los tres caminos que NO pasan por GPU_SEMAPHORE "
            f"(jacobs/plan.py::_llm_plan, jacobs/executor.py::_invoke_ollama, "
            f"las_manos/motor_registry/worker.py transporte 'ollama') corren en "
            f"paralelo real contra la misma GPU. La decision de DEUDA.md se INVIERTE"
        )


def is_ollama_serve(cmdline: str) -> bool:
    """True si esta cmdline es la del servidor de Ollama.

    Existe como funcion aparte porque `pgrep -f` matchea contra la linea de
    comando COMPLETA y por lo tanto se engancha a cualquier cosa que mencione
    el patron -- incluido el propio `bash -c` que lo invoca, cosa observada de
    verdad al medir este item. Colar un falso positivo aca no da un error
    ruidoso: da un environ ajeno y un veredicto sobre el proceso equivocado.
    """
    parts = cmdline.split()
    if len(parts) < 2:
        return False
    exe = parts[0].rsplit("/", 1)[-1]
    return exe == "ollama" and parts[1] == "serve"


def find_ollama_pid() -> int:
    """Ubica el `ollama serve` vivo. Cero o mas de uno es rojo, no una eleccion."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", r"ollama\s+serve"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise TripwireError(f"no se pudo ejecutar pgrep: {exc}") from exc
    pids = [int(p) for p in out.stdout.split() if p.isdigit()]
    # pgrep -f se ve a si mismo si el patron aparece en la propia linea de
    # comando; filtramos por el ejecutable real para no contar de mas.
    real = []
    for pid in pids:
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if is_ollama_serve(cmdline.strip()):
            real.append(pid)
    if not real:
        raise TripwireError(
            "no hay ningun proceso `ollama serve` vivo. Este chequeo mide el "
            "entorno de un proceso corriendo; sin proceso no hay nada que medir "
            "y eso es rojo, no 'no aplica'"
        )
    if len(real) > 1:
        raise TripwireError(
            f"hay {len(real)} procesos `ollama serve` vivos ({real}). El invariante "
            f"se afirma sobre 'el' servidor; con varios no se sabe cual atiende"
        )
    return real[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--environ-file",
        help="ruta a un environ en formato /proc (por defecto: el del `ollama serve` vivo)",
    )
    args = parser.parse_args(argv)

    try:
        if args.environ_file:
            source = args.environ_file
        else:
            source = f"/proc/{find_ollama_pid()}/environ"
        env = read_environ(source)
        check_environ(env)
    except TripwireError as exc:
        print(f"ROJO: {exc}", file=sys.stderr)
        print(f"  fuente: {args.environ_file or 'proceso `ollama serve` vivo'}", file=sys.stderr)
        return 1

    print(f"VERDE: {VAR}={EXPECTED} en el entorno del proceso vivo ({source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
