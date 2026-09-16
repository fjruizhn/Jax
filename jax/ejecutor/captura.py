"""Corre un comando y guarda su salida COMPLETA con procedencia.

Spec: §3.2. El truncado se marca, nunca se esconde: `cita.verificar`
rechaza toda afirmación que cite una captura truncada.

Decisiones que no estaban en el plan (2026-09-16, al implementar la Task 2),
cada una con su test en tests/test_ejecutor_captura.py:

- Se lee en BYTES y se decodifica con `errors="replace"`. Con `text=True` un
  byte que no es UTF-8 hacía lanzar a `correr` y se perdía la captura entera
  (el piso de §2.3); además `\\r\\n` se reescribía a `\\n` y `bytes_totales`
  mentía. `errors="ignore"` tampoco: borra el byte y pega lo de los lados
  (`12\\xff4` pasaría a ser `124`, un número que nadie imprimió).
- Hay plazo (`timeout_s`). Al vencer se mata el GRUPO de procesos entero:
  `subprocess.run(timeout=...)` con `shell=True` mata al shell y deja vivo al
  nieto. Una salida cortada por plazo es truncada.
- El tope acota la MEMORIA, no sólo lo que se guarda: se lee por bloques y lo
  que pasa del tope se cuenta y se descarta. `subprocess.run` cargaba la
  salida entera antes de cortarla.
- stderr también tiene tope y también trunca la captura.
- La captura dice POR QUÉ se truncó (`motivos_truncado`), como pide §3.2.
- stdin es /dev/null: un comando que pide datos termina en vez de esperar.
"""
from __future__ import annotations

import codecs
import os
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from jax.ejecutor.cita import Captura

TOPE_BYTES_POR_DEFECTO = 1_000_000
TIMEOUT_S_POR_DEFECTO = 120.0
# Después de matar el grupo, cuánto se espera a que cierren las tuberías. Un
# proceso que se escapó del grupo (setsid) puede tenerlas abiertas para
# siempre: sin este límite, el plazo no acotaría nada.
GRACIA_S = 2.0

TRUNCADO_POR_TOPE = "tope_bytes"
TRUNCADO_POR_TIMEOUT = "timeout"

_BLOQUE = 65_536


@dataclass(frozen=True)
class CapturaCompleta:
    maquina: str
    comando: str
    # None si el proceso no terminó ni después de matar su grupo.
    codigo: int | None
    salida: str
    stderr: str
    truncada: bool
    # Bytes que el comando escribió en stdout, aunque no se hayan guardado.
    bytes_totales: int
    # Momento en que EMPEZÓ a correr: es el lado seguro para un TTL.
    momento: str
    bytes_totales_stderr: int = 0
    motivos_truncado: tuple[str, ...] = ()


def _ahora_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _decodificar(datos: bytes, cortado: bool) -> str:
    # Si se cortó por tope, el último carácter puede haber quedado a medias:
    # `final=False` lo deja fuera en vez de convertirlo en basura.
    decodificador = codecs.getincrementaldecoder("utf-8")(errors="replace")
    return decodificador.decode(datos, final=not cortado)


def _matar_grupo(proceso: subprocess.Popen) -> None:
    try:
        os.killpg(proceso.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):  # fail-soft: el grupo ya no existe o tiene un miembro ajeno (sudo); el plazo igual se marca y GRACIA_S acota la espera
        return


def correr(comando: str, maquina: str, tope_bytes: int = TOPE_BYTES_POR_DEFECTO,
           timeout_s: float = TIMEOUT_S_POR_DEFECTO) -> CapturaCompleta:
    momento = _ahora_iso()
    proceso = subprocess.Popen(
        comando, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True)
    guardado = {proceso.stdout: bytearray(), proceso.stderr: bytearray()}
    totales = {proceso.stdout: 0, proceso.stderr: 0}
    vencido = False
    limite = time.monotonic() + timeout_s

    selector = selectors.DefaultSelector()
    try:
        for tuberia in guardado:
            selector.register(tuberia, selectors.EVENT_READ)
        while selector.get_map():
            restante = limite - time.monotonic()
            if restante <= 0:
                if vencido:
                    break  # ni matando el grupo se cerraron las tuberías
                _matar_grupo(proceso)
                vencido = True
                limite = time.monotonic() + GRACIA_S
                continue
            for clave, _ in selector.select(restante):
                bloque = os.read(clave.fd, _BLOQUE)
                if not bloque:
                    selector.unregister(clave.fileobj)
                    continue
                totales[clave.fileobj] += len(bloque)
                falta = tope_bytes - len(guardado[clave.fileobj])
                if falta > 0:
                    guardado[clave.fileobj] += bloque[:falta]
    finally:
        selector.close()
        proceso.stdout.close()
        proceso.stderr.close()

    # Las tuberías pueden cerrarse antes de que el proceso termine.
    try:
        proceso.wait(timeout=max(0.0, limite - time.monotonic()) if not vencido else GRACIA_S)
    except subprocess.TimeoutExpired:
        if not vencido:
            _matar_grupo(proceso)
            vencido = True
            try:
                proceso.wait(timeout=GRACIA_S)
            except subprocess.TimeoutExpired:  # fail-soft: el proceso no murió ni con SIGKILL al grupo; codigo queda None y la captura ya va marcada por timeout
                pass

    salida_cortada = totales[proceso.stdout] > tope_bytes
    stderr_cortado = totales[proceso.stderr] > tope_bytes
    motivos = []
    if salida_cortada or stderr_cortado:
        motivos.append(TRUNCADO_POR_TOPE)
    if vencido:
        motivos.append(TRUNCADO_POR_TIMEOUT)
    return CapturaCompleta(
        maquina=maquina, comando=comando, codigo=proceso.returncode,
        salida=_decodificar(bytes(guardado[proceso.stdout]), salida_cortada),
        stderr=_decodificar(bytes(guardado[proceso.stderr]), stderr_cortado),
        truncada=bool(motivos), bytes_totales=totales[proceso.stdout],
        momento=momento, bytes_totales_stderr=totales[proceso.stderr],
        motivos_truncado=tuple(motivos))


def a_captura(completa: CapturaCompleta) -> Captura:
    return Captura(comando=completa.comando, salida=completa.salida,
                   truncada=completa.truncada)
