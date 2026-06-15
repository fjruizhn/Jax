"""
JAX 2.0 — VoiceEngine v2.1: voz en STREAMING por oraciones (fix pipelining).

Historia del fix (jun-4):
  - v2 sufria UNDERRUNS (silencios entre oraciones): el buffer del pipe de
    aplay (~64KB = ~1.3s de audio) hacia que el drain de la oracion N
    bloqueara hasta que N casi terminara de sonar, y RECIEN ahi se pedia
    la N+1 (que tarda 1-6s en generarse). El pipelining era ilusorio.
  - v2.1: PREFETCH REAL — el request de la oracion N+1 se envia al worker
    ANTES de escribir la N al aplay. El worker genera N+1 EN PARALELO
    mientras N suena. Como la generacion es ~4x mas rapida que la
    reproduccion (medido), un solo frame de prefetch elimina los gaps.
  - Ademas: oraciones kilometricas (Jekyll y sus punto-y-comas) se trocean
    por comas a ~35 palabras — frames mas chicos, arranque mas rapido.
  - stderr de aplay silenciado (los "underrun!!!" no ensucian la pantalla).

Cancelacion: a lo sumo UN request en vuelo (el prefetch). /callate mata el
aplay y marca cancel; el speak descarta ese unico frame pendiente y termina.
El lock se libera en lo que tarde esa generacion (1-6s, aceptable).

Igual que siempre: lazy, texto completo SIEMPRE en pantalla, limpieza de
markdown/codigo/URLs, recorte a MAX_PALABRAS_VOZ en oraciones completas,
la voz jamas tumba el latido.

En memoria de Jairo Urbina.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

KOKORO_PYTHON = Path.home() / "kokoro-test" / ".venv" / "bin" / "python"
WORKER_SCRIPT = Path(__file__).parent / "kokoro_worker.py"
AUDIO_DEVICE = "plughw:2,0"  # ALC897 Analog (conector verde de hall9000)
SAMPLE_RATE = "24000"
MAX_PALABRAS_VOZ = 300   # decidido jun-4 (local = gratis; el limite es paciencia)
MAX_PALABRAS_FRAME = 35  # oraciones mas largas se trocean por comas


def limpiar_para_voz(texto: str) -> str:
    """Quita lo que suena horrible en voz: codigo, URLs, markdown, emojis."""
    # Fonetica del nombre: en pantalla "JAX", en la boca "Yax"
    # (Kokoro en espanol leeria "JAX" con jota de jamon).
    texto = re.sub(r"\bjax\b", "Yax", texto, flags=re.IGNORECASE)
    texto = re.sub(r"```.*?```", " ", texto, flags=re.DOTALL)
    texto = re.sub(r"`[^`]+`", " ", texto)
    texto = re.sub(r"https?://\S+|www\.\S+", " ", texto)
    texto = re.sub(r"\*\*([^*]+)\*\*", r"\1", texto)
    texto = re.sub(r"\*([^*]+)\*", r"\1", texto)
    texto = re.sub(r"^#+\s*", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"[^\w\s.,;:¿?¡!()%'\"-]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _trocear(oracion: str, max_palabras: int = MAX_PALABRAS_FRAME) -> list[str]:
    """Parte una oracion kilometrica en trozos por comas/punto-y-comas.
    Frames chicos = arranque mas rapido y prefetch mas efectivo."""
    if len(oracion.split()) <= max_palabras:
        return [oracion]
    partes = re.split(r"(?<=[,;:])\s+", oracion)
    trozos: list[str] = []
    actual: list[str] = []
    acum = 0
    for p in partes:
        n = len(p.split())
        if actual and acum + n > max_palabras:
            trozos.append(" ".join(actual))
            actual, acum = [], 0
        actual.append(p)
        acum += n
    if actual:
        trozos.append(" ".join(actual))
    return trozos


def partir_oraciones(texto: str) -> list[str]:
    """Parte el texto en oraciones (la granularidad del streaming),
    troceando las kilometricas."""
    oraciones = re.split(r"(?<=[.!?])\s+", texto)
    resultado: list[str] = []
    for o in oraciones:
        o = o.strip()
        if o:
            resultado.extend(_trocear(o))
    return resultado


def recortar_oraciones(oraciones: list[str], max_palabras: int) -> list[str]:
    """Conserva oraciones COMPLETAS hasta el limite; avisa si recorto."""
    total = sum(len(o.split()) for o in oraciones)
    if total <= max_palabras:
        return oraciones
    elegidas: list[str] = []
    acum = 0
    for o in oraciones:
        n = len(o.split())
        if elegidas and acum + n > max_palabras:
            break
        elegidas.append(o)
        acum += n
        if acum >= max_palabras:
            break
    elegidas.append("El resto te lo dejo en pantalla.")
    return elegidas


class VoiceEngine:
    def __init__(self) -> None:
        self.worker: asyncio.subprocess.Process | None = None
        self.aplay: asyncio.subprocess.Process | None = None
        self.enabled = False
        self._lock = asyncio.Lock()
        self._cancel = asyncio.Event()

    # ----------------------------------------------------------------- #
    async def _ensure_worker(self) -> None:
        """Arranca (o revive) el worker de Kokoro con SU venv."""
        if self.worker is not None and self.worker.returncode is None:
            return
        self.worker = await asyncio.create_subprocess_exec(
            str(KOKORO_PYTHON), str(WORKER_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def _enviar(self, texto: str, voz: str, velocidad: float) -> None:
        """Envia UNA oracion al worker (no espera la respuesta)."""
        req = json.dumps({"text": texto, "voice": voz, "speed": velocidad})
        self.worker.stdin.write((req + "\n").encode())
        await self.worker.stdin.drain()

    async def _leer_frame(self) -> bytes:
        """Lee la respuesta del worker: 4 bytes tamano + PCM. b'' si error."""
        size_raw = await self.worker.stdout.readexactly(4)
        size = int.from_bytes(size_raw, "little")
        if size == 0:
            return b""
        return await self.worker.stdout.readexactly(size)

    # ----------------------------------------------------------------- #
    async def speak(self, texto: str, voz: str = "em_alex",
                    velocidad: float = 1.0) -> None:
        """Locucion en streaming con prefetch real. Pensada como task de
        fondo. Si habia una locucion en curso, la corta (interrumpir > encolar)."""
        if not self.enabled:
            return
        oraciones = recortar_oraciones(
            partir_oraciones(limpiar_para_voz(texto)), MAX_PALABRAS_VOZ
        )
        if not oraciones:
            return

        await self.stop_playing()

        pendiente = False  # hay un request en vuelo sin respuesta leida
        try:
            async with self._lock:
                self._cancel.clear()
                await self._ensure_worker()

                # UN solo aplay raw para toda la locucion: PCM continuo.
                self.aplay = await asyncio.create_subprocess_exec(
                    "aplay", "-q", "-t", "raw",
                    "-r", SAMPLE_RATE, "-f", "S16_LE", "-c", "1",
                    "-D", AUDIO_DEVICE,
                    stdin=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )

                # Arrancar la generacion de la PRIMERA oracion ya.
                await self._enviar(oraciones[0], voz, velocidad)
                pendiente = True

                for i in range(len(oraciones)):
                    if self._cancel.is_set():
                        break
                    pcm = await self._leer_frame()
                    pendiente = False
                    if self._cancel.is_set():
                        break

                    # PREFETCH: pedir la siguiente ANTES de escribir esta.
                    # El worker genera i+1 EN PARALELO mientras i suena.
                    if i + 1 < len(oraciones):
                        await self._enviar(oraciones[i + 1], voz, velocidad)
                        pendiente = True

                    if not pcm:
                        continue  # esa oracion fallo; seguimos con la otra
                    try:
                        self.aplay.stdin.write(pcm)
                        await self.aplay.stdin.drain()  # backpressure: ok,
                        # la siguiente YA se esta generando en el worker.
                    except (BrokenPipeError, ConnectionResetError):
                        break  # aplay murio (/callate); salir limpio

                # Drenar el unico frame en vuelo (protocolo siempre en sync).
                if pendiente:
                    try:
                        await self._leer_frame()
                    except Exception:
                        pass

                # Cierre: sin mas datos, aplay termina solo lo que tenga.
                if self.aplay is not None:
                    try:
                        if self.aplay.stdin is not None:
                            self.aplay.stdin.close()
                        if not self._cancel.is_set():
                            await self.aplay.wait()
                    except Exception:
                        pass
                    self.aplay = None
        except Exception:
            # La voz es un plus: jamas tumba la conversacion.
            pass

    # ----------------------------------------------------------------- #
    async def stop_playing(self) -> None:
        """Corta la locucion en curso (/callate o nueva locucion).
        NO toma el lock: marca cancel y mata el aplay; el speak en curso
        despierta (drain roto o frame descartado) y libera el lock solo."""
        self._cancel.set()
        proc = self.aplay
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

    async def shutdown(self) -> None:
        """Cierre limpio al salir de JAX: corta audio y mata el worker."""
        await self.stop_playing()
        if self.worker is not None and self.worker.returncode is None:
            self.worker.terminate()
            try:
                await asyncio.wait_for(self.worker.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.worker.kill()
                await self.worker.wait()
