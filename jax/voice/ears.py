"""
JAX 2.0 — EarEngine: el OIDO de JAX (cliente del whisper_worker).

Flujo de /escucha:
  1. main.py corta la locucion en curso (que JAX no se oiga a si mismo).
  2. listen(): graba N segundos del microfono con arecord (jack trasero
     de la ALC897, calibrado: Capture 70%, Rear Mic Boost 0).
  3. Manda el WAV al whisper_worker (lazy, vive en el venv de los
     sentidos) y espera la transcripcion.
  4. Devuelve el texto, o None con motivo (silencio, alucinacion, error).

Sin huerfanos: el proceso arecord se guarda y se mata en shutdown() si JAX
muere a media grabacion. El worker tambien se cierra limpio.

En memoria de Jairo Urbina.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

WHISPER_PYTHON = Path.home() / "kokoro-test" / ".venv" / "bin" / "python"
WORKER_SCRIPT = Path(__file__).parent / "whisper_worker.py"
RECORD_DEVICE = "plughw:2,0"  # mismo jack ALC897 (entrada trasera calibrada)
TRANSCRIBE_TIMEOUT = 90.0  # generoso: incluye la carga del modelo la 1ra vez


class EarEngine:
    def __init__(self) -> None:
        self.worker: asyncio.subprocess.Process | None = None
        self.arecord: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self.last_reason: str | None = None  # por que se descarto lo ultimo

    # ----------------------------------------------------------------- #
    async def _ensure_worker(self) -> None:
        """Arranca (o revive) el whisper_worker con SU venv. La primera vez
        carga el modelo (unos segundos); las siguientes es inmediato."""
        if self.worker is not None and self.worker.returncode is None:
            return
        self.worker = await asyncio.create_subprocess_exec(
            str(WHISPER_PYTHON), str(WORKER_SCRIPT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

    async def _grabar(self, segundos: int) -> str | None:
        """Graba del microfono a un WAV temporal. None si arecord fallo."""
        fd_path = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="jax_mic_", delete=False
        )
        fd_path.close()
        ruta = fd_path.name

        self.arecord = await asyncio.create_subprocess_exec(
            "arecord", "-q", "-D", RECORD_DEVICE,
            "-f", "S16_LE", "-r", "16000", "-c", "1",
            "-d", str(segundos), ruta,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await self.arecord.wait()
        self.arecord = None
        if rc != 0:
            Path(ruta).unlink(missing_ok=True)
            return None
        return ruta

    # ----------------------------------------------------------------- #
    async def listen(self, segundos: int = 8) -> str | None:
        """Graba y transcribe. Devuelve el texto entendido, o None
        (el motivo queda en self.last_reason). El WAV lo borra el worker."""
        self.last_reason = None
        try:
            async with self._lock:
                ruta = await self._grabar(segundos)
                if ruta is None:
                    self.last_reason = "fallo de grabacion (arecord)"
                    return None

                await self._ensure_worker()
                req = json.dumps({"wav_path": ruta})
                self.worker.stdin.write((req + "\n").encode())
                await self.worker.stdin.drain()

                linea = await asyncio.wait_for(
                    self.worker.stdout.readline(), timeout=TRANSCRIBE_TIMEOUT
                )
                resp = json.loads(linea.decode().strip())
                texto = (resp.get("text") or "").strip()
                if not texto:
                    self.last_reason = resp.get("reason") or "sin texto"
                    return None
                return texto
        except asyncio.TimeoutError:
            self.last_reason = "el oido tardo demasiado (timeout)"
            return None
        except Exception as e:
            self.last_reason = f"error del oido: {e}"
            return None

    # ----------------------------------------------------------------- #
    async def shutdown(self) -> None:
        """Cierre limpio: mata arecord a media grabacion (sin huerfanos)
        y el worker de Whisper."""
        if self.arecord is not None and self.arecord.returncode is None:
            self.arecord.terminate()
            try:
                await asyncio.wait_for(self.arecord.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                self.arecord.kill()
                await self.arecord.wait()
        if self.worker is not None and self.worker.returncode is None:
            self.worker.terminate()
            try:
                await asyncio.wait_for(self.worker.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.worker.kill()
                await self.worker.wait()
