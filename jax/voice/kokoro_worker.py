#!/usr/bin/env python3
"""
JAX 2.0 — kokoro_worker v2: proceso dedicado de voz (TTS), modo STREAMING.

Corre con el Python del venv de Kokoro (~/kokoro-test/.venv), NUNCA con el
venv de JAX. JAX lo lanza como subproceso long-lived y le habla por pipes.

PROTOCOLO v2 (un request = UNA ORACION = un frame):
    stdin (una linea JSON):  {"text": "Una oracion.", "voice": "em_alex", "speed": 1.0}
    stdout (binario):        4 bytes little-endian (tamano) + PCM crudo.
                             Tamano 0 = error (detalle por stderr).
    PCM: int16 little-endian, 24000 Hz, mono, SIN header WAV — listo para
    alimentar un aplay en modo raw (-t raw -r 24000 -f S16_LE -c 1).

Por que una oracion por request (decision Fernando+Deep+Claude, jun-4):
  - El split en oraciones lo hace JAX (VoiceEngine): control total de la
    granularidad (medido: Kokoro por defecto NO parte por oraciones).
  - Con a lo sumo UN frame en vuelo, la cancelacion (/callate) es trivial:
    no pedir mas + descartar el unico frame pendiente. Sin drenados
    complejos, sin desincronizacion posible.
  - El pipelining (generar N+1 mientras suena N) lo da el flujo natural:
    JAX escribe el PCM de N al aplay (que suena en paralelo) y pide N+1.

Correcciones sobre el diseno de Deep:
  - np.clip antes de convertir a int16 (sin clip, picos >1.0 hacen wrap y
    suenan a chasquido).
  - Sin VOICE_MAP (era identidad: codigo muerto). Voces directas.
  - El audio se acumula de TODOS los chunks del pipeline (su version hacia
    return dentro del for y botaba el resto).

En memoria de Jairo Urbina.
"""

import json
import sys

import numpy as np
from kokoro import KPipeline

# 'e' = espanol. Verificado en hall9000. ('a' seria ingles americano.)
PIPELINE = KPipeline(lang_code="e")

SAMPLE_RATE = 24000  # Kokoro genera a 24 kHz (medido)


def generar_pcm(texto: str, voz: str, velocidad: float) -> bytes:
    """Genera el audio de UNA oracion como PCM crudo int16 LE mono."""
    chunks = []
    for _, _, audio in PIPELINE(texto, voice=voz, speed=velocidad):
        chunks.append(np.asarray(audio, dtype=np.float32))
    if not chunks:
        return b""
    combinado = np.concatenate(chunks)
    # Clip a [-1, 1] y a int16. x86 es little-endian: tobytes() ya es S16_LE.
    pcm = (np.clip(combinado, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm.tobytes()


def main() -> None:
    for linea in sys.stdin:
        linea = linea.strip()
        if not linea:
            continue
        try:
            req = json.loads(linea)
            pcm = generar_pcm(
                req["text"],
                req.get("voice", "em_alex"),
                float(req.get("speed", 1.0)),
            )
            sys.stdout.buffer.write(len(pcm).to_bytes(4, "little"))
            if pcm:
                sys.stdout.buffer.write(pcm)
            sys.stdout.buffer.flush()
        except Exception as e:  # el worker nunca muere por una peticion mala
            print(f"[kokoro_worker] error: {e}", file=sys.stderr, flush=True)
            sys.stdout.buffer.write((0).to_bytes(4, "little"))
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
