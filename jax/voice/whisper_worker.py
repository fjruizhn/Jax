#!/usr/bin/env python3
"""
JAX 2.0 — whisper_worker: el OIDO de JAX (transcripcion local).

Corre con el Python del venv de Kokoro/sentidos (~/kokoro-test/.venv), NUNCA
con el venv de JAX. JAX lo lanza lazy como subproceso long-lived.

PROTOCOLO (texto, una linea por mensaje):
    stdin:  {"wav_path": "/tmp/jax_mic_XXXX.wav"}
    stdout: {"text": "...", "lang_prob": 0.99, "reason": null}
            text vacio + reason explica por que se descarto.
    El WAV se BORRA siempre tras procesarlo (exito o error).

DEFENSAS CONTRA ALUCINACIONES (correccion al diseno de Deep):
  1. GATE DE RMS (la defensa real): si el audio es casi silencio, NI SE
     TRANSCRIBE. Medido en hall9000: voz sana RMS ~0.09; el "¡Suscribete!"
     alucinado salio de audio mudo. Umbral: 0.01.
     (El filtro lang_prob de Deep NO servia: nuestra alucinacion real tuvo
     prob 1.00 en espanol — medido.)
  2. Lista de frases-alucinacion tipicas de Whisper (segunda red).

Modelo: "small" int8 CPU (cambiar MODEL_SIZE a "base" si la latencia medida
en hall9000 resulta alta — es un string).

En memoria de Jairo Urbina.
"""

import json
import os
import re
import sys

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

MODEL_SIZE = "small"
RMS_MIN = 0.01  # por debajo de esto = silencio (voz sana medida: ~0.09)

# Vocabulario de la casa: condiciona a Whisper para que reconozca los
# nombres propios. Medido jun-4: sin esto, "JAX" (Yax) lo normaliza a "John".
INITIAL_PROMPT = ("Conversación en casa con JAX, el asistente del hogar. "
                  "Hablan Fernando y Claudia. Facetas: JAX, Jekyll, Hyde, Hipatia.")

# Frases tipicas que Whisper alucina sobre audio vacio/ruido.
ALUCINACIONES = (
    "suscríbete", "suscribete", "subscribe", "thanks for watching",
    "gracias por ver", "dale like", "like y suscríbete", "comparte este video",
    "nos vemos en el próximo video", "hasta la próxima",
)

# Cargar el modelo UNA vez al arrancar el worker (el worker se lanza lazy
# desde JAX, asi que esta carga ocurre recien en el primer /escucha).
MODEL = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print(f"[whisper_worker] modelo {MODEL_SIZE} cargado", file=sys.stderr, flush=True)


def normalizar_nombres(texto: str) -> str:
    """El oido oye "Yax" (o variantes) y debe escribir "JAX".
    Incluye "John": Whisper normaliza el nombre desconocido a John
    (medido jun-4 con Claudia). Si algun dia hablan de un John real,
    quitarlo de esta lista."""
    return re.sub(r"\b(yax|yaks|yacks|jacks|jaks|llax|john)\b",
                  "JAX", texto, flags=re.IGNORECASE)


def procesar(wav_path: str) -> dict:
    # Gate de silencio ANTES de transcribir (mata alucinaciones de raiz).
    audio, _sr = sf.read(wav_path)
    rms = float(np.sqrt((np.asarray(audio, dtype=np.float64) ** 2).mean()))
    if rms < RMS_MIN:
        return {"text": "", "lang_prob": 0.0,
                "reason": f"silencio (rms {rms:.4f})"}

    segments, info = MODEL.transcribe(wav_path, language="es", beam_size=5,
                                      initial_prompt=INITIAL_PROMPT)
    texto = " ".join(s.text.strip() for s in segments).strip()
    texto = normalizar_nombres(texto)

    if not texto:
        return {"text": "", "lang_prob": info.language_probability,
                "reason": "transcripcion vacia"}

    bajo = texto.lower()
    if any(a in bajo for a in ALUCINACIONES) and len(texto.split()) <= 8:
        return {"text": "", "lang_prob": info.language_probability,
                "reason": f"alucinacion tipica descartada: {texto!r}"}

    return {"text": texto, "lang_prob": info.language_probability,
            "reason": None}


def main() -> None:
    for linea in sys.stdin:
        linea = linea.strip()
        if not linea:
            continue
        wav_path = None
        try:
            req = json.loads(linea)
            wav_path = req.get("wav_path")
            if not wav_path or not os.path.exists(wav_path):
                resp = {"text": "", "lang_prob": 0.0, "reason": "wav inexistente"}
            else:
                resp = procesar(wav_path)
        except Exception as e:  # el worker nunca muere por una peticion mala
            print(f"[whisper_worker] error: {e}", file=sys.stderr, flush=True)
            resp = {"text": "", "lang_prob": 0.0, "reason": f"error: {e}"}
        finally:
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:  # fail-soft: unlink de temporal en finally; nadie depende de que el borrado haya funcionado
                    pass
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
