"""
JAX 2.0 — OllamaMuscle (JAX local).

La faceta por defecto de JAX: corre LOCAL en la GPU de hall9000 (AMD Radeon
AI PRO R9700 32GB via ROCm), sin nube, privada. Es JAX en su modo de confianza.

Decisiones firmes (Fernando + DeepSeek + Claude), formato verificado en hall9000
contra Ollama 0.24.0:
  - API HTTP local en localhost:11434/api/chat. SIN api key (es local).
  - La respuesta viene en message.content (verificado con curl).
  - Semaforo GPU de 1: UNA sola inferencia local a la vez, para no saturar la
    VRAM de 32GB ni degradar tok/s. Protege la regla de concurrencia=1 en GPU.
    NOTA (jul-2026): con 32GB hay margen para revisar si concurrencia=1 sigue
    siendo necesario, o si 2 inferencias simultaneas caben sin degradar tok/s
    — pendiente de medir, no asumido.
    El semaforo es de modulo (compartido por toda carga GPU futura).
  - System prompt inyectado como mensaje role=system: sin el, el modelo 3b se
    comporta erratico (llega a rechazar saludos). Con identidad responde bien.
  - Modelo es parametro, validado con fallo duro igual que las demas facetas.
  - Memoria: el historial previo (compartido) se inserta entre el system y el
    mensaje actual, en el mismo formato role/content que ya usa Ollama.

En memoria de Jairo Urbina.
"""

from __future__ import annotations

import asyncio

import httpx

from jax.muscles.base import Muscle, MuscleInvocationError, MuscleTimeoutError

# Semaforo global de GPU: UNA inferencia local a la vez en hall9000.
# Compartido a nivel de modulo para que cualquier carga GPU futura lo respete.
GPU_SEMAPHORE = asyncio.Semaphore(1)

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"


class OllamaMuscle(Muscle):
    def __init__(
        self,
        name: str,
        model_default: str,
        models_allowed: list[str],
        system_prompt: str,
        timeout: float,
        api_url: str = DEFAULT_OLLAMA_URL,
        authority_origin: str = "",
    ) -> None:
        super().__init__(
            name, model_default, models_allowed, system_prompt, timeout,
            authority_origin=authority_origin,
        )
        self.api_url = api_url

    async def _call(
        self, prompt: str, model: str, history: list[dict] | None = None
    ) -> str:
        # messages = system + historial previo + mensaje actual.
        # El historial ya viene en {"role": "user"|"assistant", ...}, que es
        # justo lo que Ollama (/api/chat estilo OpenAI) espera. Sin duplicar.
        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
        }

        # Una sola inferencia local a la vez. Si la GPU esta ocupada, espera turno.
        async with GPU_SEMAPHORE:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(self.api_url, json=payload)
                if resp.status_code != 200:
                    raise MuscleInvocationError(
                        f"[{self.name}] Ollama HTTP {resp.status_code}: "
                        f"{resp.text[:200]}"
                    )
                data = resp.json()
                try:
                    return data["message"]["content"]
                except (KeyError, TypeError) as exc:
                    raise MuscleInvocationError(
                        f"[{self.name}] respuesta inesperada de Ollama: "
                        f"{str(data)[:200]}"
                    ) from exc
