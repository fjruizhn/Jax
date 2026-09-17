"""
JAX 2.0 — OllamaMuscle (JAX local).

La faceta por defecto de JAX: corre LOCAL en la GPU de hall9000 (AMD Radeon
AI PRO R9700 32GB via ROCm), sin nube, privada. Es JAX en su modo de confianza.

Decisiones firmes (Fernando + DeepSeek + Claude), formato verificado en hall9000
contra Ollama 0.24.0:
  - API HTTP local en $JAX_OLLAMA_URL/api/chat (E-21). SIN api key (es local).
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

from jax.core.contrato_dispatch import ModelDispatchConfigError, limite_de_salida
from jax.core.model_catalog import record_resolved_version_safe
from jax.core.redaccion import recortar_redactado
from jax.core.cliente_http_compartido import obtener_cliente_http
from jax.muscles.base import DispatchConfigMuscleError, Muscle, MuscleInvocationError

# Semaforo de GPU -- UNA inferencia local a la vez DENTRO DE ESTE PROCESO.
#
# Es un asyncio.Semaphore de proceso, NO cross-proceso. El comentario que
# estaba aca antes decia "compartido a nivel de modulo para que cualquier
# carga GPU futura lo respete", y era FALSO: Jacobs corre en el proceso
# jax-las-manos y tiene 3 caminos propios a Ollama que no pasan por aca
# (jacobs/plan.py::_llm_plan, jacobs/executor.py::_invoke_ollama,
# las_manos/motor_registry/worker.py con transporte 'ollama').
#
# Medido el 2026-08-28 (scripts/gpu_concurrency_probe.py): no hace falta
# exclusion mutua cross-proceso HOY, pero no porque la GPU aguante --
# porque Ollama serializa (OLLAMA_NUM_PARALLEL=1) y la exclusion ya existe
# afuera.
#
# Ese valor SI lo fija alguien: Environment=OLLAMA_NUM_PARALLEL=1 en la
# unidad de ollama (el comentario anterior decia lo contrario y ya no es
# cierto). Pero la unidad declara, no garantiza -- el proceso puede correr
# con otra cosa por un drop-in, un set-environment o un arranque a mano. El
# invariante se verifica contra el ENTORNO DEL PROCESO VIVO:
# scripts/check_ollama_num_parallel.py, job `ollama-num-parallel` en CI.
# Si ese tripwire se pone rojo, esta decision no se degrada: se INVIERTE.
#
# Veredicto completo y que lo reabre:
# docs/superpowers/specs/2026-08-25-gpu-concurrency-resultado.md
GPU_SEMAPHORE = asyncio.Semaphore(1)


class OllamaMuscle(Muscle):
    # provider_id del catálogo del modelo (lo pone build_muscles desde el
    # registro); con él se lee el contrato de la fila. PR-K ronda 2 (M3).
    provider_id: str = ""

    def __init__(
        self,
        name: str,
        model_default: str,
        models_allowed: list[str],
        system_prompt: str,
        timeout: float,
        *,
        api_url: str,
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

        # PR-K ronda 2 (M3): límite de salida de la fila de `model`, como
        # options.num_predict de /api/chat (doc: github.com/ollama/ollama
        # docs/api.md). Antes no mandaba ninguno.
        try:
            limite = await limite_de_salida("ollama", self.provider_id, model)
        except ModelDispatchConfigError as e:
            raise DispatchConfigMuscleError(f"[{self.name}] dispatch abortado: {e}") from e
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            **limite,
        }

        # Una sola inferencia local a la vez. Si la GPU esta ocupada, espera turno.
        async with GPU_SEMAPHORE:
            resp = await obtener_cliente_http().post(self.api_url, json=payload, timeout=self.timeout)
            if resp.status_code != 200:
                raise MuscleInvocationError(
                    f"[{self.name}] Ollama HTTP {resp.status_code}: "
                    f"{recortar_redactado(resp.text, 200)}"
                )
            data = resp.json()
            try:
                contenido = data["message"]["content"]
            except (KeyError, TypeError) as exc:
                # JSON ya parseado de un 200 del Ollama local (sin credencial): fuera de E-16, declarado.
                raise MuscleInvocationError(
                    f"[{self.name}] respuesta inesperada de Ollama: "
                    f"{str(data)[:200]}"
                ) from exc

        # D1.2 — capturado por consistencia con los transportes HTTP; ver
        # CONTEXT.md ("decision previa al wiring de resolved_version en
        # REPL/Jacobs") para la limitacion real: los tags de Ollama no son
        # alias moviles del proveedor, un drift de PESOS bajo el mismo tag
        # no es detectable con este campo (haria falta /api/show + digest).
        await record_resolved_version_safe(self.name, data.get("model"))
        return contenido
