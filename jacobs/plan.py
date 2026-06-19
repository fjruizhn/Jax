"""
Jacobs — PlanBuilder.

Genera un plan (lista de Steps) desde un objetivo textual.
v0.2: llama a JAX Local (qwen3:14b via Ollama) para descomponer el objetivo.
Fallback: 3 steps genéricos si Ollama no responde o devuelve JSON inválido.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import json
import uuid
import logging

import httpx

from jacobs.models import Step

logger = logging.getLogger("jacobs.plan")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"
OLLAMA_TIMEOUT = 120  # segundos — qwen3:14b puede tardar

VALID_FACETS = frozenset({
    "hipatia", "jekyll", "thot", "ada", "kimi", "hyde", "jax_local",
})

_PLAN_SYSTEM = (
    "Eres Jacobs, el Director. Tu único trabajo es generar planes de ejecución "
    "como JSON. RESPONDE SOLO CON JSON VÁLIDO. Sin explicaciones, sin markdown, "
    "sin bloques de código. El JSON debe ser un array de objetos."
)


class PlanBuilder:
    """Construye un plan de steps desde un objetivo."""

    async def build(
        self,
        pipeline_id: str,
        objective: str,
        max_steps: int = 20,
        steps_spec: list[dict] | None = None,
    ) -> list[Step]:
        if steps_spec:
            return self._from_spec(pipeline_id, steps_spec)
        return await self._from_objective(pipeline_id, objective, max_steps)

    def _from_spec(self, pipeline_id: str, specs: list[dict]) -> list[Step]:
        steps = []
        for i, spec in enumerate(specs):
            input_data = dict(spec.get("input", {}))
            # Si viene prompt al nivel del spec (no dentro de input), lo movemos.
            if spec.get("prompt") and "prompt" not in input_data:
                input_data["prompt"] = spec["prompt"]
            steps.append(Step(
                step_id=str(uuid.uuid4()),
                pipeline_id=pipeline_id,
                step_index=i,
                facet=spec.get("facet", "jax_local"),
                capability=spec.get("capability", "reason"),
                input=input_data,
                timeout_seconds=spec.get("timeout_seconds", 300),
                skip_on_fail=spec.get("skip_on_fail", False),
            ))
        return steps

    async def _from_objective(
        self,
        pipeline_id: str,
        objective: str,
        max_steps: int,
    ) -> list[Step]:
        specs = await self._llm_plan(objective, max_steps)
        if not specs:
            specs = self._fallback_plan(objective)
        return self._from_spec(pipeline_id, specs)

    async def _llm_plan(self, objective: str, max_steps: int) -> list[dict] | None:
        prompt = (
            f"Dado este objetivo: {objective}\n\n"
            f"Genera un plan de ejecución con MÁXIMO {max_steps} steps.\n"
            f"Cada step debe tener: facet, capability, prompt específico.\n"
            f"Facetas disponibles: hipatia (investigar/research), jekyll (analizar), "
            f"thot (criticar/critique), ada (diseñar arquitectura), "
            f"kimi (coding), hyde (ejecutar cambios — requiere aprobación).\n"
            f"Responde SOLO con un array JSON. Ejemplo:\n"
            f'[{{"facet":"hipatia","capability":"research","prompt":"Investiga X"}},'
            f'{{"facet":"jekyll","capability":"analysis","prompt":"Analiza Y"}}]'
        )
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                resp = await client.post(OLLAMA_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                return self._parse_plan_json(content, max_steps)
        except Exception as exc:  # noqa: BLE001
            logger.warning("JAX Local no disponible para planificación: %s", exc)
            return None

    @staticmethod
    def _parse_plan_json(text: str, max_steps: int) -> list[dict] | None:
        # Extraer el primer bloque JSON del texto (puede venir con markdown o texto extra)
        text = text.strip()
        # Quitar bloques markdown ```json ... ```
        if "```" in text:
            for line in text.split("\n"):
                if line.strip().startswith("["):
                    text = line.strip()
                    break
            else:
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end != -1:
                    text = text[start:end + 1]
        elif not text.startswith("["):
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1:
                text = text[start:end + 1]

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(data, list):
            return None

        # Validar y limpiar cada step
        valid = []
        for item in data[:max_steps]:
            if not isinstance(item, dict):
                continue
            facet = item.get("facet", "")
            if facet not in VALID_FACETS:
                facet = "jax_local"
            valid.append({
                "facet": facet,
                "capability": str(item.get("capability", "reason"))[:50],
                "prompt": str(item.get("prompt", ""))[:2000],
            })

        return valid if valid else None

    @staticmethod
    def _fallback_plan(objective: str) -> list[dict]:
        return [
            {
                "facet": "hipatia",
                "capability": "research",
                "prompt": f"Investiga lo siguiente: {objective}",
            },
            {
                "facet": "jekyll",
                "capability": "analysis",
                "prompt": "Analiza la investigación anterior desde una perspectiva humanista.",
            },
            {
                "facet": "thot",
                "capability": "critique",
                "prompt": "Critica el análisis anterior. ¿Qué riesgos no se mencionaron?",
            },
        ]
