"""
Jacobs — PlanBuilder.

Genera un plan (lista de Steps) desde un objetivo textual.
v0.2: llama a JAX Local (qwen3:14b via Ollama) para descomponer el objetivo.
Fallback: 3 steps genéricos si Ollama no responde o devuelve JSON inválido.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import json
import os
import uuid
import logging

import httpx

from jacobs.models import Step

logger = logging.getLogger("jacobs.plan")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3:14b"
OLLAMA_TIMEOUT = 120  # segundos — qwen3:14b puede tardar

ADA_URL = "https://api.z.ai/api/paas/v4/chat/completions"
ADA_MODEL = "glm-5.2"
ADA_TIMEOUT = 180  # Ada puede tardar más con razonamiento encendido

# Heurística v1 para clasificar objetivos — Fase D la refina con ejemplos de oro.
_FORMAL_KEYWORDS = frozenset({
    "contrato", "módulo", "módulos", "invariante", "esquema",
    "arquitectura", "especificación", "spec", "tipos comunes",
    "dependencias", "formaliza", "capabilities",
})

VALID_FACETS = frozenset({
    "hipatia", "jekyll", "thot", "ada", "kimi", "hyde", "jax_local",
})

_PLAN_SYSTEM = (
    "Eres Jacobs, el Director. Tu único trabajo es generar planes de ejecución "
    "como JSON. RESPONDE SOLO CON JSON VÁLIDO. Sin explicaciones, sin markdown, "
    "sin bloques de código. El JSON debe ser un array de objetos. "
    "PRINCIPIO DE EVIDENCIA (innegociable): al planificar no asumas ni inventes "
    "hechos no verificados; si un dato es incógnita, el plan debe incluir un step "
    "que lo verifique con evidencia real, nunca darlo por cierto. 'El que supone "
    "se equivoca.' (Tu salida sigue siendo SOLO el array JSON.)"
)

_PLAN_SYSTEM_MODULAR = (
    "Eres Jacobs, el Director, planificando trabajo FORMAL COMPLEJO. Generás un plan de "
    "ejecución como JSON (array de objetos), SOLO JSON, sin markdown ni explicaciones.\n\n"
    "Patrón OBLIGATORIO para trabajo formal (compilador de especificaciones):\n"
    "1. El PRIMER step SIEMPRE produce 'common_types': define UNA vez todos los tipos, enums "
    "e identificadores compartidos. Todos los demás módulos los referencian, ninguno los redefine.\n"
    "2. Luego los módulos en ORDEN DE DEPENDENCIA: cada módulo declara de qué steps anteriores "
    "depende (campo depends_on: lista de step_index). Un módulo va DESPUÉS de aquellos que necesita.\n"
    "3. Las piezas que referencian a todo (invariantes, validaciones globales) van AL FINAL.\n"
    "4. El ANTEPENÚLTIMO step es 'validación de consistencia' (facet thot, capability "
    "'validate_consistency'): revisa nombres huérfanos, tipos no definidos, referencias rotas. "
    "Devuelve SOLO discrepancias con referencia al step y nombre.\n"
    "5. El PENÚLTIMO step es 'reconciliación' (facet ada, capability 'reconcile'): recibe los "
    "hallazgos del validador y los módulos afectados, y produce SOLO los PARCHES puntuales que "
    "corrigen cada hallazgo (ej: agregar el método faltante a un módulo). NO reescribe los módulos "
    "completos — solo los fragmentos a corregir, identificando módulo y ubicación.\n"
    "6. El ÚLTIMO step es 'ensamble' (facet ada, capability 'assemble'): describe el manifest del "
    "paquete (orden de módulos, versiones, índice). El ensamble FÍSICO de los módulos lo hace el "
    "sistema mecánicamente; este step solo produce el manifest/índice, NO el documento completo.\n\n"
    "Cada step: {\"facet\",\"capability\",\"prompt\",\"depends_on\":[indices]}.\n"
    "- facet para diseño formal/tipos/arquitectura: 'ada'. Para crítica/auditoría: 'thot'. "
    "Para investigación: 'hipatia'. Para código: 'kimi'.\n"
    "- depends_on lista los step_index (0-based) de los steps cuyos OUTPUTS este step necesita.\n"
    "- El prompt de cada step debe ser autocontenido y referir explícitamente a sus dependencias "
    "(\"usando los tipos comunes del step 0 y las capabilities del step 1, definí...\").\n\n"
    "PRINCIPIO DE EVIDENCIA (innegociable): no asumas hechos no verificados; si un dato es "
    "incógnita, incluí un step que lo verifique. 'El que supone se equivoca.'\n\n"
    "Salida: SOLO el array JSON."
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
                depends_on=spec.get("depends_on", []),
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
        dificultad = self._classify_difficulty(objective)
        if dificultad == "formal" and (os.environ.get("ZAI_API_KEY") or os.environ.get("ZHIPU_API_KEY")):
            logger.info("Jacobs cerebro=Ada (formal) objective=%r", objective[:80])
            specs = await self._ada_plan(objective, max_steps)
            if not specs:
                logger.warning("Ada falló planificando, cayendo a qwen local")
                specs = await self._llm_plan(objective, max_steps)
        else:
            logger.info("Jacobs cerebro=qwen (trivial) objective=%r", objective[:80])
            specs = await self._llm_plan(objective, max_steps)
        if not specs:
            specs = self._fallback_plan(objective)
        return self._from_spec(pipeline_id, specs)

    @staticmethod
    def _classify_difficulty(objective: str) -> str:
        obj_lower = objective.lower()
        if len(objective) > 200:
            return "formal"
        if any(kw in obj_lower for kw in _FORMAL_KEYWORDS):
            return "formal"
        return "trivial"

    async def _ada_plan(self, objective: str, max_steps: int) -> list[dict] | None:
        api_key = os.environ.get("ZAI_API_KEY") or os.environ.get("ZHIPU_API_KEY", "")
        if not api_key:
            return None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        prompt = (
            f"Dado este objetivo formal: {objective}\n\n"
            f"Genera un plan de ejecución modular con MÁXIMO {max_steps} steps.\n"
            f"Seguí el patrón compilador OBLIGATORIO: common_types primero, módulos en orden "
            f"de dependencia, validación de consistencia (thot/validate_consistency) como antepenúltimo step, "
            f"reconciliación (ada/reconcile) como penúltimo, ensamble (ada/assemble) al final.\n"
            f"Cada step DEBE incluir el campo 'depends_on' con la lista de step_index "
            f"(0-based) de los que depende (lista vacía [] si no depende de ninguno).\n\n"
            f"Facetas disponibles: hipatia (investigar/research), jekyll (analizar), "
            f"thot (criticar/critique), ada (diseñar arquitectura/tipos), "
            f"kimi (coding), hyde (ejecutar cambios — requiere aprobación).\n\n"
            f"Ejemplo de forma esperada (no de contenido):\n"
            f'[{{"facet":"ada","capability":"design",'
            f'"prompt":"Definí los tipos comunes: enums, identificadores, estructuras base compartidas.",'
            f'"depends_on":[]}},'
            f'{{"facet":"ada","capability":"design",'
            f'"prompt":"Usando los tipos del step 0, definí el módulo de capabilities.",'
            f'"depends_on":[0]}},'
            f'{{"facet":"ada","capability":"design",'
            f'"prompt":"Usando tipos (0) y capabilities (1), definí las invariantes.",'
            f'"depends_on":[0,1]}},'
            f'{{"facet":"thot","capability":"validate_consistency",'
            f'"prompt":"Validá consistencia: nombres huérfanos, tipos no definidos, referencias rotas entre steps 0-2. Devolvé SOLO discrepancias.",'
            f'"depends_on":[0,1,2]}},'
            f'{{"facet":"ada","capability":"reconcile",'
            f'"prompt":"Aplicá SOLO los parches puntuales para corregir las discrepancias del step 3. Identificá módulo y ubicación de cada corrección.",'
            f'"depends_on":[3]}},'
            f'{{"facet":"ada","capability":"assemble",'
            f'"prompt":"Generá el manifest del paquete: orden de módulos, versiones, índice. El ensamble físico lo hace el sistema.",'
            f'"depends_on":[0,1,2,3,4]}}]\n\n'
            f"Responde SOLO con el array JSON."
        )
        messages = [
            {"role": "system", "content": _PLAN_SYSTEM_MODULAR},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": ADA_MODEL,
            "messages": messages,
            "stream": True,
            "max_tokens": 131072,
        }
        try:
            async with httpx.AsyncClient(timeout=ADA_TIMEOUT) as client:
                async with client.stream("POST", ADA_URL, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        logger.warning("Ada HTTP %s: %r", resp.status_code, body[:200])
                        return None
                    partes = []
                    async for linea in resp.aiter_lines():
                        if not linea or not linea.startswith("data:"):
                            continue
                        chunk_str = linea[5:].strip()
                        if chunk_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(chunk_str)
                        except json.JSONDecodeError:
                            continue
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        pieza = delta.get("content")
                        if pieza:
                            partes.append(pieza)
                    content = "".join(partes)
            # Fase D: aquí se capturará el plan de Ada como ejemplo de oro
            return self._parse_plan_json(content, max_steps)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ada no disponible para planificación: %s", exc)
            return None

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
        for idx, item in enumerate(data[:max_steps]):
            if not isinstance(item, dict):
                continue
            facet = item.get("facet", "")
            if facet not in VALID_FACETS:
                facet = "jax_local"
            # depends_on: filtrar valores no-enteros y fuera de rango (0 <= dep < idx)
            raw_deps = item.get("depends_on", [])
            depends_on = [
                int(x) for x in raw_deps
                if str(x).lstrip("-").isdigit() and 0 <= int(x) < idx
            ]
            valid.append({
                "facet": facet,
                "capability": str(item.get("capability", "reason"))[:50],
                "prompt": str(item.get("prompt", ""))[:2000],
                "depends_on": depends_on,
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
