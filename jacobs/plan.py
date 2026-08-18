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
from dataclasses import dataclass, field

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

# Espejo de VALID_FACETS para capabilities (FASE A §3.3). Vocabulario CERRADO
# que el planner puede emitir: todo lo de aquí tiene destino conocido en el
# catálogo (vía executor._CAPABILITY_MAP) o es mecánico ('assemble'). Una
# capability fuera de este conjunto se degrada a 'reason' (genérica, segura) en
# _parse_plan_json — nunca se deja pasar cruda. Debe mantenerse en sync con las
# claves de _CAPABILITY_MAP + 'assemble'.
VALID_CAPABILITIES = frozenset({
    # capabilities del catálogo (las_manos/config.toml)
    "generate", "reason", "design", "validate_consistency", "reconcile", "critique",
    "refactor", "pipeline_analysis", "implementation", "code_swarm",
    "bug_hunt", "architecture_review",
    # alias semánticos que _CAPABILITY_MAP traduce a catálogo
    "analysis", "research", "review", "code", "implement",
    # mecánico: cortocircuito en executor._dispatch_step (no toca motor)
    "assemble",
})


@dataclass(frozen=True)
class CapabilityUnbound:
    """Rechazo tipado — REFORMAS-v3.md R3.4. El scheduler lo intercepta y
    reenruta a uno de los candidates; el usuario nunca ve este estado."""
    required: list[str]
    candidates: list[str]
    task_id: str
    status: str = field(default="CAPABILITY_UNBOUND", init=False)

    def to_dict(self) -> dict:
        # frozen=True es shallow: las listas internas siguen siendo mutables.
        # Devolver copias evita que un caller mute self.required/candidates
        # a través del dict devuelto.
        return {
            "status": self.status,
            "required": list(self.required),
            "candidates": list(self.candidates),
            "task_id": self.task_id,
        }


# Timeout por capability (segundos). El default cubre design/validate/critique,
# que procesan contexto acotado y completan holgados en ~50-130s. Las capabilities
# que acumulan el contexto COMPLETO de N dependencias (reconcile recibe todos los
# módulos + hallazgos del validador y genera parches) necesitan más tiempo: con
# ~22K tokens de entrada, 300s no alcanza y el step muere en asyncio.wait_for.
# 'assemble' NO va aquí: es mecánico (executor._assemble_mechanical, sin LLM) y
# completa en milisegundos, así que mantiene el default.
_DEFAULT_TIMEOUT_SECONDS = 300
_CAPABILITY_TIMEOUT_SECONDS = {
    "reconcile": 900,
}

_PLAN_SYSTEM = (
    "Eres Jacobs, el Director. Tu único trabajo es generar planes de ejecución "
    "como JSON. RESPONDE SOLO CON JSON VÁLIDO. Sin explicaciones, sin markdown, "
    "sin bloques de código. El JSON debe ser un array de objetos. "
    "PRINCIPIO DE EVIDENCIA (innegociable): al planificar no asumas ni inventes "
    "hechos no verificados; si un dato es incógnita, el plan debe incluir un step "
    "que lo verifique con evidencia real, nunca darlo por cierto. 'El que supone "
    "se equivoca.' (Tu salida sigue siendo SOLO el array JSON.)"
)

_CLEANROOM_RULE = (
    "\nREGLA DE AUDITORÍA INDEPENDIENTE (clean-room): el step que audita, valida "
    "o critica el trabajo de otros steps DEBE usar un facet DISTINTO al de los "
    "steps que revisa. Un facet no se audita a sí mismo. Si los módulos backend "
    "y frontend fueron diseñados por 'ada' y 'kimi', su auditor debe ser 'thot' "
    "(u otro facet que no sea ada ni kimi). La revisión independiente es la "
    "garantía de calidad: quien produce no es quien aprueba.\n"
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
    "incógnita, incluí un step que lo verifique. 'El que supone se equivoca.'\n"
    + _CLEANROOM_RULE +
    "\nSalida: SOLO el array JSON."
)


_AUDIT_CAPABILITIES = frozenset({
    "validate_consistency", "critique", "review", "audit",
})


def _check_cleanroom(steps: list) -> list[str]:
    """Devuelve lista de violaciones clean-room (auditor con facet de un dep).
    No modifica el plan — solo reporta. 'El que supone se equivoca': declarar,
    no asumir que está bien."""
    warnings = []
    by_index = {s.step_index: s for s in steps}
    for s in steps:
        if s.capability not in _AUDIT_CAPABILITIES:
            continue
        for dep in s.depends_on:
            dep_step = by_index.get(dep)
            if dep_step and dep_step.facet == s.facet:
                warnings.append(
                    f"step {s.step_index} ({s.facet}/{s.capability}) audita al "
                    f"step {dep} que es del MISMO facet '{s.facet}' — no es "
                    f"auditoría independiente"
                )
    return warnings


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
            capability = spec.get("capability", "reason")
            # Default de timeout según capability; un timeout_seconds explícito en el
            # spec siempre tiene prioridad.
            default_timeout = _CAPABILITY_TIMEOUT_SECONDS.get(
                capability, _DEFAULT_TIMEOUT_SECONDS
            )
            steps.append(Step(
                step_id=str(uuid.uuid4()),
                pipeline_id=pipeline_id,
                step_index=i,
                facet=spec.get("facet", "jax_local"),
                capability=capability,
                input=input_data,
                depends_on=spec.get("depends_on", []),
                timeout_seconds=spec.get("timeout_seconds", default_timeout),
                skip_on_fail=spec.get("skip_on_fail", False),
            ))
        for w in _check_cleanroom(steps):
            logger.warning("Jacobs clean-room: %s", w)
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
            # capability CERRADA al vocabulario conocido (espejo de la mecánica
            # facet→jax_local de arriba). Fuera del conjunto → degradar a 'reason'.
            capability = str(item.get("capability", "reason"))[:50]
            if capability not in VALID_CAPABILITIES:
                logger.warning(
                    "Jacobs planner: capability '%s' fuera de VALID_CAPABILITIES "
                    "→ degradada a 'reason' (segura)", capability,
                )
                capability = "reason"
            valid.append({
                "facet": facet,
                "capability": capability,
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
