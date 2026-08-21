"""
Jacobs — PlanBuilder.

Genera un plan (lista de Steps) desde un objetivo textual.
v0.2: llama a JAX Local (facet jax_local via Ollama) para descomponer el
objetivo. Modelo resuelto desde facet_binding, no hardcodeado — ver
executor.py::_invoke_ollama, mismo patrón.
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

from jacobs.models import MOTOR_FACETS, Step
from credential_resolver import resolve_credential_instrumented, CredentialUnavailableError
from facet_resolver import resolve_facet, FacetUnavailableError
from model_catalog import record_resolved_version_safe

logger = logging.getLogger("jacobs.plan")

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TIMEOUT = 120  # segundos — el modelo local puede tardar
# T1 (2026-08-19): con think:false medido en 6 corridas reales (3 objetivos
# x think true/false), eval_count del path think:false fue 181-1159 (el mas
# alto: objetivo complejo de calculadora, 20 steps). Margen ~2.6x sobre el
# maximo medido -- cinturon y tirantes: acota el peor caso si algun
# objetivo dispara generacion inesperada, sin recortar ningun plan real
# observado. NO es el limite para un eventual fallback con thinking
# habilitado -- ese path necesitaria su propio budget, mucho mayor (el
# thinking solo midio 14780-19846 caracteres, ~3700-5000 tokens).
_LLM_PLAN_NUM_PREDICT = 3000

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


# Timeout por capability (segundos). El default (300s) cubre las capabilities
# sin evidencia de necesitar mas (validado con jacobs_steps.started_at/
# finished_at + las_manos/logs/motor_jobs.jsonl, 2026-08-20 -- ver
# jax-platform backend/db/migrations.py::_CAPABILITY_SEED, mismo alineamiento
# aplicado a capability.max_execution_minutes en esa misma fecha).
# 'assemble' NO va aqui: es mecanico (executor._assemble_mechanical, sin LLM)
# y completa en milisegundos, asi que mantiene el default.
#
# reconcile/design/reason SI necesitan mas: las tres acumulan contexto
# amplio (reconcile: todos los modulos + hallazgos del validador; design/
# reason: idem segun el objetivo). reconcile ya tenia el fix (incidente
# aec827a0, ~22K tokens de entrada, 300s no alcanzaba). design/reason se
# agregan ahora (ronda 4, T2.a) con la MISMA evidencia real que le faltaba
# al comentario anterior ("completan holgados en ~50-130s" -- correcto para
# la mayoria de corridas, pero FALSO como garantia: jacobs_steps muestra
# 1 step 'design' (de 36) y 1 'reason' (de 6) fallando de verdad, exacto en
# el techo de 300s, dur=300.0s=timeout_seconds, status=failed).
_DEFAULT_TIMEOUT_SECONDS = 300
_CAPABILITY_TIMEOUT_SECONDS = {
    "reconcile": 900,
    "design": 900,
    "reason": 900,
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


@dataclass(frozen=True)
class PlanViolation:
    """Una razón concreta por la que un step del plan no puede ejecutarse.
    Cada campo existe para que el mensaje de rechazo (T2: 'el error debe
    decir qué falta') pueda nombrar step/facet/motor/capability sin que el
    caller tenga que re-derivarlo."""
    step_index: int
    facet: str
    motor: str | None
    capability: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index, "facet": self.facet,
            "motor": self.motor, "capability": self.capability,
            "reason": self.reason,
        }


class PlanRejected(Exception):
    """T2 (2026-08-21, diagnóstico pipeline 19ad2c42-cdf): un plan
    inejecutable no se persiste. routes.py atrapa esto ANTES de
    store.pipeline_create() -- ni jacobs_pipelines ni jacobs_steps llegan a
    tener fila para este plan. El mensaje nombra step/motor/capability/razón
    para cada violación (no un rechazo opaco) y viaja tal cual al frontend
    vía HTTPException.detail."""

    def __init__(self, violations: list[PlanViolation]):
        self.violations = violations
        detail = "; ".join(
            f"step {v.step_index} (facet={v.facet}, motor={v.motor or v.facet}, "
            f"capability={v.capability}): {v.reason}"
            for v in violations
        )
        super().__init__(f"Plan rechazado -- {len(violations)} step(s) inejecutable(s): {detail}")


def _check_cleanroom(steps: list) -> list[PlanViolation]:
    """Devuelve violaciones clean-room (un step audita el trabajo de un dep
    del MISMO facet -- quien produce no puede ser quien aprueba).

    T3 (2026-08-21): hasta acá esto solo emitía logger.warning, dentro de
    _from_spec (nunca corría para planes del LLM, _from_objective). Ahora
    build() (abajo) trata cada violación como bloqueante -- mismo mecanismo
    que _validate_plan_capabilities (PlanRejected). Verificado contra los 59
    pipelines históricos en jacobs_steps antes de este cambio: 0 violaciones
    -- ningún flujo que hoy funciona se rompe."""
    violations = []
    by_index = {s.step_index: s for s in steps}
    for s in steps:
        if s.capability not in _AUDIT_CAPABILITIES:
            continue
        for dep in s.depends_on:
            dep_step = by_index.get(dep)
            if dep_step and dep_step.facet == s.facet:
                violations.append(PlanViolation(
                    s.step_index, s.facet, s.motor, s.capability,
                    f"audita al step {dep}, que es del MISMO facet '{s.facet}' "
                    f"-- no es auditoría independiente (cleanroom)",
                ))
    return violations


# T2: capabilities cuya ejecución real implica que el MOTOR llame una tool
# (TOOLS_CATALOG en las_manos/motor_registry/tools_catalog.py) durante el
# job, no que devuelva texto/JSON que otra pieza aplique después. Hoy
# TOOLS_CATALOG solo define read_file/write_file -- este conjunto crece si
# GAP2 agrega tools nuevas (no antes: no hay forma de inferirlo del nombre
# de la capability). 'implementation'/'refactor'/etc. quedan AFUERA a
# propósito: un motor sin tools puede satisfacerlas devolviendo un
# code_patch.v1 (ver output_validator.py) sin escribir nada él mismo --
# meterlas acá bloquearía flujos legítimos que hoy funcionan así (kimi/
# refactor, confirmado contra jax_memory real).
_TOOL_CAPABILITIES = frozenset({"file_read", "file_write"})


async def _validate_plan_capabilities(steps: list) -> None:
    """T2: valida cada step contra la DB real (capability_motor +
    motor.has_tool_access, jacobs/store.py::get_motor_governance) ANTES de
    que build() devuelva el plan. Solo aplica a steps cuyo dispatch real
    pasa por el Motor Registry de LAS MANOS (jacobs.models.MOTOR_FACETS) --
    los facets HTTP directos (hipatia/jekyll/thot/ada) no tienen fila de
    gobernanza que verificar acá, executor.py los despacha sin pasar por
    worker.py."""
    from jacobs import store as _store  # import diferido: evita ciclo store<->plan al import time

    relevant = [s for s in steps if (s.motor or s.facet) in MOTOR_FACETS]
    if not relevant:
        return
    governance = await _store.get_motor_governance()
    violations: list[PlanViolation] = []
    for step in relevant:
        motor_key = step.motor or step.facet
        entry = governance.get(motor_key)
        if entry is None:
            violations.append(PlanViolation(
                step.step_index, step.facet, step.motor, step.capability,
                f"motor '{motor_key}' no tiene fila en la tabla `motor`",
            ))
            continue
        if step.capability not in entry["allowed_capabilities"]:
            violations.append(PlanViolation(
                step.step_index, step.facet, step.motor, step.capability,
                f"capability '{step.capability}' no está en capability_motor para motor '{motor_key}'",
            ))
            continue
        if step.capability in _TOOL_CAPABILITIES and not entry["has_tool_access"]:
            violations.append(PlanViolation(
                step.step_index, step.facet, step.motor, step.capability,
                f"capability '{step.capability}' requiere ejecutar una tool, pero "
                f"motor '{motor_key}' no tiene has_tool_access (worker.py no le "
                f"entrega TOOLS_CATALOG -- ver motor_registry/worker.py)",
            ))
    if violations:
        raise PlanRejected(violations)


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
            steps = self._from_spec(pipeline_id, steps_spec)
        else:
            steps = await self._from_objective(pipeline_id, objective, max_steps)
        # T2/T3 (2026-08-21): gate único para AMBOS caminos -- vive acá, no
        # dentro de _from_spec ni _from_objective, para que ningún origen de
        # plan pueda saltárselo. cleanroom antes solo corría dentro de
        # _from_spec (nunca para planes del LLM) y solo advertía; ahora
        # bloquea para los dos caminos, mismo mecanismo que capabilities.
        cleanroom_violations = _check_cleanroom(steps)
        if cleanroom_violations:
            raise PlanRejected(cleanroom_violations)
        await _validate_plan_capabilities(steps)
        return steps

    def _from_spec(self, pipeline_id: str, specs: list[dict]) -> list[Step]:
        steps = []
        for i, spec in enumerate(specs):
            input_data = dict(spec.get("input", {}))
            # Si viene prompt al nivel del spec (no dentro de input), lo movemos.
            if spec.get("prompt") and "prompt" not in input_data:
                input_data["prompt"] = spec["prompt"]
            capability = spec.get("capability", "reason")
            # Default de timeout según capability; un timeout_seconds explícito en el
            # spec siempre tiene prioridad. `spec.get("timeout_seconds")` (sin default)
            # y no `spec.get("timeout_seconds", default_timeout)`: esta ultima forma
            # devuelve el VALOR de la clave si esta PRESENTE, aunque sea None -- y
            # routes.py arma `specs` con StepSpec.model_dump(), que SIEMPRE incluye
            # la clave. Antes de este fix (ronda 4, T2.a) StepSpec.timeout_seconds
            # tenia default=300 a nivel Pydantic, asi que la clave llegaba poblada
            # con 300 aunque el caller nunca la hubiera tocado, pisando el default
            # por-capability en cualquier pipeline con steps explicitos (no LLM-
            # planeado). Confirmado en produccion: jacobs_steps muestra 'reconcile'
            # con timeout_seconds=300 en 1 de 3 corridas reales pese a estar en el
            # dict con valor 900. `is not None` distingue "ausente/None" de
            # "0 explicito" (0 es un timeout real, agotado de inmediato).
            default_timeout = _CAPABILITY_TIMEOUT_SECONDS.get(
                capability, _DEFAULT_TIMEOUT_SECONDS
            )
            explicit_timeout = spec.get("timeout_seconds")
            steps.append(Step(
                step_id=str(uuid.uuid4()),
                pipeline_id=pipeline_id,
                step_index=i,
                facet=spec.get("facet", "jax_local"),
                motor=spec.get("motor"),
                capability=capability,
                input=input_data,
                depends_on=spec.get("depends_on", []),
                timeout_seconds=explicit_timeout if explicit_timeout is not None else default_timeout,
                skip_on_fail=spec.get("skip_on_fail", False),
            ))
        # T3 (2026-08-21): el chequeo de cleanroom se movió a build() -- corre
        # ahí para AMBOS caminos (acá y _from_objective) y bloquea en vez de
        # solo loguear. Ver _check_cleanroom / PlanRejected.
        return steps

    async def _from_objective(
        self,
        pipeline_id: str,
        objective: str,
        max_steps: int,
    ) -> list[Step]:
        dificultad = self._classify_difficulty(objective)
        try:
            _ada_disponible = bool(await resolve_credential_instrumented("zhipu"))
        except CredentialUnavailableError:
            _ada_disponible = False
        if dificultad == "formal" and _ada_disponible:
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
        try:
            api_key = await resolve_credential_instrumented("zhipu")
        except CredentialUnavailableError:
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
        try:
            f = await resolve_facet("jax_local")
        except FacetUnavailableError as exc:
            logger.warning("JAX Local no disponible para planificación: %s", exc)
            return None

        payload = {
            "model": f.model,
            "think": False,
            "options": {"num_predict": _LLM_PLAN_NUM_PREDICT},
            "messages": [
                {"role": "system", "content": _PLAN_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        # f.base_url ("http://localhost:11434/v1") es para el path OpenAI-compat
        # generico (_call_openai_compat) — el endpoint nativo /api/chat que este
        # payload espera (respuesta en data["message"]["content"]) es siempre
        # local y fijo. Solo el modelo viene del facet, nunca la URL.
        # OJO: GPU_SEMAPHORE (jax/muscles/ollama_muscle.py:37) es un
        # asyncio.Semaphore de PROCESO del REPL de JAX -- esta llamada corre
        # en el proceso de jax-las-manos (Jacobs) y le pega a Ollama directo
        # por httpx, sin pasar por ese semáforo. No hay exclusión mutua real
        # entre el REPL y Jacobs para el acceso a la GPU (verificado
        # 2026-08-19, sonda T0.a/T1 de latencia de _llm_plan).
        try:
            async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
                resp = await client.post(OLLAMA_URL, json=payload)
                resp.raise_for_status()
                data = resp.json()
                await record_resolved_version_safe(f.key, data.get("model"))
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
