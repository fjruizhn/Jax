"""
Task 5 (2026-09-18, historial-y-arreglos-de-pipeline) -- permiso de lectura
para los pasos.

**Corrección de rumbo, con evidencia (no la premisa original del brief).**
El brief pedía "habilitar" a `jacobs` en `allowed_callers` de `file_read`.
Verificado contra `jax_memory` real (solo SELECT) y contra el seed de
`jax-platform/backend/db/migrations.py` (`_FILE_CAPABILITY_SEED`): `jacobs`
YA tenía 'jacobs' en `allowed_callers` de `file_read` **y** de `file_write`
desde GAP2 Fase2 (2026-08-19) -- no era el hueco. El hueco real era
`motor.has_tool_access`: de los cuatro motores que corren pipelines, solo
`jax_local` lo tenía en 1 -- `ada`/`kimi`/`thot` en 0, así que nunca
recibían el catálogo de tools sin importar el permiso del caller. Medido con
una llamada real por faceta contra su proveedor real: `ada` (glm-5.3, zhipu)
y `kimi` (kimi-k3, moonshot) responden HTTP 200 y llaman a `read_file`;
`thot` (gpt-6-astra, openai) responde HTTP 400 ("Function tools with
reasoning_effort are not supported for gpt-6-astra in
/v1/chat/completions"). La migración real de esta tarea es
`_seed_ada_kimi_has_tool_access` (jax-platform), no un cambio de
`allowed_callers`.

Estos dos tests quedan como FRENOS PERMANENTES:
  - el primero documenta que el permiso de lectura de `jacobs` es HISTORIA
    (GAP2 Fase2), no algo que esta tarea otorgó -- si alguna vez sale rojo,
    alguien cerró `allowed_callers` sin que la cadena de autoridad completa
    (plan.py, executor.py, tool_authority.py) se haya actualizado con él.
  - el segundo es un DETECTOR DE CAMBIO, no una afirmación de que la
    escritura está cerrada: hoy `jacobs` SÍ puede pedir `file_write` (Fase4,
    2026-08-19, `requires_human_gate=0` -- protección real es jail +
    forbidden_paths + git + auditoría posterior, no un gate de admisión).
    Es escritura al WORKSPACE con jail (`resolve_jailed_path`,
    `las_manos/motor_registry/tool_authority.py`) y `.env`/`secrets/`/
    `private_keys/`/`credentials/` en `forbidden_paths` -- NO escritura al
    repo git de `jax`/`jax-platform`; cerrarla hoy rompería los pipelines
    que ya producen documentos. Queda registrada como deuda para que
    Fernando decida (los tres contratos que el brief original pedía --
    clon en el jail, rama propia, compuerta humana -- siguen sin existir).
    Si este test sale rojo, es porque ESE valor cambió -- en cualquier
    dirección -- y alguien tiene que mirar por qué antes de seguir.

Corre en el job jacobs-gobernanza-db (DB con las migraciones de jax-platform
ya aplicadas). Solo lee.
"""
from __future__ import annotations

import asyncio

from las_manos.motor_registry.catalog import MotorCatalog


def test_jacobs_puede_pedir_file_read():
    """Historia, no una habilitación de esta tarea: 'jacobs' está en
    allowed_callers de file_read desde GAP2 Fase2 (2026-08-19), verificado
    contra jax_memory real antes de escribir código nuevo."""
    catalogo = asyncio.run(MotorCatalog.from_db())
    cap = catalogo.get_capability("file_read")
    assert cap is not None, "capability 'file_read' no sembrada -- ¿migraciones de jax-platform aplicadas?"
    assert "jacobs" in cap.allowed_callers


def test_detector_de_cambio_en_permiso_de_escritura_de_jacobs():
    """NO es una afirmación de que la escritura está cerrada -- hoy no lo
    está (Fase4, 2026-08-19, decisión registrada, deuda pendiente de que
    Fernando decida). Es un detector: si 'jacobs' alguna vez DEJA de estar
    en allowed_callers de file_write, o si alguien reabre esto pensando que
    estaba cerrado, este assert cambia de lado y hay que mirar por qué --
    en cualquiera de las dos direcciones."""
    catalogo = asyncio.run(MotorCatalog.from_db())
    cap = catalogo.get_capability("file_write")
    assert cap is not None, "capability 'file_write' no sembrada -- ¿migraciones de jax-platform aplicadas?"
    assert "jacobs" in cap.allowed_callers, (
        "el estado conocido y registrado (deuda pendiente de decisión de Fernando) "
        "es que jacobs SÍ puede pedir file_write hoy -- si esto cambió a False, "
        "alguien cerró la escritura: confirmar que fue intencional antes de seguir."
    )
