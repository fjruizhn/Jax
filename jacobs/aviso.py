"""
Jacobs — Aviso gobernado por Telegram cuando un pipeline termina.

Dónde encaja (Task 6, plan 2026-09-18-historial-y-arreglos-de-pipeline): el
correo lo manda jax-platform (Task 8, `backend/aviso_pipeline.py`) -- Telegram
lo manda este repo, porque es el único que tiene ese canal
(`send_telegram_alert`, jacobs/reaper.py:82, ya reusado por
las_manos/motor_registry/worker.py). Este módulo NO escribe un segundo
cliente de Telegram: arma el texto y reusa `send_telegram_alert` tal cual.

La superficie se conecta a la frontera de salida gobernada: F2-B acredita
PIPELINE_STATUS desde la persistencia canónica de Jacobs, F2-C renderiza el
texto y F2-D valida la unidad de transporte antes de Telegram. El destino del
bot sigue siendo el chat de operador configurado por `send_telegram_alert`.

Por qué es fire-and-forget: `send_telegram_alert` hace su POST con
timeout=10.0 (reaper.py:105). `_correr_pipeline` llama a
`avisar_fin_pipeline` justo después de haber GANADO la escritura atómica del
status final (`store.pipeline_update_status_si_epoca` devolvió True) -- para
ese momento el pipeline YA quedó completed/aborted, de verdad, en la DB. Que
el pipeline pueda considerarse terminado no puede depender de que Telegram
conteste a tiempo: `avisar_fin_pipeline` agenda el envío con
`asyncio.create_task` y devuelve sin esperarlo. La referencia fuerte a la
tarea en `_AVISO_TASKS` (con auto-limpieza al terminar) es el mismo remedio
que usa `MemoryDB._pending_tasks` (jax/memory/db.py) contra el gotcha
conocido de que el garbage collector se lleve una tarea fire-and-forget a
mitad de camino.

Por qué no hace falta una tabla de "ya avisado" (como la
`pipeline_aviso_enviado` de la Task 8 en jax-platform): acá el punto de
disparo YA es un reclamo atómico de una sola vez. `_correr_pipeline` sólo
llama a `avisar_fin_pipeline` dentro del `if` que sigue a un
`pipeline_update_status_si_epoca(...)` que devolvió True -- esa función hace
un UPDATE condicional (`WHERE pipeline_id=? AND run_epoch=? AND status IN
(...)`, jacobs/store.py:1436) y sólo una llamada concurrente puede ganarlo:
una segunda corrida de la misma época que perdió la carrera recibe False y
se va por `_perdio_la_epoca` (executor.py) SIN pasar por el punto de aviso.
Es el mismo patrón que ya usa el resto del ejecutor para status/steps -- no
uno nuevo. No hay reintento automático de `_correr_pipeline` sobre una
época ya cerrada en este repo (grep de `run_pipeline(` sin resultados fuera
de su propia definición y de los tests): el único llamador de producción es
externo a este repo (jax-platform), y ese llamador no puede reabrir una
época que el UPDATE condicional ya cerró.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("jacobs.aviso")

#: Fire-and-forget: referencia fuerte a las tareas en vuelo para que el GC no
#: se las lleve antes de que terminen -- mismo gotcha y mismo remedio que
#: `MemoryDB._pending_tasks` en jax/memory/db.py.
_AVISO_TASKS: set[asyncio.Task] = set()

async def _avisar(pipeline_id: str, nombre: str, estado: str) -> None:
    """El envío real, en background. Nunca lanza -- fail-soft CON rastro:
    cualquier fallo (excepción de red/timeout, o un `ok=False` de Telegram
    ya degradado por `send_telegram_alert`) queda en el log, nunca sube al
    caller ni se descarta en silencio."""
    # Telegram recibe únicamente el texto ya renderizado y revalidado por la
    # frontera compartida F2-C/F2-D. `nombre` y el estado del llamador no se
    # interpolan: PIPELINE_STATUS se vuelve a resolver desde Jacobs canónico.
    del nombre
    try:
        from policy.governance.external_output import (
            ExternalOutputChannelId,
            GovernedExternalOutputAdapter,
        )
        from jacobs.governed_aviso import compose_pipeline_notice

        envelope, context = await compose_pipeline_notice(
            pipeline_id=pipeline_id, requested_status=estado,
        )
        adapter = GovernedExternalOutputAdapter.for_channel(
            ExternalOutputChannelId.JACOBS_PIPELINE_NOTICE_TEXT_V1,
        )
        prepared = adapter.prepare_text(envelope, context)
        committed = await adapter.commit(prepared)
    except asyncio.CancelledError:
        # MENOR (revisión final 2026-09-18): `except Exception` de abajo NO
        # atrapa esto -- desde Python 3.8 CancelledError hereda de
        # BaseException, no de Exception. Si el proceso se apaga mientras
        # este aviso está en vuelo, la cancelación se propagaba sin dejar
        # rastro: el aviso se perdía en silencio, justo el caso que el resto
        # de esta función existe para evitar. Queda logueado ANTES de dejar
        # que la cancelación siga su curso -- NUNCA se traga: una tarea
        # cancelada tiene que seguir cancelada.
        logger.warning(
            "Aviso Telegram de fin de pipeline %s: cancelado antes de completarse "
            "(el proceso se está apagando) -- el aviso no salió",
            pipeline_id,
        )
        raise
    except Exception:  # fail-soft: send_telegram_alert no debería lanzar, pero esto es la última barrera -- ya se loguea con exc_info arriba, nunca sube al caller
        logger.error(
            "Aviso Telegram de fin de pipeline %s: excepción inesperada llamando a send_telegram_alert",
            pipeline_id, exc_info=True,
        )
        return
    if committed.state.value != "OUTPUT_COMMITTED_TO_TRANSPORT":
        logger.error(
            "Aviso Telegram gobernado de fin de pipeline %s: estado de transporte %s",
            pipeline_id, committed.state.value,
        )


def avisar_fin_pipeline(*, pipeline_id: str, nombre: str, estado: str) -> asyncio.Task:
    """Agenda el aviso de Telegram SIN esperar la respuesta (fire-and-forget
    -- ver "Por qué es fire-and-forget" en el docstring del módulo).

    El caller (`_correr_pipeline`) llama a esta función SIN `await`, apenas
    ganó la escritura atómica del status final: el estado del pipeline ya
    quedó escrito antes de que esto corra, así que un aviso que falla NUNCA
    puede cambiarlo -- ni de vuelta, porque no hay nada que revertir.

    Devuelve el Task para que un test (u otro caller que sí quiera esperar)
    pueda hacerlo -- mismo patrón que `MemoryDB.save_message`."""
    task = asyncio.create_task(_avisar(pipeline_id, nombre, estado))
    _AVISO_TASKS.add(task)
    task.add_done_callback(_AVISO_TASKS.discard)
    return task
