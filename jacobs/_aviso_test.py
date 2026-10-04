#!/usr/bin/env python3
"""Task 6 (2026-09-18, historial-y-arreglos-de-pipeline): aviso por Telegram
cuando un pipeline termina.

`send_telegram_alert` (jacobs/reaper.py:82) ya existe, no se escribe un
segundo cliente de Telegram. El texto pasa por la frontera compartida F2-C/D.

Tests de la frontera de transporte: `send_telegram_alert` se reemplaza con
un doble (ningún test manda un Telegram de verdad). La composición F2-A/B/C
real está cubierta en `_governed_aviso_test.py`.

Corre con:
  cd /home/fruiz/worktrees/jax-historial && PYTHONPATH=.:las_manos python -m pytest -v jacobs/_aviso_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from jacobs.aviso import avisar_fin_pipeline


def _composicion_estatica_de_prueba(monkeypatch, texto="Aviso gobernado de prueba."):
    from jacobs import governed_aviso
    from policy.governance.governed_renderer import RenderContext, WebChatGovernanceAdapter
    from policy.governance.response import GovernanceReceipt, ResponseScope

    scope = ResponseScope("test", "tenant-test", None, "user-test", "service:test",
        "operator", "jacobs-aviso", "request-test", "trace-test")
    receipt = GovernanceReceipt("policy-test", "vocabulary-test", "sha256:" + "a" * 64,
        "validator-test", "renderer-test")
    envelope = WebChatGovernanceAdapter(scope, receipt, producer="jacobs-aviso").seal_non_governed_candidate(
        response_id="response-test", candidate_text=texto)
    context = RenderContext(None, now=lambda: datetime.now(timezone.utc))

    async def compose(*, pipeline_id, requested_status):
        assert pipeline_id == "abc-123"
        assert requested_status in {"completed", "aborted"}
        return envelope, context

    monkeypatch.setattr(governed_aviso, "compose_pipeline_notice", compose)


def test_avisar_fin_pipeline_no_espera_la_respuesta_de_telegram(monkeypatch):
    """El POST real de send_telegram_alert tiene timeout=10.0 (reaper.py:105).
    Agendar el aviso no puede costarle esos 10s a quien llama -- es
    fire-and-forget: devuelve un Task ya corriendo en background."""
    liberar = asyncio.Event()

    async def _lento(mensaje):
        await liberar.wait()
        return {"ok": True, "message_id": 1, "error": None}

    _composicion_estatica_de_prueba(monkeypatch)
    monkeypatch.setattr("jacobs.reaper.send_telegram_alert", AsyncMock(side_effect=_lento))

    async def escenario():
        task = avisar_fin_pipeline(pipeline_id="abc-123", nombre="ERP", estado="completed")
        # Si avisar_fin_pipeline hubiera esperado el POST, este wait_for ya
        # habria disparado TimeoutError -- _lento sigue bloqueado en
        # liberar.wait().
        await asyncio.wait_for(asyncio.sleep(0), timeout=0.2)
        assert not task.done()
        liberar.set()
        await task  # limpieza: no dejar la tarea en vuelo al terminar el test

    asyncio.run(escenario())


def test_avisar_fin_pipeline_fail_soft_si_telegram_lanza(monkeypatch, caplog):
    """Un fallo de red/Telegram no puede subir al llamador -- fail-soft con
    rastro: no se traga en silencio, queda en el log."""
    _composicion_estatica_de_prueba(monkeypatch)
    monkeypatch.setattr(
        "jacobs.reaper.send_telegram_alert",
        AsyncMock(side_effect=RuntimeError("red caida")),
    )

    async def escenario():
        task = avisar_fin_pipeline(pipeline_id="abc-123", nombre="ERP", estado="completed")
        await task  # no debe relanzar RuntimeError

    with caplog.at_level(logging.ERROR, logger="jacobs.aviso"):
        asyncio.run(escenario())
    assert "abc-123" in caplog.text


def test_avisar_fin_pipeline_registra_el_rechazo_de_telegram(monkeypatch, caplog):
    """send_telegram_alert nunca lanza -- cuando degrada a ok=False (token
    rotado, chat_id invalido, TELEGRAM_BOT_TOKEN sin setear), tambien queda
    en el log: no alcanza con "no lanzo" para considerarlo entregado."""
    _composicion_estatica_de_prueba(monkeypatch)
    monkeypatch.setattr(
        "jacobs.reaper.send_telegram_alert",
        AsyncMock(return_value={"ok": False, "message_id": None, "error": "TELEGRAM_BOT_TOKEN/CHAT_ID no configurados"}),
    )

    async def escenario():
        task = avisar_fin_pipeline(pipeline_id="abc-123", nombre="ERP", estado="completed")
        await task

    with caplog.at_level(logging.ERROR, logger="jacobs.aviso"):
        asyncio.run(escenario())
    assert "abc-123" in caplog.text
    assert "TRANSPORT_OUTCOME_UNKNOWN" in caplog.text


def test_avisar_fin_pipeline_deja_rastro_si_lo_cancelan(monkeypatch, caplog):
    """MENOR (revisión final 2026-09-18): `except Exception` (fail-soft de
    arriba) NO atrapa `asyncio.CancelledError` -- desde Python 3.8 hereda de
    BaseException, no de Exception. Si el proceso se apaga mientras el aviso
    está en vuelo, la cancelación se propagaba sin dejar rastro: el aviso se
    perdía en silencio. Ahora queda logueado ANTES de que la cancelación
    siga su curso (nunca se traga -- una tarea cancelada tiene que seguir
    cancelada, no convertirse en 'terminó bien')."""
    async def _cancelado(mensaje):
        raise asyncio.CancelledError()

    _composicion_estatica_de_prueba(monkeypatch)
    monkeypatch.setattr("jacobs.reaper.send_telegram_alert", AsyncMock(side_effect=_cancelado))

    async def escenario():
        task = avisar_fin_pipeline(pipeline_id="abc-123", nombre="ERP", estado="completed")
        with pytest.raises(asyncio.CancelledError):
            await task

    with caplog.at_level(logging.WARNING, logger="jacobs.aviso"):
        asyncio.run(escenario())
    assert "abc-123" in caplog.text
    assert "cancel" in caplog.text.lower()


def test_avisar_fin_pipeline_manda_el_mensaje_correcto(monkeypatch):
    _composicion_estatica_de_prueba(monkeypatch, "Aviso F2-C enviado.")
    doble = AsyncMock(return_value={"ok": True, "message_id": 7, "error": None})
    monkeypatch.setattr("jacobs.reaper.send_telegram_alert", doble)

    async def escenario():
        task = avisar_fin_pipeline(pipeline_id="abc-123", nombre="ERP", estado="aborted")
        await task

    asyncio.run(escenario())
    doble.assert_awaited_once()
    (mensaje,), _ = doble.call_args
    assert mensaje == "Aviso F2-C enviado."
