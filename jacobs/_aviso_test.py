#!/usr/bin/env python3
"""Task 6 (2026-09-18, historial-y-arreglos-de-pipeline): aviso por Telegram
cuando un pipeline termina.

`send_telegram_alert` (jacobs/reaper.py:82) ya existe, nunca lanza y degrada
a `ok=False` sin TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID -- se reusa, no se
escribe un segundo cliente de Telegram.

Ruling 4 del ledger de esta ronda (progress.md, 2026-09-18):
TELEGRAM_CHAT_ID es UNO SOLO para todo el sistema (reaper.py:96-97) -- con
más de un usuario, el aviso de cualquiera llega al mismo chat. El mensaje
lleva SOLO nombre, estado y enlace, NUNCA contenido del pipeline.

Sin DB, sin red: `send_telegram_alert` se reemplaza con un doble en TODOS
estos tests (ningún test manda un Telegram de verdad).

Corre con:
  cd /home/fruiz/worktrees/jax-historial && PYTHONPATH=.:las_manos python -m pytest -v jacobs/_aviso_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock

from jacobs.aviso import avisar_fin_pipeline, mensaje_de_fin


def test_el_mensaje_no_lleva_contenido_del_pipeline():
    """TELEGRAM_CHAT_ID es uno solo: el contenido no sale por ahi."""
    texto = mensaje_de_fin(pipeline_id="abc-123", nombre="ERP",
                           estado="completed", salida="SECRETO DEL CLIENTE")
    assert "SECRETO DEL CLIENTE" not in texto
    assert "abc-123" in texto and "ERP" in texto


def test_el_enlace_sale_de_una_variable_de_entorno_no_hardcodeada(monkeypatch):
    monkeypatch.setenv("JAX_FRONTEND_ORIGIN", "https://mi-dominio.invalid")
    monkeypatch.setenv("JAX_PIPELINE_DETAIL_PATH", "/x/{pipeline_id}")
    texto = mensaje_de_fin(pipeline_id="abc-123", nombre="ERP", estado="completed")
    assert "https://mi-dominio.invalid/x/abc-123" in texto


def test_sin_variables_de_entorno_usa_un_default_razonable(monkeypatch):
    monkeypatch.delenv("JAX_FRONTEND_ORIGIN", raising=False)
    monkeypatch.delenv("JAX_PIPELINE_DETAIL_PATH", raising=False)
    texto = mensaje_de_fin(pipeline_id="abc-123", nombre="ERP", estado="completed")
    assert "abc-123" in texto
    assert "http" in texto  # hay ALGUN enlace, sin que la funcion explote


def test_avisar_fin_pipeline_no_espera_la_respuesta_de_telegram(monkeypatch):
    """El POST real de send_telegram_alert tiene timeout=10.0 (reaper.py:105).
    Agendar el aviso no puede costarle esos 10s a quien llama -- es
    fire-and-forget: devuelve un Task ya corriendo en background."""
    liberar = asyncio.Event()

    async def _lento(mensaje):
        await liberar.wait()
        return {"ok": True, "message_id": 1, "error": None}

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
    assert "no configurados" in caplog.text


def test_avisar_fin_pipeline_manda_el_mensaje_correcto(monkeypatch):
    doble = AsyncMock(return_value={"ok": True, "message_id": 7, "error": None})
    monkeypatch.setattr("jacobs.reaper.send_telegram_alert", doble)

    async def escenario():
        task = avisar_fin_pipeline(pipeline_id="abc-123", nombre="ERP", estado="aborted")
        await task

    asyncio.run(escenario())
    doble.assert_awaited_once()
    (mensaje,), _ = doble.call_args
    assert "abc-123" in mensaje and "ERP" in mensaje
