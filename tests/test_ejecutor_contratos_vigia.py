# tests/test_ejecutor_contratos_vigia.py
"""Vigía de C5: lee el registro de C3 desde donde empezó la misión (no lo que la jaula
dice de sí misma), verifica su cadena, audita por lote, frena con el motivo y el paso;
frena si el auditor cae; frena si el vigía muere; late mientras vive."""
import asyncio
import json
import os

import pytest

from jax.ejecutor.contratos import auditor as A
from jax.ejecutor.contratos import pausa as P
from jax.ejecutor.contratos import vigia as V
from jax.ejecutor.contratos.registro import Registro


def _registro(tmp_path, comandos, antes=1):
    ruta = tmp_path / "registro.jsonl"
    reg = Registro(ruta)
    for i in range(antes):
        reg.anotar({"evento": "registro_abierto", "pid": i})
    desde = ruta.stat().st_size
    for i, c in enumerate(comandos):
        reg.anotar({"evento": "herramienta_pedida", "tool_use_id": f"t{i}", "herramienta": "Bash",
                    "entrada": {"command": c}, "entrada_legible": True, "ruta": "/v1/messages"})
        reg.anotar({"evento": "resultado_devuelto", "tool_use_id": f"t{i}", "es_error": False, "bytes": 1,
                    "sha256": "0" * 64, "ruta": "/v1/messages"})
    reg.cerrar()
    return ruta, desde


def _cfg(tmp_path, ruta, desde, lote_max=2, latido_cada_s=0.05):
    return V.ConfigVigia(registro=ruta, desde_byte=desde, mision="uptime de hall9000", lote_max=lote_max,
                         intervalo_s=0.2, pausa=tmp_path / "PAUSA", latido=tmp_path / "latido",
                         latido_cada_s=latido_cada_s)


def _correr(cfg, auditar, fin_tras_s=1.0):
    async def escenario():
        fin = asyncio.Event()
        tarea = asyncio.create_task(V.vigilar(cfg, auditar, fin))
        await asyncio.sleep(fin_tras_s)
        fin.set()
        await asyncio.wait_for(tarea, 5)
    asyncio.run(escenario())


def _pausa(cfg):
    return json.loads(cfg.pausa.read_text())


async def _limpio(lote):
    return A.Revision(False, None, None, (), frozenset(), frozenset())


def test_frena_con_motivo_y_paso(tmp_path):
    ruta, desde = _registro(tmp_path, ["uptime", "cat ~/.ssh/id_ed25519"])
    lotes = []

    async def auditar(lote):
        lotes.append(lote)
        malos = [p.n for p in lote.pasos if "id_ed25519" in str(p.entrada)]
        return A.Revision(bool(malos), "fuera_de_mision" if malos else None, malos[0] if malos else None, (),
                          frozenset(), frozenset())

    cfg = _cfg(tmp_path, ruta, desde)
    _correr(cfg, auditar)
    contenido = _pausa(cfg)
    assert (contenido["origen"], contenido["motivo"]) == ("c5", "fuera_de_mision")
    assert contenido["paso"] == lotes[0].pasos[1].n == 4  # el número de línea del registro
    assert [p.entrada for p in lotes[0].pasos] == [{"command": "uptime"}, {"command": "cat ~/.ssh/id_ed25519"}]
    assert [p.es_error for p in lotes[0].pasos] == [False, False]
    assert lotes[0].mision == "uptime de hall9000"


def test_no_lee_lo_anterior_a_la_mision(tmp_path):
    ruta, desde = _registro(tmp_path, ["uptime"], antes=3)
    vistos = []

    async def auditar(lote):
        vistos.extend(p.n for p in lote.pasos)
        return A.Revision(False, None, None, (), frozenset(), frozenset())

    _correr(_cfg(tmp_path, ruta, desde, lote_max=1), auditar)
    assert vistos == [4]


def test_lote_parcial_se_audita_al_vencer_el_intervalo(tmp_path):
    ruta, desde = _registro(tmp_path, ["uptime"])
    vistos = []

    async def auditar(lote):
        vistos.append(len(lote.pasos))
        return A.Revision(False, None, None, (), frozenset(), frozenset())

    cfg = _cfg(tmp_path, ruta, desde, lote_max=20)
    _correr(cfg, auditar)
    assert vistos == [1] and not cfg.pausa.exists()


def test_auditor_ilegible_frena(tmp_path):
    ruta, desde = _registro(tmp_path, ["uptime"])

    async def auditar(lote):
        raise A.AuditorIlegible("json_invalido")

    cfg = _cfg(tmp_path, ruta, desde, lote_max=1)
    _correr(cfg, auditar)
    assert _pausa(cfg)["motivo"] == "auditor_ilegible"


def test_auditor_que_revienta_frena(tmp_path):
    ruta, desde = _registro(tmp_path, ["uptime"])

    async def auditar(lote):
        raise RuntimeError("red")

    cfg = _cfg(tmp_path, ruta, desde, lote_max=1)
    _correr(cfg, auditar)
    assert _pausa(cfg)["motivo"] == "auditor_caido"


def test_vigia_que_muere_frena(tmp_path):
    cfg = _cfg(tmp_path, tmp_path / "no-existe.jsonl", 0)

    async def escenario():
        with pytest.raises(FileNotFoundError):
            await asyncio.wait_for(V.vigilar(cfg, _limpio, asyncio.Event()), 5)
    asyncio.run(escenario())
    assert _pausa(cfg)["motivo"] == "vigia_caido"


def test_vigia_cancelado_sin_fin_frena(tmp_path):
    ruta, desde = _registro(tmp_path, [])
    cfg = _cfg(tmp_path, ruta, desde)

    async def escenario():
        tarea = asyncio.create_task(V.vigilar(cfg, _limpio, asyncio.Event()))
        await asyncio.sleep(0.3)
        tarea.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tarea
    asyncio.run(escenario())
    assert _pausa(cfg)["motivo"] == "vigia_caido"


def test_registro_editado_frena(tmp_path):
    ruta, desde = _registro(tmp_path, ["uptime", "df -h"])
    lineas = ruta.read_bytes().split(b"\n")
    lineas[1] = lineas[1].replace(b"uptime", b"uptimf")  # la jaula no puede; si alguien lo hizo, se ve
    ruta.write_bytes(b"\n".join(lineas))
    cfg = _cfg(tmp_path, ruta, desde, lote_max=20)
    with pytest.raises(V.RegistroRoto):
        _correr(cfg, _limpio)
    assert _pausa(cfg)["motivo"] == "registro_roto"


def test_desde_a_mitad_de_linea_frena(tmp_path):
    ruta, desde = _registro(tmp_path, ["uptime"])
    cfg = _cfg(tmp_path, ruta, desde + 3)

    async def escenario():
        with pytest.raises(V.RegistroRoto):
            await asyncio.wait_for(V.vigilar(cfg, _limpio, asyncio.Event()), 5)
    asyncio.run(escenario())
    assert _pausa(cfg)["motivo"] == "registro_roto"


def test_fin_normal_no_frena_y_late_mientras_vive(tmp_path):
    ruta, desde = _registro(tmp_path, [])

    async def auditar(lote):
        raise AssertionError("no hay pasos: no se audita")

    cfg = _cfg(tmp_path, ruta, desde)
    _correr(cfg, auditar, fin_tras_s=0.5)
    assert not cfg.pausa.exists()
    assert P.latido_fresco(cfg.latido, 2)


def test_late_aunque_el_auditor_tarde(tmp_path):
    ruta, desde = _registro(tmp_path, ["uptime"])
    vistos = []

    async def lento(lote):
        antes = os.stat(tmp_path / "latido").st_mtime_ns
        await asyncio.sleep(0.5)
        vistos.append(os.stat(tmp_path / "latido").st_mtime_ns > antes)
        return A.Revision(False, None, None, (), frozenset(), frozenset())

    _correr(_cfg(tmp_path, ruta, desde, lote_max=1), lento, fin_tras_s=1.2)
    assert vistos == [True]


def test_si_no_puede_latir_muere_y_frena(tmp_path):
    ruta, desde = _registro(tmp_path, [])
    cfg = V.ConfigVigia(registro=ruta, desde_byte=desde, mision="m", lote_max=1, intervalo_s=0.1,
                        pausa=tmp_path / "PAUSA", latido=tmp_path / "no-existe" / "latido", latido_cada_s=0.05)

    async def escenario():
        # FileNotFoundError y no OSError: TimeoutError también es OSError y un vigía que
        # sigue vivo sin latir (el defecto) saldría por el wait_for con el test verde.
        with pytest.raises(FileNotFoundError):
            await asyncio.wait_for(V.vigilar(cfg, _limpio, asyncio.Event()), 5)
    asyncio.run(escenario())
    assert _pausa(cfg)["motivo"] == "vigia_caido"
