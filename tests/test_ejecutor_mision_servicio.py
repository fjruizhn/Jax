# tests/test_ejecutor_mision_servicio.py
"""SP2: el proceso que lanza la plataforma para un turno. Pedido por stdin, eventos por stdout
(una línea JSON cada uno) y SIEMPRE una última línea `resultado`, también cuando el pedido es
ilegible, falta configuración o algo revienta: la plataforma nunca queda adivinando."""
import asyncio
import io
import json
import uuid

import pytest

from jax.ejecutor import mision as M
from jax.ejecutor import mision_servicio as S
from jax.ejecutor.contratos.cuenta_axioma import CuentaSinConfigurar

TURNO = {"mision_id": str(uuid.uuid4()), "n": 1, "sesion": str(uuid.uuid4()), "objetivo": "x",
         "instruccion": "x", "hosts": ["ejecutor-prueba"]}
ENV = {"JAX_EJECUTOR_TURNO_TOPE_S": "900", "JAX_EJECUTOR_VIGIA_ESPERA_S": "120"}


def _correr(monkeypatch, entrada: bytes, correr_turno=None, env=ENV):
    salida = io.StringIO()
    if correr_turno is not None:
        monkeypatch.setattr(M, "correr_turno", correr_turno)
    monkeypatch.setattr(S, "dependencias_reales", lambda env, turno, **topes: "deps")
    rc = S.principal(io.BytesIO(entrada), salida, dict(env))
    lineas = [json.loads(l) for l in salida.getvalue().splitlines()]
    return rc, lineas


def test_turno_completo_sale_0_con_los_eventos_y_el_resultado_al_final(monkeypatch):
    async def turno(t, deps, emitir):
        assert deps == "deps" and t.sesion == TURNO["sesion"]
        emitir(M.evento("turno_lanzado", 1))
        return {"estado": "completado", "codigo": None}
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), turno)
    assert rc == 0
    assert lineas == [{"evento": "turno_lanzado", "turno": 1, "datos": {}},
                      {"evento": "resultado", "turno": 1, "datos": {"estado": "completado", "codigo": None}}]


def test_turno_fallido_o_rechazado_sale_1(monkeypatch):
    async def turno(t, deps, emitir):
        return {"estado": "rechazado", "codigo": "arranque_rechazado"}
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), turno)
    assert rc == 1 and lineas[-1]["datos"]["estado"] == "rechazado"


def test_pedido_ilegible_da_resultado_con_codigo_y_sale_2(monkeypatch):
    rc, lineas = _correr(monkeypatch, b"{no json")
    assert rc == 2
    assert lineas == [{"evento": "resultado", "turno": None,
                       "datos": {"estado": "fallido", "codigo": "turno_ilegible", "detalle": "turno_no_es_json"}}]


@pytest.mark.parametrize("variable", sorted(ENV))
def test_sin_topes_configurados_no_se_lanza_nada(monkeypatch, variable):
    llamado = []

    async def turno(*a):
        llamado.append(1)
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), turno,
                         env={k: v for k, v in ENV.items() if k != variable})
    assert rc == 2 and llamado == []
    assert lineas[-1]["datos"] == {"estado": "fallido", "codigo": "sin_configurar", "detalle": variable}


@pytest.mark.parametrize("valor", ["0", "-1", "x", "inf", "nan"])
def test_tope_invalido_es_sin_configurar(monkeypatch, valor):
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), env={**ENV, "JAX_EJECUTOR_TURNO_TOPE_S": valor})
    assert rc == 2 and lineas[-1]["datos"]["codigo"] == "sin_configurar"


def test_configuracion_de_la_cuenta_ausente_es_sin_configurar(monkeypatch):
    async def turno(t, deps, emitir):
        raise CuentaSinConfigurar("JAX_EJECUTOR_CUENTA")
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), turno)
    assert rc == 2 and lineas[-1]["datos"] == {"estado": "fallido", "codigo": "sin_configurar",
                                               "detalle": "JAX_EJECUTOR_CUENTA"}


def test_una_excepcion_inesperada_es_runner_error_con_el_tipo_y_sin_el_mensaje(monkeypatch):
    async def turno(t, deps, emitir):
        raise RuntimeError("secreto que no debe salir")
    rc, lineas = _correr(monkeypatch, json.dumps(TURNO).encode(), turno)
    assert rc == 1 and lineas[-1]["datos"] == {"estado": "fallido", "codigo": "runner_error", "tipo": "RuntimeError"}
    assert "secreto" not in json.dumps(lineas)


def test_leer_pausa_fail_closed(tmp_path):
    ruta = tmp_path / "pausa"
    assert S.leer_pausa(ruta) == {"puesta": False, "legible": True, "origen": None, "motivo": None, "paso": None,
                                  "momento": None}
    ruta.write_text(json.dumps({"origen": "c5", "motivo": "prohibido", "paso": 4, "momento": "2026-09-17T10:00:00+00:00"}))
    assert S.leer_pausa(ruta) == {"puesta": True, "legible": True, "origen": "c5", "motivo": "prohibido", "paso": 4,
                                  "momento": "2026-09-17T10:00:00+00:00"}
    ruta.write_text("{roto")
    assert S.leer_pausa(ruta)["puesta"] is True and S.leer_pausa(ruta)["legible"] is False
    ruta.write_text(json.dumps({"motivo": 3, "paso": "x"}))
    assert S.leer_pausa(ruta) == {"puesta": True, "legible": True, "origen": None, "motivo": None, "paso": None,
                                  "momento": None}


def test_el_vigia_se_abre_con_el_archivo_de_mision_y_se_cierra_con_sigterm(tmp_path, monkeypatch):
    """El vigía de verdad es `vigia_servicio`; acá un proceso falso que imprime lo que el vigía
    imprime al cerrar y termina con SIGTERM, para probar el manejo del proceso y del archivo."""
    import sys
    falso = tmp_path / "vigia_falso.py"
    falso.write_text("import signal, sys, time\n"
                     "signal.signal(signal.SIGTERM, lambda *a: (print('arranco=true cerrada=true', flush=True), sys.exit(0)))\n"
                     "print(open(sys.argv[-1]).read(), file=sys.stderr, flush=True)\n"
                     "time.sleep(30)\n")

    async def probar():
        v = await S.abrir_vigia(tmp_path, "m-t1", "texto", frozenset({"b", "a"}),
                                argv=[sys.executable, str(falso)])
        await asyncio.sleep(0.3)
        assert v.vive()
        assert json.loads((tmp_path / "m-t1.json").read_text()) == {"mision": "texto", "hosts": ["a", "b"]}
        rc, salida = await v.cerrar()
        return rc, salida
    rc, salida = asyncio.run(probar())
    assert rc == 0 and "cerrada=true" in salida and not (tmp_path / "m-t1.json").exists()
