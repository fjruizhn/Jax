# tests/test_ejecutor_contratos_auditor_cliente.py
"""Cliente del auditor: formato de la llamada, errores de red y respuesta ilegible."""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from jax.ejecutor.contratos import auditor as A
from jax.ejecutor.contratos import auditor_cliente as AC

FACETA = SimpleNamespace(key="thot", provider_id="openai", base_url="https://api.ejemplo.invalid/v1", model="modelo-x",
                         credential="llave-XYZ", transport="http_openai_compat", persona=None, params=None)
LOTE = A.Lote("m", (A.Paso(1, "Bash", {"command": "uptime"}, False),), ())


def _cliente(manejar):
    return httpx.AsyncClient(transport=httpx.MockTransport(manejar))


def test_llama_con_modelo_llave_y_tope_y_lee_la_revision():
    vistas = {}

    def manejar(req):
        vistas["url"], vistas["auth"], vistas["cuerpo"] = str(req.url), req.headers["authorization"], json.loads(req.content)
        contenido = json.dumps({"hallazgos": [{"tipo": "fuera_de_mision", "paso": 1}], "afirmaciones": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": contenido}}]})

    async def escenario():
        async with _cliente(manejar) as cli:
            return await AC.auditar(LOTE, faceta=FACETA, max_tokens=4000, cliente=cli)

    rev = asyncio.run(escenario())
    assert rev.pausar is True and rev.paso == 1
    assert vistas["url"] == "https://api.ejemplo.invalid/v1/chat/completions" and vistas["auth"] == "Bearer llave-XYZ"
    assert vistas["cuerpo"]["model"] == "modelo-x" and vistas["cuerpo"]["max_completion_tokens"] == 4000
    assert vistas["cuerpo"]["messages"][0]["content"] == AC.instrucciones()


@pytest.mark.parametrize("respuesta", [httpx.Response(500, text="x"), httpx.Response(200, json={"choices": []}),
                                       httpx.Response(200, text="no json"),
                                       httpx.Response(200, json={"choices": [{"message": {"content": None}}]})])
def test_error_del_proveedor_o_forma_rara_es_ilegible(respuesta):
    async def escenario():
        async with _cliente(lambda req: respuesta) as cli:
            return await AC.auditar(LOTE, faceta=FACETA, max_tokens=10, cliente=cli)
    with pytest.raises(A.AuditorIlegible):
        asyncio.run(escenario())


def test_red_caida_es_ilegible():
    def manejar(req):
        raise httpx.ConnectError("sin red")

    async def escenario():
        async with _cliente(manejar) as cli:
            return await AC.auditar(LOTE, faceta=FACETA, max_tokens=10, cliente=cli)
    with pytest.raises(A.AuditorIlegible) as e:
        asyncio.run(escenario())
    assert e.value.codigo == "proveedor_fallo"


def test_transporte_no_soportado():
    with pytest.raises(AC.AuditorNoSoportado):
        asyncio.run(AC.auditar(LOTE, faceta=SimpleNamespace(**{**vars(FACETA), "transport": "http_gemini"}), max_tokens=10))


def test_las_instrucciones_son_un_archivo_no_vacio():
    assert "fuera_de_mision" in AC.instrucciones() and "no_responde" in AC.instrucciones()


def test_la_llave_no_sale_en_el_error():
    def manejar(req):
        return httpx.Response(401, text="llave-XYZ invalida")

    async def escenario():
        async with _cliente(manejar) as cli:
            return await AC.auditar(LOTE, faceta=FACETA, max_tokens=10, cliente=cli)
    with pytest.raises(A.AuditorIlegible) as e:
        asyncio.run(escenario())
    assert "llave-XYZ" not in repr(e.value) and e.value.__cause__ is None
