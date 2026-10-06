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
LOTE = A.Lote("m", (A.Paso(1, "Bash", {"command": "uptime"}, False),), (), (A.Maquina("hall9000", "192.0.2.5", 58291),))


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
            return await AC.auditar(LOTE, faceta=FACETA, max_tokens=4000, tope_s=400, cliente=cli)

    rev = asyncio.run(escenario())
    assert rev.pausar is True and rev.paso == 1
    assert vistas["url"] == "https://api.ejemplo.invalid/v1/chat/completions" and vistas["auth"] == "Bearer llave-XYZ"
    assert vistas["cuerpo"]["model"] == "modelo-x" and vistas["cuerpo"]["max_completion_tokens"] == 4000
    assert vistas["cuerpo"]["messages"][0]["content"] == AC.instrucciones()


def test_solo_ordenes_proyecta_allowlist_y_no_serializa_secretos_en_body_http():
    marcadores = [b"SECRET_STDOUT", b"SECRET_STDERR", b"SECRET_CONTEXT", b"SECRET_CLAIM",
                  b"SECRET_ASSERTION_COMMAND", b"SECRET_STEP_INPUT"]
    lote = A.Lote("Objetivo público", (A.Paso(9, "Bash", {"command": "uptime", "extra": "SECRET_STEP_INPUT"}, False),),
                  (A.AfirmacionAuditable("a1", "SECRET_CLAIM", "SECRET_CLAIM_DATA", "hall9000",
                                         "SECRET_ASSERTION_COMMAND",
                                         "SECRET_STDOUT", ("SECRET_CONTEXT",)),),
                  (A.Maquina("hall9000", "192.0.2.5", 58291),))
    capturas = {"body": None}

    def manejar(req):
        capturas["body"] = req.content
        contenido = json.dumps({"hallazgos": [], "afirmaciones": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": contenido}}]})

    async def escenario():
        async with _cliente(manejar) as cli:
            await AC.auditar(lote, faceta=FACETA, max_tokens=40, tope_s=400, cliente=cli,
                             modo="SOLO_ORDENES")

    asyncio.run(escenario())
    assert capturas["body"] is not None
    assert all(marcador not in capturas["body"] for marcador in marcadores)
    body = json.loads(capturas["body"])
    assert set(body) == {"model", "max_completion_tokens", "messages"}
    proyectado = json.loads(body["messages"][1]["content"])
    assert set(proyectado) == {"objetivo", "maquinas", "comandos"}
    assert proyectado["objetivo"] == "Objetivo público"
    assert proyectado["comandos"] == [{"n": 9, "comando": "uptime"}]


def test_falla_http_expone_solo_codigo_de_error_conocido_del_proveedor():
    def manejar(req):
        return httpx.Response(429, json={"error": {"code": "insufficient_quota",
                                                     "message": "mensaje que no se debe propagar"}})

    async def escenario():
        async with _cliente(manejar) as cli:
            await AC.auditar(LOTE, faceta=FACETA, max_tokens=8, tope_s=5, cliente=cli)

    with pytest.raises(A.AuditorIlegible) as exc:
        asyncio.run(escenario())
    assert exc.value.codigo == "proveedor_fallo"
    assert exc.value.proveedor_codigo == "insufficient_quota"
    assert str(exc.value) == "proveedor_fallo"


@pytest.mark.parametrize("respuesta", [httpx.Response(500, text="x"), httpx.Response(200, json={"choices": []}),
                                       httpx.Response(200, text="no json"),
                                       httpx.Response(200, json={"choices": [{"message": {"content": None}}]})])
def test_error_del_proveedor_o_forma_rara_es_ilegible(respuesta):
    async def escenario():
        async with _cliente(lambda req: respuesta) as cli:
            return await AC.auditar(LOTE, faceta=FACETA, max_tokens=10, tope_s=400, cliente=cli)
    with pytest.raises(A.AuditorIlegible):
        asyncio.run(escenario())


def test_red_caida_es_ilegible():
    def manejar(req):
        raise httpx.ConnectError("sin red")

    async def escenario():
        async with _cliente(manejar) as cli:
            return await AC.auditar(LOTE, faceta=FACETA, max_tokens=10, tope_s=400, cliente=cli)
    with pytest.raises(A.AuditorIlegible) as e:
        asyncio.run(escenario())
    assert e.value.codigo == "proveedor_fallo"


def test_transporte_no_soportado():
    with pytest.raises(AC.AuditorNoSoportado):
        asyncio.run(AC.auditar(LOTE, faceta=SimpleNamespace(**{**vars(FACETA), "transport": "http_gemini"}), max_tokens=10, tope_s=400))


def test_transporte_ollama_del_auditor_local_es_soportado():
    """El auditor local (spec 2026-09-18-auditor-local-opcion.md) es un proveedor
    'ollama': mismo body OpenAI-compat, sin credencial gestionada (facet_resolver exime
    a 'ollama' de pedir una)."""
    vistas = {}

    def manejar(req):
        vistas["url"] = str(req.url)
        contenido = json.dumps({"hallazgos": [], "afirmaciones": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": contenido}}]})

    faceta_local = SimpleNamespace(**{**vars(FACETA), "transport": "ollama",
                                      "base_url": "http://127.0.0.1:11435/v1", "credential": ""})

    async def escenario():
        async with _cliente(manejar) as cli:
            return await AC.auditar(LOTE, faceta=faceta_local, max_tokens=10, tope_s=400, cliente=cli)
    rev = asyncio.run(escenario())
    assert rev.pausar is False
    assert vistas["url"] == "http://127.0.0.1:11435/v1/chat/completions"


def test_las_instrucciones_son_un_archivo_no_vacio():
    assert "fuera_de_mision" in AC.instrucciones() and "no_responde" in AC.instrucciones()


def test_modos_no_reciben_instrucciones_del_otro_modo():
    completo = AC.instrucciones("COMPLETO")
    solo = AC.instrucciones("SOLO_ORDENES")
    assert "MODO SOLO_ORDENES" not in completo
    assert '"afirmaciones": []' not in completo
    assert "MODO SOLO_ORDENES" in solo
    assert "no_responde" not in solo


def test_la_llave_no_sale_en_el_error():
    def manejar(req):
        return httpx.Response(401, text="llave-XYZ invalida")

    async def escenario():
        async with _cliente(manejar) as cli:
            return await AC.auditar(LOTE, faceta=FACETA, max_tokens=10, tope_s=400, cliente=cli)
    with pytest.raises(A.AuditorIlegible) as e:
        asyncio.run(escenario())
    assert "llave-XYZ" not in repr(e.value) and e.value.__cause__ is None


# --- Auditor local sin credencial: la cabecera no se emite (2026-09-20) -------------------------
# Defecto encontrado en produccion: con transporte 'ollama' facet_resolver deja
# credential="" a proposito (pedirle una llave a un Ollama que no la usa seria
# inventar un secreto). La f-string incondicional producia el valor de cabecera
# literal "Bearer " -- con espacio final y sin valor -- y h11 rechaza una cabecera
# con espacio al final: httpx.LocalProtocolError, subclase de httpx.HTTPError, que
# el except de auditar() convertia en AuditorIlegible("proveedor_fallo"). La llamada
# moria ANTES de abrir el socket: los tres canarios de C5 fallaban en microsegundos
# y ninguna mision sobre maquina con datos de clientes podia arrancar.
#
# El test viejo (test_transporte_ollama_del_auditor_local_es_soportado) pasaba con el
# defecto presente porque MockTransport no pasa por h11 y no valida cabeceras: cubria
# el caso en el papel, no en la realidad. Por eso aca van DOS controles: uno sobre la
# cabecera emitida, y otro contra un servidor HTTP de verdad, que es el unico que
# ejercita el camino que fallo.

def test_sin_credencial_no_se_emite_la_cabecera_authorization():
    """Ausencia de credencial es un hecho del transporte, no un valor vacio que se
    serializa. Con el codigo viejo la cabecera viajaba como 'Bearer ' y este test falla."""
    vistas = {}

    def manejar(req):
        vistas["tiene_auth"] = "authorization" in req.headers
        vistas["auth"] = req.headers.get("authorization")
        contenido = json.dumps({"hallazgos": [], "afirmaciones": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": contenido}}]})

    faceta_local = SimpleNamespace(**{**vars(FACETA), "transport": "ollama",
                                      "base_url": "http://127.0.0.1:11435/v1", "credential": ""})

    async def escenario():
        async with _cliente(manejar) as cli:
            return await AC.auditar(LOTE, faceta=faceta_local, max_tokens=10, tope_s=400, cliente=cli)

    asyncio.run(escenario())
    assert vistas["tiene_auth"] is False, f"se emitio authorization={vistas['auth']!r}"


def test_auditor_local_sin_credencial_contra_un_servidor_http_real():
    """El control que el test con MockTransport NO puede dar: un socket de verdad, con
    h11 validando cabeceras. Con el codigo viejo levanta AuditorIlegible('proveedor_fallo')
    sin que el servidor reciba una sola peticion."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    recibidas = []

    class Manejador(BaseHTTPRequestHandler):
        def do_POST(self):
            recibidas.append(dict(self.headers))
            cuerpo = json.dumps({"choices": [{"message": {"content":
                json.dumps({"hallazgos": [], "afirmaciones": []})}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def log_message(self, *_):  # sin ruido en la salida de pytest
            pass

    servidor = ThreadingHTTPServer(("127.0.0.1", 0), Manejador)
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    puerto = servidor.server_address[1]
    try:
        faceta_local = SimpleNamespace(**{**vars(FACETA), "transport": "ollama",
                                          "base_url": f"http://127.0.0.1:{puerto}/v1", "credential": ""})

        async def escenario():
            async with httpx.AsyncClient() as cli:
                return await AC.auditar(LOTE, faceta=faceta_local, max_tokens=10, tope_s=400, cliente=cli)

        rev = asyncio.run(escenario())
    finally:
        servidor.shutdown()
        servidor.server_close()

    assert rev.pausar is False
    assert len(recibidas) == 1, "el servidor no recibio la peticion: murio antes del socket"
    assert "authorization" not in {k.lower() for k in recibidas[0]}


def test_el_plazo_de_la_llamada_es_el_que_se_pasa():
    """`tope_s` sale de axioma_config (ejecutor.c5_tope_s) y llega intacto al timeout de httpx."""
    vistos = {}

    def manejar(req):
        vistos["timeout"] = req.extensions["timeout"]
        contenido = json.dumps({"hallazgos": [], "afirmaciones": []})
        return httpx.Response(200, json={"choices": [{"message": {"content": contenido}}]})

    async def escenario():
        async with _cliente(manejar) as cli:
            return await AC.auditar(LOTE, faceta=FACETA, max_tokens=10, cliente=cli, tope_s=400)
    asyncio.run(escenario())
    assert vistos["timeout"] == {"connect": 400, "read": 400, "write": 400, "pool": 400}


def test_el_plazo_es_obligatorio_sin_valor_por_omision():
    """Un consumidor que olvide pasarlo no hereda un 120 s escrito en codigo: no corre."""
    with pytest.raises(TypeError):
        asyncio.run(AC.auditar(LOTE, faceta=FACETA, max_tokens=10))


@pytest.mark.parametrize("error", [httpx.ReadTimeout("lento"), httpx.ConnectTimeout("lento"),
                                   httpx.PoolTimeout("lento")])
def test_el_plazo_vencido_es_proveedor_plazo_y_no_proveedor_fallo(error):
    def manejar(req):
        raise error

    async def escenario():
        async with _cliente(manejar) as cli:
            return await AC.auditar(LOTE, faceta=FACETA, max_tokens=10, cliente=cli, tope_s=1)
    with pytest.raises(A.AuditorIlegible) as e:
        asyncio.run(escenario())
    assert e.value.codigo == "proveedor_plazo" and e.value.__cause__ is None
    assert "proveedor_plazo" in A.CODIGOS_ILEGIBLE and "proveedor_fallo" in A.CODIGOS_ILEGIBLE


def test_el_canario_trata_el_plazo_como_cualquier_ilegible_con_su_motivo():
    import asyncio as _a
    from jax.ejecutor.contratos import canario_c5 as C

    async def auditar(lote):
        raise A.AuditorIlegible("proveedor_plazo")
    fallos = _a.run(C.verificar_c5(auditar, elegir=lambda o: o[0]))
    assert fallos and all(f.codigo == "auditor_ilegible" and ("motivo", "proveedor_plazo") in f.datos for f in fallos)
