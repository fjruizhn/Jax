# tests/test_ejecutor_contratos_lectura.py
"""Lectura de la API de mensajes para C3: un tool_use se reconoce COMPLETO en el
trozo que trae su content_block_stop, sin importar cómo llegue partido."""
import hashlib
import json

from jax.ejecutor.contratos import lectura as L
from jax.ejecutor.contratos.canario_upstream import _stream_tool_use, guion_bash

SSE = _stream_tool_use(guion_bash("toolu_1", "df -h /"))
ESPERADA = [L.HerramientaPedida("toolu_1", "Bash", {"command": "df -h /"}, True)]


def test_stream_entero():
    assert L.LectorSSE().alimentar(SSE) == ESPERADA


def test_byte_a_byte_da_lo_mismo_y_solo_al_final():
    lector, vistas, momento = L.LectorSSE(), [], None
    for i in range(len(SSE)):
        nuevas = lector.alimentar(SSE[i:i + 1])
        if nuevas and momento is None:
            momento = i
        vistas += nuevas
    assert vistas == ESPERADA
    assert momento >= SSE.index(b"content_block_stop"), "se anotó antes de que el bloque estuviera completo"


def test_crlf():
    assert L.LectorSSE().alimentar(SSE.replace(b"\n", b"\r\n")) == ESPERADA


def test_cr_solo_tambien_termina_lineas_aunque_llegue_partido():
    # SSE admite CR, LF y CRLF como fin de línea: si el arnés lo entiende y el proxy no,
    # la herramienta correría sin anotar. Byte a byte, un CR al final de un trozo no se decide todavía.
    for sse in (SSE.replace(b"\n", b"\r"), SSE.replace(b"\n", b"\r\n")):
        lector, vistas = L.LectorSSE(), []
        for i in range(len(sse)):
            vistas += lector.alimentar(sse[i:i + 1])
        assert vistas == ESPERADA


def test_crlf_partido_no_se_come_una_linea_vacia_real():
    # "\r" | "\n" | "\n": el primer LF completa el CRLF; el segundo es una línea vacía de verdad
    # y despacha el evento en ese trozo, no después.
    ev = {"type": "content_block_start", "index": 0,
          "content_block": {"type": "tool_use", "id": "t", "name": "Read", "input": {"file_path": "/x"}}}
    lector = L.LectorSSE()
    assert lector.alimentar(f"data: {json.dumps(ev)}\n\ndata: ".encode()
                            + json.dumps({"type": "content_block_stop", "index": 0}).encode() + b"\r") == []
    assert lector.alimentar(b"\n") == []
    assert lector.alimentar(b"\n") == [L.HerramientaPedida("t", "Read", {"file_path": "/x"}, True)]


def test_datos_que_no_son_json_se_ignoran():
    assert L.LectorSSE().alimentar(b"data: trozo-0\n\ndata: trozo-1\n\n") == []


def test_entrada_rota_se_anota_como_ilegible():
    sse = SSE.replace(b'{\\"command\\": \\"df -h /\\"}', b'{\\"command\\": ')
    (pedida,) = L.LectorSSE().alimentar(sse)
    assert (pedida.tool_use_id, pedida.entrada_legible, pedida.entrada) == ("toolu_1", False, '{"command": ')


def test_entrada_en_el_bloque_de_inicio():
    ev = {"type": "content_block_start", "index": 0,
          "content_block": {"type": "tool_use", "id": "t", "name": "Read", "input": {"file_path": "/etc/hosts"}}}
    sse = f"data: {json.dumps(ev)}\n\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n".encode()
    assert L.LectorSSE().alimentar(sse) == [L.HerramientaPedida("t", "Read", {"file_path": "/etc/hosts"}, True)]


def test_mensaje_no_stream():
    cuerpo = json.dumps({"content": [{"type": "text", "text": "x"},
                                     {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "ls"}}]})
    assert L.herramientas_de_mensaje(cuerpo.encode()) == [L.HerramientaPedida("t2", "Bash", {"command": "ls"}, True)]
    assert L.herramientas_de_mensaje(b"no json") is None


def test_resultados_en_cualquier_mensaje():
    contenido = [{"type": "text", "text": "12G libres"}]
    cuerpo = json.dumps({"messages": [
        {"role": "user", "content": "hola"},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": contenido}]},
        {"role": "system", "content": [{"type": "text", "text": "recordatorio"}]},
    ]}).encode()
    canonico = json.dumps(contenido, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    assert L.resultados_de_peticion(cuerpo) == [
        L.ResultadoDevuelto("t1", False, len(canonico), hashlib.sha256(canonico).hexdigest())]
    assert L.resultados_de_peticion(b"[1") is None


def test_evento_de_pedida_con_entrada_enorme_guarda_huella():
    enorme = L.HerramientaPedida("t", "Write", {"file_path": "/x", "content": "a" * (L.TOPE_ENTRADA_BYTES + 1)}, True)
    ev = L.evento_de_pedida(enorme, "/v1/messages")
    assert "entrada" not in ev
    assert ev["entrada_bytes"] > L.TOPE_ENTRADA_BYTES and len(ev["entrada_sha256"]) == 64
    assert len(ev["entrada_inicio"]) == 16384
    chica = L.evento_de_pedida(ESPERADA[0], "/v1/messages")
    assert chica == {"evento": "herramienta_pedida", "tool_use_id": "toolu_1", "herramienta": "Bash",
                     "entrada": {"command": "df -h /"}, "entrada_legible": True, "ruta": "/v1/messages"}
