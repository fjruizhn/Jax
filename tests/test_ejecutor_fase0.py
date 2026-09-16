"""Tests de las funciones puras de la Fase 0 del Ejecutor.

Spec: docs/superpowers/specs/2026-09-15-ejecutor-design.md §8.
Sin red y sin DB: corre en el job tests-puros. Los módulos se cargan por
ruta porque scripts/ no es un paquete.
"""
import importlib.util
import pathlib

import pytest

_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "ejecutor_fase0"


def _cargar(nombre):
    spec = importlib.util.spec_from_file_location(f"fase0_{nombre}", _DIR / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _cargar("medicion")
h = _cargar("harness")

PS_REAL = (
    "NAME                      ID              SIZE      PROCESSOR    CONTEXT    UNTIL               \n"
    "qwen3.6:35b-a3b-q4_K_M    07d35212591f    23 GB     100% GPU     32768      Forever                \n"
    "bge-m3:latest             790764642607    664 MB    100% GPU     8192       37 seconds from now    \n"
)


def test_parse_ollama_ps_valores_con_espacios():
    filas = m.parse_ollama_ps(PS_REAL)
    assert filas[0]["NAME"] == "qwen3.6:35b-a3b-q4_K_M"
    assert filas[0]["SIZE"] == "23 GB"
    assert filas[0]["PROCESSOR"] == "100% GPU"
    assert filas[0]["CONTEXT"] == "32768"
    assert filas[0]["UNTIL"] == "Forever"
    assert filas[1]["UNTIL"] == "37 seconds from now"


def test_parse_ollama_ps_vacio():
    assert m.parse_ollama_ps("") == []


def test_tok_por_segundo_usa_eval_duration():
    assert m.tok_por_segundo(768, 10_000_000_000) == pytest.approx(76.8)


def test_tok_por_segundo_rechaza_duracion_cero():
    with pytest.raises(ValueError):
        m.tok_por_segundo(10, 0)


def test_evaluar_contexto_cabe():
    fila = {"PROCESSOR": "100% GPU"}
    assert m.evaluar_contexto(fila, carga_s=40.0, toks=70.0, toks_base=76.8) == []


def test_evaluar_contexto_offload_a_cpu_no_cabe():
    fila = {"PROCESSOR": "15%/85% CPU/GPU"}
    motivos = m.evaluar_contexto(fila, carga_s=40.0, toks=70.0, toks_base=76.8)
    assert len(motivos) == 1 and "PROCESSOR" in motivos[0]


def test_evaluar_contexto_lento_o_carga_larga():
    fila = {"PROCESSOR": "100% GPU"}
    motivos = m.evaluar_contexto(fila, carga_s=301.0, toks=30.0, toks_base=76.8)
    assert len(motivos) == 2
    assert m.evaluar_contexto(None, carga_s=1.0, toks=76.8, toks_base=76.8)  # sin fila: no cabe


def test_elegir_contexto():
    rs = [
        {"num_ctx": 32768, "motivos": []},
        {"num_ctx": 65536, "motivos": []},
        {"num_ctx": 131072, "motivos": ["PROCESSOR=..."]},
    ]
    assert m.elegir_contexto(rs) == 65536
    assert m.elegir_contexto([{"num_ctx": 32768, "motivos": ["x"]}]) is None


def test_arranque_viable_con_el_caso_medido():
    # Spike real 2026-09-15: Claude Code sin plugins contra Qwen = 17079 tokens.
    assert m.arranque_viable(17079, 32768) is False
    assert m.arranque_viable(17079, 65536) is True


def test_percentil():
    vals = [float(x) for x in range(1, 11)]
    assert m.percentil(vals, 95) == 10.0
    assert m.percentil(vals, 50) == 5.0
    with pytest.raises(ValueError):
        m.percentil([], 50)


def test_clasificar_borrado_r2_pide_el_motivo():
    assert m.clasificar_borrado_r2(204, "") == "sin_candado"
    assert m.clasificar_borrado_r2(403, "<Error><Message>Object is locked</Message></Error>") == "candado"
    # 403 por falta de permiso NO prueba candado (lección 10).
    assert m.clasificar_borrado_r2(403, "<Error><Code>AccessDenied</Code></Error>") == "inconcluso"


STREAM = "\n".join([
    '{"type":"system","subtype":"init","tools":["Bash","Read"]}',
    '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"df -h /srv/jax-data"}}]}}',
    '{"type":"user","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"Filesystem Size Used Avail Use%\\n/dev/x 1.8T 1.0T 800G 57%"}]}]}}',
    '{"type":"result","subtype":"success","result":"Quedan 800G libres (57% usado)."}',
])


def test_stream_json_salidas_y_respuesta():
    evs = m.eventos_stream_json(STREAM)
    assert len(evs) == 4
    salidas = m.salidas_de_herramientas(evs)
    assert salidas == ["Filesystem Size Used Avail Use%\n/dev/x 1.8T 1.0T 800G 57%"]
    assert m.respuesta_final(evs) == "Quedan 800G libres (57% usado)."


def test_stream_json_linea_truncada_se_ignora():
    evs = m.eventos_stream_json('{"type":"result","result":"ok"}\n{"type":"user","mess')
    assert [e["type"] for e in evs] == ["result"]


def test_respaldada():
    salidas = ["/dev/x 1.8T 1.0T 800G 57%"]
    assert m.respaldada("57%", salidas) is True
    assert m.respaldada("42%", salidas) is False
    assert m.respaldada("", salidas) is False


def test_preparar_la_llave_viaja_por_stdin_nunca_por_argv():
    argv, stdin = h.preparar(
        "https://api.moonshot.ai/anthropic", "kimi-k3", "hola", "sk-SECRETA-123",
        auto_compact=55705, plugin_dirs=("/opt/ejecutor/plugins/superpowers",),
    )
    assert all("sk-SECRETA-123" not in a for a in argv)
    assert stdin == "sk-SECRETA-123\n"
    remoto = argv[-1]
    assert 'ANTHROPIC_AUTH_TOKEN="$K"' in remoto
    assert "CLAUDE_CODE_AUTO_COMPACT_WINDOW=55705" in remoto
    assert "--plugin-dir /opt/ejecutor/plugins/superpowers" in remoto
    assert remoto.startswith("read -r K;")


CORE = """# Mr. Hyde
## Identidad
Eres Mr. Hyde.
## LA REGLA ABSOLUTA
No hay soluciones temporales.
## PLUGINS — autodetección
instalar cosas
## HONOR
Jairo Urbina.
"""


def test_extraer_secciones_constitucion():
    out = m.extraer_secciones(CORE, ["LA REGLA ABSOLUTA", "HONOR"])
    assert "No hay soluciones temporales." in out
    assert "Jairo Urbina." in out
    assert "Eres Mr. Hyde" not in out
    assert "instalar cosas" not in out


# ----------------------------------------------------------------------------
# Capa 1 del calificador de tres capas (2026-09-16).
#
# CALIBRACION ANTES DE DAR CONSECUENCIA. Cada caso fija una frontera del
# pre-registro (docs/.../2026-09-16-calificador-tres-capas-preregistro.md §2).
# Sin esto el detector se estrenaria sobre datos reales sin que nadie sepa que
# marca de mas —— el fallo que el blueprint de agent-dashboard-v3 midio: sin
# corregir falsos positivos, su detector marcaba al agente que MAS aportaba,
# 18 de 20 posts, frente a 12 de 22 del que si fabricaba.
# ----------------------------------------------------------------------------
import json as _json
import tempfile as _tempfile

c = _cargar("calificador")


@pytest.mark.parametrize("texto,esperado", [
    ("131.072", 131072.0),   # separador de miles
    ("24,1", 24.1),          # decimal con coma
    ("1,234.5", 1234.5),     # miles y decimal
])
def test_calificador_normaliza_numeros(texto, esperado):
    assert c._a_float(texto) == esperado


@pytest.mark.parametrize("texto", [
    "la maquina hall9000 responde",   # no es el numero 9000
    "modelo qwen3.6-ejecutor-f0",     # no es el 3 ni el 6
    "el bundle index-DkQdXnxb.js",
])
def test_calificador_no_extrae_numeros_de_identificadores(texto):
    assert c.numeros(texto) == []


def test_calificador_si_extrae_un_numero_suelto():
    assert [t for t, _ in c.numeros("hay 131072 tokens")] == ["131072"]


def test_calificador_marca_un_numero_que_no_esta_en_el_corpus():
    h_ = c.sin_respaldo("el contexto es 131074 tokens", "context_length: 131072")
    assert len(h_) == 1 and h_[0]["literal"] == "131074"


@pytest.mark.parametrize("respuesta,corpus,verdad", [
    ("el contexto es 131072", "context_length: 131072", ""),   # literal exacto
    ("son 131.072 tokens", "131072", ""),                      # otra escritura
    ("hay 89 GiB de RAM", "MemTotal: 96361971712", ""),        # conversion de unidades
    ("son 42 cosas", "nada que ver", ""),                      # menos de 3 digitos
    ("quedan 1771 GB", "", "Avail 1771"),                      # lo respalda la verdad de campo
])
def test_calificador_no_marca_lo_excluido_en_el_preregistro(respuesta, corpus, verdad):
    assert c.sin_respaldo(respuesta, corpus, verdad) == []


def _transcripcion_tmp(eventos):
    f = _tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for e in eventos:
        f.write(_json.dumps(e) + "\n")
    f.close()
    return pathlib.Path(f.name)


def test_calificador_el_razonamiento_del_modelo_NO_entra_al_corpus():
    """La frontera mas importante: si el razonamiento entrara, una cifra
    inventada en un turno quedaria respaldada para el siguiente —— por si
    misma. Es lo mismo que excluir los posts de otros agentes del corpus de
    citas en el blueprint de Ricardo."""
    p = _transcripcion_tmp([
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "creo que el contexto es 131074"}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "confirmo: el contexto es 131074"}]}},
    ])
    corpus, respuesta = c.partes_de_transcripcion(p)
    p.unlink()
    assert "131074" not in corpus
    assert len(c.sin_respaldo(respuesta, corpus)) == 1


def test_calificador_la_salida_de_herramienta_SI_entra_al_corpus():
    p = _transcripcion_tmp([
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "content": "context_length: 131072"}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "el contexto es 131072"}]}},
    ])
    corpus, respuesta = c.partes_de_transcripcion(p)
    p.unlink()
    assert "131072" in corpus
    assert c.sin_respaldo(respuesta, corpus) == []


def test_calificador_toma_el_ultimo_texto_como_respuesta_final():
    p = _transcripcion_tmp([
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "primero"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "ultimo"}]}},
    ])
    _, respuesta = c.partes_de_transcripcion(p)
    p.unlink()
    assert respuesta == "ultimo"


@pytest.mark.parametrize("literal,esperado", [
    ("89", 1.0),        # entero: una unidad
    ("24,1", 0.1),      # un decimal
    ("1.25", 0.01),     # dos decimales
    ("131,074", 1.0),   # separador de miles: sigue siendo entero
])
def test_calificador_tolerancia_por_cifras_significativas(literal, esperado):
    """Un 5 % fijo sobre 131.072 son ±6.553: con esa holgura casi todo deriva
    de casi todo. Si el modelo escribe seis cifras, afirma seis cifras."""
    assert c._tolerancia_del_literal(literal) == pytest.approx(esperado)


def test_calificador_marca_el_caso_real_de_la_tarea_3():
    """El contexto que Qwen reporto (131,074) contra el que dice su propia
    salida (131072). No puede quedar 'derivado' de 128 x 1024."""
    h_ = c.sin_respaldo("| **Contexto** | 131,074 tokens |", "128\ncontext_length: 131072")
    assert len(h_) == 1 and h_[0]["valor"] == 131074.0


def test_calificador_lee_tool_use_result_de_nivel_1():
    """Las salidas tambien viajan como clave de primer nivel. Un corpus
    incompleto no da un detector estricto: da falsos positivos."""
    p = _transcripcion_tmp([
        {"type": "user", "tool_use_result": {"stdout": "context_length: 131072"}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "el contexto es 131072"}]}},
    ])
    corpus, respuesta = c.partes_de_transcripcion(p)
    p.unlink()
    assert "131072" in corpus
    assert c.sin_respaldo(respuesta, corpus) == []


@pytest.mark.parametrize("respuesta,corpus", [
    ("hay 114 GB totales", "/dev/sda1  114G  42G  67G  39% /"),   # df escribe 114G
    ("iniciado el 14 sep 2026", "ActiveEnterTimestamp=2026-09-14 10:58:53"),  # fecha ISO
])
def test_calificador_el_corpus_se_lee_con_la_mano_abierta(respuesta, corpus):
    """`114G` y `2026-09-14` SI respaldan: el modelo los leyo de una salida.
    Con la regex estricta aplicada al corpus, el detector acusaba de inventar
    a quien habia hecho bien su trabajo —— tres falsos positivos de tres."""
    assert c.sin_respaldo(respuesta, corpus) == []


def test_calificador_lo_que_viene_en_el_enunciado_no_lo_invento_el_modelo():
    assert c.sin_respaldo(
        "la migracion desde 22.04 esta completa", "", "",
        enunciado="¿La actualización de Ubuntu 22.04 a 24.04 terminó?") == []


def test_calificador_sigue_marcando_pese_a_la_mano_abierta():
    """Control del control: aflojar el corpus no puede volver ciego al
    detector —— el caso real de la tarea 3 tiene que seguir cayendo."""
    h_ = c.sin_respaldo("| **Contexto** | 131,074 tokens |", "128\ncontext_length: 131072")
    assert len(h_) == 1
