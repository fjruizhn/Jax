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
