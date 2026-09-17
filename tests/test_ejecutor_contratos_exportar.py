# tests/test_ejecutor_contratos_exportar.py
"""Exportador de la política (C1/C2): filas de la DB -> documento firmado y validado
-> escritura atómica. Sin DB: las filas se construyen a mano."""
import json
import os
import stat
from datetime import datetime

import pytest

from jax.ejecutor.contratos import exportar, politica

HOSTS = [("hall9000", "192.0.2.5", 58291, "hypervisor", 1), ("bridge", "192.0.2.20", 58291, "clientes", 0)]
_EJ = json.dumps([{"tool_name": "Bash", "tool_input": {"command": "echo ejecutor-canario-c1"}}])
_NO = json.dumps([{"tool_name": "Bash", "tool_input": {"command": "echo x"}}])
REGLAS = [
    (1, "canario_c1", "prohibido", "Bash", "command", "ejecutor-canario-c1", None, None, 1, _EJ, _NO),
    (2, "migrate", "prohibido", "Bash", "command", r"\bmigrate:fresh\b", None, "produccion,clientes", 0,
     json.dumps([{"tool_name": "Bash", "tool_input": {"command": "ssh -tt axioma@bridge 'php artisan migrate:fresh'"}}]),
     "[]"),
]
RESPALDOS = [("bridge", datetime(2026, 9, 17, 6, 0))]


def test_documento_firmado_y_legible():
    doc = exportar.documento(HOSTS, REGLAS, RESPALDOS, "86400", "2026-09-17T12:00:00+00:00")
    p = politica.validar(doc)
    assert [r.codigo for r in p.reglas] == ["canario_c1", "migrate"]
    assert p.reglas[1].ambito_roles == frozenset({"produccion", "clientes"})
    assert doc["respaldos"] == {"bridge": "2026-09-17T06:00:00+00:00"}
    assert politica.autoprueba(p) == ()


@pytest.mark.parametrize("edad", [None, "", "cero", "0", "-5"])
def test_sin_edad_de_c2_no_se_exporta(edad):
    with pytest.raises(exportar.ExportacionImposible) as e:
        exportar.documento(HOSTS, REGLAS, RESPALDOS, edad, "2026-09-17T12:00:00+00:00")
    assert e.value.codigo == "c2_edad_max_invalida"


def test_una_regla_rota_no_se_publica():
    rota = REGLAS + [(3, "rota", "destructivo", "Bash", "command", "(sin cerrar", None, None, 0, _EJ, "[]")]
    with pytest.raises(exportar.ExportacionImposible) as e:
        exportar.documento(HOSTS, rota, RESPALDOS, "86400", "2026-09-17T12:00:00+00:00")
    assert e.value.codigo == "politica_invalida"


def test_escritura_atomica_con_permisos(tmp_path):
    ruta = tmp_path / "politica.json"
    ruta.write_text("vieja")
    doc = exportar.documento(HOSTS, REGLAS, RESPALDOS, "86400", "2026-09-17T12:00:00+00:00")
    exportar.escribir_atomico(ruta, doc)
    assert json.loads(ruta.read_text()) == doc
    assert stat.S_IMODE(os.stat(ruta).st_mode) == 0o640
    assert [p.name for p in tmp_path.iterdir()] == ["politica.json"]


def test_principal_importa_una_funcion_de_store_que_existe(monkeypatch):
    """Regresión: frente F retiró jacobs.store.get_conn y principal() lo seguía
    importando, así que el exportador de producción reventaba con ImportError antes
    de validar nada. Sin la variable de ruta, lo correcto es salir con 2."""
    from jax.ejecutor.contratos import exportar
    monkeypatch.delenv("JAX_EJECUTOR_POLITICA", raising=False)
    assert exportar.principal() == 2
