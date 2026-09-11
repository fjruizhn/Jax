"""scripts/check_memory_schema_drift.py -- la logica de comparacion, sin DB.

Lo que la corrida real contra produccion no puede probar sola: que el
comparador SI grita cuando hay drift. Un chequeo que nunca vio un rojo es
una hipotesis (octava leccion). Cada caso de drift de abajo es una forma
real en la que el archivo ya diverge o divergio.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_memory_schema_drift as drift  # noqa: E402

_T = """CREATE TABLE `conversations` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=358 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"""


def test_el_archivo_real_trae_las_nueve_tablas():
    tablas = drift.parse_schema(drift.SCHEMA_PATH.read_text(encoding="utf-8"))
    assert set(tablas) == {
        "conversations", "messages", "facts", "decisions", "projects",
        "errors", "people", "action_items", "jax_metadata",
    }


def test_el_contador_auto_increment_no_es_drift():
    archivo = drift.parse_schema(_T.replace(" AUTO_INCREMENT=358", "") + ";")
    assert drift.compare(archivo, {"conversations": _T}) == []


def test_una_columna_agregada_a_mano_en_la_base_es_drift():
    """El caso que motivo todo: user_id/project_id agregados en produccion y
    nunca en el archivo."""
    archivo = drift.parse_schema(_T.replace("  `user_id` int(11) DEFAULT NULL,\n", ""))
    problemas = drift.compare(archivo, {"conversations": _T})
    assert len(problemas) == 1 and "DIFIERE" in problemas[0]
    assert "user_id" in problemas[0]


def test_un_cambio_de_tipo_es_drift():
    archivo = drift.parse_schema(_T.replace("`user_id` int(11)", "`user_id` bigint(20)"))
    assert drift.compare(archivo, {"conversations": _T})


def test_una_tabla_que_falta_en_la_base_es_drift():
    archivo = drift.parse_schema(_T)
    assert drift.compare(archivo, {"conversations": None}) == [
        "conversations: esta en el archivo y NO existe en la base"]


def test_sin_lectura_de_la_base_no_hay_verde(monkeypatch):
    """Fail-closed: si la base no responde, exit 2 -- nunca 0."""
    def _falla(_tablas):
        raise ConnectionRefusedError("base caida")

    monkeypatch.setattr(drift, "_leer_vivo", _falla)
    assert drift.main() == 2
