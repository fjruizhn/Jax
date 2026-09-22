# tests/test_ejecutor_generar_claude_md.py
"""El generador del CLAUDE.md de axioma (§6.1 del spec: nunca se escribe a mano).

2026-09-22: la identidad decía «SOLO LEES» y ya es falso -- Fernando le dio sudo real
en las cuatro máquinas (`~/ejecutor-producto/LEDGER.md`, "SUDO EN LAS CUATRO"). Este
archivo prueba que el generador dice la verdad (sudo por máquina, C1-C6, GO por misión,
autoridad de Fernando), que exige `machine-id` antes de operar, y que empaqueta las
skills declaradas. Los módulos se cargan por ruta porque scripts/ no es un paquete
(mismo criterio que tests/test_ejecutor_fase0.py)."""
import importlib.util
import pathlib

import pytest
import tomllib

_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "ejecutor_fase0"


def _cargar(nombre):
    spec = importlib.util.spec_from_file_location(f"fase0_{nombre}", _DIR / f"{nombre}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


g = _cargar("generar_claude_md")

MAQUINAS = tomllib.loads((_DIR / "maquinas.toml").read_text())
CEREBROS = tomllib.loads((_DIR / "cerebros.toml").read_text())

# `generar()` lee la constitución REAL desde una ruta absoluta de ESTA máquina
# (cerebros.toml: /home/fruiz/claude-skills/common/CLAUDE.md.core) -- mismo criterio que
# el resto de scripts/ejecutor_fase0 (host-bound, Fase 0). Un runner de CI sin esa ruta
# se salta las pruebas que llaman a `generar()`; las que sólo leen los .toml no.
_FUENTE = pathlib.Path(CEREBROS["constitucion"]["fuente"])
requiere_constitucion_real = pytest.mark.skipif(
    not _FUENTE.is_file(), reason=f"{_FUENTE} no existe en este runner (host-bound, Fase 0)")

MACHINE_ID_ESPERADO = {
    "prod": "da476dce01ea4c3e9e72a8078a3ffd48",
    "atemai": "95e56bf6da0d41f993a3e36869699af1",
    "bridge": "ee090efa28cd46a7a0bff22d34e57eb4",
    "hall9000": "37ce158242c649fa80804a8c17b83ca4",
}


@requiere_constitucion_real
def test_ya_no_dice_solo_lees():
    assert "SOLO LEES" not in g.generar()


def test_las_cuatro_maquinas_tienen_machine_id_y_sudo_en_el_toml():
    for nombre, esperado in MACHINE_ID_ESPERADO.items():
        assert MAQUINAS[nombre]["machine_id"] == esperado, nombre
        assert MAQUINAS[nombre]["sudo"] is True, nombre


@requiere_constitucion_real
def test_cada_maquina_aparece_con_su_sudo_y_su_machine_id_en_el_documento():
    doc = g.generar()
    for nombre, machine_id in MACHINE_ID_ESPERADO.items():
        assert nombre in doc
        assert machine_id in doc, f"machine-id de {nombre} no aparece en el documento"


@requiere_constitucion_real
def test_la_identidad_menciona_los_seis_contratos():
    doc = g.generar()
    for contrato in ("C1", "C2", "C3", "C4", "C5", "C6"):
        assert contrato in doc, contrato


@requiere_constitucion_real
def test_la_identidad_dice_go_por_mision_y_autoridad_de_fernando():
    doc = g.generar()
    assert "por misión" in doc or "por mision" in doc
    assert "Fernando" in doc


@requiere_constitucion_real
def test_la_regla_de_machine_id_esta_escrita():
    doc = g.generar()
    assert "machine-id" in doc or "machine_id" in doc


def test_las_tres_skills_declaradas():
    assert CEREBROS["constitucion"]["skills"] == [
        "migrando-sin-romper", "desde-la-fuente", "endureciendo",
    ]


@requiere_constitucion_real
def test_seis_impossibles_incluido_pero_no_plugins():
    doc = g.generar()
    assert "LOS SEIS IMPOSIBLES" in doc
    assert "PLUGINS" not in doc.split("LOS SEIS IMPOSIBLES", 1)[0].split("## ")[-1]
