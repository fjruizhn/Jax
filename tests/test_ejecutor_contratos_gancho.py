# tests/test_ejecutor_contratos_gancho.py
"""Gancho PreToolUse del Ejecutor y su envoltorio fail-closed (C1/C2).

El programa que llama al gancho NO bloquea si el gancho sale con 1, 127, se cae o
vence su tiempo. Estos tests prueban que el envoltorio convierte TODO eso en 2."""
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import pytest

from jax.ejecutor.contratos import gancho, instalacion
from tests.test_ejecutor_contratos_politica import OTRO_UID, doc_base, escribir

AHORA = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)


def _llamar(monkeypatch, tmp_path, evento, argv_extra=(), doc=None, uid=OTRO_UID):
    ruta = escribir(tmp_path, doc or doc_base())
    monkeypatch.setattr(gancho.os, "getuid", lambda: uid)
    salida, errores = io.StringIO(), io.StringIO()
    rc = gancho.principal([str(ruta), *argv_extra], entrada=io.StringIO(json.dumps(evento)),
                          salida=salida, errores=errores, ahora=AHORA)
    return rc, salida.getvalue(), errores.getvalue()


def test_permitido_sale_0_sin_ruido(monkeypatch, tmp_path):
    assert _llamar(monkeypatch, tmp_path, {"tool_name": "Bash", "tool_input": {"command": "uptime"}}) == (0, "", "")


def test_canario_sale_2_con_su_codigo(monkeypatch, tmp_path):
    rc, _, err = _llamar(monkeypatch, tmp_path, {"tool_name": "Bash", "tool_input": {"command": "echo ejecutor-canario-c1"}})
    assert rc == 2
    assert err == 'contrato="c1" codigo="prohibido" regla="canario_c1" hosts=["hall9000"]\n'


def test_destructivo_sin_respaldo_dice_c2(monkeypatch, tmp_path):
    rc, _, err = _llamar(monkeypatch, tmp_path, {"tool_name": "Bash", "tool_input": {"command": "rm -rf /tmp/x"}})
    assert rc == 2 and err.startswith('contrato="c2" codigo="destructivo_sin_respaldo"')


def test_politica_de_la_propia_cuenta_bloquea_todo(monkeypatch, tmp_path):
    rc, _, err = _llamar(monkeypatch, tmp_path, {"tool_name": "Bash", "tool_input": {"command": "uptime"}},
                         uid=os.getuid())
    assert rc == 2 and 'codigo="politica_ilegible" motivo="duenio_es_la_cuenta"' in err


@pytest.mark.parametrize("crudo", ["", "no json", "[]"])
def test_entrada_ilegible_bloquea(monkeypatch, tmp_path, crudo):
    ruta = escribir(tmp_path, doc_base())
    monkeypatch.setattr(gancho.os, "getuid", lambda: OTRO_UID)
    errores = io.StringIO()
    assert gancho.principal([str(ruta)], entrada=io.StringIO(crudo), salida=io.StringIO(), errores=errores,
                            ahora=AHORA) == 2
    assert 'codigo="entrada_ilegible"' in errores.getvalue()


def test_autoprueba_sana_sale_0_y_lista_vacia(monkeypatch, tmp_path):
    assert _llamar(monkeypatch, tmp_path, {}, argv_extra=("autoprueba",))[:2] == (0, "[]")


@pytest.mark.parametrize("argv", [[], ["a", "b"], ["a", "autoprueba", "c"]])
def test_argumentos_invalidos_bloquean(argv):
    assert gancho.principal(argv, entrada=io.StringIO("{}"), salida=io.StringIO(), errores=io.StringIO()) == 2


# --- el envoltorio sh, con una biblioteca falsa ---------------------------------

def _lib_falsa(tmp_path, cuerpo):
    lib = tmp_path / "lib"
    paquete = lib / "jax" / "ejecutor" / "contratos"
    paquete.mkdir(parents=True)
    for d in (lib / "jax", lib / "jax" / "ejecutor", paquete):
        (d / "__init__.py").write_text("")
    (paquete / "gancho.py").write_text(f"def principal(argv):\n{cuerpo}\n")
    guion = tmp_path / "gancho.sh"
    guion.write_text(instalacion.renderizar_gancho(str(lib), 1))
    guion.chmod(0o755)
    return guion


@pytest.mark.parametrize("cuerpo, esperado", [
    ("    return 0", 0),
    ("    return 2", 2),
    ("    return 1", 2),
    ("    return 3", 2),
    ("    raise RuntimeError('x')", 2),
    ("    import sys; sys.exit(127)", 2),
])
def test_el_envoltorio_convierte_todo_lo_que_no_es_0_en_2(tmp_path, cuerpo, esperado):
    guion = _lib_falsa(tmp_path, cuerpo)
    r = subprocess.run([str(guion), "x"], input=b"{}", capture_output=True, timeout=20)
    assert r.returncode == esperado


def test_el_envoltorio_corta_por_tiempo_y_bloquea(tmp_path):
    guion = _lib_falsa(tmp_path, "    import time; time.sleep(30); return 0")
    inicio = time.monotonic()
    r = subprocess.run([str(guion), "x"], input=b"{}", capture_output=True, timeout=20)
    assert r.returncode == 2
    assert time.monotonic() - inicio < 10


def test_el_envoltorio_ignora_el_arranque_que_controla_la_cuenta(tmp_path):
    """La cuenta controla su entorno. Un usercustomize que sale con 0 antes de
    evaluar nada sería un permiso universal; `python3 -I` no lo carga."""
    guion = _lib_falsa(tmp_path, "    return 2")
    version = subprocess.run(["/usr/bin/python3", "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
    sitio = tmp_path / "base" / "lib" / f"python{version}" / "site-packages"
    sitio.mkdir(parents=True)
    (sitio / "usercustomize.py").write_text("import os\nos._exit(0)\n")
    env = {**os.environ, "PYTHONUSERBASE": str(tmp_path / "base"), "PYTHONPATH": str(sitio)}
    control = subprocess.run(["/usr/bin/python3", "-c", "print('vivo')"], capture_output=True, timeout=20, env=env)
    assert (control.returncode, control.stdout) == (0, b""), "la trampa no se dispara: este test no mediría nada"
    r = subprocess.run([str(guion), "x"], input=b"{}", capture_output=True, timeout=20, env=env)
    assert r.returncode == 2
