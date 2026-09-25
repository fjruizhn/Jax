"""ops/rutas-de-produccion.sh -- las claves de RUTA de /etc/jax/.env ya no
apuntan al checkout de TRABAJO de un agente (2026-09-25).

HECHOS verificados por la sesion principal el mismo dia: los servicios de
produccion (jax-las-manos, jax-platform, jax-ejecutor-proxy,
jax-memory-worker, jax-memory-synthesis; todos User=jaxsvc,
EnvironmentFile=/etc/jax/.env) tenian JAX_CONFIG_PATH, JAX_AUDIT_LOG_PATH y
JAX_REPO_BASE apuntando a /home/fruiz/jax/... -- el checkout de trabajo de
UN agente, hoy en otra rama, no la copia versionada de /srv/jax-prod/jax ni
los datos de /srv/jax-data. `ops/rutas-de-produccion.sh --verificar`
detecta esto (y, para esas tres claves, que jaxsvc pueda leer -- y en el
log de auditoria y REPO_BASE/documents, escribir) y sale distinto de cero
mientras siga asi.

Auditoría de escalón 3 (PR jax#277): `ops/rutas-de-produccion.sh` es ahora
un envoltorio FINO que sólo valida el argumento y delega en
`ops/rutas_de_produccion_verificador.py` (módulo Python importable y
probado aparte, ver tests/test_rutas_de_produccion_verificador.py -- ahí
viven los 6 casos donde la versión anterior en bash fallaba abierto). Este
archivo prueba el envoltorio: que exista, tenga el bit ejecutable, rechace
argumentos inválidos sin sudo, y que el guion completo (el envoltorio +
el módulo al que delega) siga fallando hoy contra producción real.

Dos partes, mismo criterio que tests/test_arranque_instalado.py
(ops/versionar-drop-ins, PR jax#274):

(a) SIEMPRE corre, tambien en CI sin ningun host de produccion cerca: la
    forma del guion es correcta -- existe, tiene el bit ejecutable EN GIT
    (no solo en el filesystem de quien corrio el checkout), rechaza
    cualquier argumento que no sea --verificar SIN necesitar sudo, y las
    excepciones documentadas (JAX_MISSIONS_DIR, JAX_BIN -- tienen
    consumidor real pero no se mueven en este cambio; viven ahora en
    `EXCEPCIONES_FASE_A` de ops/rutas_de_produccion_verificador.py, no en
    el propio .sh) siguen con motivo escrito.

(b) SOLO en el host de produccion de jax (existe /srv/jax-prod/jax): corre
    el guion DE VERDAD y exige codigo 0. Fuera de ese host, skip con el
    motivo explicito.

Verificado en rojo HOY (2026-09-25, antes del corte de produccion descrito
en docs/runbooks/rutas-de-produccion.md): corriendo el guion en hall9000
contra el /etc/jax/.env real de HOY, sale 1 y lista exactamente
JAX_CONFIG_PATH, JAX_AUDIT_LOG_PATH y JAX_REPO_BASE apuntando bajo
/home/fruiz/jax/. `test_verificar_rutas_de_produccion_da_cero_en_produccion`
reproduce esto: en hall9000, HOY, FALLA -- se pondra verde recien cuando la
sesion principal ejecute el runbook (con el GO de Fernando) y las tres
claves de /etc/jax/.env apunten a /srv/jax-prod y /srv/jax-data.

Piso medido en hall9000 con `sudo unshare --mount` tapando /srv/jax-prod
(mismo metodo que ops/versionar-drop-ins, PR jax#274, para reproducir lo
que ve un runner de GitHub Actions sin ese directorio): 4 passed, 1 skipped.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops" / "rutas-de-produccion.sh"
MODULO_VERIFICADOR = ROOT / "ops" / "rutas_de_produccion_verificador.py"

RAIZ_PRODUCCION = "/srv/jax-prod/jax"

# Mismas claves y mismo motivo que EXCEPCIONES_FASE_A del propio módulo --
# si alguna vez divergen, este test lo nota.
CLAVES_EXCEPCION = ("JAX_MISSIONS_DIR", "JAX_BIN")


def test_el_guion_existe():
    assert SCRIPT.is_file(), f"no existe {SCRIPT}"


def test_el_guion_tiene_bit_ejecutable_en_git():
    salida = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-s", "ops/rutas-de-produccion.sh"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert salida.startswith("100755"), f"modo en git no es ejecutable: {salida!r}"


def test_el_guion_rechaza_cualquier_argumento_que_no_sea_verificar():
    """Rechazar un argumento invalido no deberia necesitar sudo -- si el
    guion cambiara de forma y sudo se volviera necesario antes de validar
    el argumento, esto se pondria lento o pediria password en vez de fallar
    rapido con rc=2."""
    for argumentos in ([], ["--otra-cosa"], ["-v"]):
        resultado = subprocess.run(
            [str(SCRIPT), *argumentos], capture_output=True, text=True, timeout=10)
        assert resultado.returncode == 2, (argumentos, resultado.stdout, resultado.stderr)
        assert "--verificar" in resultado.stderr


def test_las_excepciones_del_guion_siguen_con_motivo_escrito():
    """JAX_MISSIONS_DIR y JAX_BIN tienen un consumidor real (verificado
    2026-09-25: jax-platform backend/api/command.py) pero no se mueven en
    este cambio -- moverlos decide desde que checkout corre cada mision, una
    decision de arquitectura mayor que Fernando no tomo todavia. Desde la
    auditoria de escalon 3 (PR jax#277) el motivo vive en
    EXCEPCIONES_FASE_A de ops/rutas_de_produccion_verificador.py, no en el
    .sh -- este test sigue esas claves ahi, no las deja omitirse en
    silencio."""
    fuente = MODULO_VERIFICADOR.read_text(encoding="utf-8")
    for clave in CLAVES_EXCEPCION:
        assert clave in fuente, (
            f"{clave} ya no aparece en ops/rutas_de_produccion_verificador.py -- "
            "si se retiro del .env o se decidio su destino, este test y el "
            "modulo tienen que actualizarse juntos, no quedar desincronizados")
    assert "pendiente" in fuente.lower(), (
        "el modulo ya no marca JAX_MISSIONS_DIR/JAX_BIN como pendientes de "
        "decision -- ¿se resolvio? actualiza este test y el modulo")


def _motivo_de_skip_fuera_de_produccion() -> str | None:
    if not Path(RAIZ_PRODUCCION).is_dir():
        return f"esta maquina no tiene {RAIZ_PRODUCCION} -- no es el host de produccion de jax"
    return None


def test_verificar_rutas_de_produccion_da_cero_en_produccion():
    motivo = _motivo_de_skip_fuera_de_produccion()
    if motivo:
        pytest.skip(motivo)
    resultado = subprocess.run(
        [str(SCRIPT), "--verificar"], capture_output=True, text=True, timeout=30)
    assert resultado.returncode == 0, (
        f"ops/rutas-de-produccion.sh --verificar salio {resultado.returncode}:\n"
        f"stdout: {resultado.stdout}\nstderr: {resultado.stderr}\n\n"
        "Si esto corre ANTES del corte de docs/runbooks/rutas-de-produccion.md, "
        "es el rojo esperado (ver el docstring de este archivo): "
        "JAX_CONFIG_PATH/JAX_AUDIT_LOG_PATH/JAX_REPO_BASE de /etc/jax/.env "
        "todavia apuntan bajo /home/fruiz/jax/."
    )
