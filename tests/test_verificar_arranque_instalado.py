"""ops/verificar-arranque-instalado.sh -- el chequeo de "ningún archivo de
más participa del arranque" (auditoría escalón 3, rondas 2 y 3).

Cuatro frentes:

(a) RAIZ_PRUEBA con un árbol de 19 archivos completo y correcto -> 0.
(b) RAIZ_PRUEBA con cada uno de los 5 intrusos reales que la ronda 2 de la
    auditoría reprodujo contra la versión vieja del guion (4 del auditor +
    uno nuevo en /run/systemd/system/service.d/, MAJOR-3 ronda 3) -> cada
    uno detectado.
(c) BLOCK-1 (ronda 3): un drop-in LEGÍTIMO cuyo CONTENIDO tiene una línea
    que empieza con "# /" (como el bug real: un comentario dentro de
    z-pythonpath.conf que una versión vieja del guion, basada en
    `systemctl cat` + grep, confundía con la cabecera que antepone
    systemd) NO cuenta como archivo de más -- el guion actual nunca mira
    contenido, sólo nombres.
(d) MAJOR-1 (ronda 3): en producción, lo CARGADO por systemd (`systemctl
    show`) y lo que hay en DISCO son dos chequeos independientes -- un
    `systemctl` que informa mal (aunque el disco esté perfecto) tiene que
    hacer fallar el guion igual. Se prueba con un `systemctl` de mentira
    en el PATH que responde mal SÓLO para jax-las-manos.service (que hoy
    está instalado correctamente de verdad) y delega al real para
    cualquier otra unidad -- nunca se escribe en /etc.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from manifiesto_arranque import ROOT, leer_manifiesto

SCRIPT = ROOT / "ops" / "verificar-arranque-instalado.sh"
SYSTEMCTL_FALSO = ROOT / "tests" / "fixtures" / "systemctl-falso-para-pruebas.sh"

# Los tests con systemctl de mentira necesitan que jax-las-manos.service
# esté REALMENTE instalado y correcto en /etc (para que el disco, que no
# pasa por el systemctl falso, coincida con el manifiesto) -- eso es
# hall9000/producción, no un runner de CI nuevo. Motivo explícito, no un
# skip silencioso.
_NO_ES_PRODUCCION = not Path("/srv/jax-prod/jax").is_dir()
_MOTIVO_SKIP_PRODUCCION = "esta máquina no tiene /srv/jax-prod/jax -- no es el host de producción de jax"


@pytest.fixture(autouse=True)
def _limpiar_root(tmp_path):
    """Los árboles de prueba se instalan con `sudo install -o root -g root`
    -- pytest (sin ser root) no puede borrarlos solo. Se limpia acá, con
    sudo, antes de que la limpieza automática de pytest lo intente y
    falle (mismo criterio que tests/test_instalar_dropins_de_servicio.py)."""
    yield
    subprocess.run(["sudo", "rm", "-rf", str(tmp_path)], check=False)


def _construir_arbol_completo(destino: Path) -> None:
    """Los 19 archivos del manifiesto, copiados byte a byte bajo `destino`
    con dueño/modo reales -- exactamente lo que RAIZ_PRUEBA espera
    encontrar para dar 0."""
    for repo_abs, instalada in leer_manifiesto():
        destino_archivo = destino / str(instalada).lstrip("/")
        subprocess.run(
            ["sudo", "install", "-d", "-o", "root", "-g", "root", "-m", "0755", str(destino_archivo.parent)],
            check=True,
        )
        modo = "0755" if repo_abs.suffix == ".sh" else "0644"
        subprocess.run(
            ["sudo", "install", "-o", "root", "-g", "root", "-m", modo, str(repo_abs), str(destino_archivo)],
            check=True,
        )


def _sembrar_intruso(ruta: Path, contenido: str = "[Service]\n") -> None:
    """El directorio final (`ruta.parent`) puede terminar root:root 755 --
    fruiz no puede escribir ahí directo. Se prepara el contenido en un
    archivo temporal PROPIO (bajo /tmp, fuera del árbol root-owned) y se
    copia con `sudo install`, igual que _construir_arbol_completo."""
    subprocess.run(["sudo", "install", "-d", "-o", "root", "-g", "root", "-m", "0755", str(ruta.parent)], check=True)
    fd, tmp_str = tempfile.mkstemp(prefix="intruso-")
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(contenido)
        subprocess.run(["sudo", "install", "-o", "root", "-g", "root", "-m", "0644", str(tmp), str(ruta)], check=True)
    finally:
        tmp.unlink(missing_ok=True)


def _correr(tmp_path: Path) -> subprocess.CompletedProcess:
    entorno = dict(os.environ, RAIZ_PRUEBA=str(tmp_path))
    return subprocess.run([str(SCRIPT)], capture_output=True, text=True, env=entorno)


def test_arbol_completo_da_cero(tmp_path):
    _construir_arbol_completo(tmp_path)
    resultado = _correr(tmp_path)
    assert resultado.returncode == 0, resultado.stderr
    assert "19 archivos" in resultado.stdout


@pytest.mark.parametrize("ruta_relativa", [
    # Los 4 del auditor (ronda 2).
    "etc/systemd/system/jax-.service.d/zz-malo.conf",
    "etc/systemd/system.control/jax-las-manos.service.d/z-pythonpath.conf",
    "usr/local/lib/systemd/system/jax-memory-worker.service.d/zz.conf",
    "run/systemd/generator/jax-ejecutor-proxy.service.d/runtime-intruso.conf",
    # El quinto, nuevo en esta ronda (MAJOR-3, ronda 3): genérico de tipo
    # bajo /run, una ruta de las 12 que la primera enumeración no probaba.
    "run/systemd/system/service.d/top.conf",
])
def test_cada_intruso_se_detecta(tmp_path, ruta_relativa):
    _construir_arbol_completo(tmp_path)
    _sembrar_intruso(tmp_path / ruta_relativa)
    resultado = _correr(tmp_path)
    assert resultado.returncode != 0, (
        f"el intruso {ruta_relativa} NO se detectó -- stdout: {resultado.stdout}"
    )
    assert "DE MÁS" in resultado.stderr, resultado.stderr


def test_intruso_con_nombre_no_conf_no_cuenta(tmp_path):
    """Control negativo del propio control negativo: un archivo que NO
    termina en .conf (systemd no lo lee como drop-in) no tiene que
    disparar nada -- si esto fallara, el guion estaría contando de más."""
    _construir_arbol_completo(tmp_path)
    _sembrar_intruso(tmp_path / "etc/systemd/system/jax-las-manos.service.d/notas.txt")
    resultado = _correr(tmp_path)
    assert resultado.returncode == 0, resultado.stderr


def test_block1_comentario_con_hash_slash_en_el_contenido_no_cuenta(tmp_path):
    """BLOCK-1 (ronda 3): el bug real -- una línea de COMENTARIO dentro de
    un .conf legítimo que empieza con "# /" (como
    "# /srv/jax-prod/jax/.venv/bin/python (el mismo intérprete...)" en el
    z-pythonpath.conf del proxy) no puede hacer que el guion cuente ESE
    MISMO archivo como si fuera otro drop-in de más, ni que aparezca
    ningún archivo fantasma. El guion actual nunca lee contenido -- esto
    prueba que un contenido adversarial tampoco lo hace tropezar."""
    _construir_arbol_completo(tmp_path)
    # Sobrescribe un drop-in YA ESPERADO (no agrega uno nuevo) con contenido
    # que incluye la línea ofensora real, palabra por palabra.
    ruta = tmp_path / "etc/systemd/system/jax-ejecutor-proxy.service.d/z-pythonpath.conf"
    contenido_con_bug = (
        "# comentario con una ruta que empieza con # /, como el bug real:\n"
        "# /srv/jax-prod/jax/.venv/bin/python (el mismo intérprete de ExecStart).\n"
        "[Service]\n"
        "Environment=PYTHONPATH=/srv/jax-prod/jax:/srv/jax-prod/jax/las_manos\n"
    )
    _sembrar_intruso(ruta, contenido_con_bug)

    resultado = _correr(tmp_path)
    # El cmp byte a byte SÍ va a fallar (el contenido ya no es idéntico al
    # repo) -- eso es correcto y esperado. Lo que NO puede pasar es que
    # aparezca como "DE MÁS": es el mismo archivo esperado, sólo con
    # contenido distinto.
    assert "DE MÁS" not in resultado.stderr, resultado.stderr
    assert "DIFIERE" in resultado.stderr, resultado.stderr


def _correr_con_systemctl_falso(modo: str) -> subprocess.CompletedProcess:
    """Antepone al PATH un `systemctl` de mentira (tests/fixtures/) que
    sólo intercepta `show ... jax-las-manos.service` -- delega al real
    para todo lo demás, incluida la enumeración de disco (que ni siquiera
    pasa por systemctl). Nunca escribe en /etc. RAIZ_PRUEBA deliberadamente
    AUSENTE: es el único camino donde se consulta systemctl."""
    with tempfile.TemporaryDirectory() as d:
        bin_falso = Path(d)
        enlace = bin_falso / "systemctl"
        shutil.copy(SYSTEMCTL_FALSO, enlace)
        enlace.chmod(0o755)
        entorno = dict(os.environ)
        entorno["PATH"] = f"{bin_falso}:{entorno['PATH']}"
        entorno["SYSTEMCTL_FALSO_MODO"] = modo
        entorno.pop("RAIZ_PRUEBA", None)
        return subprocess.run([str(SCRIPT)], capture_output=True, text=True, env=entorno)


@pytest.mark.skipif(not SYSTEMCTL_FALSO.is_file(), reason="falta tests/fixtures/systemctl-falso-para-pruebas.sh")
@pytest.mark.skipif(_NO_ES_PRODUCCION, reason=_MOTIVO_SKIP_PRODUCCION)
def test_systemctl_que_informa_mal_hace_fallar_aunque_el_disco_este_bien():
    """MAJOR-1 (ronda 3): lo CARGADO por systemd y lo que hay en DISCO son
    dos chequeos independientes. jax-las-manos.service está instalado
    correctamente de verdad en esta máquina (disco perfecto) -- un
    `systemctl show` de mentira que "olvida" reportar un drop-in real
    tiene que hacer fallar el guion igual, aunque el disco esté bien."""
    resultado = _correr_con_systemctl_falso("incompleto")
    assert resultado.returncode != 0
    assert "DIFERENCIA (cargado por systemd) entre lo que jax-las-manos.service" in resultado.stderr, resultado.stderr
    # El disco (real, jamás tocado) SÍ coincide -- aísla que el fallo vino
    # del systemctl falso, no de un problema real en /etc.
    assert "DIFERENCIA (disco) entre lo que jax-las-manos.service" not in resultado.stderr, resultado.stderr


@pytest.mark.skipif(not SYSTEMCTL_FALSO.is_file(), reason="falta tests/fixtures/systemctl-falso-para-pruebas.sh")
@pytest.mark.skipif(_NO_ES_PRODUCCION, reason=_MOTIVO_SKIP_PRODUCCION)
def test_systemctl_que_falla_dice_algo_claro_y_sigue_con_las_demas():
    """MINOR-2 (ronda 3): si `systemctl show` falla para una unidad
    (inexistente, systemctl roto), el guion dice algo claro Y sigue
    revisando el resto -- no aborta toda la corrida por una sola unidad.
    Las otras 3 unidades (memory-worker, memory-synthesis, proxy) siguen
    apareciendo en la salida con sus propios resultados."""
    resultado = _correr_con_systemctl_falso("falla")
    assert resultado.returncode != 0
    assert "SYSTEMCTL SHOW FALLÓ para jax-las-manos.service" in resultado.stderr, resultado.stderr
    # Siguió: las otras unidades del manifiesto también se mencionan (no se
    # cortó la corrida en la primera que falló).
    for otra in ("jax-memory-worker.service", "jax-memory-synthesis.service", "jax-ejecutor-proxy.service"):
        assert otra in resultado.stderr, f"{otra} no aparece -- ¿el guion abortó antes de llegar a ella?"
