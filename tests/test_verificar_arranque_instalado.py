"""ops/verificar-arranque-instalado.sh -- el chequeo de "ningún archivo de
más participa del arranque" (auditoría escalón 3, rondas 2 a 5).

Ronda 5 -- UN SOLO CAMINO: la capa DISCO (`enumerar_dropins_en_disco`) es
la MISMA función, con la MISMA lógica, tanto en producción (RAIZ="/",
recorriendo /etc, /run y /usr reales) como en los tests de este archivo
(RAIZ=árbol bajo /tmp vía RAIZ_PRUEBA) -- ya no hay dos implementaciones
separadas (el `systemd-delta` de las rondas 3-4 se quitó del todo: no veía
generator/transient/system.control ni los prefijos con guion, y como sólo
corría en producción, ningún test lo ejercitaba de verdad). Eso significa
que TODOS los tests de aquí, incluidos los que sólo usan RAIZ_PRUEBA bajo
/tmp, prueban el código que corre en producción -- no una simulación aparte.

Frentes cubiertos:

(a) RAIZ_PRUEBA con un árbol de 19 archivos completo y correcto -> 0.
(b) RAIZ_PRUEBA con cada intruso real reproducido en las rondas 2, 3 y 5
    de la auditoría (fragmento y drop-in de más, en cada una de las 12
    rutas de systemd-analyze unit-paths que aplica, incluidas
    system.control, generator, transient y los genéricos service.d/
    timer.d bajo /run) -> cada uno detectado.
(c) BLOCK-1 (ronda 3): un drop-in LEGÍTIMO cuyo CONTENIDO tiene una línea
    que empieza con "# /" (como el bug real: un comentario dentro de
    z-pythonpath.conf que una versión vieja del guion, basada en
    `systemctl cat` + grep, confundía con la cabecera que antepone
    systemd) NO cuenta como archivo de más -- el guion actual nunca mira
    contenido, sólo nombres.
(d) MAJOR-1 (ronda 3) / MAJOR-2 (ronda 5): en producción, lo CARGADO por
    systemd (`systemctl show -p FragmentPath -p DropInPaths -p
    NeedDaemonReload`) y lo que hay en DISCO son dos chequeos
    independientes -- un `systemctl` que informa mal, o que devuelve
    NeedDaemonReload=yes aunque los DropInPaths estén completos, tiene que
    hacer fallar el guion igual. Se prueba con un `systemctl` de mentira
    en el PATH que responde mal SÓLO para jax-las-manos.service (que hoy
    está instalado correctamente de verdad) y delega al real para
    cualquier otra unidad -- nunca se escribe en /etc.
(e) MINOR-1 (ronda 5): RAIZ_PRUEBA="/" se normaliza a modo producción
    (capa cargado incluida), no a un modo de prueba sin capa cargado.
(f) La lista fija UNIT_PATHS_SYSTEMD (versionada en el guion porque
    systemd-analyze no se puede correr contra un árbol de prueba) se
    compara contra la real, sólo en producción, para que no derive.
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
    """RAIZ_PRUEBA ahora es un ARGUMENTO posicional, no una variable de
    entorno (ronda 4): el guion se re-ejecuta a sí mismo como root vía
    `sudo -n`, y `sudo` resetea el entorno a `secure_path` -- una variable
    de entorno puesta ACÁ no sobreviviría ese salto, un argumento sí."""
    return subprocess.run([str(SCRIPT), str(tmp_path)], capture_output=True, text=True)


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
    # El quinto, nuevo en la ronda 3 (MAJOR-3): genérico de tipo bajo
    # /run, una ruta de las 12 que la primera enumeración no probaba.
    "run/systemd/system/service.d/top.conf",
    # Ronda 5 (punto 3 del coordinador) -- FRAGMENTO (el archivo <unidad>
    # en sí, no un drop-in) de más en una ruta de MÁS prioridad que
    # /etc/systemd/system: de verdad reemplazaría al fragmento real.
    "etc/systemd/system.control/jax-las-manos.service",
    # Fragmento de más en una ruta de MENOS prioridad que
    # /etc/systemd/system (system.control > ... > etc/systemd/system >
    # ... > run/systemd/system, ver UNIT_PATHS_SYSTEMD en el guion): hoy
    # no reemplaza a nada -- pero es un duplicado dormido que se
    # activaría solo si el de /etc alguna vez desaparece, y el manifiesto
    # declara UNA sola ubicación válida.
    "run/systemd/system/jax-las-manos.service",
    # Drop-in en transient (una de las 12 rutas, ronda 5).
    "run/systemd/transient/jax-las-manos.service.d/rogue.conf",
    # El de timer en /run que pidió el coordinador -- genérico de tipo
    # (timer.d/, no service.d/) bajo una ruta de /run, para una unidad
    # .timer del manifiesto.
    "run/systemd/system/timer.d/rogue.conf",
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


def test_directorio_ilegible_y_vacio_da_rc_1(tmp_path):
    """MAJOR-A (ronda 4, RECHAZO de la ronda 3): la ronda 3 hacía
    `fallo=1` DESDE DENTRO de una función invocada vía `$(...)` -- esa
    asignación vive en un SUBSHELL (la sustitución de comando) y se pierde
    en cuanto termina, así que el `rc=1` real dependía de que ALGÚN OTRO
    chequeo (una lista incompleta, por ejemplo) tropezara con el mismo
    problema. Acá se aísla el caso EXACTO donde eso no pasa: una ruta de
    la enumeración que DEBERÍA ser un directorio de drop-ins pero es un
    ARCHIVO regular -- no se puede LISTAR (`[ -d ]` da falso), pero si
    pudiera leerse igual no aportaría ningún .conf (no es un directorio).
    NOTA: `chmod 000` NO sirve para esto -- el guion corre como root desde
    la ronda 4, y root ignora los bits de permiso de un directorio (los
    probé: un directorio 000 root:root sigue siendo listable por root).
    Un archivo donde se espera un directorio SÍ falla para cualquiera,
    root incluido. Ni con la ruta "legible" ni con esta cambia el
    CONTENIDO de "lo real" (cero archivos en los dos casos) -- la ÚNICA
    señal de que algo anda mal es el mensaje "NO SE PUDO LEER" y el código
    de salida de la función que lo emite. Si alguien reintroduce el patrón
    `fallo=1` dentro de `$(...)`, este test pasa de rc=1 a rc=0 y falla."""
    _construir_arbol_completo(tmp_path)
    # Ruta real de la enumeración (jax-.service.d/ es el prefijo con guion
    # común a las 4 unidades .service) bajo una de las 12 rutas -- un
    # ARCHIVO, no un directorio.
    ruta = tmp_path / "run/systemd/generator/jax-.service.d"
    subprocess.run(["sudo", "install", "-d", "-o", "root", "-g", "root", "-m", "0755", str(ruta.parent)], check=True)
    subprocess.run(["sudo", "install", "-o", "root", "-g", "root", "-m", "0644", "/dev/null", str(ruta)], check=True)
    resultado = _correr(tmp_path)
    assert resultado.returncode == 1, (
        f"una ruta que debería ser directorio y es un archivo tiene que dar rc=1 -- "
        f"si esto da 0, el fallo se perdió en un subshell (MAJOR-A). stdout={resultado.stdout!r} stderr={resultado.stderr!r}"
    )
    assert "NO SE PUDO LEER" in resultado.stderr, resultado.stderr


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


def _correr_con_systemctl_falso(modo: str, raiz: str = "") -> subprocess.CompletedProcess:
    """Antepone al PATH un `systemctl` de mentira (tests/fixtures/) que
    sólo intercepta la consulta CARGADA (`-p ... NeedDaemonReload`) de
    jax-las-manos.service -- delega al real para todo lo demás. Nunca
    escribe en /etc. `raiz` es el argumento posicional que recibe el
    guion (RAIZ_PRUEBA); por default "" (producción).

    Ronda 4: el guion se re-ejecuta a sí mismo como root vía `sudo -n`, y
    ESE `sudo` resetea el PATH a `secure_path` (confirmado en hall9000) --
    un PATH de mentira puesto en el entorno de ESTE proceso no
    sobreviviría ese salto. Por eso acá se invoca `sudo -n env PATH=...`
    DIRECTAMENTE: el guion arranca ya como root (UID 0) y salta su propio
    re-exec, así que el PATH que `sudo -n env` fija sí llega intacto a la
    corrida real."""
    with tempfile.TemporaryDirectory() as d:
        bin_falso = Path(d)
        enlace = bin_falso / "systemctl"
        shutil.copy(SYSTEMCTL_FALSO, enlace)
        enlace.chmod(0o755)
        path_con_falso = f"{bin_falso}:{os.environ.get('PATH', '')}"
        return subprocess.run(
            ["sudo", "-n", "env", f"PATH={path_con_falso}", f"SYSTEMCTL_FALSO_MODO={modo}", str(SCRIPT), raiz],
            capture_output=True, text=True,
        )


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

    Las otras unidades del manifiesto están instaladas correctamente de
    verdad en esta máquina -- así que "siguió revisándolas" NO se prueba
    buscando sus nombres en stderr (si están bien, no imprimen nada, punto
    y punto es la señal correcta, no la ausencia de nombre). La prueba
    real es que el guion llega a su ÚLTIMA línea (el resumen final, que
    sólo se imprime DESPUÉS del bucle completo sobre las 6 unidades) --
    si hubiera abortado a mitad de camino (el bug que MAJOR-A de la ronda
    3 dejó posible con `fallo=1` perdido en un subshell), esa línea final
    nunca aparecería."""
    resultado = _correr_con_systemctl_falso("falla")
    assert resultado.returncode != 0
    assert "SYSTEMCTL SHOW FALLÓ para jax-las-manos.service" in resultado.stderr, resultado.stderr
    assert "verificar-arranque-instalado: hay diferencias entre el repo y lo instalado" in resultado.stderr, (
        f"no se ve la línea final del guion -- ¿abortó a mitad de la corrida?: {resultado.stderr!r}"
    )
    # Ninguna OTRA unidad reportó una DIFERENCIA propia (todas están bien
    # instaladas) -- confirma que el único problema sembrado fue el de
    # jax-las-manos.service, no que el guion se haya salteado al resto sin
    # revisarlas.
    lineas_diferencia = [l for l in resultado.stderr.splitlines() if l.startswith("DIFERENCIA")]
    for linea in lineas_diferencia:
        assert "jax-las-manos.service" in linea, f"unidad inesperada con DIFERENCIA propia: {linea!r}"


@pytest.mark.skipif(not SYSTEMCTL_FALSO.is_file(), reason="falta tests/fixtures/systemctl-falso-para-pruebas.sh")
@pytest.mark.skipif(_NO_ES_PRODUCCION, reason=_MOTIVO_SKIP_PRODUCCION)
def test_need_daemon_reload_yes_hace_fallar_con_todo_lo_demas_correcto():
    """MAJOR-2 (ronda 5): el chequeo de NeedDaemonReload se prueba
    AISLADO -- el systemctl falso en modo "necesita_reload" devuelve el
    FragmentPath y los 3 DropInPaths reales completos (nada falta, nada
    sobra) pero NeedDaemonReload=yes. Si el chequeo de NeedDaemonReload se
    borrara del guion, este test pasaría a dar 0 -- lo reproduje a mano
    quitando esas 4 líneas de una COPIA del guion (obtener_reales_cargado
    sin el `if [ "$need_reload" != no ]; then ... fi`) y confirmé que ESTE
    test específico falla contra esa copia (restaurada de inmediato, sin
    afectar la rama)."""
    resultado = _correr_con_systemctl_falso("necesita_reload")
    assert resultado.returncode != 0, (
        f"NeedDaemonReload=yes con todo lo demás correcto tiene que fallar -- "
        f"si da 0, el chequeo se perdió. stdout={resultado.stdout!r} stderr={resultado.stderr!r}"
    )
    assert "NeedDaemonReload=yes" in resultado.stderr, resultado.stderr
    assert "jax-las-manos.service" in resultado.stderr, resultado.stderr
    # Aísla que el fallo vino del chequeo NeedDaemonReload y no de un
    # desacuerdo de contenido (DropInPaths incompleto u otra cosa): en este
    # modo lo cargado coincide byte a byte con el manifiesto.
    assert "DIFERENCIA (cargado por systemd)" not in resultado.stderr, resultado.stderr


def test_raiz_prueba_que_resuelve_a_raiz_activa_la_capa_cargado():
    """MINOR-1 (ronda 5): un RAIZ_PRUEBA que resuelve a "/" tiene que
    tratarse EXACTAMENTE como si no se hubiera pasado nada -- modo
    PRODUCCIÓN, con la capa CARGADO incluida. Se prueba con el systemctl
    de mentira en modo "incompleto" (miente sobre jax-las-manos.service,
    el disco real está bien) pasando "/" EXPLÍCITO como RAIZ_PRUEBA: si la
    capa cargado no se activara con "/", esto daría 0 (sólo miraría el
    disco, que está perfecto); si se activa, tiene que dar 1, igual que
    sin pasar nada."""
    if not SYSTEMCTL_FALSO.is_file():
        pytest.skip("falta tests/fixtures/systemctl-falso-para-pruebas.sh")
    if _NO_ES_PRODUCCION:
        pytest.skip(_MOTIVO_SKIP_PRODUCCION)
    resultado_barra = _correr_con_systemctl_falso("incompleto", raiz="/")
    assert resultado_barra.returncode != 0, (
        f"RAIZ_PRUEBA='/' tiene que activar la capa cargado -- si da 0, se está "
        f"tratando como modo de prueba (sólo disco). stdout={resultado_barra.stdout!r}"
    )
    assert "cargado por systemd" in resultado_barra.stderr, resultado_barra.stderr


def _unit_paths_versionadas() -> list[str]:
    """Extrae el array UNIT_PATHS_SYSTEMD del propio guion -- para
    comparar la lista fija contra la real sin mantener una copia manual
    que pudiera desincronizarse."""
    texto = SCRIPT.read_text(encoding="utf-8")
    inicio = texto.index("UNIT_PATHS_SYSTEMD=(")
    fin = texto.index(")", inicio)
    bloque = texto[inicio:fin]
    rutas = []
    for linea in bloque.splitlines()[1:]:
        linea = linea.strip().strip('"')
        if linea:
            rutas.append(linea)
    return rutas


def test_unit_paths_versionadas_coinciden_con_la_real():
    """Ronda 5: UNIT_PATHS_SYSTEMD está fijo (versionado) en el guion
    porque `systemd-analyze` no se puede correr contra un árbol de prueba
    -- pero eso significa que puede desincronizarse de la realidad si una
    versión nueva de systemd cambia el orden (el orden ES la prioridad) o
    agrega/quita una ruta. Sólo corre en el host de producción real,
    compara la lista fija contra `systemd-analyze unit-paths`, línea por
    línea Y EN EL MISMO ORDEN."""
    if _NO_ES_PRODUCCION:
        pytest.skip(_MOTIVO_SKIP_PRODUCCION)
    real = subprocess.run(
        ["sudo", "-n", "systemd-analyze", "unit-paths"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    versionada = _unit_paths_versionadas()
    assert versionada == real, (
        f"UNIT_PATHS_SYSTEMD (guion) != systemd-analyze unit-paths (real):\n"
        f"  guion: {versionada}\n  real:  {real}"
    )
