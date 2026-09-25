"""ops/verificar-arranque-instalado.sh -- el chequeo de "ningún archivo de
más participa del arranque" (auditoría escalón 3, rondas 2 a 7).

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
    NeedDaemonReload -p Id -p Names`) y lo que hay en DISCO son dos
    chequeos independientes -- un `systemctl` que informa mal, o que
    devuelve NeedDaemonReload=yes aunque los DropInPaths estén completos,
    tiene que hacer fallar el guion igual. Desde la ronda 7 se prueba de
    forma HERMÉTICA (`SYSTEMCTL_DE_PRUEBA` + `RAIZ_PRUEBA`, ver (j)) --
    el shadowing de `systemctl` por PATH en producción de las rondas 3-6
    se quitó, porque MINOR-B (ronda 7) hace que producción resuelva
    `systemctl` por ruta ABSOLUTA verificada, nunca por PATH.
(e) MINOR-1 (ronda 5): RAIZ_PRUEBA="/" se normaliza a modo producción
    (capa cargado incluida), no a un modo de prueba sin capa cargado.
(f) La lista fija UNIT_PATHS_SYSTEMD (versionada en el guion porque
    systemd-analyze no se puede correr contra un árbol de prueba) se
    compara contra la real, sólo en producción, para que no derive.
(g) MAJOR-1 (ronda 6): un symlink de PRIMER NIVEL cuyo destino sea (por
    nombre base) una unidad del manifiesto es un ALIAS no declarado --
    `man systemd.unit(5)`, "drop-ins for the aliased name and all
    aliases are loaded". Se prueba en DOS capas independientes: disco
    (`otro-alias.service -> jax-las-manos.service`, con y sin drop-in
    propio) y cargado (`Names` != `Id` con un systemctl de mentira, con
    reproducción contra el guion sin el chequeo antes de cerrarlo).
(h) MINOR-1 (ronda 6): la capa cargado es inyectable con
    `SYSTEMCTL_DE_PRUEBA` (SOLO en modo prueba, nunca en producción) --
    esto hace que las pruebas de capa cargado, NeedDaemonReload y alias
    cargado corran en CUALQUIER runner (ubuntu-latest de CI incluido),
    activándose sobre un árbol de RAIZ_PRUEBA en vez de depender de que
    jax-las-manos.service esté instalado de verdad en /etc.
(i) MINOR-2 (ronda 6): `systemctl show` ya no se llama con `--value` --
    medido en hall9000, el orden de salida NO respeta el orden de los
    `-p` pedidos -- se parsea `Clave=valor` por clave.
(j) MAJOR-1 (ronda 7): los alias también pueden ser OCULTOS (nombre con
    "." inicial -- el glob de la ronda 6 no los veía, confirmado cargados
    de verdad por el auditor con systemd-analyze) y pueden formar
    CADENAS (`a -> b -> unidad`, `a -> .b -> unidad`) -- el mensaje
    nombra TODOS los eslabones, no sólo el último.
(k) MINOR-A (ronda 7): `SYSTEMCTL_DE_PRUEBA` nunca se lee en producción,
    ni siquiera si está puesta -- probado con prueba de mutación
    (reintroducir la lectura en una copia del guion hace fallar el test).
(l) MINOR-B (ronda 7): en producción, `systemctl` se resuelve por ruta
    ABSOLUTA fija (/usr/bin/systemctl o /bin/systemctl), verificada
    root:root y sin escritura de grupo/otros -- nunca por PATH. Esto
    retiró el shadowing por PATH que las rondas 3-6 usaban para probar la
    capa cargado en "producción" -- esas pruebas se reemplazaron por la
    capa cargado HERMÉTICA (`SYSTEMCTL_DE_PRUEBA` + `RAIZ_PRUEBA`, ver
    (d)), que corre en cualquier runner.
(m) MINOR-C (ronda 7): `Id`/`Names` vacíos, o `Id` distinto de la unidad
    pedida, fallan explícitamente -- con la ronda 6 sola, un systemctl
    que omitiera los dos campos daba `"" != ""` = sin diferencia.
(n) MINOR-D (ronda 7): el systemctl de mentira NUNCA delega al real
    (`exec "$REAL"` se quitó del todo) -- cualquier unidad desconocida,
    o cualquier invocación que no sea la consulta cargado esperada, sale
    con error. Las unidades "conocidas" se leen de
    ops/manifiesto-arranque-instalado.tsv, no de una lista copiada a
    mano.
"""
from __future__ import annotations

import os
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


def _sembrar_symlink(link: Path, destino: str) -> None:
    """Symlink de PRIMER NIVEL apuntando a `destino` (nombre relativo, en
    el mismo directorio) -- para reproducir un alias no declarado. El
    directorio padre puede terminar root:root 755 -- fruiz no puede crear
    el link ahí directo, mismo criterio que _sembrar_intruso."""
    subprocess.run(["sudo", "install", "-d", "-o", "root", "-g", "root", "-m", "0755", str(link.parent)], check=True)
    subprocess.run(["sudo", "ln", "-s", destino, str(link)], check=True)


def test_alias_de_primer_nivel_con_dropin_propio_se_detecta(tmp_path):
    """MAJOR-1 (ronda 6): `man systemd.unit(5)` -- "drop-ins for the
    aliased name and all aliases are loaded". Un symlink de PRIMER NIVEL
    (`otro-alias.service -> jax-las-manos.service`) más SU PROPIO drop-in
    (`otro-alias.service.d/evil.conf`) hace que systemd cargue ESE
    drop-in también para jax-las-manos.service -- el manifiesto no
    declara alias (hoy ninguno existe), así que esto tiene que dar rc!=0.

    **Reproducido contra el código de la ronda 5 (commit 5c52fe5, antes
    de este fix)**: el mismo árbol daba `rc=0` -- el guion nunca miraba
    symlinks de primer nivel que no fueran el nombre EXACTO de la unidad
    (confirmado a mano contra la rama sin el fix, antes de escribir el
    fix; ver el informe de la ronda 6)."""
    _construir_arbol_completo(tmp_path)
    base = tmp_path / "etc/systemd/system"
    _sembrar_symlink(base / "otro-alias.service", "jax-las-manos.service")
    _sembrar_intruso(base / "otro-alias.service.d" / "evil.conf")
    resultado = _correr(tmp_path)
    assert resultado.returncode != 0, f"alias no detectado -- stdout: {resultado.stdout}"
    assert "alias no declarado de jax-las-manos.service" in resultado.stderr, resultado.stderr


def test_alias_sin_dropin_propio_tambien_se_detecta(tmp_path):
    """El symlink SOLO (sin ningún drop-in propio todavía) también cuenta
    -- la sola presencia del alias es la diferencia, no hace falta que ya
    tenga un drop-in cargado para ser un riesgo: alguien podría agregarle
    uno más tarde sin que el manifiesto se entere, y para entonces ya
    seria tarde si el guion sólo mirara "¿tiene drop-in el alias?"."""
    _construir_arbol_completo(tmp_path)
    base = tmp_path / "etc/systemd/system"
    _sembrar_symlink(base / "otro-alias-vacio.service", "jax-las-manos.service")
    resultado = _correr(tmp_path)
    assert resultado.returncode != 0, f"alias no detectado -- stdout: {resultado.stdout}"
    assert "alias no declarado de jax-las-manos.service" in resultado.stderr, resultado.stderr


def test_alias_oculto_en_etc_se_detecta(tmp_path):
    """MAJOR-1 (ronda 7, auditor): un alias OCULTO (nombre que empieza
    con ".") también se carga de verdad -- confirmado por el auditor con
    `systemd-analyze` en hall9000 (systemd 259). El glob `"$dir_padre"/*`
    de la ronda 6 NUNCA ve nombres que empiecen con "." (comportamiento
    estándar de shell -- no depende de systemd), así que
    `.oculto.service -> jax-las-manos.service` se colaba sin detectar."""
    _construir_arbol_completo(tmp_path)
    base = tmp_path / "etc/systemd/system"
    _sembrar_symlink(base / ".oculto.service", "jax-las-manos.service")
    resultado = _correr(tmp_path)
    assert resultado.returncode != 0, f"alias oculto no detectado -- stdout: {resultado.stdout}"
    assert "alias no declarado de jax-las-manos.service" in resultado.stderr, resultado.stderr


def test_alias_oculto_en_run_se_detecta(tmp_path):
    """Mismo caso que arriba, pero bajo /run (una ruta de mayor prioridad
    dinámica, no la que declara el manifiesto) -- confirma que la
    detección de ocultos no está atada a una sola de las 12 rutas."""
    _construir_arbol_completo(tmp_path)
    base = tmp_path / "run/systemd/system"
    _sembrar_symlink(base / ".oculto.service", "jax-las-manos.service")
    resultado = _correr(tmp_path)
    assert resultado.returncode != 0, f"alias oculto en /run no detectado -- stdout: {resultado.stdout}"
    assert "alias no declarado de jax-las-manos.service" in resultado.stderr, resultado.stderr


def test_cadena_de_alias_con_intermedio_oculto_se_detecta(tmp_path):
    """MAJOR-1 (ronda 7): una CADENA de alias -- `a -> .b -> unidad` --
    no sólo un salto directo. `man systemd.unit(5)` no distingue "cuántos
    saltos": todos los nombres de la cadena quedan aliaseados. El mensaje
    tiene que nombrar los DOS eslabones (a y .b), no sólo el último."""
    _construir_arbol_completo(tmp_path)
    base = tmp_path / "etc/systemd/system"
    _sembrar_symlink(base / ".intermedio-oculto.service", "jax-las-manos.service")
    _sembrar_symlink(base / "cabeza-de-cadena.service", ".intermedio-oculto.service")
    resultado = _correr(tmp_path)
    assert resultado.returncode != 0, f"cadena de alias no detectada -- stdout: {resultado.stdout}"
    assert "cabeza-de-cadena.service" in resultado.stderr, resultado.stderr
    assert ".intermedio-oculto.service" in resultado.stderr, resultado.stderr


def test_cadena_de_alias_visible_se_detecta(tmp_path):
    """Misma cadena, pero con los dos nombres VISIBLES (`a -> b ->
    unidad`, sin ningún "." de por medio) -- para separar "sigue
    cadenas" de "ve ocultos": este caso prueba sólo lo primero."""
    _construir_arbol_completo(tmp_path)
    base = tmp_path / "etc/systemd/system"
    _sembrar_symlink(base / "eslabon-b.service", "jax-las-manos.service")
    _sembrar_symlink(base / "eslabon-a.service", "eslabon-b.service")
    resultado = _correr(tmp_path)
    assert resultado.returncode != 0, f"cadena de alias no detectada -- stdout: {resultado.stdout}"
    assert "eslabon-a.service" in resultado.stderr, resultado.stderr
    assert "eslabon-b.service" in resultado.stderr, resultado.stderr


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


# Ronda 7, MINOR-B: `systemctl` en producción se resuelve por RUTA
# ABSOLUTA fija (/usr/bin/systemctl o /bin/systemctl), NUNCA por PATH --
# el shadowing por PATH que usaban las pruebas de las rondas 3-5
# (`_correr_con_systemctl_falso`, ahora borrado) YA NO FUNCIONA contra
# producción, A PROPÓSITO: es exactamente lo que MINOR-B existe para
# impedir. Las pruebas de la capa cargado que necesitan un systemctl de
# mentira usan `SYSTEMCTL_DE_PRUEBA` bajo `RAIZ_PRUEBA`
# (`_correr_con_capa_cargado_de_prueba`, más abajo) -- HERMÉTICAS,
# corren en cualquier runner, ver ronda 6.


@pytest.mark.skipif(not SYSTEMCTL_FALSO.is_file(), reason="falta tests/fixtures/systemctl-falso-para-pruebas.sh")
@pytest.mark.skipif(_NO_ES_PRODUCCION, reason=_MOTIVO_SKIP_PRODUCCION)
def test_produccion_ignora_systemctl_de_prueba_aunque_este_puesta():
    """MINOR-A (ronda 7): `SYSTEMCTL_DE_PRUEBA` NUNCA se lee en
    producción -- ni siquiera si está puesta en el entorno. Se corre el
    guion en modo PRODUCCIÓN de verdad (RAIZ_PRUEBA vacío) con
    `SYSTEMCTL_DE_PRUEBA` apuntando al systemctl de mentira en modo
    "falla" (que, si se leyera, haría fallar TODO con "SYSTEMCTL SHOW
    FALLÓ"): el resultado tiene que ser IDÉNTICO al de correr sin esa
    variable -- rc=0, 19 archivos.

    **Prueba de mutación**: en una COPIA del guion se cambió la rama de
    producción para que leyera `SYSTEMCTL_CMD="${SYSTEMCTL_DE_PRUEBA:-$SYSTEMCTL_CMD}"`
    en vez de resolver por ruta absoluta (la vulnerabilidad que este test
    existe para atrapar) y se confirmó que ESTE test específico pasa a
    fallar contra esa copia -- restaurada de inmediato y confirmada byte
    a byte idéntica con `diff` (ver el informe de la ronda 7)."""
    resultado = subprocess.run(
        ["sudo", "-n", "env", f"SYSTEMCTL_DE_PRUEBA={SYSTEMCTL_FALSO}", "SYSTEMCTL_FALSO_MODO=falla", str(SCRIPT), ""],
        capture_output=True, text=True,
    )
    assert resultado.returncode == 0, (
        f"producción con SYSTEMCTL_DE_PRUEBA puesta (modo falla) tiene que dar 0 igual -- "
        f"si falla, la variable se está leyendo en producción. stdout={resultado.stdout!r} stderr={resultado.stderr!r}"
    )
    assert "19 archivos" in resultado.stdout, resultado.stdout


def test_raiz_prueba_que_resuelve_a_raiz_activa_la_capa_cargado():
    """MINOR-1 (ronda 5): un RAIZ_PRUEBA que resuelve a "/" tiene que
    tratarse EXACTAMENTE como si no se hubiera pasado nada -- modo
    PRODUCCIÓN, con la capa CARGADO incluida.

    Ronda 7: ya no se prueba con un systemctl de mentira vía PATH (ver
    MINOR-B, arriba -- ese shadowing ya no funciona contra producción a
    propósito). En cambio, se compara la corrida REAL sin argumento
    contra la corrida REAL con "/" -- tienen que dar EXACTAMENTE lo
    mismo (mismo rc, mismo stdout): si "/" saltara la capa cargado, la
    corrida sería más rápida pero el resultado sería indistinguible en
    este host (todo está bien instalado) -- así que la prueba real no es
    el resultado, es que las DOS corridas sean IDÉNTICAS entre sí, no
    sólo "las dos con rc=0"."""
    if _NO_ES_PRODUCCION:
        pytest.skip(_MOTIVO_SKIP_PRODUCCION)
    sin_argumento = subprocess.run(["sudo", "-n", str(SCRIPT), ""], capture_output=True, text=True)
    con_barra = subprocess.run(["sudo", "-n", str(SCRIPT), "/"], capture_output=True, text=True)
    assert sin_argumento.returncode == 0, sin_argumento.stderr
    assert con_barra.returncode == sin_argumento.returncode
    assert con_barra.stdout == sin_argumento.stdout, (
        f"RAIZ_PRUEBA='/' tiene que dar EXACTAMENTE el mismo resultado que sin argumento -- "
        f"sin argumento: {sin_argumento.stdout!r}; con '/': {con_barra.stdout!r}"
    )


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


def _correr_con_capa_cargado_de_prueba(tmp_path: Path, modo: str) -> subprocess.CompletedProcess:
    """MINOR-1 (ronda 6): activa la capa cargado sobre un árbol de
    RAIZ_PRUEBA (NO el /etc real) con `SYSTEMCTL_DE_PRUEBA` apuntando
    DIRECTO al systemctl de mentira -- corre en CUALQUIER runner, CI
    incluido, sin depender de que jax-las-manos.service esté instalado de
    verdad. `tmp_path` tiene que tener ya el árbol completo (llamar
    `_construir_arbol_completo` antes). A diferencia de
    `_correr_con_systemctl_falso` (que shadowea `systemctl` en el PATH
    para la corrida SIN RAIZ_PRUEBA, producción real), acá no hace falta
    PATH-shadowing: el guion invoca `"$SYSTEMCTL_CMD"` por su ruta
    absoluta, así que basta con pasarle esa ruta directo."""
    return subprocess.run(
        ["sudo", "-n", "env", f"SYSTEMCTL_DE_PRUEBA={SYSTEMCTL_FALSO}", f"SYSTEMCTL_FALSO_MODO={modo}",
         str(SCRIPT), str(tmp_path)],
        capture_output=True, text=True,
    )


@pytest.mark.skipif(not SYSTEMCTL_FALSO.is_file(), reason="falta tests/fixtures/systemctl-falso-para-pruebas.sh")
def test_capa_cargado_de_prueba_correcta_da_cero(tmp_path):
    """MINOR-1 (ronda 6): activar la capa cargado bajo RAIZ_PRUEBA, por sí
    solo, no tiene que hacer fallar nada cuando el systemctl de mentira
    coincide de verdad con el árbol -- corre en ubuntu-latest (CI), sin
    /srv/jax-prod/jax."""
    _construir_arbol_completo(tmp_path)
    resultado = _correr_con_capa_cargado_de_prueba(tmp_path, "correcto")
    assert resultado.returncode == 0, (
        f"con disco Y cargado perfectos, esto tiene que dar 0 -- "
        f"stdout={resultado.stdout!r} stderr={resultado.stderr!r}"
    )


@pytest.mark.skipif(not SYSTEMCTL_FALSO.is_file(), reason="falta tests/fixtures/systemctl-falso-para-pruebas.sh")
def test_capa_cargado_de_prueba_detecta_desacuerdo(tmp_path):
    """MINOR-1 (ronda 6): mismo caso que
    test_systemctl_que_informa_mal_hace_fallar_aunque_el_disco_este_bien
    (ronda 3) pero corriendo en CUALQUIER runner -- disco perfecto
    (RAIZ_PRUEBA), cargado incompleto (systemctl de mentira)."""
    _construir_arbol_completo(tmp_path)
    resultado = _correr_con_capa_cargado_de_prueba(tmp_path, "incompleto")
    assert resultado.returncode != 0
    assert "DIFERENCIA (cargado por systemd) entre lo que jax-las-manos.service" in resultado.stderr, resultado.stderr
    assert "DIFERENCIA (disco) entre lo que jax-las-manos.service" not in resultado.stderr, resultado.stderr


@pytest.mark.skipif(not SYSTEMCTL_FALSO.is_file(), reason="falta tests/fixtures/systemctl-falso-para-pruebas.sh")
def test_capa_cargado_de_prueba_need_daemon_reload(tmp_path):
    """MINOR-1 (ronda 6): mismo caso que
    test_need_daemon_reload_yes_hace_fallar_con_todo_lo_demas_correcto
    (ronda 5) pero corriendo en CUALQUIER runner."""
    _construir_arbol_completo(tmp_path)
    resultado = _correr_con_capa_cargado_de_prueba(tmp_path, "necesita_reload")
    assert resultado.returncode != 0
    assert "NeedDaemonReload=yes" in resultado.stderr, resultado.stderr
    assert "DIFERENCIA (cargado por systemd)" not in resultado.stderr, resultado.stderr


@pytest.mark.skipif(not SYSTEMCTL_FALSO.is_file(), reason="falta tests/fixtures/systemctl-falso-para-pruebas.sh")
def test_capa_cargado_de_prueba_detecta_alias_cargado(tmp_path):
    """MAJOR-1 (ronda 6): `Names` != `Id` (un alias cargado que el
    manifiesto no declara) tiene que fallar SOLO, aislado de cualquier
    otro desacuerdo -- el systemctl de mentira en modo "alias_cargado"
    devuelve los 3 drop-ins reales completos y NeedDaemonReload=no, pero
    Names trae un segundo nombre.

    **Verificado contra el código sin el chequeo antes de darlo por
    bueno**: se quitó el bloque `if [ "$nombres" != "$id" ]; then ...` de
    una COPIA del guion (nunca de la rama) y se confirmó que ESTE test
    específico pasa a fallar contra esa copia (el guion daba 0 con un
    alias cargado y todo lo demás perfecto) -- restaurada de inmediato y
    confirmada byte a byte idéntica con `diff` (ver el informe de la
    ronda 6)."""
    _construir_arbol_completo(tmp_path)
    resultado = _correr_con_capa_cargado_de_prueba(tmp_path, "alias_cargado")
    assert resultado.returncode != 0, (
        f"Names != Id con todo lo demás correcto tiene que fallar -- "
        f"si da 0, el chequeo se perdió. stdout={resultado.stdout!r} stderr={resultado.stderr!r}"
    )
    assert "Names=jax-las-manos.service otro-alias.service != Id=jax-las-manos.service" in resultado.stderr, resultado.stderr


@pytest.mark.skipif(not SYSTEMCTL_FALSO.is_file(), reason="falta tests/fixtures/systemctl-falso-para-pruebas.sh")
def test_capa_cargado_de_prueba_systemctl_falla(tmp_path):
    """Reemplaza a `test_systemctl_que_falla_dice_algo_claro_y_sigue_con_las_demas`
    (rondas 3-6, borrado en la ronda 7 porque dependía de shadowear
    `systemctl` por PATH en producción -- MINOR-B ya no lo permite) --
    misma propiedad (si `systemctl show` falla del todo, el guion sigue
    revisando las demás unidades y da rc=1 al final, con un mensaje
    claro), probada de forma HERMÉTICA con RAIZ_PRUEBA +
    SYSTEMCTL_DE_PRUEBA en vez de necesitar producción real."""
    _construir_arbol_completo(tmp_path)
    resultado = _correr_con_capa_cargado_de_prueba(tmp_path, "falla")
    assert resultado.returncode != 0
    assert "SYSTEMCTL SHOW FALLÓ para jax-las-manos.service" in resultado.stderr, resultado.stderr
    assert "verificar-arranque-instalado: hay diferencias entre el repo y lo instalado" in resultado.stderr, (
        f"no se ve la línea final del guion -- ¿abortó a mitad de la corrida?: {resultado.stderr!r}"
    )
    lineas_diferencia = [l for l in resultado.stderr.splitlines() if l.startswith("DIFERENCIA")]
    for linea in lineas_diferencia:
        assert "jax-las-manos.service" in linea, f"unidad inesperada con DIFERENCIA propia: {linea!r}"


@pytest.mark.skipif(not SYSTEMCTL_FALSO.is_file(), reason="falta tests/fixtures/systemctl-falso-para-pruebas.sh")
def test_capa_cargado_de_prueba_id_names_vacios(tmp_path):
    """MINOR-C (ronda 7): un `systemctl show` que omite `Id` y `Names`
    (los deja vacíos) tiene que fallar EXPLÍCITAMENTE -- con la ronda 6
    sola, `"$nombres" != "$id"` daba `"" != ""` = FALSO, así que este
    caso pasaba desapercibido."""
    _construir_arbol_completo(tmp_path)
    resultado = _correr_con_capa_cargado_de_prueba(tmp_path, "sin_id_names")
    assert resultado.returncode != 0, (
        f"Id/Names vacíos tiene que fallar -- si da 0, el chequeo no cubre este caso. "
        f"stdout={resultado.stdout!r} stderr={resultado.stderr!r}"
    )
    assert "Id/Names VACÍOS para jax-las-manos.service" in resultado.stderr, resultado.stderr
