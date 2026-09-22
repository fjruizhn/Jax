# tests/test_ejecutor_huella_sh.py
"""ejecutor-huella (arreglo del bug de producción jax#260, 2026-09-22): el comando
forzado de la llave PROPIA del servicio -- reemplaza el camino viejo
(`revocacion.argv_admin` + `huella.comando_huella()` armado en Python y mandado por
ssh como el administrador), que `jaxsvc` no puede satisfacer porque no puede leer la
llave personal de `fruiz`.

Mismo patrón de instalación que `ejecutor-freno-remoto`/`ejecutor-revocar`
(`ops/ejecutor/instalar_en_maquina.sh`): un script de root, ESTÁTICO, sin parámetros.
`comando_huella()`/`RUTAS_CONTROLES` (jax/ejecutor/contratos/huella.py) siguen siendo la
definición en Python de QUÉ se mide; este archivo mantiene sincronizado el script real
contra esa definición -- una sola fuente de verdad, dos formas (Python armaba un `sh -c`
para mandarlo por ssh, el script ahora corre ESTÁTICO en la remota, pero mide las
MISMAS rutas)."""
import re
import subprocess
from pathlib import Path

from jax.ejecutor.contratos import huella as H

RAIZ = Path(__file__).resolve().parents[1]
GUION = RAIZ / "ops" / "ejecutor" / "ejecutor-huella"
TEXTO = GUION.read_text()


def test_el_script_pasa_sh_menos_n():
    assert subprocess.run(["sh", "-n", str(GUION)]).returncode == 0


def test_el_script_es_ejecutable():
    assert GUION.stat().st_mode & 0o111 == 0o111


def test_el_script_no_lee_argv_del_script_ignora_lo_que_le_manden():
    """No hay superficie de ataque: lo que mide es fijo, nunca lo que alguien mande por
    ssh -- el `command=` forzado tampoco se lo dejaría pedir, pero el script no depende
    de eso para ser seguro. `tramo()` usa `$1` como parámetro de SU PROPIA función (las
    seis rutas fijas que le pasa el cuerpo del script, nunca argv del script) -- por eso
    esto se prueba corriéndolo con basura en argv y comparando la salida, no con un grep
    de `$1` (que daría un falso positivo contra `tramo()`)."""
    normal = subprocess.run([str(GUION)], capture_output=True, timeout=30)
    con_argv_hostil = subprocess.run(
        [str(GUION), "/etc/shadow", "--", "; rm -rf /"], capture_output=True, timeout=30)
    assert con_argv_hostil.returncode == normal.returncode
    assert con_argv_hostil.stdout == normal.stdout


# --- sincronización con jax.ejecutor.contratos.huella (una sola fuente de verdad) -------

def test_el_script_mide_exactamente_las_mismas_rutas_que_rutas_controles():
    for ruta in H.RUTAS_CONTROLES:
        assert f'tramo {ruta}\n' in TEXTO or TEXTO.count(f"tramo {ruta}") >= 1, ruta
    # Y nada de más: cada línea `tramo <ruta>` del script está en RUTAS_CONTROLES.
    rutas_del_script = re.findall(r"^\s{2}tramo (\S+)$", TEXTO, flags=re.MULTILINE)
    assert set(rutas_del_script) == set(H.RUTAS_CONTROLES)


def test_el_script_mide_el_glob_de_ejecutor_en_usr_local_sbin():
    assert "/usr/local/sbin" in TEXTO
    assert "ejecutor-*" in TEXTO


def test_el_script_no_mide_passwd_group_shadow():
    """Mismo criterio que `comando_huella()` (ronda 7): crear cuentas de sistema es
    administración legítima."""
    for ruta in ("/etc/passwd", "/etc/group", "/etc/shadow"):
        assert ruta not in TEXTO, ruta


def test_el_script_usa_binarios_por_ruta_absoluta():
    """Mismo LÍMITE documentado en huella.py: sin `secure_path`, un root remoto podría
    reemplazar `sha256sum`/`find`/`sort` en el PATH."""
    for binario in ("/usr/bin/find", "/usr/bin/sha256sum", "/usr/bin/sort"):
        assert binario in TEXTO, binario


def test_el_script_termina_con_sort():
    """Mismo formato que `comando_huella()`: todo se ordena al final -- la comparación
    `cambio()`/`hallazgos()` no debe depender del orden en que el remoto haya listado
    los archivos."""
    assert TEXTO.rstrip().splitlines()[-1].strip() == '} | "$SORT"'


# --- comportamiento: correr el script REAL, sin privilegios (mismo criterio que
# test_ejecutor_revocar_sh.py) -- lo que SÍ se puede leer sin root (/etc/ssh/sshd_config)
# tiene que aparecer; lo que no se puede leer sin root se salta en silencio (2>/dev/null,
# igual que documenta huella.py -- una ruta ausente/no legible cuenta como "no existe",
# no como error) y el script de todos modos sale con éxito. -----------------------------

def test_correr_sin_privilegios_sale_con_exito_y_es_deterministico():
    """El orden que da `sort(1)` depende de LC_COLLATE de la máquina -- no tiene que
    coincidir con el de Python (huella_valida()/cambio() sólo exigen que sea el MISMO
    entre dos corridas del MISMO proceso, no un orden en particular)."""
    r1 = subprocess.run([str(GUION)], capture_output=True, timeout=30)
    r2 = subprocess.run([str(GUION)], capture_output=True, timeout=30)
    assert r1.returncode == 0
    assert r1.stdout  # /etc/ssh/sshd_config es legible por cualquiera: nunca queda vacío
    assert r1.stdout == r2.stdout


def test_correr_sin_privilegios_incluye_sshd_config_legible():
    r = subprocess.run([str(GUION)], capture_output=True, timeout=30)
    assert any(linea.endswith(" /etc/ssh/sshd_config") for linea in r.stdout.decode().splitlines())


def test_correr_sin_privilegios_no_revienta_por_rutas_no_legibles():
    """`/etc/sudoers` (0440 root) no es legible por el usuario del test -- el script NO
    debe fallar por eso: el `2>/dev/null` de cada `tramo()` lo trata como "no existe",
    igual que documenta `huella.py` (fail-closed vive en `huella_valida()`, del lado de
    Python, que trata una huella VACÍA como no medible -- un script que revienta por un
    `find` sin permiso sería peor: ni siquiera se sabría que no midió nada)."""
    r = subprocess.run([str(GUION)], capture_output=True, timeout=30)
    assert r.returncode == 0
    assert r.stderr == b""
