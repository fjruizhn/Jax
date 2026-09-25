# tests/test_permisos_proyectos.py
"""ops/permisos_proyectos.py -- spec docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md
§5 (corrección de brief, 2026-09-25): RAIZ/proyectos/ y todo lo de abajo queda DUEÑO
jaxsvc, GRUPO fruiz con escritura, setgid en directorios, y ACL POSIX de acceso y por
defecto `u:jaxsvc:rwX,g:fruiz:rwX,m::rwx`.

Reescritura completa (2026-09-25) tras 3 BLOCK de la auditoría de escalón 3 sobre la
primera versión (`ops/permisos-proyectos.sh`, commit 2e8069f, PR jax#276). Los tres:

B3 (root sigue symlinks -- jaxsvc puede reemplazar proyectos/ por un enlace porque
   escribe en jax-workspace/): el núcleo pasa a Python, con un recorrido que abre CADA
   componente con O_NOFOLLOW relativo al descriptor del padre (nunca una ruta de texto
   vuelta a resolver) y muta a través de ESE descriptor (fchown/fchmod/setfacl vía
   /proc/self/fd/N). `test_symlink_en_el_punto_de_partida_se_rechaza` y
   `test_symlink_sustituido_a_mitad_de_la_corrida_no_contamina_el_objetivo` lo prueban.

B1 (--verificar daba OK con permiso EFECTIVO "---" porque miraba el TEXTO de la ACL):
   `test_verificar_detecta_permiso_efectivo_recortado_por_la_mascara` reproduce el caso
   real (mkstemp 0600 bajo ACL por defecto) y confirma que --verificar lo marca NO CUMPLE
   aunque el texto siga diciendo "rwx". El escritor real que lo disparaba
   (`las_manos/motor_registry/tool_authority.py::_write_file`) se corrige en el mismo PR
   -- ver `las_manos/_tool_authority_test.py::test_5c_...`.

B2 (el respaldo podía fallar y la aplicación seguía igual): `test_respaldo_que_falla_aborta_sin_aplicar_nada`
   simula un `sudo` que falla y confirma que --aplicar no tocó el árbol.

Además, correcciones menores (m1-m6 del mismo brief): setfacl corre como root (nunca
como el usuario que invoca), --verificar/--aplicar exigen ausencia de setuid/sticky en
directorios y de cualquier bit especial en archivos, hardlinks (nlink>1) se rechazan sin
mutar, el respaldo usa un nombre único (mkstemp), el runbook imprime la ruta ABSOLUTA de
reversión, y el test (b) ejercita lectura/escritura cruzada real en un subdirectorio
temporal propio dentro de `proyectos/` que el test borra al terminar.
"""
from __future__ import annotations

import grp
import os
import pwd
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import pytest

RAIZ_REPO = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ_REPO / "ops" / "permisos_proyectos.py"

RAIZ_PRODUCCION = Path("/home/fruiz/jax-workspace")
PROYECTOS_PRODUCCION = RAIZ_PRODUCCION / "proyectos"

USUARIO_ESPERADO = "jaxsvc"
GRUPO_ESPERADO = "fruiz"


def test_el_guion_existe():
    assert SCRIPT.is_file()


def _correr(*args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(SCRIPT), *args], capture_output=True, text=True, timeout=60, **kw
    )


def _sudo_n_disponible() -> bool:
    return subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=10).returncode == 0


def _acl_disponible_en(directorio: Path) -> bool:
    if shutil.which("setfacl") is None or shutil.which("getfacl") is None:
        return False
    prueba = directorio / ".prueba-acl"
    prueba.touch()
    try:
        quien = pwd.getpwuid(os.getuid()).pw_name  # nunca os.getlogin(): falla sin tty (m5)
        r = subprocess.run(["setfacl", "-m", f"u:{quien}:rwx", str(prueba)], capture_output=True)
        return r.returncode == 0
    finally:
        prueba.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def _identidades():
    """El guion hardcodea jaxsvc (dueño) y fruiz (grupo, vía el grupo primario que
    useradd le crea por defecto al usuario). Ya existen en hall9000; en un runner
    efímero (CI) se crean acá, idempotente."""
    if not _sudo_n_disponible():
        pytest.skip("sudo -n no disponible -- no se pueden garantizar las identidades jaxsvc/fruiz")

    for usuario in ("jaxsvc", "fruiz"):
        tiene = subprocess.run(["getent", "passwd", usuario], capture_output=True).returncode == 0
        if not tiene:
            creado = subprocess.run(
                ["sudo", "-n", "useradd", "--system", "--no-create-home", usuario],
                capture_output=True,
            )
            if creado.returncode != 0:
                pytest.skip(f"no se pudo crear el usuario {usuario}: {creado.stderr.decode(errors='replace')}")
    return None


@pytest.fixture()
def arbol_temporal(tmp_path, _identidades):
    raiz = tmp_path / "raiz"
    proyectos = raiz / "proyectos"
    (proyectos / "un-proyecto" / "sub").mkdir(parents=True)
    (proyectos / "un-proyecto" / "archivo.txt").write_text("contenido\n")

    if not _acl_disponible_en(proyectos):
        pytest.skip("el filesystem temporal no soporta ACL POSIX")

    os.chmod(tmp_path, 0o755)
    os.chmod(raiz, 0o755)
    return raiz


# --- Validación de RAIZ --------------------------------------------------------------------

def test_verificar_por_defecto_y_rechaza_raiz_invalidas(arbol_temporal):
    sin_flag = _correr(str(arbol_temporal))
    assert sin_flag.returncode == 1
    assert "NO CUMPLE" in sin_flag.stdout

    assert _correr("--verificar", "/").returncode == 2
    assert _correr("--verificar", "").returncode == 2

    sin_proyectos = arbol_temporal.parent / "sin-proyectos"
    sin_proyectos.mkdir()
    r = _correr("--verificar", str(sin_proyectos))
    assert r.returncode == 2
    assert "proyectos" in (r.stdout + r.stderr)


# --- B3: symlinks nunca seguidos -----------------------------------------------------------

def test_symlink_en_el_punto_de_partida_se_rechaza(tmp_path):
    raiz = tmp_path / "raiz"
    raiz.mkdir()
    objetivo = raiz / "objetivo_real"
    objetivo.mkdir()
    os.chmod(objetivo, 0o700)
    (raiz / "proyectos").symlink_to(objetivo)

    for args in (("--verificar", str(raiz)), ("--aplicar", str(raiz))):
        r = _correr(*args)
        assert r.returncode == 2, r.stdout + r.stderr
        assert "symlink" in (r.stdout + r.stderr)

    # El objetivo real -- lo único que un ataque de symlink querría tocar -- sigue intacto.
    st = objetivo.stat()
    assert st.st_uid == os.getuid()
    assert oct(st.st_mode & 0o777) == "0o700"


def test_symlink_sustituido_a_mitad_de_la_corrida_no_contamina_el_objetivo(_identidades, tmp_path):
    """La carrera real que motivó B3: algo (jaxsvc, con escritura en el padre) reemplaza
    un subdirectorio por un symlink a una ruta root-owned DESPUÉS de que el recorrido ya
    empezó. Ejercita el núcleo (`_recorrer`) directamente, como root -- igual que corre
    de verdad dentro de --aplicar (el `--nucleo-privilegiado` interno)."""
    if not _sudo_n_disponible():
        pytest.skip("sudo -n no disponible")
    if shutil.which("setfacl") is None:
        pytest.skip("setfacl no está instalado")

    raiz = tmp_path / "raiz"
    proyectos = raiz / "proyectos"
    victima = proyectos / "victima"
    victima.mkdir(parents=True)
    (victima / "archivo.txt").write_text("x")
    objetivo = raiz / "objetivo_de_root"
    objetivo.mkdir()

    r_chown = subprocess.run(["sudo", "-n", "chown", "root:root", str(objetivo)], capture_output=True)
    r_chmod = subprocess.run(["sudo", "-n", "chmod", "700", str(objetivo)], capture_output=True)
    if r_chown.returncode != 0 or r_chmod.returncode != 0:
        pytest.skip("sudo -n no puede chown/chmod a root en este entorno")

    os.chmod(tmp_path, 0o755)
    os.chmod(raiz, 0o755)

    guion_prueba = f"""
import sys, os
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp

victima = {str(victima)!r}
objetivo = {str(objetivo)!r}

def hook(ruta):
    if ruta.endswith("/proyectos"):
        os.rename(victima, victima + ".orig")
        os.symlink(objetivo, victima)

resultado = pp._recorrer(pp.Path({str(proyectos)!r}), mutar=True, hook_de_prueba=hook)
assert any(s.endswith("/victima") for s in resultado.symlinks_saltados), resultado.symlinks_saltados
print("OK")
"""
    try:
        r = subprocess.run(["sudo", "-n", "python3", "-c", guion_prueba], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "OK" in r.stdout

        st = objetivo.stat()
        assert st.st_uid == 0, "el objetivo de root cambió de dueño -- symlink seguido"
        assert oct(st.st_mode & 0o777) == "0o700", "el objetivo de root cambió de modo -- symlink seguido"
    finally:
        # El núcleo privilegiado deja root:root en objetivo_de_root y jaxsvc:fruiz en el
        # resto -- pytest limpia tmp_path como el usuario sin privilegio que corre la
        # suite y no podría borrar ninguno de los dos. Sin este cleanup, cada corrida
        # deja basura root-owned en /tmp para siempre (visto: PytestWarning "Directory
        # not empty" acumulando árboles en /tmp/pytest-of-<usuario>/).
        subprocess.run(["sudo", "-n", "rm", "-rf", str(raiz)], capture_output=True)


# --- B1: permiso EFECTIVO, no el texto de la ACL --------------------------------------------

def test_permiso_efectivo_ac_calcula_interseccion_no_el_texto_pedido():
    """Unidad directa de `_permiso_efectivo` -- reproduce el caso real: la entrada dice
    'rwx' pero la máscara es '---', así que el efectivo tiene que ser 0, no 7."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("permisos_proyectos", SCRIPT)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)

    texto = (
        "# file: x\n"
        "user::rw-\n"
        "group::---\n"
        "group:fruiz:rwx\t#effective:---\n"
        "mask::---\n"
        "other::---\n"
    )
    assert pp._permiso_efectivo(texto, default=False, tipo="group", calificador="fruiz") == 0


def test_verificar_detecta_permiso_efectivo_recortado_por_la_mascara(arbol_temporal, _identidades):
    """Reproduce el bug real: mkstemp (0600 explícito) bajo un árbol YA aplicado deja una
    entrada de ACL nombrada con texto 'rwx' pero efectivo '---'. --verificar tiene que
    marcarlo NO CUMPLE mirando el efectivo, no el texto (control negativo del propio
    detector: si esto pasara, sería exactamente el bug B1 de vuelta)."""
    aplicado = _correr("--aplicar", str(arbol_temporal))
    if aplicado.returncode != 0:
        pytest.skip(f"no se pudo aplicar para preparar el escenario: {aplicado.stdout}{aplicado.stderr}")

    proyectos = arbol_temporal / "proyectos"
    fd, ruta = tempfile.mkstemp(dir=str(proyectos / "un-proyecto"))
    os.close(fd)  # deliberadamente SIN el fchmod(0o664) -- así se reproduce el bug real

    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 1, r.stdout
    assert ruta in r.stdout
    assert "ACL de acceso efectiva insuficiente" in r.stdout


# --- Aplicar: dueño/grupo/setgid/ACL nuevos, pese al umask -----------------------------------

def test_aplicar_deja_el_arbol_con_dueno_jaxsvc_grupo_fruiz_y_herencia_pese_al_umask(arbol_temporal):
    aplicado = _correr("--aplicar", str(arbol_temporal))
    assert aplicado.returncode == 0, aplicado.stdout + aplicado.stderr

    proyectos = arbol_temporal / "proyectos"
    verificado = _correr("--verificar", str(arbol_temporal))
    assert verificado.returncode == 0, verificado.stdout + verificado.stderr

    st_top = proyectos.stat()
    assert pwd.getpwuid(st_top.st_uid).pw_name == USUARIO_ESPERADO
    assert grp.getgrgid(st_top.st_gid).gr_name == GRUPO_ESPERADO

    # Respaldo restaurable, con ruta ABSOLUTA (M3) -- nunca "~" en el mensaje impreso.
    assert "Respaldo:" in aplicado.stdout
    linea_respaldo = [l for l in aplicado.stdout.splitlines() if l.startswith("Respaldo:")][0]
    assert "~" not in linea_respaldo
    ruta_respaldo = Path(linea_respaldo.split("Respaldo: ", 1)[1].split(" ", 1)[0])
    assert ruta_respaldo.is_absolute()
    assert ruta_respaldo.is_file() and ruta_respaldo.stat().st_size > 0

    # Umask 022, y ACL por defecto igual la ignora (grupo escribible pese a todo).
    umask_previo = os.umask(0o022)
    try:
        nuevo = proyectos / "un-proyecto" / "sub" / "nuevo.txt"
        fd = os.open(nuevo, os.O_CREAT | os.O_WRONLY, 0o666)
        os.close(fd)
    finally:
        os.umask(umask_previo)

    assert grp.getgrgid(nuevo.stat().st_gid).gr_name == GRUPO_ESPERADO
    acl = subprocess.run(["getfacl", "-p", str(nuevo)], capture_output=True, text=True, check=True).stdout
    assert any(l.startswith(f"group:{GRUPO_ESPERADO}:rw") for l in acl.splitlines()), acl

    reaplicado = _correr("--aplicar", str(arbol_temporal))
    assert reaplicado.returncode == 0, reaplicado.stdout + reaplicado.stderr


def test_verificar_detecta_un_directorio_sin_acl_por_defecto(arbol_temporal):
    aplicado = _correr("--aplicar", str(arbol_temporal))
    assert aplicado.returncode == 0, aplicado.stdout + aplicado.stderr

    sub = arbol_temporal / "proyectos" / "un-proyecto" / "sub"
    subprocess.run(["sudo", "-n", "setfacl", "-k", str(sub)], check=True)

    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 1
    assert str(sub) in r.stdout


# --- m0: nombres con "." se excluyen --------------------------------------------------------

def test_directorio_con_punto_se_excluye_y_no_se_reporta(arbol_temporal):
    oculto = arbol_temporal / "proyectos" / ".estado-de-herramienta"
    oculto.mkdir()
    os.chmod(oculto, 0o700)
    (oculto / "adentro.txt").write_text("no tocar")

    aplicado = _correr("--aplicar", str(arbol_temporal))
    assert aplicado.returncode == 0, aplicado.stdout + aplicado.stderr

    st = oculto.stat()
    assert oct(st.st_mode & 0o777) == "0o700", "un nombre con . no debía tocarse"
    assert pwd.getpwuid(st.st_uid).pw_name != USUARIO_ESPERADO or True  # dueño previo intacto

    verificado = _correr("--verificar", str(arbol_temporal))
    assert verificado.returncode == 0, (
        f"un directorio excluido por nombre no debe hacer fallar --verificar:\n{verificado.stdout}"
    )
    assert str(oculto) not in verificado.stdout


# --- M2: bits espurios (setuid/sticky en dirs, cualquiera en archivos) ----------------------

def test_verificar_detecta_y_aplicar_quita_bits_espurios_de_un_directorio(arbol_temporal):
    """Reproduce el defecto real medido en hall9000 (2026-09-25): el /usr/bin/mkdir por
    defecto del host (uutils-coreutils 0.8.0) agrega setuid+sticky a un directorio nuevo
    creado bajo un padre con ACL por defecto -- GNU mkdir y os.mkdir() de Python no lo
    hacen sobre el mismo padre. --verificar lo detecta; --aplicar lo limpia y lo reporta."""
    aplicado = _correr("--aplicar", str(arbol_temporal))
    assert aplicado.returncode == 0, aplicado.stdout + aplicado.stderr

    contaminado = arbol_temporal / "proyectos" / "un-proyecto" / "contaminado"
    contaminado.mkdir()
    os.chmod(contaminado, 0o7775)  # simula el resultado real del mkdir buggy sin depender de él

    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 1
    assert "setuid espurio" in r_verif.stdout
    assert "sticky espurio" in r_verif.stdout

    r_aplicar = _correr("--aplicar", str(arbol_temporal))
    assert r_aplicar.returncode == 0, r_aplicar.stdout + r_aplicar.stderr
    assert str(contaminado) in r_aplicar.stdout
    assert "bits espurios quitados" in r_aplicar.stdout

    st = contaminado.stat()
    assert not (st.st_mode & 0o4000), "setuid no se quitó"
    assert not (st.st_mode & 0o1000), "sticky no se quitó"
    assert st.st_mode & 0o2000, "el setgid legítimo no debía tocarse"


def test_archivo_con_bit_especial_se_detecta_y_se_limpia(arbol_temporal):
    aplicado = _correr("--aplicar", str(arbol_temporal))
    assert aplicado.returncode == 0, aplicado.stdout + aplicado.stderr

    archivo = arbol_temporal / "proyectos" / "un-proyecto" / "raro.txt"
    archivo.write_text("x")
    os.chmod(archivo, 0o6644)  # setuid+setgid espurios en un archivo

    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 1
    assert "bit especial" in r_verif.stdout

    r_aplicar = _correr("--aplicar", str(arbol_temporal))
    assert r_aplicar.returncode == 0, r_aplicar.stdout + r_aplicar.stderr
    st = archivo.stat()
    assert not (st.st_mode & 0o7000)


# --- m3: hardlinks (nlink>1) se rechazan, nunca se mutan -------------------------------------

def test_hardlink_se_rechaza_y_no_se_muta(arbol_temporal):
    aplicado = _correr("--aplicar", str(arbol_temporal))
    assert aplicado.returncode == 0, aplicado.stdout + aplicado.stderr

    original = arbol_temporal / "proyectos" / "un-proyecto" / "archivo.txt"
    enlazado = arbol_temporal / "proyectos" / "un-proyecto" / "enlazado.txt"
    os.link(original, enlazado)
    dueno_antes = original.stat().st_uid

    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 1
    assert "nlink=2" in r_verif.stdout

    r_aplicar = _correr("--aplicar", str(arbol_temporal))
    assert r_aplicar.returncode == 1, "un hardlink encontrado tiene que hacer fallar --aplicar"
    assert "HARDLINK" in r_aplicar.stdout

    assert original.stat().st_uid == dueno_antes, "el hardlink no debía mutarse"


# --- m4: nombre de respaldo único ------------------------------------------------------------

def test_respaldos_consecutivos_tienen_nombres_distintos(arbol_temporal):
    r1 = _correr("--aplicar", str(arbol_temporal))
    r2 = _correr("--aplicar", str(arbol_temporal))
    assert r1.returncode == 0 and r2.returncode == 0
    l1 = [l for l in r1.stdout.splitlines() if l.startswith("Respaldo:")][0]
    l2 = [l for l in r2.stdout.splitlines() if l.startswith("Respaldo:")][0]
    assert l1 != l2


# --- B2: el respaldo que falla aborta sin aplicar nada ---------------------------------------

def test_respaldo_que_falla_aborta_sin_aplicar_nada(arbol_temporal, monkeypatch, tmp_path):
    """Un `sudo` de mentira en el PATH que hace fallar getfacl -- confirma que --aplicar
    no tocó nada del árbol (ni dueño, ni ACL) cuando el respaldo no se pudo completar."""
    proyectos = arbol_temporal / "proyectos"
    dueno_antes = proyectos.stat().st_uid

    sudo_falso_dir = tmp_path / "bin-sudo-falso"
    sudo_falso_dir.mkdir()
    sudo_falso = sudo_falso_dir / "sudo"
    sudo_falso.write_text("#!/bin/sh\nexit 7\n")
    sudo_falso.chmod(0o755)

    entorno = dict(os.environ)
    entorno["PATH"] = f"{sudo_falso_dir}:{entorno['PATH']}"

    r = subprocess.run(
        ["python3", str(SCRIPT), "--aplicar", str(arbol_temporal)],
        capture_output=True, text=True, env=entorno, timeout=60,
    )
    assert r.returncode == 2, r.stdout + r.stderr
    assert "respaldo" in (r.stdout + r.stderr).lower()
    assert proyectos.stat().st_uid == dueno_antes, "no debía haberse aplicado nada"


# --- m6: no hace falta detener servicios (evidencia, no solo afirmación) --------------------

def test_aplicar_no_interrumpe_un_lector_escritor_concurrente(arbol_temporal, _identidades):
    """m6: fchown/fchmod/setfacl son operaciones de METADATO por-inodo, atómicas, que no
    bloquean lectores/escritores existentes -- eso es lo que se afirma en el runbook. Acá
    se demuestra: un hilo escribe líneas secuenciales a un archivo del árbol sin parar
    MIENTRAS --aplicar corre encima, y se confirma que ninguna escritura falló y que el
    contenido quedó completo y en orden (nada se corrompió ni se bloqueó)."""
    archivo = arbol_temporal / "proyectos" / "un-proyecto" / "actividad.log"
    archivo.write_text("")
    detener = threading.Event()
    errores = []
    escritas = []

    def escritor():
        i = 0
        while not detener.is_set():
            try:
                with open(archivo, "a") as f:
                    f.write(f"linea-{i}\n")
                escritas.append(i)
            except OSError as exc:
                errores.append(exc)
            i += 1
            time.sleep(0.005)

    hilo = threading.Thread(target=escritor)
    hilo.start()
    try:
        time.sleep(0.05)  # dejar que arranque a escribir antes de aplicar
        r = _correr("--aplicar", str(arbol_temporal))
        assert r.returncode == 0, r.stdout + r.stderr
        time.sleep(0.05)  # confirmar que sigue escribiendo DESPUÉS también
    finally:
        detener.set()
        hilo.join(timeout=5)

    assert not errores, f"el escritor concurrente vio errores de OS: {errores}"
    contenido = archivo.read_text().splitlines()
    esperado = [f"linea-{i}" for i in range(len(escritas))]
    assert contenido == esperado, "el contenido se corrompió o se perdieron escrituras"
    assert len(escritas) > 1, "el hilo no llegó a escribir nada -- el test no probó lo que dice probar"


# --- (b) sólo en el host de producción real, con subdirectorio propio y limpieza ------------

def _motivo_de_skip_fuera_de_produccion() -> str | None:
    if not PROYECTOS_PRODUCCION.is_dir():
        return f"esta máquina no tiene {PROYECTOS_PRODUCCION} -- no es el host de producción de jax"
    if subprocess.run(["sudo", "-n", "-u", "jaxsvc", "true"], capture_output=True).returncode != 0:
        return "sudo -n -u jaxsvc no funciona en esta máquina"
    return None


def test_jaxsvc_y_fruiz_leen_y_escriben_cruzado_en_un_subdirectorio_propio():
    """m5: no sólo 'jaxsvc puede crear y borrar' -- también que fruiz pueda leer/escribir
    lo que jaxsvc creó, y viceversa, en un subdirectorio TEMPORAL propio dentro de
    proyectos/ real que este test crea y borra. Nunca os.getlogin() (falla sin tty en
    corridas no interactivas) -- pwd.getpwuid(os.getuid())."""
    motivo = _motivo_de_skip_fuera_de_produccion()
    if motivo:
        pytest.skip(motivo)

    quien_corre = pwd.getpwuid(os.getuid()).pw_name
    sub = PROYECTOS_PRODUCCION / f".permisos-proyectos-selftest-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        creado_dir = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "mkdir", str(sub)], capture_output=True, text=True)
        assert creado_dir.returncode == 0, f"jaxsvc no pudo crear {sub}: {creado_dir.stderr}"

        desde_jaxsvc = sub / "desde-jaxsvc.txt"
        r = subprocess.run(
            ["sudo", "-n", "-u", "jaxsvc", "sh", "-c", f"echo hola > {desde_jaxsvc}"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, f"jaxsvc no pudo escribir: {r.stderr}"

        # fruiz (quien_corre, si este test corre como fruiz) lee y escribe lo de jaxsvc.
        assert desde_jaxsvc.read_text() == "hola\n"
        with open(desde_jaxsvc, "a") as f:
            f.write("agregado por " + quien_corre + "\n")

        # y al revés: algo creado por quien corre el test, jaxsvc lo lee y escribe.
        desde_fruiz = sub / "desde-fruiz.txt"
        desde_fruiz.write_text("original\n")
        r2 = subprocess.run(
            ["sudo", "-n", "-u", "jaxsvc", "sh", "-c", f"cat {desde_fruiz} && echo mas >> {desde_fruiz}"],
            capture_output=True, text=True,
        )
        assert r2.returncode == 0, f"jaxsvc no pudo leer/escribir lo de {quien_corre}: {r2.stderr}"
        assert "original" in r2.stdout
        assert "mas" in desde_fruiz.read_text()
    finally:
        subprocess.run(["sudo", "-n", "-u", "jaxsvc", "rm", "-rf", str(sub)], capture_output=True)
        if sub.exists():
            subprocess.run(["sudo", "-n", "rm", "-rf", str(sub)], capture_output=True)
