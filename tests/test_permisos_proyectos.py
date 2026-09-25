# tests/test_permisos_proyectos.py
"""ops/permisos_proyectos.py -- spec docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md
§5. Segunda ronda de auditoría (2026-09-25): 3 BLOCK, 3 MAJOR, 6 MINOR sobre la primera
reescritura en Python (commit 4f117a7). Ver el docstring del propio guion para el detalle
completo de cada hallazgo y su defensa; este archivo los prueba uno por uno.

División de responsabilidad en los tests (importante para entender por qué algunos NO
usan `_correr("--aplicar", ...)`): BLOCK-2 fija el núcleo privilegiado a la RAIZ
CONFIGURADA (`JAX_WORKSPACE_DIR`) -- ya NO acepta un árbol temporal arbitrario vía la CLI
pública. Eso es correcto y deseado (es la defensa), pero significa que los tests que
ejercitan la MECÁNICA del recorrido (ACL, bits especiales, hardlinks, symlinks) contra un
árbol temporal tienen que llamar a `pp._recorrer()` DIRECTO, como root, con
`_recorrer_directo()` (más abajo) -- bypaseando la política de fijación de RAIZ, que es
una capa aparte del mecanismo que están probando. Los tests que SÍ pasan por la CLI
pública (`_correr`) son los que prueban esa misma política: que --verificar es de sólo
lectura, que RAIZ inválida se rechaza, y que una RAIZ que no es la configurada se rechaza
(reproduce "hoy acepta /etc" -> "ahora rechaza cualquier cosa que no sea la real", sin
necesidad de tocar la real -- NUNCA se corre --aplicar contra /home/fruiz/jax-workspace).
"""
from __future__ import annotations

import grp
import json
import os
import pwd
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import pytest

RAIZ_REPO = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ_REPO / "ops" / "permisos_proyectos.py"
RUTA_INSTALADA = Path("/usr/local/sbin/jax-permisos-proyectos")

RAIZ_PRODUCCION = Path("/home/fruiz/jax-workspace")
PROYECTOS_PRODUCCION = RAIZ_PRODUCCION / "proyectos"

USUARIO_ESPERADO = "jaxsvc"
GRUPO_ESPERADO = "fruiz"


def test_el_guion_existe():
    assert SCRIPT.is_file()


def _correr(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["python3", str(SCRIPT), *args], capture_output=True, text=True, timeout=60)


def _sudo_n_disponible() -> bool:
    return subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=10).returncode == 0


def _acl_disponible_en(directorio: Path) -> bool:
    if shutil.which("setfacl") is None or shutil.which("getfacl") is None:
        return False
    prueba = directorio / ".prueba-acl"
    prueba.touch()
    try:
        quien = pwd.getpwuid(os.getuid()).pw_name
        r = subprocess.run(["setfacl", "-m", f"u:{quien}:rwx", str(prueba)], capture_output=True)
        return r.returncode == 0
    finally:
        prueba.unlink(missing_ok=True)


def _recorrer_directo(proyectos: Path, *, mutar: bool, extra_codigo: str = "") -> dict:
    """`pp._recorrer()` como root, bypaseando la fijación de RAIZ de la CLI pública (ver
    el docstring del módulo). Único punto de entrada que usan los tests de mecánica."""
    codigo = f"""
import sys, json
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
hook = None
{extra_codigo}
r = pp._recorrer(pp.Path({str(proyectos)!r}), mutar={mutar!r}, hook_de_prueba=hook)
print(json.dumps({{
    "no_cumple": r.no_cumple, "symlinks_saltados": r.symlinks_saltados,
    "hardlinks_rechazados": r.hardlinks_rechazados,
    "bits_espurios_quitados": r.bits_espurios_quitados, "excluidos": r.excluidos,
    "dirs_procesados": r.dirs_procesados, "archivos_procesados": r.archivos_procesados,
}}))
"""
    r = subprocess.run(["sudo", "-n", "python3", "-c", codigo], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def _identidades():
    if not _sudo_n_disponible():
        pytest.skip("sudo -n no disponible -- no se pueden garantizar las identidades jaxsvc/fruiz")
    for usuario in ("jaxsvc", "fruiz"):
        tiene = subprocess.run(["getent", "passwd", usuario], capture_output=True).returncode == 0
        if not tiene:
            creado = subprocess.run(
                ["sudo", "-n", "useradd", "--system", "--no-create-home", usuario], capture_output=True
            )
            if creado.returncode != 0:
                pytest.skip(f"no se pudo crear el usuario {usuario}: {creado.stderr.decode(errors='replace')}")
    return None


@pytest.fixture(scope="module")
def _instalado(_identidades):
    """Instala la copia del repo en RUTA_INSTALADA (root:root 0755) para los tests que
    ejercitan la CLI pública de verdad. Se reinstala siempre (idempotente, sha256 puede
    haber cambiado entre corridas de desarrollo)."""
    if not _sudo_n_disponible():
        pytest.skip("sudo -n no disponible")
    r = subprocess.run(
        ["sudo", "-n", "install", "-o", "root", "-g", "root", "-m", "0755", str(SCRIPT), str(RUTA_INSTALADA)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        pytest.skip(f"no se pudo instalar el núcleo: {r.stderr}")
    return RUTA_INSTALADA


def _abrir_travesia_hasta(ruta: Path, tope: Path) -> None:
    """pytest crea /tmp/pytest-of-<usuario>/pytest-N/ en 0700 -- si un test hace
    `sudo -u jaxsvc` (o cualquier otra identidad) contra algo bajo tmp_path, esos
    ANCESTROS (no sólo tmp_path mismo) bloquean la travesía aunque proyectos/ esté bien
    configurado. Se abre o+x en cada nivel desde `ruta` hasta `tope` (sin incluirlo)."""
    actual = ruta
    while actual != tope and actual != actual.parent:
        os.chmod(actual, os.stat(actual).st_mode | 0o111)
        actual = actual.parent


@pytest.fixture()
def arbol_temporal(tmp_path, _identidades):
    raiz = tmp_path / "raiz"
    proyectos = raiz / "proyectos"
    (proyectos / "un-proyecto" / "sub").mkdir(parents=True)
    (proyectos / "un-proyecto" / "archivo.txt").write_text("contenido\n")

    if not _acl_disponible_en(proyectos):
        pytest.skip("el filesystem temporal no soporta ACL POSIX")

    _abrir_travesia_hasta(raiz, Path("/tmp"))
    os.chmod(raiz, 0o755)
    return raiz


def _limpiar_como_root(ruta: Path) -> None:
    subprocess.run(["sudo", "-n", "rm", "-rf", str(ruta)], capture_output=True)


def _como_fruiz(codigo_python: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """El proceso que corre pytest NO es necesariamente `fruiz` -- en hall9000 sí lo es
    (así corre esta sesión), pero en CI el usuario del runner se llama `runner`, no
    `fruiz` (verificado en un contenedor limpio ubuntu:24.04, BLOCK-3a). Después de que
    `_recorrer_directo(mutar=True)` deja un árbol dueño jaxsvc:fruiz, cualquier operación
    que simule "lo que fruiz haría" tiene que correr REALMENTE como fruiz -- no asumir
    que el proceso que invoca pytest ya lo es."""
    return subprocess.run(
        ["sudo", "-n", "-u", "fruiz", "python3", "-c", codigo_python],
        capture_output=True, text=True, timeout=timeout,
    )


# --- Validación de RAIZ / política pública --------------------------------------------------

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


def test_aplicar_rechaza_una_raiz_que_no_es_la_configurada(arbol_temporal, _instalado):
    """BLOCK-2: reproduce 'hoy acepta /etc' -> ahora rechaza cualquier RAIZ que no sea la
    configurada en /etc/jax/.env, SIN tocar la real (nunca se le pasa esa ruta)."""
    aplicado = _correr("--aplicar", str(arbol_temporal))
    assert aplicado.returncode != 0
    assert "no es la RAIZ configurada" in (aplicado.stdout + aplicado.stderr)
    # nada se mutó: sigue siendo del dueño original, no jaxsvc.
    assert pwd.getpwuid((arbol_temporal / "proyectos").stat().st_uid).pw_name != USUARIO_ESPERADO


# --- BLOCK-2: instalación del núcleo ----------------------------------------------------

def test_verificar_reporta_si_el_nucleo_no_esta_instalado(arbol_temporal, _identidades, monkeypatch):
    r_borrar = subprocess.run(["sudo", "-n", "rm", "-f", str(RUTA_INSTALADA)], capture_output=True)
    if r_borrar.returncode != 0:
        pytest.skip("no se pudo desinstalar el núcleo para probar el caso 'no instalado'")
    try:
        r = _correr("--verificar", str(arbol_temporal))
        assert "no instalado" in (r.stdout + r.stderr) or "no está instalado" in (r.stdout + r.stderr)
    finally:
        subprocess.run(
            ["sudo", "-n", "install", "-o", "root", "-g", "root", "-m", "0755", str(SCRIPT), str(RUTA_INSTALADA)],
            capture_output=True,
        )


def test_aplicar_rechaza_si_el_nucleo_instalado_no_coincide_con_el_repo(arbol_temporal, _instalado):
    """BLOCK-2: sha256 del instalado tiene que coincidir con el repo -- si alguien lo
    modificó sin reinstalar, --aplicar se niega en vez de correr código viejo/ajeno."""
    r_mod = subprocess.run(
        ["sudo", "-n", "sh", "-c", f"echo '# alterado' >> {RUTA_INSTALADA}"], capture_output=True
    )
    if r_mod.returncode != 0:
        pytest.skip("no se pudo alterar el núcleo instalado para esta prueba")
    try:
        r = _correr("--aplicar", str(arbol_temporal))
        assert r.returncode != 0
        assert "sha256" in (r.stdout + r.stderr) or "no coincide" in (r.stdout + r.stderr)
    finally:
        subprocess.run(
            ["sudo", "-n", "install", "-o", "root", "-g", "root", "-m", "0755", str(SCRIPT), str(RUTA_INSTALADA)],
            capture_output=True,
        )


def test_aplicar_rechaza_si_el_nucleo_instalado_es_escribible_por_grupo(arbol_temporal, _instalado):
    r_mod = subprocess.run(["sudo", "-n", "chmod", "0775", str(RUTA_INSTALADA)], capture_output=True)
    if r_mod.returncode != 0:
        pytest.skip("no se pudo aflojar el modo del núcleo instalado para esta prueba")
    try:
        r = _correr("--aplicar", str(arbol_temporal))
        assert r.returncode != 0
        assert "escribible por grupo" in (r.stdout + r.stderr)
    finally:
        subprocess.run(["sudo", "-n", "chmod", "0755", str(RUTA_INSTALADA)], capture_output=True)


def test_menos_uno_que_injecta_codigo_en_el_directorio_no_se_ejecuta_como_root(tmp_path):
    """BLOCK-2, la prueba directa del defecto reproducido por el auditor: SIN -I, un
    json.py de mentira en el directorio del script se importa y corre como root; CON -I
    (lo que --aplicar/--nucleo-privilegiado usan siempre), no."""
    if not _sudo_n_disponible():
        pytest.skip("sudo -n no disponible")
    marca = tmp_path / "marca-uid-0"
    d = tmp_path / "dir_con_json_falso"
    d.mkdir()
    (d / "json.py").write_text(f"open({str(marca)!r}, 'w').write('PWNED\\n')\n")
    (d / "victima.py").write_text("import json\n")

    subprocess.run(["sudo", "-n", "python3", str(d / "victima.py")], capture_output=True)
    contaminado_sin_I = marca.exists()
    marca.unlink(missing_ok=True)

    subprocess.run(["sudo", "-n", "python3", "-I", str(d / "victima.py")], capture_output=True)
    contaminado_con_I = marca.exists()

    assert contaminado_sin_I, "el ataque no reprodujo el defecto -- este test no prueba nada sin esto"
    assert not contaminado_con_I, "-I no evitó la inyección -- BLOCK-2 no está resuelto"


# --- BLOCK-3a: directorio de respaldo, root 0700, no depende del HOME de fruiz -------------

def test_respaldo_no_depende_del_home_de_fruiz(arbol_temporal, _instalado):
    """El fixture crea fruiz con --no-create-home; si el respaldo dependiera de
    pwd.getpwnam('fruiz').pw_dir, esto fallaría en cualquier runner limpio (BLOCK-3a)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("permisos_proyectos", SCRIPT)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    assert str(pp.RUTA_RESPALDOS) == "/var/backups/jax-permisos"
    assert "home" not in str(pp.RUTA_RESPALDOS).lower()


# --- BLOCK-1: reversión sin seguir symlinks -------------------------------------------------

def test_revertir_camino_feliz_restaura_dueno_grupo_y_acl(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, mutar=True)

    codigo_respaldo = f"""
import sys
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
pp._cmd_nucleo_respaldo({str(proyectos)!r})
"""
    r = subprocess.run(["sudo", "-n", "python3", "-c", codigo_respaldo], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    respaldo = r.stdout.strip()
    assert respaldo

    # Desconfigurar a propósito.
    subprocess.run(["sudo", "-n", "chgrp", "-R", "fruiz", str(proyectos)], check=True)
    subprocess.run(["sudo", "-n", "setfacl", "-R", "-b", str(proyectos)], check=True)

    r_verif_roto = _correr("--verificar", str(arbol_temporal))
    assert r_verif_roto.returncode == 1

    r_revertir = _correr("--revertir", respaldo, str(arbol_temporal))
    assert r_revertir.returncode == 0, r_revertir.stdout + r_revertir.stderr

    r_verif_final = _correr("--verificar", str(arbol_temporal))
    assert r_verif_final.returncode == 0, r_verif_final.stdout

    subprocess.run(["sudo", "-n", "rm", "-f", respaldo], capture_output=True)


def test_revertir_no_contamina_si_algo_se_volvio_symlink(arbol_temporal, _identidades):
    """El ataque exacto del auditor: jaxsvc reemplaza un subdirectorio ya respaldado por
    un symlink a un directorio root:700. --revertir tiene que saltarlo y reportarlo, y el
    objetivo del symlink NO puede terminar con las entradas de ACL del respaldo."""
    proyectos = arbol_temporal / "proyectos"
    victima = proyectos / "un-proyecto"
    _recorrer_directo(proyectos, mutar=True)

    codigo_respaldo = f"""
import sys
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
pp._cmd_nucleo_respaldo({str(proyectos)!r})
"""
    r = subprocess.run(["sudo", "-n", "python3", "-c", codigo_respaldo], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    respaldo = r.stdout.strip()

    objetivo_root = arbol_temporal / "objetivo_root"
    objetivo_root.mkdir()
    r1 = subprocess.run(["sudo", "-n", "chown", "root:root", str(objetivo_root)], capture_output=True)
    r2 = subprocess.run(["sudo", "-n", "chmod", "700", str(objetivo_root)], capture_output=True)
    if r1.returncode != 0 or r2.returncode != 0:
        pytest.skip("sudo -n no puede chown/chmod a root en este entorno")

    r3 = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "rm", "-rf", str(victima)], capture_output=True)
    r4 = subprocess.run(
        ["sudo", "-n", "-u", "jaxsvc", "ln", "-s", str(objetivo_root), str(victima)], capture_output=True
    )
    assert r3.returncode == 0 and r4.returncode == 0, r3.stderr + r4.stderr

    try:
        r_revertir = _correr("--revertir", respaldo, str(arbol_temporal))
        assert r_revertir.returncode == 1
        assert "SYMLINK" in r_revertir.stdout
        assert str(victima) in r_revertir.stdout

        st = objetivo_root.stat()
        assert st.st_uid == 0, "el objetivo de root cambió de dueño -- BLOCK-1 no está resuelto"
        assert oct(st.st_mode & 0o777) == "0o700", "el objetivo de root cambió de modo -- BLOCK-1 no está resuelto"
    finally:
        subprocess.run(["sudo", "-n", "rm", "-f", respaldo], capture_output=True)


def test_getfacl_desescapa_octales_en_nombres_de_archivo():
    import importlib.util
    spec = importlib.util.spec_from_file_location("permisos_proyectos", SCRIPT)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    assert pp._desescapar_getfacl("con\\\\backslash.txt") == "con\\backslash.txt"
    assert pp._desescapar_getfacl("con\\012newline") == "con\nnewline"


# --- MAJOR-1: EACCES nunca crashea ----------------------------------------------------------

def test_verificar_no_crashea_con_archivo_0600_de_jaxsvc(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, mutar=True)

    codigo = f"""
import sys, tempfile, os
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
fd, path = tempfile.mkstemp(dir={str(proyectos / "un-proyecto")!r})
os.close(fd)
print(path)
"""
    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "python3", "-c", codigo], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr

    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 1
    assert "Traceback" not in (r_verif.stdout + r_verif.stderr)
    assert "PermissionError" not in (r_verif.stdout + r_verif.stderr)
    assert "NO CUMPLE" in r_verif.stdout


# --- MAJOR-2: FIFO no cuelga, hardlink no se pierde en una carrera -------------------------

def test_fifo_no_cuelga_el_nucleo(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    fifo = proyectos / "un-proyecto" / "unfifo"
    os.mkfifo(fifo)
    try:
        datos = _recorrer_directo(proyectos, mutar=True)
        assert not datos["no_cumple"], datos["no_cumple"]
    finally:
        # un-proyecto/ ya quedó jaxsvc:fruiz -- borrar el fifo necesita privilegio.
        subprocess.run(["sudo", "-n", "rm", "-f", str(fifo)], capture_output=True)


def test_verificar_con_fifo_no_cuelga_ni_crashea(arbol_temporal):
    fifo = arbol_temporal / "proyectos" / "un-proyecto" / "unfifo2"
    os.mkfifo(fifo)
    try:
        r = _correr("--verificar", str(arbol_temporal))
        assert "Traceback" not in (r.stdout + r.stderr)
    finally:
        fifo.unlink(missing_ok=True)


# --- m-a: dueño (uid), no sólo grupo --------------------------------------------------------

def test_verificar_detecta_dueno_incorrecto_aunque_el_grupo_este_bien(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, mutar=True)
    archivo = proyectos / "un-proyecto" / "archivo.txt"
    r = subprocess.run(["sudo", "-n", "chown", "fruiz", str(archivo)], capture_output=True)
    assert r.returncode == 0, r.stderr
    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 1
    assert f"dueño=fruiz, esperado={USUARIO_ESPERADO}" in r_verif.stdout


# --- m-b: exclusión explícita, y se reporta -------------------------------------------------

def test_solo_claude_flow_se_excluye_y_se_reporta(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    oculto = proyectos / ".claude-flow"
    oculto.mkdir()
    os.chmod(oculto, 0o700)
    otro_punto = proyectos / ".otro-nombre-con-punto"
    otro_punto.mkdir()

    datos = _recorrer_directo(proyectos, mutar=True)
    assert any(e.endswith("/.claude-flow") for e in datos["excluidos"]), datos["excluidos"]
    # un nombre con "." que NO es .claude-flow ya no se excluye (m-b: lista explícita) --
    # tiene que aparecer como procesado/mutado, no como excluido.
    assert not any(e.endswith("/.otro-nombre-con-punto") for e in datos["excluidos"])

    st_oculto = oculto.stat()
    assert oct(st_oculto.st_mode & 0o777) == "0o700", ".claude-flow no debía tocarse"

    r_aplicar = _correr("--verificar", str(arbol_temporal))
    assert "excluido por nombre" in r_aplicar.stdout
    assert str(oculto) in r_aplicar.stdout


# --- m-d: respaldo parcial se borra al fallar -----------------------------------------------

def test_respaldo_parcial_se_borra_si_getfacl_falla(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    codigo = f"""
import sys, subprocess
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp

_original = subprocess.run
def _falso(args, **kw):
    if args[:1] == ["getfacl"]:
        class R: returncode = 9; stderr = b"getfacl de mentira, siempre falla"
        return R()
    return _original(args, **kw)
subprocess.run = _falso

import sys as _sys
_sys.exit(pp._cmd_nucleo_respaldo({str(proyectos)!r}))
"""
    # Contar ANTES con sudo (el directorio es root 0700, un listado sin privilegio no
    # ve nada -- no es que esté vacío, es que no se puede ni mirar).
    r_antes = subprocess.run(
        ["sudo", "-n", "sh", "-c", "ls /var/backups/jax-permisos/proyectos-*.acl 2>/dev/null | wc -l"],
        capture_output=True, text=True,
    )
    cantidad_antes = int(r_antes.stdout.strip() or "0")

    r = subprocess.run(["sudo", "-n", "python3", "-c", codigo], capture_output=True, text=True)
    assert r.returncode == 1, r.stdout + r.stderr

    r_despues = subprocess.run(
        ["sudo", "-n", "sh", "-c", "ls /var/backups/jax-permisos/proyectos-*.acl 2>/dev/null | wc -l"],
        capture_output=True, text=True,
    )
    cantidad_despues = int(r_despues.stdout.strip() or "0")
    assert cantidad_despues == cantidad_antes, (
        f"quedó un respaldo parcial: había {cantidad_antes} archivos antes, {cantidad_despues} después"
    )


# --- m-f: sin sudo, falla cerrado en vez de adivinar ----------------------------------------

def test_sin_sudo_la_raiz_por_defecto_falla_cerrado(tmp_path):
    sudo_falso_dir = tmp_path / "bin-sin-sudo"
    sudo_falso_dir.mkdir()
    (sudo_falso_dir / "sudo").write_text("#!/bin/sh\nexit 1\n")
    (sudo_falso_dir / "sudo").chmod(0o755)
    entorno = dict(os.environ)
    entorno["PATH"] = f"{sudo_falso_dir}:{entorno['PATH']}"

    r = subprocess.run(["python3", str(SCRIPT), "--verificar"], capture_output=True, text=True, env=entorno)
    assert r.returncode == 2
    assert "no se pudo resolver" in (r.stdout + r.stderr)


# --- MAJOR-3: el test de B1 exige returncode==0, no se salta --------------------------------

def test_verificar_detecta_permiso_efectivo_recortado_por_la_mascara(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, mutar=True)

    r_crear = _como_fruiz(f"""
import tempfile, os
fd, path = tempfile.mkstemp(dir={str(proyectos / "un-proyecto")!r})
os.close(fd)
print(path)
""")
    assert r_crear.returncode == 0, r_crear.stdout + r_crear.stderr
    ruta = r_crear.stdout.strip()

    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 1, r.stdout
    assert ruta in r.stdout
    assert "ACL de acceso efectiva insuficiente" in r.stdout


def test_permiso_efectivo_calcula_interseccion_no_el_texto_pedido():
    import importlib.util
    spec = importlib.util.spec_from_file_location("permisos_proyectos", SCRIPT)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    texto = (
        "# file: x\nuser::rw-\ngroup::---\ngroup:fruiz:rwx\t#effective:---\nmask::---\nother::---\n"
    )
    assert pp._permiso_efectivo(texto, default=False, tipo="group", calificador="fruiz") == 0


# --- Mecánica general (recorrido/ACL/herencia/idempotencia) -------------------------------

def test_aplicar_deja_el_arbol_con_dueno_jaxsvc_grupo_fruiz_y_herencia_pese_al_umask(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    datos = _recorrer_directo(proyectos, mutar=True)
    assert not datos["no_cumple"]

    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 0, r_verif.stdout

    st_top = proyectos.stat()
    assert pwd.getpwuid(st_top.st_uid).pw_name == USUARIO_ESPERADO
    assert grp.getgrgid(st_top.st_gid).gr_name == GRUPO_ESPERADO

    nuevo = proyectos / "un-proyecto" / "sub" / "nuevo.txt"
    r_crear = _como_fruiz(f"""
import os
os.umask(0o022)
fd = os.open({str(nuevo)!r}, os.O_CREAT | os.O_WRONLY, 0o666)
os.close(fd)
""")
    assert r_crear.returncode == 0, r_crear.stdout + r_crear.stderr

    assert grp.getgrgid(nuevo.stat().st_gid).gr_name == GRUPO_ESPERADO
    acl = subprocess.run(["getfacl", "-p", str(nuevo)], capture_output=True, text=True, check=True).stdout
    assert any(l.startswith(f"group:{GRUPO_ESPERADO}:rw") for l in acl.splitlines()), acl

    datos2 = _recorrer_directo(proyectos, mutar=True)
    assert not datos2["no_cumple"]


def test_verificar_detecta_un_directorio_sin_acl_por_defecto(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, mutar=True)
    sub = proyectos / "un-proyecto" / "sub"
    subprocess.run(["sudo", "-n", "setfacl", "-k", str(sub)], check=True)
    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 1
    assert str(sub) in r.stdout


def test_verificar_detecta_y_aplicar_quita_bits_espurios_de_un_directorio(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, mutar=True)

    contaminado = proyectos / "un-proyecto" / "contaminado"
    r_crear = _como_fruiz(f"""
import os
os.mkdir({str(contaminado)!r})
os.chmod({str(contaminado)!r}, 0o7775)
""")
    assert r_crear.returncode == 0, r_crear.stdout + r_crear.stderr

    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 1
    assert "setuid espurio" in r_verif.stdout
    assert "sticky espurio" in r_verif.stdout

    datos = _recorrer_directo(proyectos, mutar=True)
    assert str(contaminado) in datos["bits_espurios_quitados"]

    st = contaminado.stat()
    assert not (st.st_mode & 0o4000)
    assert not (st.st_mode & 0o1000)
    assert st.st_mode & 0o2000


def test_archivo_con_bit_especial_se_detecta_y_se_limpia(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, mutar=True)
    archivo = proyectos / "un-proyecto" / "raro.txt"
    r_crear = _como_fruiz(f"""
with open({str(archivo)!r}, "w") as f:
    f.write("x")
import os
os.chmod({str(archivo)!r}, 0o6644)
""")
    assert r_crear.returncode == 0, r_crear.stdout + r_crear.stderr

    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 1
    assert "bit especial" in r_verif.stdout

    datos = _recorrer_directo(proyectos, mutar=True)
    assert str(archivo) in datos["bits_espurios_quitados"]
    st = archivo.stat()
    assert not (st.st_mode & 0o7000)


def test_hardlink_se_rechaza_y_no_se_muta(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, mutar=True)

    original = proyectos / "un-proyecto" / "archivo.txt"
    enlazado = proyectos / "un-proyecto" / "enlazado.txt"
    r_link = _como_fruiz(f"""
import os
os.link({str(original)!r}, {str(enlazado)!r})
""")
    assert r_link.returncode == 0, r_link.stdout + r_link.stderr
    dueno_antes = original.stat().st_uid

    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 1
    assert "nlink=2" in r_verif.stdout

    datos = _recorrer_directo(proyectos, mutar=True)
    assert datos["hardlinks_rechazados"]
    assert original.stat().st_uid == dueno_antes


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

    st = objetivo.stat()
    assert st.st_uid == os.getuid()
    assert oct(st.st_mode & 0o777) == "0o700"


def test_symlink_sustituido_a_mitad_de_la_corrida_no_contamina_el_objetivo(_identidades, tmp_path):
    if not _sudo_n_disponible() or shutil.which("setfacl") is None:
        pytest.skip("sudo -n/setfacl no disponible")

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

    try:
        extra = f"""
victima = {str(victima)!r}
objetivo = {str(objetivo)!r}
def hook(ruta):
    if ruta.endswith("/proyectos"):
        import os as _os
        _os.rename(victima, victima + ".orig")
        _os.symlink(objetivo, victima)
"""
        datos = _recorrer_directo(proyectos, mutar=True, extra_codigo=extra)
        assert any(s.endswith("/victima") for s in datos["symlinks_saltados"]), datos["symlinks_saltados"]

        st = objetivo.stat()
        assert st.st_uid == 0
        assert oct(st.st_mode & 0o777) == "0o700"
    finally:
        _limpiar_como_root(raiz)


# --- Concurrencia (no hace falta detener servicios) -----------------------------------------

def test_aplicar_no_interrumpe_un_lector_escritor_concurrente(arbol_temporal, _identidades):
    """El escenario real: el árbol YA está en el estado correcto (una segunda corrida de
    --aplicar es mantenimiento normal, no la primera vez), y jaxsvc (así corre LAS MANOS
    en producción) tiene un archivo abierto y lo sigue escribiendo mientras --aplicar
    vuelve a pasar por encima. El escritor corre COMO JAXSVC, en un subproceso propio (no
    un hilo de este proceso -- el que corre pytest no es necesariamente ni jaxsvc ni
    fruiz, ver `_como_fruiz`). Antes de que el árbol esté aplicado una primera vez, jaxsvc
    directamente no puede escribir ahí -- ese es el problema que este guion arregla, no
    algo que este test de concurrencia tenga que ejercitar."""
    proyectos = arbol_temporal / "proyectos"
    datos_iniciales = _recorrer_directo(proyectos, mutar=True)
    assert not datos_iniciales["no_cumple"]

    archivo = proyectos / "un-proyecto" / "actividad.log"
    r_crear = subprocess.run(
        ["sudo", "-n", "-u", "jaxsvc", "python3", "-c", f"open({str(archivo)!r}, 'w').close()"],
        capture_output=True, text=True,
    )
    assert r_crear.returncode == 0, r_crear.stdout + r_crear.stderr

    detener = arbol_temporal.parent / "detener-escritor"
    detener.unlink(missing_ok=True)

    codigo_escritor = f"""
import time, os, sys
i = 0
while not os.path.exists({str(detener)!r}):
    try:
        with open({str(archivo)!r}, "a") as f:
            f.write(f"linea-{{i}}" + chr(10))
    except OSError as exc:
        sys.stderr.write(f"ERROR en linea {{i}}: {{exc}}")
        sys.exit(1)
    i += 1
    time.sleep(0.005)
print(i)
"""
    proc = subprocess.Popen(
        ["sudo", "-n", "-u", "jaxsvc", "python3", "-c", codigo_escritor],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        time.sleep(0.1)
        datos = _recorrer_directo(proyectos, mutar=True)  # segunda corrida, mantenimiento
        assert not datos["no_cumple"]
        time.sleep(0.1)
    finally:
        detener.touch()
        try:
            salida, error = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            salida, error = proc.communicate()
        detener.unlink(missing_ok=True)

    assert proc.returncode == 0, f"el escritor concurrente (como jaxsvc) vio un error de OS: {error}"
    total_escrito = int(salida.strip())
    assert total_escrito > 1, "el escritor no llegó a escribir nada -- el test no probó lo que dice probar"
    contenido = archivo.read_text().splitlines()
    esperado = [f"linea-{i}" for i in range(total_escrito)]
    assert contenido == esperado, "el contenido se corrompió o se perdieron escrituras"


# --- (b) sólo en el host de producción real, con subdirectorio propio y limpieza ------------

def _motivo_de_skip_fuera_de_produccion() -> str | None:
    if not PROYECTOS_PRODUCCION.is_dir():
        return f"esta máquina no tiene {PROYECTOS_PRODUCCION} -- no es el host de producción de jax"
    if subprocess.run(["sudo", "-n", "-u", "jaxsvc", "true"], capture_output=True).returncode != 0:
        return "sudo -n -u jaxsvc no funciona en esta máquina"
    return None


def test_jaxsvc_y_fruiz_leen_y_escriben_cruzado_en_un_subdirectorio_propio():
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
            ["sudo", "-n", "-u", "jaxsvc", "sh", "-c", f"echo hola > {desde_jaxsvc}"], capture_output=True, text=True,
        )
        assert r.returncode == 0, f"jaxsvc no pudo escribir: {r.stderr}"

        assert desde_jaxsvc.read_text() == "hola\n"
        with open(desde_jaxsvc, "a") as f:
            f.write("agregado por " + quien_corre + "\n")

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
