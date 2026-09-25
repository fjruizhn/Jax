# tests/test_permisos_proyectos.py
"""ops/permisos_proyectos.py -- spec docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md
§5. Tercera ronda de auditoría (2026-09-25): 2 BLOCK sobre el DISEÑO de la reversión de
la ronda 2 (los MAJOR/MINOR de esa ronda quedaron cerrados por construcción, no hubo que
tocarlos de nuevo). Ver el docstring del propio guion para el detalle completo.

BLOCK-1: reconstruir-desde-respaldo se retira por completo (tenía cuatro fallas de
   diseño reales: un directorio sin ACL todavía se leía como archivo, no restauraba
   setgid/bits/máscara, se comía un espacio final en un nombre, y aceptaba un respaldo
   fabricado con cualquier ruta). `--deshacer`/`--nucleo-deshacer` es DETERMINISTA: no
   lee ningún archivo de estado, lleva el árbol al único estado medido con `stat` real en
   producción (fruiz:fruiz, sin ACL, sin bits especiales, modo derivado del rwx que el
   dueño ya tiene). El respaldo de getfacl queda como registro forense, nunca se usa para
   reconstruir nada.

BLOCK-2: las tres entradas del núcleo (`--nucleo-privilegiado`, `--nucleo-respaldo`,
   `--nucleo-deshacer`) ya no aceptan ningún argumento de ruta -- cada una resuelve
   PROYECTOS de forma independiente, siempre desde /etc/jax/.env. Un argumento de más se
   rechaza sin tocar nada. Modos repetidos también se rechazan.

MAJOR-1 (ronda 3): getfacl -R -p ... > archivo puede dar rc==0 aunque la escritura haya
   fallado (verificado: getfacl ... > /dev/full también da rc==0). El respaldo ahora
   valida su propia cantidad de entradas contra un recorrido independiente del árbol
   real antes de darse por bueno.

m1 (rondas 2/3): sin /etc/jax/.env legible, todo lo que necesita PROYECTOS falla cerrado.
m2 (ronda 3): el sha256 del núcleo instalado se compara contra HEAD commiteado (no el
   working tree), y la cadena de directorios padre usa lstat (nunca sigue un symlink).
m4 (ronda 3): NOMBRES_EXCLUIDOS sólo aplica en profundidad 2 (proyectos/<proyecto>/.claude-flow),
   nunca en proyectos/ mismo ni más profundo.

División de responsabilidad en los tests (igual que la ronda 2): --aplicar/--deshacer
PÚBLICOS (la CLI real) sólo actúan sobre la RAIZ configurada -- eso es la defensa, no un
estorbo, pero significa que los tests de MECÁNICA (ACL, bits especiales, hardlinks,
symlinks, deshacer) llaman a `pp._recorrer()` DIRECTO, como root, con
`_recorrer_directo()`, bypaseando la política de fijación de RAIZ. Los tests que SÍ pasan
por la CLI pública son los que prueban esa misma política de fijación y la validación de
argumentos.
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
DUENO_ORIGINAL = "fruiz"


def test_el_guion_existe():
    assert SCRIPT.is_file()


def _correr(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["python3", str(SCRIPT), *args], capture_output=True, text=True, timeout=60)


def _sudo_n_disponible() -> bool:
    return subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=10).returncode == 0


def _raiz_por_defecto_legible() -> bool:
    """True sólo en un host de jax real (hall9000 o equivalente) donde sudo -n puede
    leer JAX_WORKSPACE_DIR de /etc/jax/.env -- falso en un contenedor genérico de CI
    sin ese archivo."""
    r = subprocess.run(
        ["sudo", "-n", "grep", "^JAX_WORKSPACE_DIR=", "/etc/jax/.env"], capture_output=True, timeout=10,
    )
    return r.returncode == 0 and bool(r.stdout.strip())


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


def _recorrer_directo(proyectos: Path, *, accion: str, extra_codigo: str = "") -> dict:
    """pp._recorrer() como root, bypaseando la fijación de RAIZ de la CLI pública."""
    codigo = f"""
import sys, json
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
hook = None
{extra_codigo}
r = pp._recorrer(pp.Path({str(proyectos)!r}), accion={accion!r}, hook_de_prueba=hook)
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


def _limpiar_como_root(ruta: Path) -> None:
    subprocess.run(["sudo", "-n", "rm", "-rf", str(ruta)], capture_output=True)


def _como_fruiz(codigo_python: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sudo", "-n", "-u", "fruiz", "python3", "-c", codigo_python],
        capture_output=True, text=True, timeout=timeout,
    )


def _abrir_travesia_hasta(ruta: Path, tope: Path) -> None:
    """pytest crea /tmp/pytest-of-<usuario>/pytest-N/ en 0700 -- si un test hace
    sudo -u <alguien> contra algo bajo tmp_path, esos ANCESTROS bloquean la travesía."""
    actual = ruta
    while actual != tope and actual != actual.parent:
        os.chmod(actual, os.stat(actual).st_mode | 0o111)
        actual = actual.parent


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


def test_aplicar_rechaza_una_raiz_que_no_es_la_configurada(arbol_temporal):
    """--aplicar contra un árbol temporal SIEMPRE falla y nunca muta nada -- el motivo
    exacto varía con el entorno (si /etc/jax/.env es legible acá, el chequeo de
    cortesía lo agarra con "no es la RAIZ configurada"; si no -- como en un contenedor
    genérico sin ese archivo -- cae al chequeo de instalación del núcleo, que rechaza
    por otro motivo igual de válido). Lo que importa, y lo único que este test exige,
    es que NINGÚN camino termina mutando el árbol."""
    r = _correr("--aplicar", str(arbol_temporal))
    assert r.returncode != 0
    assert pwd.getpwuid((arbol_temporal / "proyectos").stat().st_uid).pw_name != USUARIO_ESPERADO


def test_aplicar_rechaza_explicitamente_por_raiz_no_configurada_cuando_env_es_legible(
        arbol_temporal, _repo_de_prueba_con_head):
    """La versión ESTRECHA del test de arriba: si /etc/jax/.env es legible (hall9000,
    cualquier host de jax real) Y el núcleo instalado es de confianza, el mensaje
    específico tiene que ser el de la RAIZ, no el de instalación -- confirma que el
    chequeo de cortesía realmente compara, no que sólo el núcleo termina rechazando por
    otra causa. Invoca la copia de `_repo_de_prueba_con_head` (no el guion real vía
    `_correr`): ese chequeo de instalación compara contra el HEAD de SU PROPIO repo, y
    el repo real puede tener esta misma ronda sin commitear todavía -- lo que se prueba
    acá es el orden de los chequeos dentro de _cmd_aplicar, no el estado de git del
    checkout real."""
    if not _raiz_por_defecto_legible():
        pytest.skip("/etc/jax/.env no es legible en este entorno (no es un host de jax real)")
    r = subprocess.run(
        ["python3", str(_repo_de_prueba_con_head), "--aplicar", str(arbol_temporal)],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "no es la RAIZ configurada" in (r.stdout + r.stderr)


# --- BLOCK-2 (ronda 3): las tres entradas del núcleo no aceptan argumentos -------------------

@pytest.mark.parametrize("modo,arg", [
    ("--nucleo-privilegiado", "/etc"),
    ("--nucleo-respaldo", "/etc/ssh"),
    ("--nucleo-deshacer", "/var/tmp/x"),
    ("--deshacer", "/var/tmp/y"),
])
def test_nucleo_no_acepta_argumentos_de_ruta(modo, arg, tmp_path):
    """Reproduce exactamente lo pedido: --nucleo-respaldo /etc/ssh, --nucleo-deshacer
    /var/tmp/x -- y de paso --deshacer público, que tampoco acepta RAIZ. rc != 0, nada
    se toca (ni siquiera se llega a intentar resolver PROYECTOS)."""
    marca_antes = tmp_path / "nada-debe-pasar-aqui"
    marca_antes.mkdir()
    st_antes = marca_antes.stat()

    r = _correr(modo, arg)
    assert r.returncode != 0, r.stdout + r.stderr
    assert "no acepta" in (r.stdout + r.stderr)

    st_despues = marca_antes.stat()
    assert st_antes.st_mtime == st_despues.st_mtime
    assert st_antes.st_uid == st_despues.st_uid


def test_modos_repetidos_se_rechazan():
    r = _correr("--aplicar", "--verificar")
    assert r.returncode == 2
    assert "repetidos" in (r.stdout + r.stderr)


# --- Heredados de la ronda 2: instalación (BLOCK-2 original) y respaldo (BLOCK-3a) ----------

def test_verificar_reporta_si_el_nucleo_no_esta_instalado(arbol_temporal, _identidades):
    """Desinstala el núcleo, comprueba el reporte, y lo REINSTALA en el finally -- si no,
    cualquier test que corra después de éste (el orden de pytest no está garantizado en
    general, pero es file-order sin plugins de aleatorización) encontraría el núcleo
    ausente por un efecto colateral de ESTE test, no por su propia configuración."""
    r_estado = subprocess.run(["sudo", "-n", "test", "-f", str(RUTA_INSTALADA)], capture_output=True)
    habia_antes = r_estado.returncode == 0
    r_borrar = subprocess.run(["sudo", "-n", "rm", "-f", str(RUTA_INSTALADA)], capture_output=True)
    if r_borrar.returncode != 0:
        pytest.skip("no se pudo desinstalar el núcleo para probar el caso 'no instalado'")
    try:
        r = _correr("--verificar", str(arbol_temporal))
        assert "no está instalado" in (r.stdout + r.stderr)
    finally:
        if habia_antes:
            subprocess.run(
                ["sudo", "-n", "install", "-o", "root", "-g", "root", "-m", "0755", str(SCRIPT), str(RUTA_INSTALADA)],
                capture_output=True,
            )


def test_aplicar_rechaza_si_el_nucleo_instalado_es_escribible_por_grupo(_repo_de_prueba_con_head, arbol_temporal):
    """RAIZ explícita (arbol_temporal), no la resuelta por defecto -- en un entorno sin
    /etc/jax/.env (un contenedor de CI genérico) _resolver_raiz(None) fallaría ANTES de
    llegar siquiera al chequeo de instalación que este test quiere ejercitar."""
    r_mod = subprocess.run(["sudo", "-n", "chmod", "0775", str(RUTA_INSTALADA)], capture_output=True)
    if r_mod.returncode != 0:
        pytest.skip("no se pudo aflojar el modo del núcleo instalado para esta prueba")
    try:
        r = _correr("--aplicar", str(arbol_temporal))
        assert r.returncode != 0
        assert "escribible por grupo" in (r.stdout + r.stderr)
    finally:
        subprocess.run(["sudo", "-n", "chmod", "0755", str(RUTA_INSTALADA)], capture_output=True)


def test_inyeccion_de_codigo_en_el_directorio_no_se_ejecuta_como_root(tmp_path):
    """BLOCK-2 (ronda 2), la prueba directa del defecto reproducido por el auditor: SIN
    -I, un json.py de mentira en el directorio del script corre como root; CON -I (lo
    que --aplicar/--nucleo-* usan siempre), no. Sigue cerrado por construcción."""
    if not _sudo_n_disponible():
        pytest.skip("sudo -n no disponible")
    marca = tmp_path / "marca-uid-0"
    d = tmp_path / "dir_con_json_falso"
    d.mkdir()
    (d / "json.py").write_text(f"open({str(marca)!r}, 'w').write('PWNED\\n')\n")
    (d / "victima.py").write_text("import json\n")

    try:
        subprocess.run(["sudo", "-n", "python3", str(d / "victima.py")], capture_output=True)
        contaminado_sin_I = marca.exists()
        marca.unlink(missing_ok=True)

        subprocess.run(["sudo", "-n", "python3", "-I", str(d / "victima.py")], capture_output=True)
        contaminado_con_I = marca.exists()

        assert contaminado_sin_I, "el ataque no reprodujo el defecto -- este test no prueba nada sin esto"
        assert not contaminado_con_I, "-I no evitó la inyección -- BLOCK-2 no está resuelto"
    finally:
        # `sudo -n python3 ... json.py` deja __pycache__/*.pyc dueño de root -- pytest
        # (sin privilegio) no puede limpiar tmp_path solo al terminar.
        subprocess.run(["sudo", "-n", "rm", "-rf", str(d)], capture_output=True)


def test_respaldo_no_depende_del_home_de_fruiz():
    """El fixture crea fruiz con --no-create-home; si el respaldo dependiera de
    pwd.getpwnam('fruiz').pw_dir, esto fallaría en cualquier runner limpio (BLOCK-3a)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("permisos_proyectos", SCRIPT)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    assert str(pp.RUTA_RESPALDOS) == "/var/backups/jax-permisos"
    assert "home" not in str(pp.RUTA_RESPALDOS).lower()


# --- m2 (ronda 3): instalación -- sha256 contra HEAD commiteado, cadena por lstat ------------

@pytest.fixture()
def _repo_de_prueba_con_head(tmp_path, _identidades):
    """Un checkout de git PROPIO del test, con ops/permisos_proyectos.py commiteado en
    HEAD -- así se puede probar el camino FELIZ de _verificar_instalacion() (sha256
    instalado == sha256 de HEAD) sin depender de que el trabajo de esta ronda ya esté
    commiteado en el repo real (no lo está -- sigue en el índice)."""
    if not _sudo_n_disponible():
        pytest.skip("sudo -n no disponible")
    repo = tmp_path / "repo-de-prueba"
    (repo / "ops").mkdir(parents=True)
    contenido = SCRIPT.read_bytes()
    copia = repo / "ops" / "permisos_proyectos.py"
    copia.write_bytes(contenido)
    copia.chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "ops/permisos_proyectos.py"],
                    cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "test"],
                    cwd=repo, check=True)
    _abrir_travesia_hasta(repo, Path("/tmp"))
    r_instalar = subprocess.run(
        ["sudo", "-n", "install", "-o", "root", "-g", "root", "-m", "0755", str(copia), str(RUTA_INSTALADA)],
        capture_output=True, text=True,
    )
    assert r_instalar.returncode == 0, r_instalar.stderr
    yield copia
    subprocess.run(["sudo", "-n", "rm", "-f", str(RUTA_INSTALADA)], capture_output=True)


def test_sha256_del_head_committeado_coincide_con_lo_instalado(_repo_de_prueba_con_head):
    import hashlib
    esperado = hashlib.sha256(_repo_de_prueba_con_head.read_bytes()).hexdigest()
    r = subprocess.run(
        ["python3", "-c", f"""
import sys
sys.path.insert(0, {str(_repo_de_prueba_con_head.parent)!r})
import permisos_proyectos as pp
sha, motivo = pp._sha256_del_head_committeado()
print(sha)
"""],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == esperado


def test_verificar_instalacion_pasa_con_head_coincidente(_repo_de_prueba_con_head):
    r = subprocess.run(
        ["python3", str(_repo_de_prueba_con_head), "--verificar", "/tmp"],
        capture_output=True, text=True,
    )
    assert "núcleo privilegiado NO instalado" not in r.stdout


def test_cadena_de_instalacion_rechaza_un_ancestro_symlink(tmp_path, _identidades):
    real = tmp_path / "real" / "sbin"
    real.mkdir(parents=True)
    subprocess.run(["sudo", "-n", "chown", "-R", "root:root", str(tmp_path / "real")], check=True)
    subprocess.run(["sudo", "-n", "chmod", "755", str(tmp_path / "real"), str(real)], check=True)
    enlace = tmp_path / "enlace_sbin"
    enlace.symlink_to(real)
    nucleo_falso = real / "nucleo_falso"
    subprocess.run(["sudo", "-n", "touch", str(nucleo_falso)], check=True)
    subprocess.run(["sudo", "-n", "chown", "root:root", str(nucleo_falso)], check=True)
    subprocess.run(["sudo", "-n", "chmod", "755", str(nucleo_falso)], check=True)

    try:
        r = subprocess.run(["python3", "-c", f"""
import sys
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
from pathlib import Path
motivo = pp._cadena_es_de_root_sin_escritura_de_grupo_u_otros(Path({str(enlace / "nucleo_falso")!r}))
assert motivo is not None and "symlink" in motivo, motivo
print("OK")
"""], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "OK" in r.stdout
    finally:
        # tmp_path/real quedó root:root -- pytest (sin privilegio) no puede limpiarlo solo.
        subprocess.run(["sudo", "-n", "rm", "-rf", str(tmp_path / "real")], capture_output=True)


# --- m1: sin sudo, falla cerrado en vez de adivinar ------------------------------------------

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


# --- m4: exclusión sólo en profundidad 2 (proyectos/<p>/.claude-flow) -----------------------

def test_exclusion_solo_en_primer_nivel_de_cada_proyecto(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    en_la_raiz = proyectos / ".claude-flow"
    en_la_raiz.mkdir()
    primer_nivel = proyectos / "un-proyecto" / ".claude-flow"
    primer_nivel.mkdir()
    mas_profundo = proyectos / "un-proyecto" / "sub" / ".claude-flow"
    mas_profundo.mkdir()

    datos = _recorrer_directo(proyectos, accion="aplicar")

    assert any(e.endswith("/un-proyecto/.claude-flow") for e in datos["excluidos"]), datos["excluidos"]
    assert not any(e.endswith("/proyectos/.claude-flow") and "un-proyecto" not in e for e in datos["excluidos"])
    assert not any(e.endswith("/sub/.claude-flow") for e in datos["excluidos"])

    # Los que NO están en la lista de exclusión SÍ se mutaron (dueño jaxsvc).
    assert en_la_raiz.stat().st_uid == pwd.getpwnam(USUARIO_ESPERADO).pw_uid
    assert mas_profundo.stat().st_uid == pwd.getpwnam(USUARIO_ESPERADO).pw_uid
    # El del primer nivel de un proyecto, NO se tocó.
    assert primer_nivel.stat().st_uid != pwd.getpwnam(USUARIO_ESPERADO).pw_uid


# --- MAJOR-1 (ronda 3): validación forense del respaldo --------------------------------------

def test_contar_objetos_reales_coincide_con_el_respaldo(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, accion="aplicar")

    r = subprocess.run(["sudo", "-n", "python3", "-c", f"""
import sys, subprocess
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
proyectos = pp.Path({str(proyectos)!r})
n_real = pp._contar_objetos_reales(proyectos)
out = subprocess.run(["getfacl", "-R", "-p", str(proyectos)], capture_output=True, text=True).stdout
n_respaldo = len(pp._parsear_respaldo(out))
assert n_real == n_respaldo, (n_real, n_respaldo)
print("OK", n_real)
"""], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


def test_getfacl_a_dev_full_da_rc_cero_pero_esta_vacio(tmp_path):
    """La razón de fondo de MAJOR-1: no confiar en el returncode. Contra un árbol propio
    y legible (no /tmp entero, que puede tener archivos ajenos ilegibles y fallar por
    OTRO motivo, enmascarando lo que este test quiere probar)."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f.txt").write_text("x")
    r = subprocess.run(["sh", "-c", f"getfacl -R -p {tmp_path}/a > /dev/full 2>/dev/null; echo $?"],
                        capture_output=True, text=True)
    assert r.stdout.strip() == "0", "si esto deja de dar 0, el hallazgo original ya no aplica -- revisar"


def test_desescapa_octales_y_preserva_espacio_final():
    import importlib.util
    spec = importlib.util.spec_from_file_location("permisos_proyectos", SCRIPT)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    assert pp._desescapar_getfacl("con\\\\backslash.txt") == "con\\backslash.txt"
    assert pp._desescapar_getfacl("con\\012newline") == "con\nnewline"
    texto = "# file: /x/nombre con espacio final \n# owner: fruiz\n# group: fruiz\nuser::rw-\n"
    rutas = pp._parsear_respaldo(texto)
    assert rutas == ["/x/nombre con espacio final "], rutas


# --- BLOCK-1 (ronda 3): --deshacer es determinista, nunca lee un respaldo -------------------

def test_deshacer_vuelve_exactamente_al_estado_medido_en_produccion(arbol_temporal, _identidades):
    """Compara stat+getfacl del árbol deshecho contra el estado REAL medido en hall9000
    el 2026-09-25 (fuera de .claude-flow): dirs fruiz:fruiz 0775 sin ACL, archivos
    fruiz:fruiz 0664 sin ACL."""
    proyectos = arbol_temporal / "proyectos"
    aplicado = _recorrer_directo(proyectos, accion="aplicar")
    assert not aplicado["no_cumple"]

    deshecho = _recorrer_directo(proyectos, accion="deshacer")
    assert not deshecho["hardlinks_rechazados"]
    assert not deshecho["symlinks_saltados"]

    for ruta_dir in [proyectos, proyectos / "un-proyecto", proyectos / "un-proyecto" / "sub"]:
        st = ruta_dir.stat()
        assert pwd.getpwuid(st.st_uid).pw_name == DUENO_ORIGINAL, ruta_dir
        assert grp.getgrgid(st.st_gid).gr_name == GRUPO_ESPERADO, ruta_dir
        assert oct(st.st_mode & 0o7777) == "0o775", ruta_dir
        acl = subprocess.run(["getfacl", "-p", str(ruta_dir)], capture_output=True, text=True, check=True).stdout
        lineas = [l for l in acl.splitlines() if l.strip() and not l.startswith("#")]
        assert lineas == ["user::rwx", "group::rwx", "other::r-x"], (ruta_dir, acl)

    archivo = proyectos / "un-proyecto" / "archivo.txt"
    st = archivo.stat()
    assert pwd.getpwuid(st.st_uid).pw_name == DUENO_ORIGINAL
    assert grp.getgrgid(st.st_gid).gr_name == GRUPO_ESPERADO
    assert oct(st.st_mode & 0o7777) == "0o664"
    acl = subprocess.run(["getfacl", "-p", str(archivo)], capture_output=True, text=True, check=True).stdout
    lineas = [l for l in acl.splitlines() if l.strip() and not l.startswith("#")]
    assert lineas == ["user::rw-", "group::rw-", "other::r--"], (archivo, acl)


def test_deshacer_con_enlace_plantado_no_lo_toca_y_reporta(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, accion="aplicar")

    objetivo = proyectos / "un-proyecto" / "objetivo_real"
    enlace = proyectos / "un-proyecto" / "enlace"
    r_crear = _como_fruiz(f"""
import os
os.mkdir({str(objetivo)!r})
os.symlink({str(objetivo)!r}, {str(enlace)!r})
""")
    assert r_crear.returncode == 0, r_crear.stdout + r_crear.stderr

    deshecho = _recorrer_directo(proyectos, accion="deshacer")
    assert any(s.endswith("/enlace") for s in deshecho["symlinks_saltados"]), deshecho["symlinks_saltados"]
    # El objetivo real SÍ se deshace normalmente (es un directorio real, no el symlink).
    st = objetivo.stat()
    assert pwd.getpwuid(st.st_uid).pw_name == DUENO_ORIGINAL


def test_deshacer_con_nombre_espacio_final_lo_deshace_bien(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    con_espacio = proyectos / "un-proyecto" / "nombre con espacio final "
    con_espacio.write_text("x")
    _recorrer_directo(proyectos, accion="aplicar")
    assert con_espacio.exists(), "el nombre con espacio final tiene que seguir existiendo tal cual"

    _recorrer_directo(proyectos, accion="deshacer")
    assert con_espacio.exists()
    st = con_espacio.stat()
    assert pwd.getpwuid(st.st_uid).pw_name == DUENO_ORIGINAL
    assert oct(st.st_mode & 0o7777) == "0o664"


def test_deshacer_con_uid_sin_nombre_no_crashea(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    raro = proyectos / "un-proyecto" / "uid_raro.txt"
    r_crear = subprocess.run(["sudo", "-n", "touch", str(raro)], capture_output=True, text=True)
    assert r_crear.returncode == 0, r_crear.stderr
    r_chown = subprocess.run(["sudo", "-n", "chown", "54321:54321", str(raro)], capture_output=True, text=True)
    assert r_chown.returncode == 0, r_chown.stderr

    datos = _recorrer_directo(proyectos, accion="deshacer")
    assert not datos["hardlinks_rechazados"]
    st = raro.stat()
    assert pwd.getpwuid(st.st_uid).pw_name == DUENO_ORIGINAL


def test_deshacer_con_hardlink_no_lo_muta_y_reporta(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, accion="aplicar")
    original = proyectos / "un-proyecto" / "archivo.txt"
    enlazado = proyectos / "un-proyecto" / "enlazado.txt"
    r_link = _como_fruiz(f"import os; os.link({str(original)!r}, {str(enlazado)!r})")
    if r_link.returncode != 0:
        r_link = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "python3", "-c",
                                  f"import os; os.link({str(original)!r}, {str(enlazado)!r})"],
                                 capture_output=True, text=True)
    assert r_link.returncode == 0, r_link.stdout + r_link.stderr
    dueno_antes = original.stat().st_uid

    datos = _recorrer_directo(proyectos, accion="deshacer")
    assert datos["hardlinks_rechazados"]
    assert original.stat().st_uid == dueno_antes


# --- Mecánica general (heredada, adaptada a accion=) ----------------------------------------

def test_aplicar_deja_el_arbol_con_dueno_jaxsvc_grupo_fruiz_y_herencia_pese_al_umask(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    datos = _recorrer_directo(proyectos, accion="aplicar")
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

    datos2 = _recorrer_directo(proyectos, accion="aplicar")
    assert not datos2["no_cumple"]


def test_verificar_detecta_un_directorio_sin_acl_por_defecto(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, accion="aplicar")
    sub = proyectos / "un-proyecto" / "sub"
    subprocess.run(["sudo", "-n", "setfacl", "-k", str(sub)], check=True)
    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 1
    assert str(sub) in r.stdout


def test_verificar_detecta_y_aplicar_quita_bits_espurios_de_un_directorio(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, accion="aplicar")

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

    datos = _recorrer_directo(proyectos, accion="aplicar")
    assert str(contaminado) in datos["bits_espurios_quitados"]

    st = contaminado.stat()
    assert not (st.st_mode & 0o4000)
    assert not (st.st_mode & 0o1000)
    assert st.st_mode & 0o2000


def test_archivo_con_bit_especial_se_detecta_y_se_limpia(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, accion="aplicar")
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

    datos = _recorrer_directo(proyectos, accion="aplicar")
    assert str(archivo) in datos["bits_espurios_quitados"]
    st = archivo.stat()
    assert not (st.st_mode & 0o7000)


def test_hardlink_se_rechaza_y_no_se_muta(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, accion="aplicar")

    original = proyectos / "un-proyecto" / "archivo.txt"
    enlazado = proyectos / "un-proyecto" / "enlazado.txt"
    r_link = _como_fruiz(f"import os; os.link({str(original)!r}, {str(enlazado)!r})")
    assert r_link.returncode == 0, r_link.stdout + r_link.stderr
    dueno_antes = original.stat().st_uid

    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 1
    assert "nlink=2" in r_verif.stdout

    datos = _recorrer_directo(proyectos, accion="aplicar")
    assert datos["hardlinks_rechazados"]
    assert original.stat().st_uid == dueno_antes


def test_verificar_no_crashea_con_archivo_0600_de_jaxsvc(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, accion="aplicar")

    codigo = f"""
import tempfile, os
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


def test_verificar_detecta_dueno_incorrecto_aunque_el_grupo_este_bien(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, accion="aplicar")
    archivo = proyectos / "un-proyecto" / "archivo.txt"
    r = subprocess.run(["sudo", "-n", "chown", "fruiz", str(archivo)], capture_output=True)
    assert r.returncode == 0, r.stderr
    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 1
    assert f"dueño=fruiz, esperado={USUARIO_ESPERADO}" in r_verif.stdout


def test_fifo_no_cuelga_el_nucleo(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    fifo = proyectos / "un-proyecto" / "unfifo"
    os.mkfifo(fifo)
    try:
        datos = _recorrer_directo(proyectos, accion="aplicar")
        assert not datos["no_cumple"], datos["no_cumple"]
    finally:
        subprocess.run(["sudo", "-n", "rm", "-f", str(fifo)], capture_output=True)


def test_verificar_con_fifo_no_cuelga_ni_crashea(arbol_temporal):
    fifo = arbol_temporal / "proyectos" / "un-proyecto" / "unfifo2"
    os.mkfifo(fifo)
    try:
        r = _correr("--verificar", str(arbol_temporal))
        assert "Traceback" not in (r.stdout + r.stderr)
    finally:
        fifo.unlink(missing_ok=True)


def test_permiso_efectivo_calcula_interseccion_no_el_texto_pedido():
    import importlib.util
    spec = importlib.util.spec_from_file_location("permisos_proyectos", SCRIPT)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    texto = "# file: x\nuser::rw-\ngroup::---\ngroup:fruiz:rwx\t#effective:---\nmask::---\nother::---\n"
    assert pp._permiso_efectivo(texto, default=False, tipo="group", calificador="fruiz") == 0


def test_verificar_detecta_permiso_efectivo_recortado_por_la_mascara(arbol_temporal, _identidades):
    """MAJOR-3 (ronda 2): el test de B1 no se saltea nunca -- se aplica por el camino de
    mecánica directamente (--aplicar público está fijado a la RAIZ configurada desde
    BLOCK-2, así que no serviría acá; eso ya lo prueba
    test_aplicar_rechaza_una_raiz_que_no_es_la_configurada, no hace falta repetirlo)."""
    proyectos = arbol_temporal / "proyectos"
    aplicado = _recorrer_directo(proyectos, accion="aplicar")
    assert not aplicado["no_cumple"]

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

    _abrir_travesia_hasta(raiz, Path("/tmp"))
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
        datos = _recorrer_directo(proyectos, accion="aplicar", extra_codigo=extra)
        assert any(s.endswith("/victima") for s in datos["symlinks_saltados"]), datos["symlinks_saltados"]

        st = objetivo.stat()
        assert st.st_uid == 0
        assert oct(st.st_mode & 0o777) == "0o700"
    finally:
        _limpiar_como_root(raiz)


def test_aplicar_no_interrumpe_un_lector_escritor_concurrente(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    datos_iniciales = _recorrer_directo(proyectos, accion="aplicar")
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
        datos = _recorrer_directo(proyectos, accion="aplicar")
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
    assert contenido == esperado


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
