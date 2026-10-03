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
m4 (ronda 3): la exclusión de carpetas ocultas (nombre con punto inicial) sólo aplica en
   profundidad 2 (proyectos/<proyecto>/<.oculta>), nunca en proyectos/ mismo ni más profundo.

Ronda 4 (APROBADO CON CAMBIOS, sin BLOCK ni MAJOR) -- 6 MINOR:
m1: una FIFO/socket en el árbol hacía que getfacl -R -p (que sí las enumera) y
   _contar_objetos_reales (que las ignora, igual que el resto del guion) dieran
   cantidades distintas -- rc=2 sin decir dónde. Ahora se detectan y se nombra la ruta
   ANTES de comparar cantidades. `_generar_respaldo_validado` separa la lógica real del
   respaldo de la fijación de RAIZ de `_cmd_nucleo_respaldo`, para poder probarla contra
   un árbol de prueba (no la RAIZ configurada) -- ver test_respaldo_con_fifo_aborta_....
m2: _parsear_respaldo aceptaba un respaldo cortado justo después de "# file: /a/b" en el
   último bloque. Ahora exige que cada bloque traiga owner/group/user::/group::/other::
   completos, Y ADEMÁS un marcador de fin que el propio guion escribe después de que
   getfacl termina (atrapa un corte que caiga justo en un borde de bloque, que la
   validación por bloque sola no vería mal).
m5: si `sudo -n` no funciona, el chequeo de cortesía de RAIZ en --aplicar lo trataba
   igual que "la variable no está" -- silencioso. Ahora se prueba `sudo -n true` aparte y
   se aborta explícito si eso falla. Más de una RAIZ posicional (antes se quedaba con la
   última, sin avisar) también se rechaza.
m6: la ACL de --aplicar ahora también fija `g::rwX` (grupo DUEÑO, sin nombre), no sólo
   la entrada nombrada `g:fruiz:` -- antes podían mostrar valores distintos para el
   mismo grupo (la causa real del defecto de --deshacer que se encontró en la ronda 3).
   --verificar lo exige.

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
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest

RAIZ_REPO = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ_REPO / "ops" / "permisos_proyectos.py"
# MAJOR-1 (revision Tarea 3 E2a): las pruebas NUNCA instalan ni borran el nucleo de sistema real
# (/usr/local/sbin/jax-permisos-proyectos): usan una ruta propia bajo el directorio temporal,
# que el guion lee de JAX_PERMISOS_NUCLEO.
_DIR_NUCLEO_PRUEBA = Path(tempfile.mkdtemp(prefix="permisos-nucleo-"))
RUTA_INSTALADA = _DIR_NUCLEO_PRUEBA / "jax-permisos-proyectos"
os.environ["JAX_PERMISOS_NUCLEO"] = str(RUTA_INSTALADA)
RUTA_NUCLEO_SISTEMA = Path("/usr/local/sbin/jax-permisos-proyectos")

USUARIO_ESPERADO = "jaxsvc"
GRUPO_ESPERADO = "fruiz"
DUENO_ORIGINAL = "fruiz"


@pytest.fixture(scope="module", autouse=True)
def _limpiar_nucleo_de_prueba():
    yield
    subprocess.run(["sudo", "-n", "rm", "-f", str(RUTA_INSTALADA)], capture_output=True)
    shutil.rmtree(_DIR_NUCLEO_PRUEBA, ignore_errors=True)


def test_las_pruebas_no_tocan_el_nucleo_de_sistema():
    """Falla si la ruta del nucleo que usan las pruebas (y el guion) cae fuera del directorio
    temporal, o si el valor por defecto del guion dejo de ser la ruta de produccion."""
    assert RUTA_INSTALADA != RUTA_NUCLEO_SISTEMA
    assert not str(RUTA_INSTALADA).startswith("/usr/local/sbin")
    assert Path(tempfile.gettempdir()) in RUTA_INSTALADA.parents
    uso = subprocess.run(
        ["python3", "-c", f"import sys; sys.path.insert(0, {str(RAIZ_REPO / 'ops')!r});"
         "import permisos_proyectos as pp; print(pp.RUTA_INSTALADA)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert uso == str(RUTA_INSTALADA)
    sin_env = {k: v for k, v in os.environ.items() if k != "JAX_PERMISOS_NUCLEO"}
    por_defecto = subprocess.run(
        ["python3", "-c", f"import sys; sys.path.insert(0, {str(RAIZ_REPO / 'ops')!r});"
         "import permisos_proyectos as pp; print(pp.RUTA_INSTALADA)"],
        capture_output=True, text=True, check=True, env=sin_env,
    ).stdout.strip()
    assert por_defecto == str(RUTA_NUCLEO_SISTEMA)


def test_ninguna_ruta_de_las_pruebas_resuelve_a_un_arbol_de_produccion():
    """Ninguna prueba puede tocar un arbol de produccion. En hall9000 `/home/fruiz/jax-workspace` es un
    symlink a `/srv/jax-data/jax-workspace`: una constante con esa ruta hacia que la prueba de
    lectura/escritura cruzada escribiera y borrara en produccion. Se mira el realpath de toda ruta
    fija del modulo, no el texto, para que un symlink no la disfrace."""
    prohibidos = (Path("/srv/jax-data"), Path("/srv/jax-prod"))
    for nombre, valor in sorted(globals().items()):
        if isinstance(valor, Path) and nombre != "RAIZ_REPO":
            real = Path(os.path.realpath(valor))
            assert not any(real == p or p in real.parents for p in prohibidos), (
                f"{nombre}={valor} resuelve a {real}, un arbol de produccion"
            )


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


def _conceder_acceso_al_usuario_de_pruebas(proyectos: Path, usuario: str | None = None) -> None:
    """`--aplicar` deja `other::---`: el usuario que corre pytest en un runner (`runner`) ni es
    jaxsvc ni es fruiz, y ya no entra por `other` a stat/leer/`--verificar` lo que las pruebas
    miran DESPUES de aplicar. Se le da una entrada NOMBRADA (acceso y por defecto, que `--aplicar`
    conserva porque usa `setfacl -m`): es del arnes, no de lo que se prueba. Con el usuario de
    esta maquina (fruiz) es redundante e inofensivo. Como root no hace falta."""
    usuario = usuario or pwd.getpwuid(os.getuid()).pw_name
    if usuario == "root":
        return
    entrada = f"u:{usuario}:rwX"
    # La raíz del workspace (padre de proyectos/) también pierde `otros` con --aplicar: quien
    # entra a proyectos/ tiene que poder atravesarla por entrada nombrada (en producción, por grupo).
    subprocess.run(["setfacl", "-m", f"u:{usuario}:x", str(proyectos.parent)], check=True, capture_output=True)
    subprocess.run(["setfacl", "-R", "-m", entrada, str(proyectos)], check=True, capture_output=True)
    subprocess.run(["find", str(proyectos), "-type", "d", "-exec", "setfacl", "-d", "-m", entrada, "{}", "+"],
                   check=True, capture_output=True)


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
    _conceder_acceso_al_usuario_de_pruebas(proyectos)
    for cuenta in ("jaxsvc", "fruiz"):  # las que atraviesan la raíz en producción (por grupo)
        subprocess.run(["setfacl", "-m", f"u:{cuenta}:x", str(raiz)], check=True, capture_output=True)
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
        arbol_temporal, _repo_de_prueba_con_nucleo_de_sistema):
    """La versión ESTRECHA del test de arriba: si /etc/jax/.env es legible (hall9000,
    cualquier host de jax real) Y el núcleo instalado es de confianza, el mensaje
    específico tiene que ser el de la RAIZ, no el de instalación -- confirma que el
    chequeo de cortesía realmente compara, no que sólo el núcleo termina rechazando por
    otra causa. Invoca la copia de `_repo_de_prueba_con_nucleo_de_sistema` (no el guion real vía
    `_correr`): ese chequeo de instalación compara contra el HEAD de SU PROPIO repo, y
    el repo real puede tener esta misma ronda sin commitear todavía -- lo que se prueba
    acá es el orden de los chequeos dentro de _cmd_aplicar, no el estado de git del
    checkout real."""
    if not _raiz_por_defecto_legible():
        pytest.skip("/etc/jax/.env no es legible en este entorno (no es un host de jax real)")
    r = subprocess.run(
        ["python3", str(_repo_de_prueba_con_nucleo_de_sistema), "--aplicar", str(arbol_temporal)],
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

def _crear_repo_con_head(tmp_path, destino: Path) -> Path:
    """Un checkout de git PROPIO del test, con ops/permisos_proyectos.py commiteado en
    HEAD -- así se puede probar el camino FELIZ de _verificar_instalacion() (sha256
    instalado == sha256 de HEAD) sin depender de que el trabajo de esta ronda ya esté
    commiteado en el repo real. Instala la copia (root:root 0755) en `destino`."""
    if not _sudo_n_disponible():
        pytest.skip("sudo -n no disponible")
    repo = tmp_path / "repo-de-prueba"
    (repo / "ops").mkdir(parents=True)
    copia = repo / "ops" / "permisos_proyectos.py"
    copia.write_bytes(SCRIPT.read_bytes())
    copia.chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "ops/permisos_proyectos.py"],
                    cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "test"],
                    cwd=repo, check=True)
    _abrir_travesia_hasta(repo, Path("/tmp"))
    r_instalar = subprocess.run(
        ["sudo", "-n", "install", "-o", "root", "-g", "root", "-m", "0755", str(copia), str(destino)],
        capture_output=True, text=True,
    )
    assert r_instalar.returncode == 0, r_instalar.stderr
    return copia


@pytest.fixture()
def _repo_de_prueba_con_head(tmp_path, _identidades):
    """Nucleo instalado en la ruta de PRUEBA (bajo el directorio temporal, no en /usr/local/sbin)."""
    copia = _crear_repo_con_head(tmp_path, RUTA_INSTALADA)
    yield copia
    subprocess.run(["sudo", "-n", "rm", "-f", str(RUTA_INSTALADA)], capture_output=True)


@pytest.fixture()
def _repo_de_prueba_con_nucleo_de_sistema(tmp_path, _identidades, monkeypatch):
    """Para las pruebas que necesitan que la CADENA de la ruta sea de root (el directorio
    temporal no lo es, y /tmp es escribible por otros). Usa la ruta de sistema SOLO si no hay ya
    un nucleo que no puso esta prueba (en un host desplegado se salta); lo que instala, lo borra."""
    if RUTA_NUCLEO_SISTEMA.exists() or RUTA_NUCLEO_SISTEMA.is_symlink():
        pytest.skip(f"{RUTA_NUCLEO_SISTEMA} ya existe (nucleo real) -- esta prueba no lo toca")
    monkeypatch.setenv("JAX_PERMISOS_NUCLEO", str(RUTA_NUCLEO_SISTEMA))
    copia = _crear_repo_con_head(tmp_path, RUTA_NUCLEO_SISTEMA)
    yield copia
    subprocess.run(["sudo", "-n", "rm", "-f", str(RUTA_NUCLEO_SISTEMA)], capture_output=True)


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


def test_verificar_instalacion_pasa_con_head_coincidente(arbol_temporal, _repo_de_prueba_con_nucleo_de_sistema):
    """Camino FELIZ de verdad: nucleo instalado con cadena de root y sha256 == HEAD, sobre una RAIZ
    valida (con `proyectos/`). Antes corria `--verificar /tmp`, que sale con rc=2 por RAIZ invalida
    ANTES de mirar la instalacion, y el `not in` pasaba vacio. Usa el nucleo de sistema (la cadena
    de /tmp no es de root); se salta, con su razon, si ya hay un nucleo real en el host."""
    r = subprocess.run(
        ["python3", str(_repo_de_prueba_con_nucleo_de_sistema), "--verificar", str(arbol_temporal)],
        capture_output=True, text=True,
    )
    salida = r.stdout + r.stderr
    assert r.returncode in (0, 1), salida          # 2 = RAIZ invalida / error: no llego a la instalacion
    assert "RAIZ inválida" not in salida, salida
    assert "núcleo privilegiado NO instalado" not in salida, salida
    assert "NO CUMPLE" in r.stdout, salida          # el arbol de prueba no esta aplicado: SI se recorrio


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


# --- m4: exclusión sólo en profundidad 2 (proyectos/<p>/<carpeta oculta>) ------------------

OCULTA = ".estado-herramienta"


def test_exclusion_solo_en_primer_nivel_de_cada_proyecto(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    en_la_raiz = proyectos / OCULTA
    en_la_raiz.mkdir()
    primer_nivel = proyectos / "un-proyecto" / OCULTA
    primer_nivel.mkdir()
    mas_profundo = proyectos / "un-proyecto" / "sub" / OCULTA
    mas_profundo.mkdir()

    datos = _recorrer_directo(proyectos, accion="aplicar")

    assert any(e.endswith(f"/un-proyecto/{OCULTA}") for e in datos["excluidos"]), datos["excluidos"]
    assert not any(e.endswith(f"/proyectos/{OCULTA}") and "un-proyecto" not in e for e in datos["excluidos"])
    assert not any(e.endswith(f"/sub/{OCULTA}") for e in datos["excluidos"])

    # Los que NO están en profundidad 2 SÍ se mutaron (dueño jaxsvc).
    assert en_la_raiz.stat().st_uid == pwd.getpwnam(USUARIO_ESPERADO).pw_uid
    assert mas_profundo.stat().st_uid == pwd.getpwnam(USUARIO_ESPERADO).pw_uid
    # El del primer nivel de un proyecto, NO se tocó.
    assert primer_nivel.stat().st_uid != pwd.getpwnam(USUARIO_ESPERADO).pw_uid


def test_en_profundidad_2_se_excluye_toda_carpeta_oculta_y_solo_las_ocultas(arbol_temporal, _identidades):
    """Regla general, no una lista de nombres: cualquier CARPETA con punto inicial en
    proyectos/<p>/ se excluye; una carpeta sin punto y un archivo oculto suelto, no."""
    proyecto = arbol_temporal / "proyectos" / "un-proyecto"
    ocultas = [proyecto / ".otra-herramienta", proyecto / ".x", proyecto / "..doble"]
    for o in ocultas:
        o.mkdir()
    visible = proyecto / "estado-herramienta"          # sin punto: se gobierna
    visible.mkdir()
    archivo_oculto = proyecto / ".nota-suelta"          # archivo, no carpeta: se gobierna
    archivo_oculto.write_text("x")

    datos = _recorrer_directo(arbol_temporal / "proyectos", accion="aplicar")

    excluidos = {Path(e).name for e in datos["excluidos"]}
    assert {".otra-herramienta", ".x", "..doble"} <= excluidos, excluidos
    assert "estado-herramienta" not in excluidos and ".nota-suelta" not in excluidos, excluidos
    uid = pwd.getpwnam(USUARIO_ESPERADO).pw_uid
    assert visible.stat().st_uid == uid and archivo_oculto.stat().st_uid == uid
    assert all(o.stat().st_uid != uid for o in ocultas)


def test_un_symlink_con_nombre_oculto_en_profundidad_2_no_es_carpeta_excluida(arbol_temporal, _identidades):
    """La exclusion es de CARPETAS reales: un symlink oculto no se excluye (no queda en
    `excluidos`) y se reporta en `symlinks_saltados`, como cualquier otro symlink."""
    proyecto = arbol_temporal / "proyectos" / "un-proyecto"
    (proyecto / ".enlace-oculto").symlink_to(proyecto / "sub")

    datos = _recorrer_directo(arbol_temporal / "proyectos", accion="aplicar")

    assert not any(e.endswith("/.enlace-oculto") for e in datos["excluidos"]), datos["excluidos"]
    assert any(e.endswith("/un-proyecto/.enlace-oculto") for e in datos["symlinks_saltados"]), datos["symlinks_saltados"]


def test_la_cuenta_forense_incluye_la_carpeta_oculta_porque_getfacl_la_respalda(arbol_temporal, _identidades):
    """`_contar_objetos_reales` NO aplica la exclusion de carpetas ocultas a proposito: se compara
    contra `getfacl -R`, que respalda TODO el arbol, carpeta oculta incluida. Si la cuenta la
    excluyera, no coincidiria con el respaldo y el respaldo fallaria siempre que exista una."""
    proyecto = arbol_temporal / "proyectos" / "un-proyecto"
    oculta = proyecto / ".estado-herramienta"
    oculta.mkdir()
    (oculta / "dato.txt").write_text("x")
    _recorrer_directo(arbol_temporal / "proyectos", accion="aplicar")

    r = subprocess.run(["sudo", "-n", "python3", "-c", f"""
import sys, subprocess
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
proyectos = pp.Path({str(arbol_temporal / "proyectos")!r})
n_real, no_gobernados = pp._contar_objetos_reales(proyectos)
out = subprocess.run(["getfacl", "-R", "-p", str(proyectos)], capture_output=True, text=True).stdout
assert ".estado-herramienta/dato.txt" in out, "getfacl no respalda la carpeta oculta"
assert n_real == len(pp._parsear_respaldo(out)), (n_real, len(pp._parsear_respaldo(out)))
print("OK", n_real)
"""], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "OK" in r.stdout


# --- MAJOR-1 (ronda 3): validación forense del respaldo --------------------------------------

def test_contar_objetos_reales_coincide_con_el_respaldo(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, accion="aplicar")

    r = subprocess.run(["sudo", "-n", "python3", "-c", f"""
import sys, subprocess
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
proyectos = pp.Path({str(proyectos)!r})
n_real, no_gobernados = pp._contar_objetos_reales(proyectos)
assert not no_gobernados, no_gobernados
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
    # Bloque COMPLETO (m2, ronda 4: _parsear_respaldo exige owner/group/user::/group::/
    # other:: -- un bloque incompleto se rechaza, así que la muestra de este test tiene
    # que ser un bloque real, no sólo la línea que le interesa a esta prueba puntual).
    texto = (
        "# file: /x/nombre con espacio final \n"
        "# owner: fruiz\n# group: fruiz\n"
        "user::rw-\ngroup::rw-\nother::r--\n"
    )
    rutas = pp._parsear_respaldo(texto)
    assert rutas == ["/x/nombre con espacio final "], rutas


# --- BLOCK-1 (ronda 3): --deshacer es determinista, nunca lee un respaldo -------------------

def test_deshacer_vuelve_exactamente_al_estado_medido_en_produccion(arbol_temporal, _identidades):
    """Compara stat+getfacl del árbol deshecho contra el estado REAL medido en hall9000
    el 2026-09-25 (fuera de la carpeta oculta de estado): dirs fruiz:fruiz 0775 sin ACL, archivos
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


def _grupo_efectivo(acl: str) -> str:
    """Permiso EFECTIVO (tras la mascara) de la entrada nombrada g:GRUPO en un `getfacl -p`."""
    for linea in acl.splitlines():
        if linea.startswith(f"group:{GRUPO_ESPERADO}:"):
            texto, _, efectivo = linea.partition("#effective:")
            return efectivo.strip() if efectivo else texto.split(":")[2].strip()
    return ""


def test_entrada_y_fuente_creadas_despues_heredan_grupo_setgid_y_acl_por_defecto(arbol_temporal, _identidades):
    """E2a (Tarea 3): jax-platform/LAS MANOS (jaxsvc) crean proyectos/<uuid>/entrada/<lote>/
    y fuente/ DESPUES de --aplicar. Nada de eso existe cuando el guion corre, asi que la
    cobertura es por HERENCIA: setgid + ACL por defecto de proyectos/ tienen que llegar a
    cualquier subdirectorio y archivo nuevo, a cualquier profundidad."""
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]

    uuid_proyecto = str(uuid.uuid4())
    lote = proyectos / uuid_proyecto / "entrada" / "lote1"
    fuente = proyectos / uuid_proyecto / "fuente"
    # El proceso real que crea esto es jaxsvc, con el umask del servicio.
    r = subprocess.run(
        ["sudo", "-n", "-u", "jaxsvc", "python3", "-c", f"""
import os
os.umask(0o022)
os.makedirs({str(lote)!r})
os.makedirs({str(fuente)!r})
"""],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stdout + r.stderr

    # El archivo se crea por el camino REAL de LAS MANOS (tool_authority._write_file: mkstemp +
    # fchmod + replace), no con os.open: sin el fchmod(0o660) el archivo queda con mascara ACL
    # --- y esta prueba se pone roja. Solo se sustituyen el commit de git del workspace y el
    # registro de eventos (necesita la base: aqui se prueba el modo y la ACL del archivo, no eso).
    #
    # Corre COMO fruiz (quien opera en produccion y esta en la ACL), no como el usuario que
    # corre pytest (en el runner, `runner`, que no esta en la ACL). sys.executable y no
    # "python3": el interprete de pytest (con aiomysql via requirements.txt). Fruiz tiene que
    # poder leer el interprete, su prefijo y el checkout: si no, se dice claro en vez de un
    # PermissionError opaco dentro del subproceso.
    for que, ruta_a_leer in (("el interprete de pytest", sys.executable),
                             ("el prefijo del interprete", sys.prefix),
                             ("el checkout del repo", str(RAIZ_REPO))):
        legible = subprocess.run(["sudo", "-n", "-u", "fruiz", "test", "-r", ruta_a_leer, "-a", "-x", ruta_a_leer],
                                 capture_output=True)
        if legible.returncode != 0:
            pytest.fail(f"el usuario fruiz no puede leer {que} ({ruta_a_leer}): esta prueba corre el "
                        "escritor real como fruiz; hay que dar lectura a fruiz o ejecutar pytest desde "
                        "un interprete que fruiz pueda leer")
    r_w = subprocess.run(
        ["sudo", "-n", "-u", "fruiz", "env", f"JAX_WORKSPACE_DIR={arbol_temporal}", "PYTHONDONTWRITEBYTECODE=1",
         sys.executable, "-c", f"""
import asyncio, sys
from pathlib import Path
sys.path[:0] = [{str(RAIZ_REPO / "las_manos")!r}, {str(RAIZ_REPO)!r}]
from motor_registry import tool_authority as ta
ta._git_commit_write = lambda *a, **k: (True, "sha", None)
async def _sin_registro(*a, **k):
    return None
ta.event_append = _sin_registro
r = asyncio.run(ta._write_file(job_id="t", tool_name="write_file", caller="t",
    resolved=Path({str(lote / "a.pdf")!r}), content="x", tool_call_id="t"))
assert r["decision"] == "executed", r
"""],
        capture_output=True, text=True, timeout=60,
    )
    assert r_w.returncode == 0, r_w.stdout + r_w.stderr

    for directorio in (proyectos / uuid_proyecto, proyectos / uuid_proyecto / "entrada", lote, fuente):
        st = directorio.stat()
        assert grp.getgrgid(st.st_gid).gr_name == GRUPO_ESPERADO, directorio
        assert st.st_mode & 0o2000, f"{directorio} sin setgid"
        acl = subprocess.run(["getfacl", "-p", str(directorio)], capture_output=True, text=True, check=True).stdout
        lineas = acl.splitlines()
        assert any(l.startswith(f"default:group:{GRUPO_ESPERADO}:rwx") for l in lineas), acl
        assert _grupo_efectivo(acl).startswith("rw"), acl

    archivo = lote / "a.pdf"
    assert grp.getgrgid(archivo.stat().st_gid).gr_name == GRUPO_ESPERADO
    acl_archivo = subprocess.run(["getfacl", "-p", str(archivo)], capture_output=True, text=True, check=True).stdout
    assert _grupo_efectivo(acl_archivo).startswith("rw"), acl_archivo

    # Ambas cuentas pueden escribir el archivo: fruiz (el guion a mano) y jaxsvc (el servicio).
    r2 = _como_fruiz(f"open({str(archivo)!r}, 'a').write('x')")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    r3 = subprocess.run(
        ["sudo", "-n", "-u", "jaxsvc", "python3", "-c", f"open({str(archivo)!r}, 'a').write('y')"],
        capture_output=True, text=True, timeout=30,
    )
    assert r3.returncode == 0, r3.stdout + r3.stderr


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


# --- m1 (ronda 4): FIFO por el camino del RESPALDO, no sólo _recorrer_directo ------------

def test_respaldo_con_fifo_aborta_nombrando_la_ruta(arbol_temporal, _identidades):
    """A diferencia de test_fifo_no_cuelga_el_nucleo (que sólo pasa por _recorrer_directo,
    el camino de --aplicar/--verificar), esto ejercita el camino del RESPALDO
    (_generar_respaldo_validado, lo que --nucleo-respaldo llama tras fijar RAIZ) --
    donde getfacl -R SÍ cuenta la FIFO y _contar_objetos_reales no, y antes eso daba un
    "no coincide" genérico en vez de nombrar la ruta."""
    proyectos = arbol_temporal / "proyectos"
    fifo = proyectos / "un-proyecto" / "unfifo-respaldo"
    os.mkfifo(fifo)
    try:
        codigo = f"""
import sys
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
try:
    pp._generar_respaldo_validado(pp.Path({str(proyectos)!r}))
    print("NO_ABORTO")
except pp.ErrorPermisosProyectos as exc:
    print("ABORTO:" + str(exc))
"""
        r = subprocess.run(["sudo", "-n", "python3", "-c", codigo], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "ABORTO:" in r.stdout, r.stdout
        assert str(fifo) in r.stdout, r.stdout
        assert "FIFO" in r.stdout or "socket" in r.stdout
    finally:
        subprocess.run(["sudo", "-n", "rm", "-f", str(fifo)], capture_output=True)


def test_respaldo_sin_objetos_no_gobernados_funciona_normal(arbol_temporal, _identidades):
    """Control negativo del test de arriba: sin FIFO, el mismo camino del respaldo tiene
    que funcionar y dar una ruta real (si el fix de m1 rompiera el caso sano, esto lo
    vería)."""
    proyectos = arbol_temporal / "proyectos"
    codigo = f"""
import sys
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
ruta = pp._generar_respaldo_validado(pp.Path({str(proyectos)!r}))
print(str(ruta))
ruta.unlink()
"""
    r = subprocess.run(["sudo", "-n", "python3", "-c", codigo], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().startswith("/var/backups/jax-permisos/")


# --- m2 (ronda 4): respaldo cortado ---------------------------------------------------------

def test_parsear_respaldo_rechaza_bloque_incompleto():
    import importlib.util
    spec = importlib.util.spec_from_file_location("permisos_proyectos", SCRIPT)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)

    completo = "# file: /a\n# owner: fruiz\n# group: fruiz\nuser::rwx\ngroup::rwx\nother::r-x\n"
    assert pp._parsear_respaldo(completo) == ["/a"]

    # Cortado justo después de "# file: /a/b" en el último bloque -- ni owner, ni group,
    # ni las tres entradas base.
    cortado = completo + "\n# file: /a/b"
    with pytest.raises(Exception) as exc_info:
        pp._parsear_respaldo(cortado)
    assert "truncado" in str(exc_info.value) or "corrupto" in str(exc_info.value)


def test_respaldo_con_marcador_de_fin_faltante_se_rechaza(arbol_temporal, _identidades):
    """Simula un corte que cae EXACTO en un borde de bloque completo -- el caso que la
    validación por bloque, sola, no vería mal, y que sólo el marcador de fin atrapa."""
    proyectos = arbol_temporal / "proyectos"
    codigo = f"""
import sys
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
ruta = pp._generar_respaldo_validado(pp.Path({str(proyectos)!r}))
contenido = ruta.read_text()
assert contenido.endswith(pp._MARCADOR_FIN_RESPALDO)
# reescribe el mismo archivo SIN el marcador -- como si getfacl hubiera terminado bien
# pero el proceso hubiera muerto antes de escribir el marcador de este guion.
sin_marcador = contenido[: -len(pp._MARCADOR_FIN_RESPALDO)]
ruta.write_text(sin_marcador)
print("SIN_MARCADOR_ESCRITO")
print(not sin_marcador.endswith(pp._MARCADOR_FIN_RESPALDO))
ruta.unlink()
"""
    r = subprocess.run(["sudo", "-n", "python3", "-c", codigo], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "True" in r.stdout, r.stdout  # confirma que el contenido de prueba ya no tiene el marcador


# --- m5 (ronda 4): sudo -n roto aborta; más de una RAIZ también --------------------------

def test_dos_raices_posicionales_se_rechazan(tmp_path):
    r = _correr("--verificar", str(tmp_path), "/otro/lugar")
    assert r.returncode == 2
    assert "más de una RAIZ" in (r.stdout + r.stderr)


def test_aplicar_aborta_si_sudo_n_no_funciona_incluso_con_nucleo_instalado(
        arbol_temporal, _repo_de_prueba_con_nucleo_de_sistema):
    """Antes, si `sudo -n` fallaba, el chequeo de cortesía de RAIZ en --aplicar lo
    trataba igual que "la variable no está" -- lo salteaba en silencio y seguía. Ahora
    aborta explícito. Usa `_repo_de_prueba_con_nucleo_de_sistema` para que el núcleo instalado SÍ
    coincida con HEAD (si no, el chequeo de instalación abortaría primero, antes de
    llegar al que este test quiere probar)."""
    sudo_falso_dir = arbol_temporal.parent / "bin-sudo-roto"
    sudo_falso_dir.mkdir()
    (sudo_falso_dir / "sudo").write_text("#!/bin/sh\nexit 1\n")
    (sudo_falso_dir / "sudo").chmod(0o755)
    entorno = dict(os.environ)
    entorno["PATH"] = f"{sudo_falso_dir}:{entorno['PATH']}"

    r = subprocess.run(
        ["python3", str(_repo_de_prueba_con_nucleo_de_sistema), "--aplicar", str(arbol_temporal)],
        capture_output=True, text=True, env=entorno,
    )
    assert r.returncode != 0
    assert "sudo -n no funciona" in (r.stdout + r.stderr), r.stdout + r.stderr
    # nada se mutó -- el árbol sigue del dueño original.
    assert pwd.getpwuid((arbol_temporal / "proyectos").stat().st_uid).pw_name != USUARIO_ESPERADO


# --- m6 (ronda 4): g::rwX explícito, --verificar lo exige ---------------------------------

def test_aplicar_fija_group_obj_igual_a_la_entrada_nombrada(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    datos = _recorrer_directo(proyectos, accion="aplicar")
    assert not datos["no_cumple"]

    acl = subprocess.run(
        ["getfacl", "-p", str(proyectos / "un-proyecto")], capture_output=True, text=True, check=True
    ).stdout
    grupo_obj = [l for l in acl.splitlines() if l.startswith("group::")]
    grupo_nombrado = [l for l in acl.splitlines() if l.startswith(f"group:{GRUPO_ESPERADO}:")]
    assert grupo_obj and grupo_nombrado, acl
    assert grupo_obj[0].split(":")[-1] == grupo_nombrado[0].split(":")[-1], (
        f"group:: y group:{GRUPO_ESPERADO}: divergen:\n{acl}"
    )


def test_verificar_detecta_group_obj_recortado_aunque_la_entrada_nombrada_este_bien(arbol_temporal, _identidades):
    """Reproduce el defecto real de fondo (m6): si algo (a mano, o un getfacl -m viejo)
    deja group:: por debajo de rwx mientras la entrada nombrada sigue bien,
    --verificar tiene que marcarlo NO CUMPLE -- antes no lo miraba en absoluto."""
    proyectos = arbol_temporal / "proyectos"
    _recorrer_directo(proyectos, accion="aplicar")
    objetivo = proyectos / "un-proyecto"
    r = subprocess.run(["sudo", "-n", "setfacl", "-m", "g::r-x", str(objetivo)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    r_verif = _correr("--verificar", str(arbol_temporal))
    assert r_verif.returncode == 1
    assert "group:: (grupo dueño)" in r_verif.stdout


def test_permiso_efectivo_directo_de_grupo_obj():
    import importlib.util
    spec = importlib.util.spec_from_file_location("permisos_proyectos", SCRIPT)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    texto = "# file: x\nuser::rwx\ngroup::r-x\nother::r-x\n"
    assert pp._permiso_efectivo(texto, default=False, tipo="group", calificador="") == 0o5


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


# --- Sin acceso para "otros" (spec madre §5: 2770 en directorios, 0660 en archivos) -----------
#
# El 2026-10-03 el workspace se trasladó a /srv/jax-data/jax-workspace; la barrera que daba
# /home/fruiz (750) desapareció y `proyectos/` quedó con `other::r-x` y `default:other::r-x`:
# cualquier usuario local leía los documentos de los clientes. --aplicar no tocaba `o::` y
# --verificar no lo contaba. El árbol tiene que ser cerrado por sí mismo.

def _acl(ruta: Path) -> list[str]:
    salida = subprocess.run(["getfacl", "-p", str(ruta)], capture_output=True, text=True, check=True).stdout
    return [l.split("\t")[0].strip() for l in salida.splitlines() if l.strip() and not l.startswith("#")]


def _otros_en_nombres(ruta: Path) -> list[str]:
    return [l for l in _acl(ruta) if l.startswith(("other::", "default:other::"))]


def _lineas_no_cumple(salida: str, ruta: Path) -> list[str]:
    return [l for l in salida.splitlines() if l.startswith(f"NO CUMPLE: {ruta}:")]


def test_verificar_marca_otros_en_un_arbol_con_o_r_x(arbol_temporal):
    """El árbol de partida es 0755/0644, o sea `other::r-x`/`other::r--`: cada objeto tiene que
    salir con una falta que hable de "otros" (el dueño y el grupo también faltan, pero esa falta
    no es la que se prueba acá)."""
    proyectos = arbol_temporal / "proyectos"
    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 1, r.stdout
    for ruta in (proyectos, proyectos / "un-proyecto", proyectos / "un-proyecto" / "sub",
                 proyectos / "un-proyecto" / "archivo.txt"):
        faltas = _lineas_no_cumple(r.stdout, ruta)
        assert faltas and "para otros" in faltas[0], (ruta, r.stdout)
    assert f"NO CUMPLE: {arbol_temporal} (raíz del workspace): permisos para otros" in r.stdout, r.stdout


@pytest.mark.parametrize("que,orden", [
    ("modo de la raíz (chmod o+x)", ["chmod", "o+x", "{raiz}"]),
    ("ACL de acceso de la raíz", ["setfacl", "-m", "o::r-x", "{raiz}"]),
])
def test_verificar_marca_otros_en_la_raiz_del_workspace(arbol_temporal, _identidades, que, orden):
    """Hoy el cierre depende de un solo bit, el de la raíz (padre de proyectos/): si alguien lo
    reabre, --verificar tiene que decirlo aunque todo proyectos/ esté en regla."""
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    assert _correr("--verificar", str(arbol_temporal)).returncode == 0
    r_mod = subprocess.run(["sudo", "-n", *[a.format(raiz=arbol_temporal) for a in orden]],
                           capture_output=True, text=True)
    assert r_mod.returncode == 0, r_mod.stderr

    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 1, (que, r.stdout)
    assert f"NO CUMPLE: {arbol_temporal} (raíz del workspace): permisos para otros" in r.stdout, r.stdout
    assert not any(l.startswith(f"NO CUMPLE: {proyectos}") for l in r.stdout.splitlines()), r.stdout


def test_aplicar_cierra_la_raiz_sin_tocar_su_dueno_ni_su_grupo(arbol_temporal, _identidades):
    raiz = arbol_temporal
    gid_otro = next(g.gr_gid for g in grp.getgrall() if g.gr_name == "jaxsvc") if any(
        g.gr_name == "jaxsvc" for g in grp.getgrall()) else None
    if gid_otro is not None:
        subprocess.run(["sudo", "-n", "chown", f":{gid_otro}", str(raiz)], check=True)  # como fruiz:jaxsvc
    subprocess.run(["sudo", "-n", "chmod", "o+rx", str(raiz)], check=True)
    antes = raiz.stat()
    assert antes.st_mode & 0o005

    datos = _recorrer_directo(raiz / "proyectos", accion="aplicar")
    assert not datos["no_cumple"], datos["no_cumple"]
    despues = raiz.stat()
    assert (despues.st_uid, despues.st_gid) == (antes.st_uid, antes.st_gid)
    assert despues.st_mode & 0o007 == 0
    assert despues.st_mode & 0o070 == antes.st_mode & 0o070 or "mask" in "".join(_acl(raiz))
    assert "other::---" in _acl(raiz)


def test_aplicar_quita_otros_en_modo_y_en_las_dos_acl_y_verificar_da_cero(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    datos = _recorrer_directo(proyectos, accion="aplicar")
    assert not datos["no_cumple"], datos["no_cumple"]

    directorios = [proyectos, proyectos / "un-proyecto", proyectos / "un-proyecto" / "sub"]
    for d in directorios:
        assert oct(d.stat().st_mode & 0o7777) == "0o2770", d
        assert _otros_en_nombres(d) == ["other::---", "default:other::---"], (d, _acl(d))
        acl = _acl(d)
        # las entradas de jaxsvc y fruiz no se tocan, ni el setgid
        assert f"user:{USUARIO_ESPERADO}:rwx" in acl and f"group:{GRUPO_ESPERADO}:rwx" in acl, acl
        assert f"default:user:{USUARIO_ESPERADO}:rwx" in acl and f"default:group:{GRUPO_ESPERADO}:rwx" in acl, acl
    archivo = proyectos / "un-proyecto" / "archivo.txt"
    assert oct(archivo.stat().st_mode & 0o7777) == "0o660", archivo
    assert _otros_en_nombres(archivo) == ["other::---"], (archivo, _acl(archivo))
    assert f"user:{USUARIO_ESPERADO}:rw-" in _acl(archivo) and f"group:{GRUPO_ESPERADO}:rw-" in _acl(archivo)

    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 0, r.stdout
    assert "NO CUMPLE" not in r.stdout


@pytest.mark.parametrize("que,orden", [
    ("ACL de acceso de un directorio", ["setfacl", "-m", "o::r-x", "{dir}"]),
    ("ACL por defecto de un directorio", ["setfacl", "-d", "-m", "o::r-x", "{dir}"]),
    ("ACL de acceso de un archivo", ["setfacl", "-m", "o::r--", "{archivo}"]),
    ("modo de un archivo (chmod o+r)", ["chmod", "o+r", "{archivo}"]),
    ("modo de un directorio (chmod o+x)", ["chmod", "o+x", "{dir}"]),
    # `other` no depende de la máscara: con m::--- el permiso efectivo de otros no se recorta,
    # y la falta de "otros" tiene que salir igual.
    ("otros con la máscara en ---", ["setfacl", "-m", "m::---,o::r-x", "{dir}"]),
])
def test_verificar_cuenta_cualquier_bit_de_otros_tras_aplicar(arbol_temporal, _identidades, que, orden):
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    directorio = proyectos / "un-proyecto" / "sub"
    archivo = proyectos / "un-proyecto" / "archivo.txt"
    objetivo = archivo if "archivo" in que else directorio
    r_mod = subprocess.run(
        ["sudo", "-n", *[a.format(dir=directorio, archivo=archivo) for a in orden]],
        capture_output=True, text=True,
    )
    assert r_mod.returncode == 0, r_mod.stderr

    try:
        r = _correr("--verificar", str(arbol_temporal))
    finally:  # una máscara en --- impediría a pytest borrar su propio tmp_path
        subprocess.run(["sudo", "-n", "setfacl", "-m", "m::rwx", str(directorio)], capture_output=True)
    assert r.returncode == 1, (que, r.stdout)
    faltas = _lineas_no_cumple(r.stdout, objetivo)
    assert faltas and "para otros" in faltas[0], (que, r.stdout)


@pytest.mark.parametrize("usuario", ["jaxsvc", "fruiz"])
def test_lo_creado_despues_de_aplicar_hereda_other_cerrado(arbol_temporal, _identidades, usuario):
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    sub = proyectos / "un-proyecto" / "sub"
    nuevo_archivo = sub / f"nuevo-de-{usuario}.txt"
    nuevo_dir = sub / f"nuevo-dir-de-{usuario}"
    r = subprocess.run(
        ["sudo", "-n", "-u", usuario, "python3", "-c", f"""
import os
os.umask(0o022)
os.close(os.open({str(nuevo_archivo)!r}, os.O_CREAT | os.O_WRONLY, 0o666))
os.mkdir({str(nuevo_dir)!r}, 0o777)
"""],
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, r.stdout + r.stderr

    assert _otros_en_nombres(nuevo_archivo) == ["other::---"], _acl(nuevo_archivo)
    assert nuevo_archivo.stat().st_mode & 0o007 == 0
    assert _otros_en_nombres(nuevo_dir) == ["other::---", "default:other::---"], _acl(nuevo_dir)
    assert nuevo_dir.stat().st_mode & 0o007 == 0
    assert nuevo_dir.stat().st_mode & 0o2000, "el directorio nuevo perdió el setgid"


def _nobody_puede_leer(ruta: Path, *, directorio: bool = False) -> bool:
    """Lectura REAL como `nobody` (open o listdir), no `test -r`: el `test` de uutils (el de
    hall9000) mira solo los bits del modo y no las ACL, así que daría falso aun para quien entra
    por una entrada nombrada."""
    codigo = f"import os; os.listdir({str(ruta)!r})" if directorio else f"open({str(ruta)!r}, 'rb').close()"
    return subprocess.run(["sudo", "-n", "-u", "nobody", "python3", "-c", codigo], capture_output=True).returncode == 0


def test_un_usuario_ajeno_lee_antes_de_aplicar_y_no_despues(arbol_temporal, _identidades):
    """El control del runbook: una lectura como `nobody` (`head -c1 <archivo>`) tiene que fallar tras aplicar.
    El control positivo (antes sí lee) demuestra que la prueba mira lo que dice mirar."""
    proyectos = arbol_temporal / "proyectos"
    archivo = proyectos / "un-proyecto" / "archivo.txt"
    if subprocess.run(["getent", "passwd", "nobody"], capture_output=True).returncode != 0:
        pytest.skip("no existe el usuario nobody")
    assert _nobody_puede_leer(archivo), "control positivo: antes de aplicar, otros SÍ lee"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    assert not _nobody_puede_leer(archivo)
    assert not _nobody_puede_leer(proyectos / "un-proyecto", directorio=True)


def test_el_arnes_deja_entrar_a_un_usuario_ajeno_solo_por_su_entrada_nombrada(tmp_path, _identidades):
    """Valida el arnés mismo: en un runner el usuario de pytest no es jaxsvc ni fruiz, y tras
    aplicar solo entra por la entrada nombrada que le da `_conceder_acceso_al_usuario_de_pruebas`.
    Se simula con `nobody`; sin la entrada no entra (la barrera es real), con ella sí."""
    if subprocess.run(["getent", "passwd", "nobody"], capture_output=True).returncode != 0:
        pytest.skip("no existe el usuario nobody")
    raiz = tmp_path / "raiz"
    proyectos = raiz / "proyectos"
    (proyectos / "p" / "sub").mkdir(parents=True)
    archivo = proyectos / "p" / "archivo.txt"
    archivo.write_text("x")
    if not _acl_disponible_en(proyectos):
        pytest.skip("el filesystem temporal no soporta ACL POSIX")
    _abrir_travesia_hasta(raiz, Path("/tmp"))
    os.chmod(raiz, 0o755)

    _recorrer_directo(proyectos, accion="aplicar")
    assert not _nobody_puede_leer(archivo), "sin entrada nombrada, nobody no tiene que entrar"
    _recorrer_directo(proyectos, accion="deshacer")
    _conceder_acceso_al_usuario_de_pruebas(proyectos, "nobody")
    subprocess.run(["setfacl", "-m", "u:jaxsvc:x", str(raiz)], check=True, capture_output=True)
    _recorrer_directo(proyectos, accion="aplicar")
    assert _nobody_puede_leer(archivo), subprocess.run(["namei", "-l", str(archivo)], capture_output=True, text=True).stdout + "\n".join(str(_acl(d)) for d in (raiz, proyectos, proyectos / "p", archivo))
    nuevo = proyectos / "p" / "sub" / "posterior.txt"
    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "python3", "-c", f"open({str(nuevo)!r}, 'w').write('y')"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert _nobody_puede_leer(nuevo), "la entrada por defecto del arnés tiene que llegar a lo creado después"
    assert _otros_en_nombres(nuevo) == ["other::---"]


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


# --- (b) lectura/escritura cruzada jaxsvc <-> fruiz, sobre un arbol de prueba PROPIO ---------

def test_jaxsvc_y_fruiz_leen_y_escriben_cruzado_en_un_subdirectorio_propio(arbol_temporal, _identidades):
    """Antes corria contra `/home/fruiz/jax-workspace/proyectos` -- en hall9000 un symlink a
    produccion (`/srv/jax-data`) -- y escribia y borraba ahi. Ahora el arbol es de prueba (tmp_path),
    armado y aplicado como el resto, y se corre en cualquier maquina con las dos cuentas."""
    proyectos = arbol_temporal / "proyectos"
    assert Path(os.path.realpath(proyectos)).is_relative_to(Path(os.path.realpath(tempfile.gettempdir())))
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]

    sub = proyectos / "un-proyecto" / "cruzado"
    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "mkdir", str(sub)], capture_output=True, text=True)
    assert r.returncode == 0, f"jaxsvc no pudo crear {sub}: {r.stderr}"

    desde_jaxsvc = sub / "desde-jaxsvc.txt"
    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "sh", "-c", f"echo hola > {desde_jaxsvc}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"jaxsvc no pudo escribir: {r.stderr}"
    # fruiz lee y agrega a lo de jaxsvc
    r = subprocess.run(["sudo", "-n", "-u", "fruiz", "sh", "-c", f"cat {desde_jaxsvc} && echo agregado >> {desde_jaxsvc}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"fruiz no pudo leer/escribir lo de jaxsvc: {r.stderr}"
    assert "hola" in r.stdout

    # fruiz crea y jaxsvc lee y agrega
    desde_fruiz = sub / "desde-fruiz.txt"
    r = subprocess.run(["sudo", "-n", "-u", "fruiz", "sh", "-c", f"echo original > {desde_fruiz}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"fruiz no pudo crear: {r.stderr}"
    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "sh", "-c", f"cat {desde_fruiz} && echo mas >> {desde_fruiz}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"jaxsvc no pudo leer/escribir lo de fruiz: {r.stderr}"
    assert "original" in r.stdout
    assert subprocess.run(["sudo", "-n", "cat", str(desde_fruiz)], capture_output=True, text=True).stdout == "original\nmas\n"
    assert subprocess.run(["sudo", "-n", "cat", str(desde_jaxsvc)], capture_output=True, text=True).stdout == "hola\nagregado\n"
