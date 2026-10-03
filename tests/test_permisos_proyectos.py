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
import shlex
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


_RESPALDOS_TEMPORALES = (
    "import tempfile, shutil, atexit\n"
    "pp.RUTA_RESPALDOS = pp.Path(tempfile.mkdtemp(prefix='respaldos-prueba-'))\n"
    "atexit.register(shutil.rmtree, pp.RUTA_RESPALDOS, ignore_errors=True)\n"
)
"""Se inserta tras `import permisos_proyectos as pp` en todo codigo de prueba que corre como root y llama a
`_generar_respaldo_validado`: sin esto escribe en /var/backups/jax-permisos y le hace chmod 0700."""


@pytest.fixture(scope="module", autouse=True)
def _ninguna_prueba_escribe_en_var_backups():
    """Red de seguridad: el contenido de /var/backups/jax-permisos (existencia, listado con fechas) es el mismo
    antes y despues de TODO el modulo. Complementa a la guarda por codigo."""
    def foto() -> str:
        r = subprocess.run(["sudo", "-n", "ls", "-la", "--time-style=full-iso", "/var/backups/jax-permisos"],
                           capture_output=True, text=True)
        return f"rc={r.returncode}\n{r.stdout}"
    if not _sudo_n_disponible():
        yield
        return
    antes = foto()
    yield
    despues = foto()
    assert despues == antes, f"alguna prueba escribio en /var/backups/jax-permisos:\n--antes--\n{antes}\n--despues--\n{despues}"


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


_PREFIJOS_DE_PRODUCCION = ("/srv/jax-data", "/srv/jax-prod", "/home/fruiz/jax-workspace")


def _es_de_produccion(texto: str) -> bool:
    if not texto.startswith("/") or any(c.isspace() for c in texto):
        return False
    real = os.path.realpath(texto)
    return any(c == pre or c.startswith(pre + "/") for pre in _PREFIJOS_DE_PRODUCCION for c in (texto, real))


def _rutas_de_produccion_en(fuente: str, *, omitir: frozenset = frozenset()) -> list[str]:
    """Toda cadena literal del codigo (no docstrings, no comentarios) con forma de ruta que, tal cual o por
    realpath, caiga en un arbol de produccion. Cubre constantes, `Path("...")` locales y partes literales de
    f-strings; lo que se arma con variables lo cubre `test_la_prueba_cruzada...` (realpath bajo el tempdir)."""
    import ast
    arbol = ast.parse(fuente)
    ignorar: set[int] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            doc = nodo.body[0] if nodo.body else None
            if isinstance(doc, ast.Expr) and isinstance(getattr(doc, "value", None), ast.Constant):
                ignorar.add(id(doc.value))
        if isinstance(nodo, ast.FunctionDef) and nodo.name in omitir:
            ignorar.update(id(n) for n in ast.walk(nodo))
        if isinstance(nodo, ast.Assign) and any(getattr(t, "id", None) in omitir for t in nodo.targets):
            ignorar.update(id(n) for n in ast.walk(nodo))
    return sorted({n.value for n in ast.walk(arbol)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in ignorar
                   and _es_de_produccion(n.value)})


def test_ninguna_ruta_de_las_pruebas_resuelve_a_un_arbol_de_produccion():
    """Ninguna prueba puede tocar un arbol de produccion. En hall9000 `/home/fruiz/jax-workspace` es un
    symlink a `/srv/jax-data/jax-workspace`: una constante con esa ruta hacia que la prueba de
    lectura/escritura cruzada escribiera y borrara en produccion. Se mira el realpath, no el texto, para que
    un symlink no la disfrace: de las constantes `Path` del modulo Y de toda cadena literal con forma de ruta."""
    prohibidos = tuple(Path(p) for p in _PREFIJOS_DE_PRODUCCION)
    for nombre, valor in sorted(globals().items()):
        if isinstance(valor, Path) and nombre != "RAIZ_REPO":
            real = Path(os.path.realpath(valor))
            assert not any(real == p or p in real.parents for p in prohibidos), (
                f"{nombre}={valor} resuelve a {real}, un arbol de produccion"
            )
    literales = _rutas_de_produccion_en(Path(__file__).read_text(), omitir=frozenset({"test_la_guarda_ve_lo_que_dice_ver", "_PREFIJOS_DE_PRODUCCION"}))
    assert not literales, f"cadenas literales que apuntan a produccion: {literales}"


def test_la_guarda_ve_lo_que_dice_ver():
    """Control negativo de la guarda: sobre un texto con las tres formas (constante, `Path` local, parte literal
    de un f-string) tiene que encontrarlas; un docstring y un comentario no cuentan."""
    fuente = (
        'from pathlib import Path\n'
        '"""doc de modulo: /srv/jax-data/x"""\n'
        'A = "/srv/jax-data/jax-workspace"\n'
        'def f():\n'
        '    """doc: /srv/jax-prod/y"""\n'
        '    # comentario /srv/jax-data/z\n'
        '    p = Path("/home/fruiz/jax-workspace/proyectos")\n'
        '    return f"/srv/jax-prod/jax/{p}"\n'
        'B = "una frase /srv/jax-data con espacios"\n'
        'C = "/tmp/algo"\n'
    )
    assert _rutas_de_produccion_en(fuente) == [
        "/home/fruiz/jax-workspace/proyectos", "/srv/jax-data/jax-workspace", "/srv/jax-prod/jax/",
    ]


def test_el_guion_existe():
    assert SCRIPT.is_file()


def _usuario_de_pruebas() -> str:
    return pwd.getpwuid(os.getuid()).pw_name


def _correr(*args: str) -> subprocess.CompletedProcess:
    """El CLI como el usuario de pytest. En un runner ese usuario no es jaxsvc ni fruiz: el arnés le da una
    entrada ACL nombrada (`_conceder_acceso_al_usuario_de_pruebas`) y, como `--verificar` marca NO CUMPLE
    toda entrada nombrada ajena, se le declara conocida con `--permitir-entrada` (solo existe en --verificar)."""
    extra = []
    if not any(a in ("--aplicar", "--deshacer") or a.startswith("--nucleo") for a in args):
        extra = [f"--permitir-entrada={_usuario_de_pruebas()}"]
    return subprocess.run(["python3", str(SCRIPT), *args, *extra], capture_output=True, text=True, timeout=60)


def _verificar_como_root(raiz: Path) -> subprocess.CompletedProcess:
    """`--verificar` como root: para los árboles armados como producción (raíz fruiz:jaxsvc 0770), donde el
    usuario de pytest no entra."""
    return subprocess.run(["sudo", "-n", "python3", str(SCRIPT), "--verificar", str(raiz)],
                          capture_output=True, text=True, timeout=60)


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


def _recorrer_directo(proyectos: Path, *, accion: str, extra_codigo: str = "", puede_fallar: bool = False,
                      conceder_al_terminar: bool = True, procesos_simulados: list | None = None) -> dict:
    """pp._recorrer() como root, bypaseando la fijación de RAIZ de la CLI pública. Con `puede_fallar`, un
    ErrorPermisosProyectos vuelve como {"error": texto} en vez de romper la prueba."""
    codigo = f"""
import sys, json
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
pp.ENTRADAS_EXTRA_PERMITIDAS = {{{_usuario_de_pruebas()!r}}}
# El arbol de pruebas no depende de los procesos reales del host (en hall9000 corren las unidades jaxsvc):
pp._procesos_de_usuario = lambda uid: {list(procesos_simulados or [])!r}
hook = None
hook_raiz = None
hook_entre = None
hook_scandir = None
{extra_codigo}
try:
    r = pp._recorrer(pp.Path({str(proyectos)!r}), accion={accion!r}, hook_de_prueba=hook, hook_antes_de_raiz=hook_raiz,
                    hook_entre_previo_y_mutacion=hook_entre, hook_tras_scandir=hook_scandir)
except pp.ErrorPermisosProyectos as exc:
    print(json.dumps({{"error": str(exc), "a_medio": pp._PROGRESO["mutando"]}}))
    sys.exit(0)
print(json.dumps({{
    "no_cumple": r.no_cumple, "symlinks_saltados": r.symlinks_saltados,
    "hardlinks_rechazados": r.hardlinks_rechazados,
    "bits_espurios_quitados": r.bits_espurios_quitados, "excluidos": r.excluidos,
    "dirs_procesados": r.dirs_procesados, "archivos_procesados": r.archivos_procesados,
}}))
"""
    r = subprocess.run(["sudo", "-n", "python3", "-c", codigo], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    datos = json.loads(r.stdout.strip().splitlines()[-1])
    assert puede_fallar or "error" not in datos, datos
    if conceder_al_terminar and "error" not in datos and accion in ("aplicar", "deshacer"):
        # `--aplicar` REEMPLAZA la ACL entera (`setfacl --set`) y `--deshacer` la quita: la entrada del arnés
        # para el usuario de pytest desaparece y se vuelve a dar, como haria quien arma el entorno.
        _conceder_acceso_al_usuario_de_pruebas(proyectos)
    return datos


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


def _setfacl_root(*args: str) -> None:
    subprocess.run(["sudo", "-n", "setfacl", *args], check=True, capture_output=True)


def _conceder_acceso_al_usuario_de_pruebas(proyectos: Path, usuario: str | None = None) -> None:
    """`--aplicar` deja `other::---`: el usuario que corre pytest en un runner (`runner`) ni es jaxsvc ni es
    fruiz, y ya no entra por `other` a stat/leer/`--verificar` lo que las pruebas miran DESPUES de aplicar.
    Se le da una entrada NOMBRADA (acceso y por defecto, que `--aplicar` conserva porque usa `setfacl -m`):
    es del arnés, no de lo que se prueba. Con sudo: tras `--aplicar`/`--deshacer` los objetos ya no son del
    usuario de pytest y un setfacl sin privilegio falla (asi fallo el primer CI de este PR). Como root no hace
    falta."""
    usuario = usuario or _usuario_de_pruebas()
    if usuario == "root":
        return
    entrada = f"u:{usuario}:rwX"
    # La raíz del workspace (padre de proyectos/) también pierde `otros` con --aplicar: quien entra a
    # proyectos/ tiene que poder atravesarla por entrada nombrada.
    # `m::rwx` explicito: un `setfacl -m` recalcula la mascara (union de las entradas de grupo) y dejaria la raiz en
    # 0750 -- lo que `--verificar` marca (la raiz es 0770).
    _setfacl_root("-m", f"u:{usuario}:x,m::rwx", str(proyectos.parent))
    _setfacl_root("-R", "-m", entrada, str(proyectos))
    subprocess.run(["sudo", "-n", "find", str(proyectos), "-type", "d", "-exec", "setfacl", "-d", "-m", entrada,
                    "{}", "+"], check=True, capture_output=True)


def _dar_paso_por_la_raiz(raiz: Path) -> None:
    """jaxsvc y fruiz tienen que poder atravesar la raíz SIN el bit de otros (`--aplicar` lo comprueba y falla
    cerrado si no). En produccion es por dueño/grupo (raiz fruiz:jaxsvc 0770); en las pruebas con arnés, por
    entrada nombrada (es mas simple de armar sin chown)."""
    for cuenta in ("jaxsvc", "fruiz"):
        _setfacl_root("-m", f"u:{cuenta}:x", str(raiz))


@pytest.fixture()
def base_propia(_identidades):
    """Directorio base PROPIO (mkdtemp, fuera de /tmp/pytest-of-<usuario>). pytest hace chmod 0700 de su
    `pytest-of-<usuario>` al iniciar CADA sesion: con dos sesiones solapadas en el mismo host, jaxsvc/fruiz
    pierden el paso al arbol de pruebas (reproducido por la auditoria: la causa del fallo intermitente de la
    prueba de concurrencia, no una carrera del guion). Aqui nadie reescribe los permisos."""
    base = Path(tempfile.mkdtemp(prefix="permisos-arbol-"))
    os.chmod(base, 0o755)
    yield base
    _limpiar_como_root(base)


@pytest.fixture(scope="module")
def _identidades():
    """jaxsvc y fruiz tienen que EXISTIR: las pruebas no crean cuentas del sistema. En CI las crea un paso del
    workflow (job `permisos-proyectos`); en un host de jax ya existen. Si falta una, la prueba FALLA con un
    mensaje claro -- no se salta en silencio."""
    if not _sudo_n_disponible():
        pytest.skip("sudo -n no disponible -- no se pueden garantizar las identidades jaxsvc/fruiz")
    faltan = [u for u in ("jaxsvc", "fruiz")
              if subprocess.run(["getent", "passwd", u], capture_output=True).returncode != 0]
    if faltan:
        pytest.fail(f"faltan las cuentas del sistema {faltan}: las pruebas no las crean. En CI las crea un paso del "
                    "workflow (job permisos-proyectos de .github/workflows/policy.yml); en otra maquina, crearlas "
                    "antes con el procedimiento del administrador.")
    return None


@pytest.fixture()
def arbol_temporal(base_propia):
    raiz = base_propia / "raiz"
    proyectos = raiz / "proyectos"
    (proyectos / "un-proyecto" / "sub").mkdir(parents=True)
    (proyectos / "un-proyecto" / "archivo.txt").write_text("contenido\n")

    if not _acl_disponible_en(proyectos):
        pytest.skip("el filesystem temporal no soporta ACL POSIX")

    _abrir_travesia_hasta(raiz, Path("/tmp"))
    os.chmod(raiz, 0o755)
    _conceder_acceso_al_usuario_de_pruebas(proyectos)
    _dar_paso_por_la_raiz(raiz)
    # La raiz del workspace es fruiz:jaxsvc (el guion no cambia dueños de la raiz y --aplicar falla cerrado si no).
    subprocess.run(["sudo", "-n", "chown", "fruiz:jaxsvc", str(raiz)], check=True)
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
    """Para las pruebas que necesitan que la CADENA de la ruta del nucleo sea de root (el directorio temporal no
    lo es, y /tmp es escribible por otros): una copia en un directorio propio de root bajo /run (tmpfs, con nombre
    unico) apuntada por JAX_PERMISOS_NUCLEO. NUNCA toca /usr/local/sbin. Se borra siempre al terminar; si la
    prueba muere antes, queda un directorio con nombre unico en una tmpfs que se vacia al reiniciar."""
    r = subprocess.run(["sudo", "-n", "mktemp", "-d", "/run/permisos-nucleo-XXXXXXXX"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"no se pudo crear un directorio propio de root bajo /run: {r.stderr.strip()}")
    directorio = Path(r.stdout.strip())
    try:
        subprocess.run(["sudo", "-n", "chmod", "755", str(directorio)], check=True)
        destino = directorio / "jax-permisos-proyectos"
        monkeypatch.setenv("JAX_PERMISOS_NUCLEO", str(destino))
        copia = _crear_repo_con_head(tmp_path, destino)
        yield copia
    finally:
        subprocess.run(["sudo", "-n", "rm", "-rf", str(directorio)], capture_output=True)


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


def _mkdir_oculta_limpia(ruta: Path) -> None:
    """Una carpeta oculta SIN bits de otros: 0700 y sin la ACL por defecto que hereda del arnes (`default:other::r-x`)."""
    ruta.mkdir(mode=0o700)
    subprocess.run(["setfacl", "-k", str(ruta)], check=True)


def test_exclusion_solo_en_primer_nivel_de_cada_proyecto(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    en_la_raiz = proyectos / OCULTA
    en_la_raiz.mkdir()
    primer_nivel = proyectos / "un-proyecto" / OCULTA
    _mkdir_oculta_limpia(primer_nivel)   # limpia: las ocultas con bits de otros detienen a --aplicar (ver sus pruebas)
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
        _mkdir_oculta_limpia(o)
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
    _mkdir_oculta_limpia(oculta)
    (oculta / "dato.txt").write_text("x")
    os.chmod(oculta / "dato.txt", 0o600)
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

def _lineas_acl_sin_arnes(ruta: Path) -> list[str]:
    """getfacl sin los comentarios y sin las entradas del arnés (el usuario de pytest), que se vuelven a dar
    tras cada --aplicar/--deshacer."""
    return [l for l in _acl(ruta) if f":{_usuario_de_pruebas()}:" not in l and not l.startswith(("mask::", "default:mask::"))]


def test_deshacer_revierte_el_dueno_conserva_a_jaxsvc_y_nunca_reabre_a_otros(arbol_temporal, _identidades):
    """`--deshacer` devuelve el dueño a fruiz:fruiz (lo de antes de E2a) pero CONSERVA `u:jaxsvc:rwx` (`rw-` en
    archivos) en la ACL de acceso y por defecto, con su mascara, y deja `other::---`: LAS MANOS (jaxsvc, que no
    es del grupo fruiz) sigue operando y nadie mas entra. Antes quitaba a jaxsvc (el servicio quedaba sin
    acceso) y antes de eso dejaba `other::r-x` (reabria los documentos a cualquier usuario local)."""
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
        assert st.st_mode & 0o007 == 0, ruta_dir
        assert oct(st.st_mode & 0o777) == "0o770", ruta_dir
        assert _lineas_acl_sin_arnes(ruta_dir) == [
            "user::rwx", f"user:{USUARIO_ESPERADO}:rwx", "group::rwx", "other::---",
            "default:user::rwx", f"default:user:{USUARIO_ESPERADO}:rwx", "default:group::rwx", "default:other::---",
        ], (ruta_dir, _acl(ruta_dir))

    archivo = proyectos / "un-proyecto" / "archivo.txt"
    st = archivo.stat()
    assert pwd.getpwuid(st.st_uid).pw_name == DUENO_ORIGINAL
    assert grp.getgrgid(st.st_gid).gr_name == GRUPO_ESPERADO
    assert oct(st.st_mode & 0o777) == "0o660"
    assert _lineas_acl_sin_arnes(archivo) == [
        "user::rw-", f"user:{USUARIO_ESPERADO}:rw-", "group::rw-", "other::---"], (archivo, _acl(archivo))


def test_tras_deshacer_jaxsvc_sigue_operando_y_otros_no_tienen_ningun_bit(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    _recorrer_directo(proyectos, accion="deshacer")
    sub = proyectos / "un-proyecto"
    archivo = sub / "archivo.txt"
    nuevo = sub / "creado-por-jaxsvc-tras-deshacer.txt"
    nuevo_dir = sub / "dir-creado-por-jaxsvc-tras-deshacer"

    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "python3", "-c", f"""
import os
os.umask(0o022)
assert open({str(archivo)!r}).read() == "contenido\\n"
open({str(archivo)!r}, "a").write("jaxsvc\\n")
open({str(nuevo)!r}, "w").write("nuevo\\n")
os.mkdir({str(nuevo_dir)!r})
"""], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, "jaxsvc (LAS MANOS) no puede operar tras --deshacer: " + r.stdout + r.stderr
    assert subprocess.run(["sudo", "-n", "cat", str(archivo)], capture_output=True, text=True).stdout == "contenido\njaxsvc\n"
    for creado in (nuevo, nuevo_dir):
        assert creado.stat().st_mode & 0o007 == 0, creado
        assert _otros_en_nombres(creado)[0] == "other::---", _acl(creado)
    # otros (ni nombrados ajenos) no tienen ningun bit en el arbol
    r = subprocess.run(["sudo", "-n", "getfacl", "-R", "-p", str(proyectos)], capture_output=True, text=True)
    otros = sorted({l.split("\t")[0] for l in r.stdout.splitlines() if l.startswith(("other::", "default:other::"))})
    assert otros == ["default:other::---", "other::---"], otros


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
    assert oct(st.st_mode & 0o7777) == "0o660"


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


def test_verificar_con_fifo_no_cuelga_ni_crashea(arbol_temporal, _identidades):
    """Con un FIFO en el arbol YA aplicado, `--verificar` termina (no cuelga), con rc 0 y el mensaje exacto de
    exito del recorrido completo -- no con un `RAIZ inválida` o un rc=2 que tambien esquivaria el Traceback. El
    FIFO no se reporta (no gobernado)."""
    proyectos = arbol_temporal / "proyectos"
    fifo = proyectos / "un-proyecto" / "unfifo2"
    os.mkfifo(fifo)
    try:
        assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
        r = _correr("--verificar", str(arbol_temporal))
        assert r.returncode == 0, r.stdout + r.stderr
        assert f"OK: {proyectos} cumple (dueño jaxsvc, grupo fruiz, setgid, sin bits espurios" in r.stdout, r.stdout
        assert "NO CUMPLE" not in r.stdout and "Traceback" not in (r.stdout + r.stderr)
        assert str(fifo) not in r.stdout
    finally:
        subprocess.run(["sudo", "-n", "rm", "-f", str(fifo)], capture_output=True)


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
{_RESPALDOS_TEMPORALES}
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
{_RESPALDOS_TEMPORALES}
ruta = pp._generar_respaldo_validado(pp.Path({str(proyectos)!r}))
print(str(ruta))
ruta.unlink()
"""
    r = subprocess.run(["sudo", "-n", "python3", "-c", codigo], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().startswith("/tmp/respaldos-prueba-"), r.stdout


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
{_RESPALDOS_TEMPORALES}
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


def _hay_nobody() -> bool:
    return subprocess.run(["getent", "passwd", "nobody"], capture_output=True).returncode == 0


def _entrada_temporal_de_nobody_en_la_raiz(raiz: Path, *, poner: bool) -> None:
    """Lo que dice el runbook: `u:nobody:x` temporal y SOLO en la raíz, y se quita comprobando que se quitó."""
    if poner:
        _setfacl_root("-m", "u:nobody:x,m::rwx", str(raiz))     # la mascara explicita: no se recalcula a r-x
        assert "user:nobody:--x" in _acl(raiz)
    else:
        _setfacl_root("-x", "u:nobody", str(raiz))
        _setfacl_root("-m", "m::rwx", str(raiz))
        assert not any("nobody" in l for l in _acl(raiz)), _acl(raiz)


def test_un_usuario_ajeno_lee_antes_de_aplicar_y_no_despues(arbol_temporal, _identidades):
    """La prueba del runbook, tal cual: nobody no atraviesa la raíz (770 en produccion), asi que se le da
    `u:nobody:x` temporal SOLO en la raíz para medir el cierre de proyectos/ por si mismo, y se quita
    despues. Control positivo coherente con eso: ANTES de aplicar (proyectos/ con o::r-x), con la misma
    entrada, nobody SI lee; DESPUES, no. Sin el control, un cierre que ya estuviera hecho por la raíz
    pasaria por bueno."""
    if not _hay_nobody():
        pytest.skip("no existe el usuario nobody")
    raiz = arbol_temporal
    proyectos = raiz / "proyectos"
    archivo = proyectos / "un-proyecto" / "archivo.txt"

    _entrada_temporal_de_nobody_en_la_raiz(raiz, poner=True)
    try:
        assert _nobody_puede_leer(archivo), "control positivo: antes de aplicar, otros SÍ lee"
        assert _nobody_puede_leer(proyectos / "un-proyecto", directorio=True)
    finally:
        _entrada_temporal_de_nobody_en_la_raiz(raiz, poner=False)

    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]

    _entrada_temporal_de_nobody_en_la_raiz(raiz, poner=True)
    try:
        assert not _nobody_puede_leer(archivo)
        assert not _nobody_puede_leer(proyectos / "un-proyecto", directorio=True)
    finally:
        _entrada_temporal_de_nobody_en_la_raiz(raiz, poner=False)
    assert _correr("--verificar", str(raiz)).returncode == 0, "tras quitar la entrada temporal, el arbol cumple"


# --- MINOR-2: entradas ACL nombradas ajenas ---------------------------------------------------

def _un_grupo_ajeno() -> str:
    for g in ("users", "nogroup", "nobody", "daemon"):
        if subprocess.run(["getent", "group", g], capture_output=True).returncode == 0:
            return g
    pytest.skip("no hay un grupo ajeno conocido")


@pytest.mark.parametrize("donde,entrada,default", [
    ("raiz", "u:nobody:r-x", False),
    ("raiz", "u:nobody:r-x", True),
    ("directorio", "u:nobody:r-x", False),
    ("directorio", "u:nobody:r-x", True),
    ("directorio", "g:GRUPO:r-x", False),
    ("archivo", "u:nobody:r--", False),
    ("archivo", "g:GRUPO:r--", False),
])
def test_verificar_marca_cualquier_entrada_nombrada_que_no_sea_jaxsvc_ni_fruiz(
        arbol_temporal, _identidades, donde, entrada, default):
    if not _hay_nobody():
        pytest.skip("no existe el usuario nobody")
    entrada = entrada.replace("GRUPO", _un_grupo_ajeno())
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    assert _correr("--verificar", str(arbol_temporal)).returncode == 0
    objetivo = {"raiz": arbol_temporal, "directorio": proyectos / "un-proyecto" / "sub",
                "archivo": proyectos / "un-proyecto" / "archivo.txt"}[donde]
    _setfacl_root(*(["-d"] if default else []), "-m", entrada, str(objetivo))

    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 1, (donde, entrada, r.stdout)
    lineas = [l for l in r.stdout.splitlines() if l.startswith("NO CUMPLE: ") and str(objetivo) in l]
    assert lineas and "entrada ACL nombrada ajena" in lineas[0], (donde, entrada, r.stdout)


def test_aplicar_falla_cerrado_ante_una_entrada_nombrada_ajena_y_no_la_borra(arbol_temporal, _identidades):
    """Una persona decide: --aplicar no borra la entrada ajena (ni la deja ampliada por `m::rwx`), la reporta
    y NO cambia nada del arbol."""
    if not _hay_nobody():
        pytest.skip("no existe el usuario nobody")
    proyectos = arbol_temporal / "proyectos"
    sub = proyectos / "un-proyecto" / "sub"
    _setfacl_root("-m", "u:nobody:r-x", str(sub))
    antes = {d: (d.stat().st_uid, oct(d.stat().st_mode & 0o7777)) for d in (proyectos, sub, arbol_temporal)}
    acl_antes = _acl(sub)

    datos = _recorrer_directo(proyectos, accion="aplicar", puede_fallar=True)
    assert "error" in datos and str(sub) in datos["error"] and "nobody" in datos["error"], datos
    assert {d: (d.stat().st_uid, oct(d.stat().st_mode & 0o7777)) for d in antes} == antes, "se mutó algo"
    assert _acl(sub) == acl_antes and "user:nobody:r-x" in _acl(sub), "la entrada ajena se tocó"
    assert pwd.getpwuid(proyectos.stat().st_uid).pw_name != USUARIO_ESPERADO


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
    _dar_paso_por_la_raiz(raiz)
    subprocess.run(["sudo", "-n", "chown", "fruiz:jaxsvc", str(raiz)], check=True)

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
    """El escritor es fruiz y el archivo ES DE FRUIZ, 0600, sin ACL (como lo deja el estado de hoy): el
    `fchown` a jaxsvc le quita la condicion de dueño, y esa es la ventana que `--aplicar` tiene que cubrir.
    (La version anterior escribia como jaxsvc un archivo de jaxsvc: el dueño no pierde nada con el chown, y la
    prueba no podia detectar nada. Su fallo intermitente NO era el guion: pytest hace chmod 0700 de
    /tmp/pytest-of-<usuario> al iniciar cada sesion, y con dos sesiones solapadas jaxsvc perdia el paso; por
    eso ahora el arbol vive en un directorio propio, ver `base_propia`.)"""
    proyectos = arbol_temporal / "proyectos"
    archivo = proyectos / "un-proyecto" / "actividad.log"
    r_crear = _como_fruiz(f"import os; os.close(os.open({str(archivo)!r}, os.O_CREAT | os.O_WRONLY, 0o600))")
    assert r_crear.returncode == 0, r_crear.stdout + r_crear.stderr
    assert archivo.stat().st_mode & 0o7777 == 0o600

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
print(i)
"""
    proc = subprocess.Popen(
        ["sudo", "-n", "-u", "fruiz", "python3", "-c", codigo_escritor],
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

    assert proc.returncode == 0, f"el escritor concurrente (como fruiz, ya no dueño) vio un error de OS: {error}"
    total_escrito = int(salida.strip())
    assert total_escrito > 1, "el escritor no llegó a escribir nada -- el test no probó lo que dice probar"
    contenido = subprocess.run(["sudo", "-n", "cat", str(archivo)], capture_output=True, text=True).stdout.splitlines()
    assert contenido == [f"linea-{i}" for i in range(total_escrito)]


def _sondear_etapas(arbol: Path, parchear: str) -> list:
    """Corre --aplicar sobre un archivo 0600 de fruiz parcheando la funcion `parchear` de pp (y os.fchown/
    os.fchmod) para sondear, tras cada una, si fruiz sigue pudiendo abrir su archivo. Devuelve [[etapa, rc]]."""
    proyectos = arbol / "proyectos"
    archivo = proyectos / "un-proyecto" / "actividad.log"
    r_crear = _como_fruiz(f"import os; os.close(os.open({str(archivo)!r}, os.O_CREAT | os.O_WRONLY, 0o600))")
    assert r_crear.returncode == 0, r_crear.stdout + r_crear.stderr
    codigo = f"""
import sys, json, os, subprocess
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
pp.ENTRADAS_EXTRA_PERMITIDAS = {{{_usuario_de_pruebas()!r}}}
pp._procesos_de_usuario = lambda uid: []     # el arbol de pruebas no depende de los procesos reales del host
objetivo_ino = os.stat({str(archivo)!r}).st_ino
sondeos = []
def _sondear(etapa, fd):
    if os.fstat(fd).st_ino != objetivo_ino:
        return
    rc = subprocess.run(["sudo", "-n", "-u", "fruiz", "python3", "-c",
                         "open(%r, 'ab').close()" % {str(archivo)!r}], capture_output=True).returncode
    sondeos.append([etapa, rc])
_fchown, _fchmod = os.fchown, os.fchmod
_acl = getattr(pp, {parchear!r})
def fchown(fd, uid, gid):
    _fchown(fd, uid, gid); _sondear("tras fchown", fd)
def fchmod(fd, modo):
    _fchmod(fd, modo); _sondear("tras fchmod", fd)
def acl(fd, entrada, *a, **k):
    _acl(fd, entrada, *a, **k); _sondear("tras setfacl", fd)
os.fchown, os.fchmod = fchown, fchmod
setattr(pp, {parchear!r}, acl)
pp._recorrer(pp.Path({str(proyectos)!r}), accion="aplicar")
print(json.dumps(sondeos))
"""
    r = subprocess.run(["sudo", "-n", "python3", "-c", codigo], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_ninguna_cuenta_pierde_acceso_entre_el_chown_y_el_setfacl(arbol_temporal, _identidades):
    """La version DETERMINISTA de la ventana: se sondea el acceso de fruiz a su propio archivo 0600 justo
    despues de cada operacion privilegiada sobre ese archivo (fchown, la ACL, fchmod). Con el orden
    chown -> ACL, fruiz deja de ser dueño y no tiene todavia entrada: el sondeo falla. Con la ACL primero
    (las entradas nombradas ya valen cuando cambia el dueño) no hay instante sin acceso. Parchea la funcion
    que la mutacion llama DE VERDAD (`_setfacl_reemplazar`) y exige haber visto las tres etapas."""
    sondeos = _sondear_etapas(arbol_temporal, "_setfacl_reemplazar")
    etapas = [e for e, _ in sondeos]
    # (el fchmod 0660 ya no se llama si la ACL --set dejo el modo en 0660: sin esa etapa no hay nada que sondear)
    assert {"tras setfacl", "tras fchown"} <= set(etapas), f"no se sondeo cada etapa: {etapas}"
    assert etapas.index("tras setfacl") < etapas.index("tras fchown"), f"la ACL tiene que ir ANTES del chown: {etapas}"
    sin_acceso = [e for e, rc in sondeos if rc != 0]
    assert not sin_acceso, f"fruiz perdió el acceso a su archivo en: {sin_acceso} (todos: {sondeos})"


def test_el_sondeo_de_etapas_falla_si_se_parchea_la_funcion_equivocada(arbol_temporal, _identidades):
    """Control negativo: parchear `_sudo_n_funciona` (que la mutacion no llama) no sondea la etapa de la ACL, y la
    comprobacion de la prueba anterior lo detecta. Sin esto, un parche mal puesto pasaria en verde."""
    etapas = [e for e, _ in _sondear_etapas(arbol_temporal, "_sudo_n_funciona")]
    assert "tras setfacl" not in etapas, etapas
    assert {"tras setfacl", "tras fchown"} - set(etapas) == {"tras setfacl"}, etapas


def test_una_entrada_agregada_entre_la_pasada_previa_y_la_mutacion_no_sobrevive_a_aplicar(arbol_temporal, _identidades):
    """Carrera: tras una primera aplicacion jaxsvc es dueño del arbol y puede agregar `u:nobody:rwx,m::---` a un
    directorio DESPUES de la pasada previa. Con `setfacl -m` la entrada sobrevivia y la mascara `m::rwx` la
    volvia efectiva. Ahora la mutacion REEMPLAZA la ACL entera (acceso y por defecto) por la canonica: la
    entrada desaparece, y la mascara nunca vuelve efectiva a una ajena."""
    if not _hay_nobody():
        pytest.skip("no existe el usuario nobody")
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    sub = proyectos / "un-proyecto" / "sub"
    marca = arbol_temporal.parent / "hook-entre-corrio"
    extra = f"""
def hook_entre():
    import subprocess
    subprocess.run(["setfacl", "-m", "u:nobody:rwx,m::---", {str(sub)!r}], check=True)
    subprocess.run(["setfacl", "-d", "-m", "u:nobody:rwx", {str(sub)!r}], check=True)
    open({str(marca)!r}, "w").write("corrio")
"""
    datos = _recorrer_directo(proyectos, accion="aplicar", extra_codigo=extra)
    assert marca.read_text() == "corrio", "el gancho de la ventana NO corrió: la prueba no probó nada"
    assert not datos["no_cumple"], datos
    acl = _acl(sub)
    assert not any("nobody" in l for l in acl), f"sobrevivió la entrada ajena: {acl}"
    assert "mask::rwx" in acl and "other::---" in acl and "default:other::---" in acl, acl
    # y nobody no tiene permiso efectivo: no hay entrada, y `other` esta cerrado
    r = subprocess.run(["sudo", "-n", "setpriv", "--reuid=nobody", "--regid=nogroup", "--clear-groups",
                        "python3", "-c", f"import os; print(os.access({str(sub)!r}, os.R_OK))"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "False", \
        f"el sondeo de nobody no corrió o dio acceso: rc={r.returncode} {r.stdout!r} {r.stderr!r}"
    assert _correr("--verificar", str(arbol_temporal)).returncode == 0


# --- el runbook: bloques de shell validos y sin accesos temporales para terceros ----------------

RUNBOOK_E2A = RAIZ_REPO / "docs" / "runbooks" / "proyectos-e2a-produccion.md"


def _bloques_bash(texto: str) -> list[str]:
    import re
    return re.findall(r"```bash\n(.*?)```", texto, re.S)


def test_la_receta_manual_de_las_ocultas_usa_una_variable_y_no_se_pega_la_ruta():
    texto = RUNBOOK_E2A.read_text()
    assert 'OCULTA="${OCULTA:?' in texto and 'chmod -R o-rwx -- "$OCULTA"' in texto
    assert "no se retipea" in texto.lower() or "no la retipees" in texto.lower() or "no retipear" in texto.lower()
    assert 'chmod -R o-rwx "<ruta' not in texto


def test_el_runbook_no_da_acceso_temporal_a_nadie_ni_usa_test_como_prueba_de_permisos():
    texto = RUNBOOK_E2A.read_text()
    assert "u:nobody" not in texto and "centinela" not in texto.lower() and "trap " not in texto
    assert "sudo -u nobody" not in texto


def test_los_bloques_del_runbook_de_permisos_son_sintacticamente_validos_con_set_euo_pipefail():
    texto = RUNBOOK_E2A.read_text()
    inicio = texto.index("### 2. Permisos de `proyectos/`")
    fin = texto.index("### 3. jax a producción")
    bloques = [b for b in _bloques_bash(texto[inicio:fin]) if "<<'" in b or "stat -c" in b or "OCULTA" in b]
    assert bloques, "no hay bloques de verificación en el paso 2"
    for b in bloques:
        assert "set -euo pipefail" in b, f"bloque sin set -euo pipefail:\n{b}"
        r = subprocess.run(["bash", "-n"], input=b, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


# --- ronda 5: carpetas ocultas (solo `otros`), raiz en --deshacer, el bloque del runbook -----------

def _crear_oculta_abierta(proyectos: Path) -> dict:
    """Como jaxsvc, DESPUES de aplicar: `.estado` 0777 con un archivo 0666, un subdirectorio 0777, un symlink a un
    directorio de fuera (que no se debe tocar) y, en la oculta, una ACL por defecto con other abierto."""
    base = proyectos / "un-proyecto"
    oculta = base / ".estado"
    fuera = proyectos.parent.parent / "fuera-del-arbol"
    fuera.mkdir()
    os.chmod(fuera, 0o755)
    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "python3", "-c", f"""
import os
os.mkdir({str(oculta)!r}); os.chmod({str(oculta)!r}, 0o777)
open({str(oculta / "dato.txt")!r}, "w").write("x"); os.chmod({str(oculta / "dato.txt")!r}, 0o666)
os.mkdir({str(oculta / "sub")!r}); os.chmod({str(oculta / "sub")!r}, 0o777)
os.symlink({str(fuera)!r}, {str(oculta / "enlace")!r})
"""], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    _setfacl_root("-d", "-m", "o::r-x", str(oculta))
    _setfacl_root("-m", "u:nobody:r-x", str(oculta / "dato.txt")) if _hay_nobody() else None
    return {"oculta": oculta, "dato": oculta / "dato.txt", "sub": oculta / "sub", "fuera": fuera}


def _otros_de(ruta: Path) -> list[str]:
    return [l for l in _acl(ruta) if l.startswith(("other::", "default:other::"))]


def test_verificar_marca_otros_en_una_carpeta_oculta_y_en_su_contenido(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    o = _crear_oculta_abierta(proyectos)
    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 1, r.stdout
    for ruta in (o["oculta"], o["dato"], o["sub"]):
        lineas = [l for l in r.stdout.splitlines() if l.startswith(f"NO CUMPLE: {ruta}:")]
        assert lineas and "para otros" in lineas[0], (ruta, r.stdout)
    assert not any("/enlace" in l and l.startswith("NO CUMPLE") for l in r.stdout.splitlines()), "se siguió un symlink"


def _foto_completa(ruta: Path) -> tuple:
    st = ruta.stat()
    return (st.st_uid, st.st_gid, st.st_mode, st.st_mtime_ns, st.st_ctime_ns, tuple(_acl(ruta)))


@pytest.mark.parametrize("accion", ["aplicar", "deshacer"])
def test_una_oculta_con_bits_de_otros_hace_fallar_cerrado_y_root_no_la_toca(arbol_temporal, _identidades, accion):
    """Root NO muta las carpetas ocultas (estado de herramientas, y un inode que jaxsvc puede enlazar desde fuera
    seria una carrera): si alguna tiene un bit de otros, `--aplicar` y `--deshacer` fallan cerrado en su pasada
    previa, ANTES de mutar nada; el mensaje nombra cada ruta y da la orden manual que ejecuta una persona."""
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    o = _crear_oculta_abierta(proyectos)
    objetos = [proyectos, proyectos / "un-proyecto", proyectos / "un-proyecto" / "sub",
               proyectos / "un-proyecto" / "archivo.txt", o["oculta"], o["dato"], o["sub"]]
    antes = {d: _foto_completa(d) for d in objetos}

    datos = _recorrer_directo(proyectos, accion=accion, puede_fallar=True, conceder_al_terminar=False)
    assert "error" in datos, datos
    for ruta in (o["oculta"], o["dato"], o["sub"]):
        assert str(ruta) in datos["error"], (ruta, datos["error"])
    assert f"chmod -R o-rwx -- {shlex.quote(str(o['oculta']))}" in datos["error"], datos["error"]
    assert {d: _foto_completa(d) for d in objetos} == antes, f"--{accion} mutó algo (incluida la oculta) pese a fallar"


@pytest.mark.parametrize("accion", ["aplicar", "deshacer"])
def test_una_oculta_limpia_no_detiene_a_aplicar_ni_a_deshacer_y_queda_intacta(arbol_temporal, _identidades, accion):
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    oculta = proyectos / "un-proyecto" / ".estado-limpio"
    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "python3", "-c", f"""
import os
os.mkdir({str(oculta)!r}, 0o700)
open({str(oculta / "dato.txt")!r}, "w").write("x"); os.chmod({str(oculta / "dato.txt")!r}, 0o600)
"""], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    _setfacl_root("-m", "u:nobody:r-x", str(oculta)) if _hay_nobody() else None
    objetos = [oculta, oculta / "dato.txt"]
    antes = {d: _foto_completa(d) for d in objetos}
    time.sleep(0.05)

    datos = _recorrer_directo(proyectos, accion=accion, puede_fallar=True, conceder_al_terminar=False)
    assert "error" not in datos, datos
    assert {d: _foto_completa(d) for d in objetos} == antes, "root tocó una oculta limpia (modo, dueño, ACL o mtime)"


def test_deshacer_falla_cerrado_si_jaxsvc_no_atraviesa_la_raiz_y_no_toca_nada(base_propia):
    """Escenario del auditor: tras --aplicar la raiz pasa de fruiz:jaxsvc a fruiz:fruiz sin ACL de jaxsvc.
    --deshacer cambiaria los objetos a fruiz:fruiz y restauraria un modo que dejaria a jaxsvc sin llegar a nada:
    se calcula ANTES con `_puede_atravesar` y, si no atraviesa, falla cerrado sin tocar un solo objeto."""
    raiz = _arbol_como_produccion(base_propia, dueno="fruiz", grupo="jaxsvc", modo=0o775)
    proyectos = raiz / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar", conceder_al_terminar=False)["no_cumple"]
    subprocess.run(["sudo", "-n", "chown", "fruiz:fruiz", str(raiz)], check=True)
    objetos = [proyectos, proyectos / "p", proyectos / "p" / "sub", proyectos / "p" / "archivo.txt", raiz]
    antes = {d: _foto(d) for d in objetos}
    acl_antes = {d: _acl(d) for d in objetos}

    datos = _recorrer_directo(proyectos, accion="deshacer", puede_fallar=True, conceder_al_terminar=False)
    assert "error" in datos and "jaxsvc" in datos["error"] and "atravesar" in datos["error"], datos
    assert {d: _foto(d) for d in objetos} == antes, "se mutó algo pese a fallar cerrado"
    assert {d: _acl(d) for d in objetos} == acl_antes


def test_deshacer_con_un_modo_guardado_que_dejaria_a_jaxsvc_fuera_falla_cerrado(arbol_temporal, _identidades):
    """Lo mismo pero por el MODO que va a restaurar: un respaldo (de confianza) con la raiz en 0700, con la raiz
    fruiz:jaxsvc sin ACL, dejaria a jaxsvc sin paso (solo entra por grupo)."""
    proyectos = arbol_temporal / "proyectos"
    out = _driver_respaldo(f"""
proy = pp.Path({str(proyectos)!r})
raiz = {str(arbol_temporal)!r}
import subprocess
subprocess.run(["setfacl", "-b", raiz], check=True)
subprocess.run(["chown", "fruiz:jaxsvc", raiz], check=True)
os.chmod(raiz, 0o770)
(pp.RUTA_RESPALDOS / "proyectos-forjado-de-prueba.acl").write_text(
    '# raiz-ruta: ' + json.dumps(raiz) + '\\n# raiz-modo: 0700\\n\\n' + pp._MARCADOR_FIN_RESPALDO)
os.chown(pp.RUTA_RESPALDOS / "proyectos-forjado-de-prueba.acl", 0, 0)
pp._recorrer(proy, accion="aplicar")
antes = os.stat(proy).st_uid, oct(os.stat(raiz).st_mode & 0o7777)
pp._raiz_configurada_privilegiada = lambda: proy
import io, contextlib
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        salida["rc"] = pp._cmd_nucleo_deshacer()
except pp.ErrorPermisosProyectos as exc:
    salida["error"] = str(exc)
salida["salida"] = buf.getvalue()
salida["despues"] = os.stat(proy).st_uid, oct(os.stat(raiz).st_mode & 0o7777)
salida["antes"] = antes
""")
    texto = out.get("error", "") + out["salida"]
    assert out.get("rc", 2) != 0 and "jaxsvc" in texto and "atravesar" in texto, out
    assert out["despues"] == out["antes"], "se mutó algo pese a fallar cerrado"
    assert "OK: deshecho" not in texto


def test_deshacer_verifica_de_nuevo_la_raiz_y_solo_dice_ok_si_los_dos_atraviesan(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    out = _driver_respaldo(f"""
proy = pp.Path({str(proyectos)!r})
pp._generar_respaldo_validado(proy)
pp._recorrer(proy, accion="aplicar")
pp._raiz_configurada_privilegiada = lambda: proy
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    salida["rc"] = pp._cmd_nucleo_deshacer()
salida["json"] = json.loads(buf.getvalue())
""")
    assert out["rc"] == 0 and out["json"]["raiz"]["paso_ok"] is True and out["json"]["raiz"]["paso_faltas"] == [], out


def _bloque_de_verificacion_del_runbook() -> str:
    import re
    texto = RUNBOOK_E2A.read_text()
    m = re.search(r"<<'VERIFICACION'\n(.*?)\nVERIFICACION", texto, re.S)
    assert m, "el runbook ya no tiene el bloque de verificacion posterior a --aplicar"
    return m.group(1)


def _correr_bloque(raiz: Path) -> subprocess.CompletedProcess:
    """EJECUTA el bloque del runbook tal cual, con la raiz sustituida por variable (nunca /srv), como root."""
    return subprocess.run(["sudo", "-n", "env", f"RAIZ={raiz}", "bash", "-s"], input=_bloque_de_verificacion_del_runbook(),
                          capture_output=True, text=True, cwd=str(RAIZ_REPO), timeout=120)


def test_el_bloque_de_verificacion_no_repite_el_control_de_la_raiz_que_ya_hace_verificar():
    bloque = _bloque_de_verificacion_del_runbook()
    assert "stat -c" not in bloque and "RAIZ_ESTADO" not in bloque, "el control (c) ya lo cubre --verificar"
    assert "--verificar" in bloque and "getfacl" in bloque


def test_el_bloque_de_verificacion_del_runbook_se_ejecuta_y_falla_cerrado(base_propia):
    assert "${RAIZ:-/srv/jax-data/jax-workspace}" in _bloque_de_verificacion_del_runbook(), \
        "la raiz del bloque tiene que ser una variable con la de produccion por defecto"
    raiz = _arbol_como_produccion(base_propia, dueno="fruiz", grupo="jaxsvc", modo=0o775)
    proyectos = raiz / "proyectos"

    antes = _correr_bloque(raiz)
    assert antes.returncode != 0 and "NO CUMPLE" in antes.stderr, antes.stdout + antes.stderr

    assert not _recorrer_directo(proyectos, accion="aplicar", conceder_al_terminar=False)["no_cumple"]
    despues = _correr_bloque(raiz)
    assert despues.returncode == 0 and "OK:" in despues.stdout, despues.stdout + despues.stderr

    _setfacl_root("-m", "o::r-x", str(proyectos / "p" / "sub"))
    reabierto = _correr_bloque(raiz)
    assert reabierto.returncode != 0 and "NO CUMPLE" in reabierto.stderr, reabierto.stdout + reabierto.stderr
    _setfacl_root("-m", "o::---", str(proyectos / "p" / "sub"))

    subprocess.run(["sudo", "-n", "chown", "fruiz:fruiz", str(raiz)], check=True)
    raiz_mal = _correr_bloque(raiz)
    assert raiz_mal.returncode != 0 and "NO CUMPLE" in raiz_mal.stderr, raiz_mal.stdout + raiz_mal.stderr
    subprocess.run(["sudo", "-n", "chown", "fruiz:jaxsvc", str(raiz)], check=True)
    assert _correr_bloque(raiz).returncode == 0

    # El control (b) del bloque, donde `--verificar` NO ve el problema: un FIFO 0666 (no gobernado, pero
    # `getfacl -R` lo enumera con other::rw-). Sin esto, sustituir el control por un `echo OK` dejaria la prueba
    # en verde. (La raiz 770 fruiz:jaxsvc ya no es un control del bloque: la exige `--verificar`, control (a).)
    fifo = proyectos / "p" / "canal"
    subprocess.run(["sudo", "-n", "mkfifo", str(fifo)], check=True)
    subprocess.run(["sudo", "-n", "chmod", "666", str(fifo)], check=True)   # con la ACL por defecto, mkfifo -m no basta
    r_fifo = _correr_bloque(raiz)
    assert r_fifo.returncode != 0 and "NO CUMPLE: other" in r_fifo.stderr, r_fifo.stdout + r_fifo.stderr
    subprocess.run(["sudo", "-n", "rm", str(fifo)], check=True)
    assert _correr_bloque(raiz).returncode == 0


# --- ronda 6: hardlinks en las ocultas, camino de fracaso de la verificacion final, fixtures sin rutas reales ---

def test_hardlink_dentro_de_una_oculta_no_se_toca_nunca_y_se_falla_cerrado(arbol_temporal, _identidades):
    """Mismo criterio que en el arbol gobernado: un archivo con st_nlink > 1 dentro de una carpeta oculta NO se toca
    (un hardlink es EL MISMO inode: un fchmod alcanzaria tambien a la ruta de fuera de proyectos/). `--verificar`
    lo marca NO CUMPLE; `--aplicar` y `--deshacer` lo reportan y fallan cerrado ANTES de mutar nada."""
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    fuera = arbol_temporal.parent / "ejecutable-de-fuera"
    fuera.write_text("#!/bin/sh\n")
    os.chmod(fuera, 0o755)
    oculta = proyectos / "un-proyecto" / ".estado"
    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "mkdir", str(oculta)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    enlace = oculta / "enlace-duro"
    subprocess.run(["sudo", "-n", "ln", str(fuera), str(enlace)], check=True)
    # y algo abierto en la misma oculta, que tampoco se cierra si falla cerrado (no se muta NADA)
    abierto = oculta / "abierto.txt"
    subprocess.run(["sudo", "-n", "-u", "jaxsvc", "sh", "-c", f"echo x > {abierto} && chmod 666 {abierto}"], check=True)
    assert fuera.stat().st_nlink == 2 and fuera.stat().st_mode & 0o7777 == 0o755

    v = _correr("--verificar", str(arbol_temporal))
    assert v.returncode == 1, v.stdout
    assert f"hardlink en carpeta oculta: {enlace}" in v.stdout, v.stdout

    for accion in ("aplicar", "deshacer"):
        datos = _recorrer_directo(proyectos, accion=accion, puede_fallar=True, conceder_al_terminar=False)
        assert "error" in datos and f"hardlink en carpeta oculta: {enlace}" in datos["error"], (accion, datos)
        assert f"chmod -R o-rwx -- {shlex.quote(str(oculta))}" in datos["error"], datos["error"]   # por `abierto.txt`
        assert fuera.stat().st_mode & 0o7777 == 0o755, f"--{accion} tocó el modo de un archivo de fuera por un hardlink"
        assert abierto.stat().st_mode & 0o007 == 0o006, f"--{accion} mutó algo pese a fallar cerrado"


def test_deshacer_dice_no_ok_y_sale_con_1_si_jaxsvc_pierde_el_paso_a_mitad(arbol_temporal, _identidades):
    """El camino de FRACASO de la verificacion final: tras la mutacion, la raiz pasa a fruiz:fruiz sin ACL (jaxsvc
    pierde el paso). --deshacer tiene que imprimir NO OK y salir con rc 1, no `OK`. Si la verificacion final
    devolviera siempre «sin faltas», esta prueba falla."""
    proyectos = arbol_temporal / "proyectos"
    out = _driver_respaldo(f"""
proy = pp.Path({str(proyectos)!r})
raiz = {str(arbol_temporal)!r}
pp._generar_respaldo_validado(proy)
pp._recorrer(proy, accion="aplicar")
pp._raiz_configurada_privilegiada = lambda: proy
import io, contextlib, subprocess
_restaurar = pp._restaurar_raiz_desde_respaldo
def restaurar_y_romper(p, *a):
    r = _restaurar(p, *a)
    subprocess.run(["setfacl", "-b", raiz], check=True)       # jaxsvc pierde su entrada de paso...
    subprocess.run(["chown", "fruiz:fruiz", raiz], check=True)  # ...y el grupo
    return r
pp._restaurar_raiz_desde_respaldo = restaurar_y_romper
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc_nucleo = pp._cmd_nucleo_deshacer()
json_nucleo = buf.getvalue()
def falso_nucleo(*a):
    return subprocess.CompletedProcess(a, 0, stdout=json_nucleo, stderr="")
pp._invocar_nucleo = falso_nucleo
sal, err = io.StringIO(), io.StringIO()
with contextlib.redirect_stdout(sal), contextlib.redirect_stderr(err):
    salida["rc"] = pp._cmd_deshacer()
salida["stdout"], salida["stderr"] = sal.getvalue(), err.getvalue()
salida["json"] = json.loads(json_nucleo)
""")
    assert out["json"]["raiz"]["paso_ok"] is False and out["json"]["raiz"]["paso_faltas"], out["json"]
    assert out["rc"] == 1, out
    assert "NO OK" in out["stderr"] and "jaxsvc" in out["stderr"], out["stderr"]
    assert "OK: deshecho" not in out["stdout"], out["stdout"]


def test_las_pruebas_no_instalan_en_usr_local_sbin_ni_crean_cuentas():
    """Las pruebas usan SIEMPRE una copia del nucleo en un directorio propio (JAX_PERMISOS_NUCLEO) y nunca crean
    cuentas del sistema: si jaxsvc o fruiz faltan, un paso del workflow las crea y la prueba falla con un mensaje
    claro. Se mira el codigo (AST): ninguna llamada instala en la ruta de sistema ni ejecuta useradd."""
    import ast
    arbol = ast.parse(Path(__file__).read_text())
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call) and getattr(nodo.func, "id", None) == "_crear_repo_con_head":
            for arg in nodo.args:
                assert getattr(arg, "id", None) != "RUTA_NUCLEO_SISTEMA", "una prueba instala el nucleo en la ruta de sistema"
    propia = next(n for n in ast.walk(arbol) if isinstance(n, ast.FunctionDef)
                  and n.name == "test_las_pruebas_no_instalan_en_usr_local_sbin_ni_crean_cuentas")
    docstrings = {id(x) for x in ast.walk(propia)}      # esta guarda nombra la orden: no se mira a si misma
    docstrings |= {id(n.body[0].value) for n in ast.walk(arbol)
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module)) and n.body
                  and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
    literales = [n.value for n in ast.walk(arbol)
                 if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
                 and "useradd" in n.value and n.value != "useradd"]
    llamadas = [n for n in ast.walk(arbol) if isinstance(n, ast.Constant) and n.value == "useradd"
                and id(n) not in docstrings]
    assert not llamadas and not literales, "una prueba crea cuentas del sistema (useradd)"


def test_ninguna_prueba_llama_a_generar_respaldo_sin_sustituir_RUTA_RESPALDOS():
    """`_generar_respaldo_validado` crea archivos en RUTA_RESPALDOS (/var/backups/jax-permisos) y le hace chmod
    0700 al directorio. Toda prueba cuyo codigo (incluido el que corre como root en un subproceso) la llame tiene
    que sustituir RUTA_RESPALDOS: por `_RESPALDOS_TEMPORALES`, por `_driver_respaldo(...)` o a mano."""
    import ast
    fuente = Path(__file__).read_text()
    culpables = []
    for nodo in ast.parse(fuente).body:
        if isinstance(nodo, ast.FunctionDef) and nodo.name != "test_ninguna_prueba_llama_a_generar_respaldo_sin_sustituir_RUTA_RESPALDOS":
            seg = ast.get_source_segment(fuente, nodo) or ""
            if "_generar_respaldo_validado" in seg and not any(
                    m in seg for m in ("_RESPALDOS_TEMPORALES", "_driver_respaldo(", "RUTA_RESPALDOS =")):
                culpables.append(nodo.name)
    assert not culpables, f"llaman a _generar_respaldo_validado sin sustituir RUTA_RESPALDOS: {culpables}"


# --- no_cumple que surge DURANTE la mutacion tiene que llegar al cliente ------------------------

def test_un_no_cumple_durante_la_mutacion_llega_al_json_y_el_cliente_no_dice_ok(arbol_temporal, _identidades):
    """`--aplicar` y `--deshacer` pueden anotar `no_cumple` mientras mutan (p. ej. un directorio que no se puede
    listar). Antes no salia en el JSON del nucleo y el cliente podia imprimir OK con un objeto sin procesar. Se
    inyecta uno sintetico DESPUES del recorrido y se exige: va en el JSON, el cliente lo imprime y no dice OK."""
    proyectos = arbol_temporal / "proyectos"
    out = _driver_respaldo(f"""
proy = pp.Path({str(proyectos)!r})
raiz = {str(arbol_temporal)!r}
pp._generar_respaldo_validado(proy)
pp._raiz_configurada_privilegiada = lambda: proy
import io, contextlib, subprocess
_recorrer = pp._recorrer
def recorrer_con_falta(*a, **k):
    r = _recorrer(*a, **k)
    if k.get("accion") in ("aplicar", "deshacer"):
        r.no_cumple.append(proy.as_posix() + "/falta-sintetica: surgio durante la mutacion")
    return r
pp._recorrer = recorrer_con_falta
resultados = {{}}
for accion in ("aplicar", "deshacer"):
    nucleo = pp._cmd_nucleo_privilegiado if accion == "aplicar" else pp._cmd_nucleo_deshacer
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        nucleo()
    datos = json.loads(buf.getvalue())
    if accion == "aplicar":
        pp._verificar_instalacion = lambda: None
        pp._sudo_n_funciona = lambda: True
        pp._raiz_por_defecto = lambda: raiz
        pp._hacer_respaldo = lambda: pp.Path("/dev/null")
        pp._validar_y_obtener_proyectos = lambda r: proy
        pp._cmd_verificar = lambda r: 0
        cliente = lambda: pp._cmd_aplicar(raiz)
    else:
        cliente = pp._cmd_deshacer
    pp._invocar_nucleo = lambda *a, _j=buf.getvalue(): subprocess.CompletedProcess(a, 0, stdout=_j, stderr="")
    sal, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(sal), contextlib.redirect_stderr(err):
        rc = cliente()
    resultados[accion] = {{"no_cumple": datos.get("no_cumple"), "rc": rc, "stdout": sal.getvalue(), "stderr": err.getvalue()}}
salida["resultados"] = resultados
""")
    for accion, r in out["resultados"].items():
        assert r["no_cumple"] and "falta-sintetica" in r["no_cumple"][0], (accion, r)
        assert r["rc"] == 1, (accion, r)
        assert "falta-sintetica" in r["stdout"] + r["stderr"], (accion, r)
        assert "OK: deshecho" not in r["stdout"] and "aplicado y verificado" not in r["stdout"], (accion, r)


# --- ronda 8: nlink antes de cada mutacion, comillas de las ordenes manuales, arbol a medio aplicar -----------

def _ciclo_nucleo_cliente(proyectos: Path, raiz: Path, accion: str, preparar: str) -> dict:
    """Corre el NUCLEO (`_cmd_nucleo_privilegiado` o `_cmd_nucleo_deshacer`) como root con `preparar` ya aplicado
    (monkeypatches de `pp`), y despues el CLIENTE (`_cmd_aplicar` o `_cmd_deshacer`) con un `_invocar_nucleo` falso
    que devuelve lo que el nucleo dijo (rc y stdout). Devuelve el JSON del nucleo y rc/stdout/stderr del cliente."""
    cliente_aplicar = f"""
pp._verificar_instalacion = lambda: None
pp._sudo_n_funciona = lambda: True
pp._raiz_por_defecto = lambda: {str(raiz)!r}
pp._hacer_respaldo = lambda: pp.Path("/dev/null")
pp._validar_y_obtener_proyectos = lambda r: proy
pp._cmd_verificar = lambda r: 0
cliente = lambda: pp._cmd_aplicar({str(raiz)!r})
""" if accion == "aplicar" else "cliente = pp._cmd_deshacer\n"
    nucleo = "pp._cmd_nucleo_privilegiado" if accion == "aplicar" else "pp._cmd_nucleo_deshacer"
    return _driver_respaldo(f"""
proy = pp.Path({str(proyectos)!r})
pp._generar_respaldo_validado(proy)
pp._raiz_configurada_privilegiada = lambda: proy
import io, contextlib, subprocess
{preparar}
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc_nucleo = {nucleo}()
texto = buf.getvalue()
{cliente_aplicar}
pp._invocar_nucleo = lambda *a: subprocess.CompletedProcess(a, rc_nucleo, stdout=texto, stderr="")
sal, err = io.StringIO(), io.StringIO()
try:
    with contextlib.redirect_stdout(sal), contextlib.redirect_stderr(err):
        rc = cliente()
except pp.ErrorPermisosProyectos as exc:
    rc = "error:" + str(exc)
salida["rc_nucleo"] = rc_nucleo
try:
    salida["json"] = json.loads(texto)
except ValueError:
    salida["json"] = None
salida["texto"] = texto
salida["rc"] = rc
salida["stdout"], salida["stderr"] = sal.getvalue(), err.getvalue()
""")


@pytest.mark.parametrize("accion", ["aplicar", "deshacer"])
def test_un_hardlink_creado_entre_la_acl_y_el_chown_no_se_muta_y_se_anota(arbol_temporal, _identidades, accion):
    """Justo antes de CADA mutacion sobre un archivo (setfacl, fchown, fchmod) se vuelve a mirar st_nlink sobre el
    descriptor. Con un gancho que crea el enlace ENTRE la mutacion de la ACL y el fchown, el objeto queda anotado en
    no_cumple (no se hace el chown ni el chmod) y el cliente sale con 1 sin decir OK."""
    proyectos = arbol_temporal / "proyectos"
    archivo = proyectos / "un-proyecto" / "archivo.txt"
    enlace = arbol_temporal.parent / "enlace-hacia-fuera"
    if accion == "deshacer":
        assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    dueno_antes = archivo.stat().st_uid
    preparar = f"""
_set = pp._setfacl_reemplazar
estado = {{"hecho": False}}
def con_enlace(fd, acl, **k):
    _set(fd, acl, **k)
    if not estado["hecho"] and os.fstat(fd).st_ino == {archivo.stat().st_ino}:
        estado["hecho"] = True
        os.link({str(archivo)!r}, {str(enlace)!r})
pp._setfacl_reemplazar = con_enlace
"""
    out = _ciclo_nucleo_cliente(proyectos, arbol_temporal, accion, preparar)
    assert enlace.exists() and archivo.stat().st_nlink == 2, "el gancho no creó el enlace: la prueba no probó nada"
    assert any("hardlink" in l and str(archivo) in l for l in (out["json"] or {}).get("no_cumple", [])), out
    assert archivo.stat().st_uid == dueno_antes, "se hizo el fchown sobre un inode que ya tenia otro enlace"
    # (el modo si cambio: la mutacion de la ACL -- anterior al enlace -- ya fija mascara y `other`; lo que no se
    # hizo despues del enlace es el fchown y el fchmod.)
    assert out["rc"] == 1, out
    assert str(archivo) in out["stdout"] + out["stderr"]
    assert "OK: deshecho" not in out["stdout"] and "aplicado y verificado" not in out["stdout"], out


def test_las_ordenes_manuales_citan_la_ruta_con_shlex_quote(arbol_temporal, _identidades):
    """La orden manual que el guion imprime la copia y ejecuta una persona (quiza como root): el nombre de una oculta
    lo controla quien la crea. Con `'` y `$(...)` en el nombre, la orden impresa pasada por shlex.split da la ruta
    exacta como UN solo argumento."""
    import shlex
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    nombre = ".es'tado$(touch pwned)`id`; rm -rf x"
    oculta = proyectos / "un-proyecto" / nombre
    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "python3", "-c",
                        f"import os; os.mkdir({str(oculta)!r}); os.chmod({str(oculta)!r}, 0o777)"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    _setfacl_root("-d", "-m", "o::r-x", str(oculta))

    datos = _recorrer_directo(proyectos, accion="aplicar", puede_fallar=True, conceder_al_terminar=False)
    assert "error" in datos, datos
    linea = next(l for l in datos["error"].splitlines() if "corregir a mano" in l)
    tokens = shlex.split(linea.split("este guion no toca las carpetas ocultas:", 1)[1])
    i = tokens.index("chmod")
    assert tokens[i:i + 5] == ["chmod", "-R", "o-rwx", "--", str(oculta)], tokens
    j = tokens.index("find")
    assert tokens[j + 1] == str(oculta), tokens
    assert not (oculta.parent / "pwned").exists()


def test_el_nucleo_que_falla_a_medio_aplicar_lo_dice_y_el_cliente_sale_con_1(arbol_temporal, _identidades):
    """Si el nucleo falla DESPUES de empezar a mutar (aqui: setfacl en el tercer objeto), el JSON lleva
    `a_medio_aplicar: true`, la ultima ruta y la instruccion; el cliente la imprime y sale con 1. Lo mismo para
    --deshacer. Antes: un rc de error generico sin decir que parte del arbol ya habia cambiado."""
    proyectos = arbol_temporal / "proyectos"
    preparar = """
_set = pp._setfacl_reemplazar
vistos = []
def falla_en_el_tercero(fd, acl, **k):
    ino = os.fstat(fd).st_ino
    if ino not in vistos:
        vistos.append(ino)
    if len(vistos) == 3 and ino == vistos[2]:
        raise pp.ErrorPermisosProyectos("setfacl falló (simulado en el tercer objeto)")
    return _set(fd, acl, **k)
pp._setfacl_reemplazar = falla_en_el_tercero
"""
    for accion in ("aplicar", "deshacer"):
        if accion == "deshacer":
            # el arbol tiene que estar aplicado para que --deshacer tenga que mutar algo
            assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
        out = _ciclo_nucleo_cliente(proyectos, arbol_temporal, accion, preparar)
        datos = out["json"]
        assert datos and datos.get("a_medio_aplicar") is True, (accion, out)
        assert datos["ultima_ruta"].startswith(str(proyectos)), (accion, datos)
        assert "parcialmente" in datos["instruccion"] and f"--{accion} es idempotente" in datos["instruccion"], datos
        assert "corregí la causa y volvé a correrlo" in datos["instruccion"], datos
        assert out["rc"] == 1, (accion, out)
        salida = out["stdout"] + out["stderr"]
        assert datos["ultima_ruta"] in salida and "parcialmente" in salida, (accion, salida)


# --- ronda 9: BaseException a medio aplicar, bit x en archivos, raiz 770 fruiz:jaxsvc, rutas sin inyectar lineas ---

@pytest.mark.parametrize("excepcion", ["KeyboardInterrupt", "SystemExit(2)"])
def test_una_interrupcion_a_medio_mutar_tambien_dice_a_medio_aplicar(arbol_temporal, _identidades, excepcion):
    """KeyboardInterrupt (y SystemExit) escapaban de `except Exception`: el nucleo salia sin el JSON ni la ultima
    ruta. Ahora desde que empieza a mutar se captura BaseException, se emite el JSON y se sale con 1."""
    proyectos = arbol_temporal / "proyectos"
    preparar = f"""
_set = pp._setfacl_reemplazar
vistos = []
def interrumpe_en_el_tercero(fd, acl, **k):
    ino = os.fstat(fd).st_ino
    if ino not in vistos:
        vistos.append(ino)
    if len(vistos) == 3 and ino == vistos[2]:
        raise {excepcion}
    return _set(fd, acl, **k)
pp._setfacl_reemplazar = interrumpe_en_el_tercero
"""
    for accion in ("aplicar", "deshacer"):
        if accion == "deshacer":
            assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
        out = _ciclo_nucleo_cliente(proyectos, arbol_temporal, accion, preparar)
        datos = out["json"]
        assert datos and datos.get("a_medio_aplicar") is True, (accion, out)
        assert datos["ultima_ruta"].startswith(str(proyectos)), (accion, datos)
        assert f"--{accion} es idempotente" in datos["instruccion"], datos
        assert out["rc_nucleo"] == 1 and out["rc"] == 1, (accion, out)
        assert datos["ultima_ruta"] in out["stdout"] + out["stderr"], (accion, out)


@pytest.mark.parametrize("que,orden", [
    ("chmod 0770", ["chmod", "770", "{archivo}"]),
    ("chmod u+x", ["chmod", "u+x", "{archivo}"]),
    ("ACL nombrada con x", ["setfacl", "-m", "u:jaxsvc:rwx", "{archivo}"]),
    ("ACL de grupo con x", ["setfacl", "-m", "g:fruiz:rwx", "{archivo}"]),
])
def test_verificar_marca_cualquier_bit_de_ejecucion_en_un_archivo_gobernado(arbol_temporal, _identidades, que, orden):
    """Un archivo gobernado es exactamente 0660: `--verificar` marca NO CUMPLE cualquier bit de ejecucion, en el modo
    o en una entrada ACL (nombrada, de grupo) con permiso efectivo. Antes solo exigia lectura y escritura."""
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    archivo = proyectos / "un-proyecto" / "archivo.txt"
    assert _correr("--verificar", str(arbol_temporal)).returncode == 0
    r_mod = subprocess.run(["sudo", "-n", *[a.format(archivo=archivo) for a in orden]], capture_output=True, text=True)
    assert r_mod.returncode == 0, r_mod.stderr
    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 1, (que, r.stdout)
    lineas = [l for l in r.stdout.splitlines() if l.startswith(f"NO CUMPLE: {archivo}:")]
    assert lineas and "ejecución" in lineas[0], (que, r.stdout)


def test_verificar_exige_exactamente_0660_en_un_archivo_gobernado(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    archivo = proyectos / "un-proyecto" / "archivo.txt"
    subprocess.run(["sudo", "-n", "chmod", "640", str(archivo)], check=True)   # group r--: tambien por la mascara
    r = _correr("--verificar", str(arbol_temporal))
    assert r.returncode == 1, r.stdout
    assert any(l.startswith(f"NO CUMPLE: {archivo}:") and "0660" in l for l in r.stdout.splitlines()), r.stdout


def test_raiz_0750_fruiz_jaxsvc_verificar_marca_y_aplicar_la_deja_en_0770(base_propia):
    raiz = _arbol_como_produccion(base_propia, dueno="fruiz", grupo="jaxsvc", modo=0o750)
    v = _verificar_como_root(raiz)
    assert v.returncode == 1, v.stdout
    assert any("(raíz del workspace)" in l and "770" in l for l in v.stdout.splitlines()), v.stdout

    datos = _recorrer_directo(raiz / "proyectos", accion="aplicar", conceder_al_terminar=False)
    assert not datos["no_cumple"], datos
    assert _foto(raiz) == (pwd.getpwnam("fruiz").pw_uid, grp.getgrnam("jaxsvc").gr_gid, 0o770)
    assert _verificar_como_root(raiz).returncode == 0


@pytest.mark.parametrize("modo_antes,modo_despues", [(0o2750, 0o2770), (0o2775, 0o2770), (0o755, 0o770)])
def test_aplicar_fija_la_raiz_en_0770_y_conserva_el_setgid_solo_si_lo_tiene(base_propia, modo_antes, modo_despues):
    raiz = _arbol_como_produccion(base_propia, dueno="fruiz", grupo="jaxsvc", modo=modo_antes)
    datos = _recorrer_directo(raiz / "proyectos", accion="aplicar", conceder_al_terminar=False)
    assert not datos["no_cumple"], datos
    assert _foto(raiz)[2] & 0o7777 == modo_despues, oct(_foto(raiz)[2])


@pytest.mark.parametrize("dueno,grupo", [("fruiz", "fruiz"), ("jaxsvc", "jaxsvc"), ("jaxsvc", "fruiz")])
def test_aplicar_falla_cerrado_si_el_dueno_o_el_grupo_de_la_raiz_no_son_fruiz_jaxsvc(base_propia, dueno, grupo):
    """No cambia dueños de la raiz: si no coinciden, falla cerrado ANTES de mutar -- aunque las dos cuentas igual la
    atraviesen (aqui por ACL nombrada), porque el modelo de acceso ya no es el esperado."""
    raiz = _arbol_como_produccion(base_propia, dueno=dueno, grupo=grupo, modo=0o750)
    _dar_paso_por_la_raiz(raiz)
    proyectos = raiz / "proyectos"
    objetos = [raiz, proyectos, proyectos / "p", proyectos / "p" / "archivo.txt"]
    antes = {d: _foto(d) for d in objetos}
    datos = _recorrer_directo(proyectos, accion="aplicar", puede_fallar=True, conceder_al_terminar=False)
    assert "error" in datos and "fruiz:jaxsvc" in datos["error"] and "no cambia dueños" in datos["error"], datos
    assert {d: _foto(d) for d in objetos} == antes, "se mutó algo pese a fallar cerrado"
    v = _verificar_como_root(raiz)
    assert v.returncode == 1 and any("(raíz del workspace)" in l and "fruiz:jaxsvc" in l for l in v.stdout.splitlines()), v.stdout


def test_ninguna_ruta_inyecta_lineas_en_los_diagnosticos(arbol_temporal, _identidades):
    """Un nombre de carpeta con un salto de linea puede fabricar una linea que parece una orden independiente
    (`sudo ...`) para quien copia un aviso trabajando como root. Toda ruta que se imprime en cualquier mensaje
    pasa por la misma funcion de presentacion (los caracteres de control se escapan): ninguna linea de la salida
    empieza con `sudo`, en `--verificar` ni en el error de `--aplicar`."""
    proyectos = arbol_temporal / "proyectos"
    assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    nombre = ".estado\nsudo touch marca-inyectada #"
    oculta = proyectos / "un-proyecto" / nombre
    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "python3", "-c",
                        f"import os; os.mkdir({str(oculta)!r}); os.chmod({str(oculta)!r}, 0o777)"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    gobernado = proyectos / "un-proyecto" / "sub" / "dato\nsudo touch marca-inyectada2 #.txt"
    r = subprocess.run(["sudo", "-n", "-u", "jaxsvc", "python3", "-c",
                        f"open({str(gobernado)!r}, 'w').write('x'); import os; os.chmod({str(gobernado)!r}, 0o666)"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr

    v = _correr("--verificar", str(arbol_temporal))
    datos = _recorrer_directo(proyectos, accion="aplicar", puede_fallar=True, conceder_al_terminar=False)
    assert "error" in datos, datos
    for nombre_salida, texto in (("verificar", v.stdout + v.stderr), ("aplicar", datos["error"])):
        assert texto.strip(), nombre_salida
        for linea in texto.splitlines():
            assert not linea.lstrip().startswith("sudo"), f"{nombre_salida}: una linea de la salida empieza con sudo: {linea!r}"
        assert "\\n" in texto, f"{nombre_salida}: el salto de linea no se escapó: {texto!r}"


# --- ronda 10: la identidad de la raiz y de proyectos/ no cambia entre la pasada previa y la mutacion ----------

def _directorio_ajeno(base: Path, nombre: str, *, con_proyectos: bool) -> Path:
    """Un directorio REAL (no un symlink) fruiz:jaxsvc 0755 que ocupara el lugar de la raiz o de proyectos/."""
    d = base / nombre
    (d / "proyectos" / "p" if con_proyectos else d / "p").mkdir(parents=True)
    subprocess.run(["sudo", "-n", "chown", "-R", "fruiz:jaxsvc", str(d)], check=True)
    subprocess.run(["sudo", "-n", "chmod", "-R", "755", str(d)], check=True)
    return d


@pytest.mark.parametrize("accion", ["aplicar", "deshacer"])
@pytest.mark.parametrize("que", ["la raiz", "proyectos"])
def test_un_directorio_real_que_sustituye_a_la_raiz_o_a_proyectos_entre_pasadas_no_se_muta(base_propia, accion, que):
    """La pasada previa guarda (st_dev, st_ino) de la raiz y de proyectos/; la mutacion vuelve a abrirlas y compara,
    junto con dueño y grupo, ANTES de cualquier fchmod. Un directorio REAL (rename, sin symlink) con dueño y grupo
    correctos que ocupe su lugar entre las dos pasadas conserva su modo, y el comando falla cerrado sin
    `a_medio_aplicar`: no habia empezado a mutar."""
    raiz = _arbol_como_produccion(base_propia, dueno="fruiz", grupo="jaxsvc", modo=0o775)
    proyectos = raiz / "proyectos"
    if accion == "deshacer":
        assert not _recorrer_directo(proyectos, accion="aplicar", conceder_al_terminar=False)["no_cumple"]
    if que == "la raiz":
        ajeno = _directorio_ajeno(base_propia, "ajeno", con_proyectos=True)
        sustituto = raiz                       # el ajeno pasara a ocupar este nombre
        original, desplazado = raiz, Path(str(raiz) + ".orig")
    else:
        ajeno = _directorio_ajeno(base_propia, "ajeno-proyectos", con_proyectos=False)
        sustituto = proyectos
        original, desplazado = proyectos, Path(str(proyectos) + ".orig")
    modo_ajeno = _foto(ajeno)[2]
    ino_ajeno = ajeno.stat().st_ino
    extra = f"""
import os
estado = {{"hecho": False}}
def hook_entre():
    if estado["hecho"]:
        return
    estado["hecho"] = True
    os.rename({str(original)!r}, {str(desplazado)!r})
    os.rename({str(ajeno)!r}, {str(original)!r})
"""
    datos = _recorrer_directo(proyectos, accion=accion, extra_codigo=extra, puede_fallar=True,
                              conceder_al_terminar=False)
    assert "error" in datos, datos
    assert sustituto.stat().st_ino == ino_ajeno, "el gancho no sustituyó el directorio: la prueba no probó nada"
    assert datos["a_medio"] is False, f"no habia empezado a mutar: no es a_medio_aplicar ({datos})"
    assert "cambió" in datos["error"], datos["error"]
    assert _foto(sustituto)[2] == modo_ajeno, "se hizo fchmod sobre el directorio sustituto"


def test_deshacer_no_restaura_el_modo_en_una_raiz_sustituida_antes_del_fchmod_final(arbol_temporal, _identidades):
    """El restaurador de la raiz (tercera apertura de `--deshacer`) tambien compara identidad antes del fchmod."""
    proyectos = arbol_temporal / "proyectos"
    base = arbol_temporal.parent
    ajeno = _directorio_ajeno(base, "ajeno-final", con_proyectos=True)
    modo_ajeno = _foto(ajeno)[2]
    out = _driver_respaldo(f"""
proy = pp.Path({str(proyectos)!r})
raiz = {str(arbol_temporal)!r}
pp._generar_respaldo_validado(proy)
pp._recorrer(proy, accion="aplicar")
pp._raiz_configurada_privilegiada = lambda: proy
import io, contextlib
orig = pp._recorrer
estado = {{"hecho": False}}
def recorrer(*a, **k):
    r = orig(*a, **k)
    if k.get("accion") == "deshacer" and not estado["hecho"]:
        estado["hecho"] = True
        os.rename(raiz, raiz + ".orig")
        os.rename({str(ajeno)!r}, raiz)
    return r
pp._recorrer = recorrer
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    salida["rc"] = pp._cmd_nucleo_deshacer()
salida["json"] = json.loads(buf.getvalue())
salida["modo_final"] = os.stat(raiz).st_mode & 0o7777
""")
    assert out["json"].get("a_medio_aplicar") is True and "cambió" in out["json"]["error"], out
    assert out["modo_final"] == modo_ajeno, "se hizo fchmod sobre la raiz sustituta"


# --- ronda 11: cada entrada abierta se compara por inode con la que enumero scandir -------------------

def _huella_de(ruta: Path) -> tuple:
    st = ruta.stat()
    return (st.st_uid, st.st_gid, st.st_mode, st.st_ctime_ns, tuple(_acl(ruta)))


def _arbol_con_oculta_y_hermana(arbol: Path) -> tuple:
    """proyectos/un-proyecto/{visible/, .claude-flow/ (limpia, de quien corre pytest, con un archivo)}."""
    proyectos = arbol / "proyectos"
    proyecto = proyectos / "un-proyecto"
    visible = proyecto / "visible"
    visible.mkdir()
    oculta = proyecto / ".claude-flow"
    _mkdir_oculta_limpia(oculta)
    (oculta / "estado.json").write_text("{}")
    os.chmod(oculta / "estado.json", 0o600)
    return proyectos, proyecto, visible, oculta


_INTERCAMBIO = """
import os
estado = {{"hecho": False}}
def hook_scandir(ruta):
    if estado["hecho"] or not ruta.endswith("/un-proyecto"):
        return
    estado["hecho"] = True
    a, b, t = {a!r}, {b!r}, {a!r} + ".tmp-intercambio"
    os.rename(a, t); os.rename(b, a); os.rename(t, b)      # RENAME_EXCHANGE con tres renames
    # el rename mismo actualiza el ctime del inode movido: se anota el de DESPUES del intercambio
    open({marca!r}, "w").write(str(os.stat(a).st_ctime_ns))
"""


@pytest.mark.parametrize("accion", ["aplicar", "deshacer"])
def test_un_intercambio_de_nombres_entre_el_scandir_y_el_open_no_hace_que_root_mute_la_oculta(
        arbol_temporal, _identidades, accion):
    """Despues de que scandir enumera `visible`, un proceso intercambia los nombres de `visible/` y `.claude-flow/`.
    Root abriria `visible` -- ahora el inode de la oculta -- y le cambiaria ACL, dueño y modo. Se compara
    (st_dev, st_ino) del descriptor abierto con lo enumerado: si no coincide, no se muta, no se desciende y se anota."""
    proyectos, proyecto, visible, oculta = _arbol_con_oculta_y_hermana(arbol_temporal)
    if accion == "deshacer":
        # el arbol se aplica primero (con la oculta ya creada: es de quien corre pytest y no se toca)
        assert not _recorrer_directo(proyectos, accion="aplicar", conceder_al_terminar=False)["no_cumple"]
    ino_oculta = oculta.stat().st_ino
    huella = _huella_de(oculta)
    huella_archivo = _huella_de(oculta / "estado.json")
    marca = arbol_temporal.parent / "ctime-tras-el-intercambio"
    extra = _INTERCAMBIO.format(a=str(visible), b=str(oculta), marca=str(marca))
    out = _ciclo_nucleo_cliente(proyectos, arbol_temporal, accion, extra.replace("hook_scandir", "_h").replace(
        "def _h(ruta):", "def _h(ruta):") + "\n_rec = pp._recorrer\ndef _con_hook(*a, **k):\n"
        "    if k.get('accion') in ('aplicar', 'deshacer'):\n        k.setdefault('hook_tras_scandir', _h)\n    return _rec(*a, **k)\npp._recorrer = _con_hook\n")
    # tras el intercambio, el inode de la oculta esta bajo el nombre `visible`
    assert visible.stat().st_ino == ino_oculta, "el gancho no intercambió los nombres: la prueba no probó nada"
    h = _huella_de(visible)
    assert h[:3] == huella[:3] and h[4] == huella[4], "root mutó la carpeta oculta (dueño, grupo, modo o ACL)"
    assert h[3] == int(marca.read_text()), "root mutó la carpeta oculta (cambió su ctime tras el intercambio)"
    assert _huella_de(visible / "estado.json") == huella_archivo, "root mutó el contenido de la oculta"
    no_cumple = (out["json"] or {}).get("no_cumple", [])
    assert any("la entrada cambió durante el recorrido" in l and "visible" in l for l in no_cumple), out
    assert out["rc"] == 1, out
    assert "la entrada cambió durante el recorrido" in out["stdout"] + out["stderr"]


def test_un_inode_de_una_oculta_vista_en_la_pasada_previa_no_se_muta_aunque_cambie_de_nombre(
        arbol_temporal, _identidades):
    """Conjunto de inodes de las ocultas y su contenido, guardado en la pasada previa: si la mutacion abre un inode
    del conjunto (aqui la oculta, movida a un lugar gobernado entre las dos pasadas, con un nombre sin punto), no lo
    muta y lo anota, aunque scandir lo enumere con ese inode."""
    proyectos, proyecto, visible, oculta = _arbol_con_oculta_y_hermana(arbol_temporal)
    movida = proyecto / "sub" / "movida"
    huella = [_huella_de(oculta)[i] for i in (0, 1, 2, 4)]     # sin ctime: el rename lo cambia
    extra = f"""
import os
def hook_entre():
    os.rename({str(oculta)!r}, {str(movida)!r})
"""
    out = _ciclo_nucleo_cliente(proyectos, arbol_temporal, "aplicar", extra + """
_rec = pp._recorrer
def _con_hook(*a, **k):
    if k.get('accion') in ('aplicar', 'deshacer'):   # solo la pasada de mutacion, no las previas internas
        k.setdefault('hook_entre_previo_y_mutacion', hook_entre)
    return _rec(*a, **k)
pp._recorrer = _con_hook
""")
    assert movida.exists() and not oculta.exists()
    assert [_huella_de(movida)[i] for i in (0, 1, 2, 4)] == huella, "root mutó una carpeta que era oculta"
    no_cumple = (out["json"] or {}).get("no_cumple", [])
    assert any("carpeta oculta" in l and "movida" in l for l in no_cumple), out
    assert out["rc"] == 1, out


@pytest.mark.parametrize("como", ["chown", "setfacl"])
def test_la_verificacion_final_detecta_una_oculta_mutada(arbol_temporal, _identidades, como):
    """Despues de mutar, el nucleo relee las ocultas y compara dueño y ACL contra lo que guardo la pasada previa: si
    cambiaron -- aunque se restituyan los nombres, aunque un rename los esconda -- es NO CUMPLE."""
    proyectos, proyecto, visible, oculta = _arbol_con_oculta_y_hermana(arbol_temporal)
    cambio = ('os.chown(%r, pwd.getpwnam("jaxsvc").pw_uid, -1)' % str(oculta) if como == "chown"
              else 'subprocess.run(["setfacl", "-m", "u:nobody:r-x", %r], check=True)' % str(oculta))
    extra = f"""
import os, pwd, subprocess
def hook_entre():
    {cambio}
"""
    out = _ciclo_nucleo_cliente(proyectos, arbol_temporal, "aplicar", extra + """
_rec = pp._recorrer
def _con_hook(*a, **k):
    if k.get('accion') in ('aplicar', 'deshacer'):   # solo la pasada de mutacion, no las previas internas
        k.setdefault('hook_entre_previo_y_mutacion', hook_entre)
    return _rec(*a, **k)
pp._recorrer = _con_hook
""")
    no_cumple = (out["json"] or {}).get("no_cumple", [])
    assert any("carpeta oculta mutada" in l and ".claude-flow" in l for l in no_cumple), out
    assert out["rc"] == 1, out


# --- ronda 12: sin procesos jaxsvc vivos no hay quien renombre mientras root recorre el arbol ----------------

_UNIDADES = ("jax-las-manos", "jax-platform", "jax-ariadna-pm", "jax-ejecutor-proxy", "jax-catalogo-modelos")


def _secuencia_de_procesos(secuencia: list) -> str:
    """Codigo de prueba: `_procesos_de_usuario` devuelve cada elemento de `secuencia` en llamadas sucesivas (el
    ultimo se repite). Llamadas de --aplicar: al empezar, justo antes de mutar y al terminar."""
    return f"""
_seq = {secuencia!r}
_n = {{"i": 0}}
def _procs(uid):
    i = min(_n["i"], len(_seq) - 1)
    _n["i"] += 1
    return list(_seq[i])
pp._procesos_de_usuario = _procs
"""


@pytest.mark.parametrize("accion", ["aplicar", "deshacer"])
def test_con_procesos_de_jaxsvc_vivos_falla_cerrado_sin_mutar_nada(arbol_temporal, _identidades, accion):
    """Todas las carreras de renombre parten de un proceso jaxsvc VIVO que renombra mientras root recorre el arbol.
    Con cualquiera, `--aplicar` y `--deshacer` fallan cerrado ANTES de la pasada previa y no cambian nada; el mensaje
    nombra los pids y las unidades que hay que detener."""
    proyectos = arbol_temporal / "proyectos"
    if accion == "deshacer":
        assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    objetos = [arbol_temporal, proyectos, proyectos / "un-proyecto", proyectos / "un-proyecto" / "archivo.txt"]
    antes = {d: _foto_completa(d) for d in objetos}
    datos = _recorrer_directo(proyectos, accion=accion, procesos_simulados=[4242, 4243], puede_fallar=True,
                              conceder_al_terminar=False)
    assert "error" in datos, datos
    assert "hay procesos de jaxsvc vivos (pids 4242, 4243)" in datos["error"], datos["error"]
    for unidad in _UNIDADES:
        assert unidad in datos["error"], (unidad, datos["error"])
    assert "timers" in datos["error"] and f"antes de {'aplicar' if accion == 'aplicar' else 'deshacer'}" in datos["error"]
    assert datos["a_medio"] is False
    assert {d: _foto_completa(d) for d in objetos} == antes, "se mutó algo pese a los procesos de jaxsvc"


@pytest.mark.parametrize("accion", ["aplicar", "deshacer"])
def test_sin_procesos_de_jaxsvc_aplica_y_deshace(arbol_temporal, _identidades, accion):
    proyectos = arbol_temporal / "proyectos"
    if accion == "deshacer":
        assert not _recorrer_directo(proyectos, accion="aplicar", procesos_simulados=[])["no_cumple"]
    datos = _recorrer_directo(proyectos, accion=accion, procesos_simulados=[])
    assert "error" not in datos and not datos["no_cumple"], datos
    esperado = "jaxsvc" if accion == "aplicar" else "fruiz"
    assert pwd.getpwuid((proyectos / "un-proyecto").stat().st_uid).pw_name == esperado


@pytest.mark.parametrize("accion", ["aplicar", "deshacer"])
def test_un_proceso_jaxsvc_que_aparece_justo_antes_de_mutar_falla_cerrado_sin_mutar(arbol_temporal, _identidades, accion):
    proyectos = arbol_temporal / "proyectos"
    if accion == "deshacer":
        assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    objetos = [arbol_temporal, proyectos, proyectos / "un-proyecto", proyectos / "un-proyecto" / "archivo.txt"]
    antes = {d: _foto_completa(d) for d in objetos}
    datos = _recorrer_directo(proyectos, accion=accion, puede_fallar=True, conceder_al_terminar=False,
                              extra_codigo=_secuencia_de_procesos([[], [777]]))
    assert "error" in datos and "hay procesos de jaxsvc vivos (pids 777)" in datos["error"], datos
    assert datos["a_medio"] is False, "no habia empezado a mutar"
    assert {d: _foto_completa(d) for d in objetos} == antes, "se mutó algo pese al proceso de jaxsvc"


@pytest.mark.parametrize("accion", ["aplicar", "deshacer"])
def test_un_proceso_jaxsvc_que_aparece_durante_la_mutacion_se_anota_y_el_cliente_no_dice_ok(
        arbol_temporal, _identidades, accion):
    proyectos = arbol_temporal / "proyectos"
    if accion == "deshacer":
        assert not _recorrer_directo(proyectos, accion="aplicar")["no_cumple"]
    out = _ciclo_nucleo_cliente(proyectos, arbol_temporal, accion, _secuencia_de_procesos([[], [], [777]]))
    no_cumple = (out["json"] or {}).get("no_cumple", [])
    assert any("procesos de jaxsvc" in l and "777" in l and "durante" in l for l in no_cumple), out
    assert out["rc"] == 1, out
    assert "OK: deshecho" not in out["stdout"] and "aplicado y verificado" not in out["stdout"], out


def test_verificar_no_exige_que_no_haya_procesos_de_jaxsvc(arbol_temporal, _identidades):
    """`--verificar` es de solo lectura: no hay carrera que cerrar."""
    datos = _recorrer_directo(arbol_temporal / "proyectos", accion="verificar", procesos_simulados=[999],
                              puede_fallar=True, conceder_al_terminar=False)
    assert "error" not in datos, datos


def test_si_la_cuenta_jaxsvc_no_existe_falla_cerrado(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    datos = _recorrer_directo(proyectos, accion="aplicar", puede_fallar=True, conceder_al_terminar=False,
                              extra_codigo='pp.USUARIO = "cuenta-que-no-existe-xyz"')
    assert "error" in datos and "no existe la cuenta cuenta-que-no-existe-xyz" in datos["error"], datos
    assert datos["a_medio"] is False


def test_la_inspeccion_de_procesos_lee_los_cuatro_uid_de_proc_status(tmp_path):
    """Real, uid, guardado y fs (los cuatro campos de `Uid:`); ignora lo que no es un pid y lo ilegible."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("permisos_proyectos", SCRIPT)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)
    casos = {"101": "Name:\tx\nUid:\t994\t0\t0\t0\n", "102": "Uid:\t0\t994\t0\t0\n", "103": "Uid:\t0\t0\t994\t0\n",
             "104": "Uid:\t0\t0\t0\t994\n", "105": "Uid:\t0\t1000\t0\t0\n", "106": "Uid:\tbasura\n",
             "self": "Uid:\t994\t994\t994\t994\n", "107": "sin linea uid\n"}
    for pid, texto in casos.items():
        (tmp_path / pid).mkdir()
        (tmp_path / pid / "status").write_text(texto)
    (tmp_path / "108").mkdir()   # sin status: el proceso termino entre el listado y la lectura
    pp.RUTA_PROC = tmp_path
    assert pp._procesos_de_usuario(994) == [101, 102, 103, 104]


def test_la_inspeccion_real_de_proc_ve_un_proceso_jaxsvc_efimero(_identidades):
    """Contra el /proc real: un `sleep` lanzado como jaxsvc aparece, y deja de aparecer al terminar."""
    codigo = f"""
import sys, json
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
print(json.dumps(pp._procesos_de_usuario(pp.pwd.getpwnam("jaxsvc").pw_uid)))
"""
    def vivos() -> set:
        r = subprocess.run(["sudo", "-n", "python3", "-c", codigo], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stdout + r.stderr
        return set(json.loads(r.stdout.strip().splitlines()[-1]))
    antes = vivos()
    proc = subprocess.Popen(["sudo", "-n", "-u", "jaxsvc", "sleep", "30"])
    try:
        nuevos = set()
        for _ in range(50):
            nuevos = vivos() - antes
            if nuevos:
                break
            time.sleep(0.1)
        assert nuevos, "la inspeccion no vio el proceso de jaxsvc recien lanzado"
    finally:
        proc.terminate()
        proc.wait(timeout=10)
    for _ in range(50):
        if not (vivos() & nuevos):
            break
        time.sleep(0.1)
    assert not (vivos() & nuevos), "el proceso terminó pero la inspeccion lo sigue viendo"


def test_ninguna_prueba_corre_la_mutacion_sin_sustituir_la_inspeccion_de_procesos():
    """En hall9000 hay procesos jaxsvc reales: toda prueba cuyo codigo llame a `pp._recorrer(` o a `_cmd_nucleo_*`
    como root tiene que pasar por un constructor que sustituye `_procesos_de_usuario`."""
    import ast
    fuente = Path(__file__).read_text()
    propia = "test_ninguna_prueba_corre_la_mutacion_sin_sustituir_la_inspeccion_de_procesos"
    culpables = []
    for nodo in ast.parse(fuente).body:
        if isinstance(nodo, ast.FunctionDef) and nodo.name not in (propia, "_recorrer_directo", "_driver_respaldo"):
            seg = ast.get_source_segment(fuente, nodo) or ""
            usa = "pp._recorrer(" in seg or "pp._cmd_nucleo" in seg
            if usa and not any(m in seg for m in ("_procesos_de_usuario", "_recorrer_directo(", "_driver_respaldo(",
                                                  "_ciclo_nucleo_cliente(", "_sondear_etapas(")):
                culpables.append(nodo.name)
    assert not culpables, f"corren la mutacion sin sustituir la inspeccion de procesos: {culpables}"


# --- MAJOR-1: la raiz se abre por descriptor, sin seguir symlinks ------------------------------

def _modo_y_acl(ruta: Path) -> tuple[str, list[str]]:
    return oct(ruta.stat().st_mode & 0o7777), _acl(ruta)


@pytest.mark.parametrize("que_se_cambia", ["la raiz", "el directorio padre de la raiz"])
def test_una_raiz_cambiada_por_un_symlink_antes_del_open_no_se_muta_fuera_del_arbol(base_propia, que_se_cambia):
    """TOCTOU: entre validar por NOMBRE que la raiz no es un symlink y abrirla, quien pueda renombrar en el
    directorio padre (jaxsvc puede en /srv/jax-data) la cambia por un symlink. El nucleo, como root, abria
    /etc siguiendolo y le hacia `chmod o-rwx`. Se prueba con el gancho de la ventana: la raiz de verdad se
    mueve y en su lugar queda un symlink a un directorio SEMEJANTE (con su proyectos/) que no es el arbol."""
    ws = base_propia / "ws"
    destino = base_propia / "ajeno"
    for nivel in (ws, destino):
        (nivel / "raiz" / "proyectos" / "p").mkdir(parents=True)
        os.chmod(nivel / "raiz", 0o755)
        os.chmod(nivel / "raiz" / "proyectos", 0o755)
    raiz = ws / "raiz"
    if que_se_cambia == "la raiz":
        cambiar, por, observado = raiz, destino / "raiz", destino / "raiz"
    else:
        cambiar, por, observado = ws, destino, destino / "raiz"
    antes = _modo_y_acl(observado)
    assert antes[0] == "0o755"

    extra = f"""
def hook_raiz(ruta):
    import os
    if os.path.islink({str(cambiar)!r}):
        return  # el gancho corre en cada recorrido: la ventana se abre una sola vez
    os.rename({str(cambiar)!r}, {str(cambiar) + ".orig"!r})
    os.symlink({str(por)!r}, {str(cambiar)!r})
"""
    datos = _recorrer_directo(raiz / "proyectos", accion="aplicar", extra_codigo=extra, puede_fallar=True)
    assert _modo_y_acl(observado) == antes, f"se mutó fuera del árbol validado: {_modo_y_acl(observado)} vs {antes}"
    assert "error" in datos, datos


# --- MAJOR-2: --aplicar no deja a jaxsvc/fruiz fuera de la raiz --------------------------------

def _arbol_como_produccion(base: Path, *, dueno: str, grupo: str, modo: int) -> Path:
    """raiz/proyectos/p/{sub,archivo.txt} SIN arnes: la raiz con el dueño, el grupo y el modo dados y sin
    ninguna ACL (como en produccion, donde la entrada es por dueño/grupo y no por entradas nombradas)."""
    raiz = base / "raiz"
    (raiz / "proyectos" / "p" / "sub").mkdir(parents=True)
    (raiz / "proyectos" / "p" / "archivo.txt").write_text("x")
    subprocess.run(["sudo", "-n", "chown", f"{dueno}:{grupo}", str(raiz)], check=True)
    subprocess.run(["sudo", "-n", "chmod", oct(modo)[2:], str(raiz)], check=True)
    return raiz


def _foto(ruta: Path) -> tuple:
    st = ruta.stat()
    return st.st_uid, st.st_gid, st.st_mode & 0o7777


def test_aplicar_falla_cerrado_y_no_toca_nada_si_la_raiz_deja_fuera_a_jaxsvc(base_propia):
    """Raiz fruiz:fruiz 0755 (lo que deja un mkdir con umask 022): jaxsvc hoy entra por `otros`. Quitarlo la
    dejaria fuera (LAS MANOS pierde todos los proyectos) y `--deshacer` no la reabre. Falla cerrado."""
    raiz = _arbol_como_produccion(base_propia, dueno="fruiz", grupo="fruiz", modo=0o755)
    proyectos = raiz / "proyectos"
    antes = {d: _foto(d) for d in (raiz, proyectos, proyectos / "p", proyectos / "p" / "archivo.txt")}

    datos = _recorrer_directo(proyectos, accion="aplicar", puede_fallar=True)
    assert "error" in datos and "jaxsvc" in datos["error"] and "atravesar" in datos["error"], datos
    assert {d: _foto(d) for d in antes} == antes, "se mutó algo pese a fallar cerrado"


def test_aplicar_con_la_raiz_de_produccion_acceso_por_grupo_aplica_sin_tocar_dueno_ni_grupo(base_propia):
    """El escenario real: raiz fruiz:jaxsvc 0775. fruiz entra COMO DUEÑO, jaxsvc POR GRUPO; ninguno por ACL
    nombrada ni por otros. Aplica, deja 0770, y no cambia ni dueño ni grupo."""
    raiz = _arbol_como_produccion(base_propia, dueno="fruiz", grupo="jaxsvc", modo=0o775)
    proyectos = raiz / "proyectos"
    uid, gid, _ = _foto(raiz)

    datos = _recorrer_directo(proyectos, accion="aplicar", puede_fallar=True, conceder_al_terminar=False)
    assert "error" not in datos and not datos["no_cumple"], datos
    assert _foto(raiz) == (uid, gid, 0o770)
    assert _otros_en_nombres(raiz) == ["other::---"]
    r = _verificar_como_root(raiz)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("dueno,grupo,sin_paso", [
    ("fruiz", "fruiz", "jaxsvc"),     # jaxsvc ni dueño ni de grupo
    ("jaxsvc", "jaxsvc", "fruiz"),    # fruiz ni dueño ni de grupo
])
def test_verificar_marca_no_cumple_si_jaxsvc_o_fruiz_no_atraviesan_la_raiz(base_propia, dueno, grupo, sin_paso):
    raiz = _arbol_como_produccion(base_propia, dueno="fruiz", grupo="jaxsvc", modo=0o775)
    assert not _recorrer_directo(raiz / "proyectos", accion="aplicar", conceder_al_terminar=False)["no_cumple"]
    assert _verificar_como_root(raiz).returncode == 0
    subprocess.run(["sudo", "-n", "chown", f"{dueno}:{grupo}", str(raiz)], check=True)

    r = _verificar_como_root(raiz)
    assert r.returncode == 1, r.stdout
    lineas = [l for l in r.stdout.splitlines() if l.startswith(f"NO CUMPLE: {raiz} (raíz del workspace)")]
    assert lineas and f"{sin_paso} no puede atravesar la raíz" in lineas[0], r.stdout


def test_puede_atravesar_calcula_dueno_grupo_acl_nombrada_y_mascara():
    import importlib.util
    spec = importlib.util.spec_from_file_location("permisos_proyectos", SCRIPT)
    pp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pp)

    def puede(modo, acl="", *, uid=1000, grupos=None, ignorar_otros=False, dueno=(1, 2)):
        return pp._puede_atravesar(modo, dueno[0], dueno[1], acl, uid=uid, nombre="u1000",
                                   grupos=grupos if grupos is not None else {1000: "g1000"},
                                   ignorar_otros=ignorar_otros)

    assert pp._puede_atravesar(0o700, 1000, 5, "", uid=1000, nombre="u", grupos={}, ignorar_otros=False) is True
    assert pp._puede_atravesar(0o070, 1000, 5, "", uid=1000, nombre="u", grupos={5: "g"}, ignorar_otros=False) is False, \
        "el dueño se decide solo por los bits de dueño"
    assert puede(0o070, grupos={2: "g2"}) is True            # por grupo dueño
    assert puede(0o700, grupos={2: "g2"}) is False           # grupo dueño sin x
    assert puede(0o005) is True                              # por otros
    assert puede(0o005, ignorar_otros=True) is False         # ...salvo que se calcule SIN otros
    acl = "user::rwx\nuser:u1000:--x\ngroup::---\nmask::r-x\nother::---\n"
    assert puede(0o750, acl) is True                         # usuario nombrado
    acl = "user::rwx\nuser:u1000:--x\ngroup::---\nmask::r--\nother::---\n"
    assert puede(0o740, acl) is False                        # ...la mascara lo recorta
    acl = "user::rwx\ngroup::---\ngroup:g1000:--x\nmask::r-x\nother::---\n"
    assert puede(0o750, acl) is True                         # grupo nombrado
    acl = "user::rwx\ngroup::---\ngroup:g1000:--x\nmask::---\nother::r-x\n"
    assert puede(0o700, acl) is False, "con un grupo que coincide pero sin permiso efectivo, `otros` no rescata"


# --- MINOR-3: la raiz queda en el respaldo forense y --deshacer restaura su modo ---------------

def _driver_respaldo(codigo_extra: str) -> dict:
    codigo = f"""
import sys, json, tempfile, os
sys.path.insert(0, {str(RAIZ_REPO / "ops")!r})
import permisos_proyectos as pp
pp.ENTRADAS_EXTRA_PERMITIDAS = {{{_usuario_de_pruebas()!r}}}
pp._procesos_de_usuario = lambda uid: []     # el arbol de pruebas no depende de los procesos reales del host
pp.RUTA_RESPALDOS = pp.Path(tempfile.mkdtemp(prefix="respaldos-prueba-"))
salida = {{}}
try:
{chr(10).join("    " + l for l in codigo_extra.splitlines())}
finally:
    import shutil
    shutil.rmtree(pp.RUTA_RESPALDOS, ignore_errors=True)
print(json.dumps(salida))
"""
    r = subprocess.run(["sudo", "-n", "python3", "-c", codigo], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_el_respaldo_registra_la_raiz_y_deshacer_restaura_su_modo_sin_reabrir_a_otros(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    subprocess.run(["sudo", "-n", "chmod", "775", str(arbol_temporal)], check=True)
    modo_antes = _foto(arbol_temporal)[2]
    out = _driver_respaldo(f"""
proy = pp.Path({str(proyectos)!r})
ruta = pp._generar_respaldo_validado(proy)
salida["texto"] = ruta.read_text(errors="replace")
pp._recorrer(proy, accion="aplicar")
salida["modo_aplicado"] = oct(os.stat({str(arbol_temporal)!r}).st_mode & 0o7777)
pp._raiz_configurada_privilegiada = lambda: proy
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    salida["rc"] = pp._cmd_nucleo_deshacer()
salida["json"] = json.loads(buf.getvalue())
""")
    assert f"# raiz-ruta: {json.dumps(str(arbol_temporal))}" in out["texto"], out["texto"][:400]
    assert f"# raiz-modo: {modo_antes:04o}" in out["texto"]
    assert "# raiz-acl: " in out["texto"]
    assert out["modo_aplicado"] == "0o770"
    # La raiz tenia bits de otros (0775): --deshacer restaura dueño y grupo del modo pero NO los de otros,
    # y avisa. Nunca reabre.
    assert out["rc"] == 0 and out["json"]["raiz"]["restaurada"] is True, out["json"]
    assert "bits de otros" in out["json"]["raiz"]["detalle"] and "no se restauran" in out["json"]["raiz"]["detalle"]
    assert _foto(arbol_temporal)[2] == modo_antes & ~0o007 == 0o770, "--deshacer reabrió la raíz a otros"


def test_deshacer_no_confia_en_un_respaldo_forjado_ni_incompleto(arbol_temporal, _identidades):
    proyectos = arbol_temporal / "proyectos"
    subprocess.run(["sudo", "-n", "chmod", "775", str(arbol_temporal)], check=True)
    out = _driver_respaldo(f"""
proy = pp.Path({str(proyectos)!r})
raiz = {str(arbol_temporal)!r}
d = pp.RUTA_RESPALDOS
marcador = pp._MARCADOR_FIN_RESPALDO
casos = {{
  "otra-ruta": '# raiz-ruta: "/etc"\\n# raiz-modo: 0777\\n\\n' + marcador,
  "modo-invalido": '# raiz-ruta: ' + json.dumps(raiz) + '\\n# raiz-modo: 9999\\n\\n' + marcador,
  "sin-marcador": '# raiz-ruta: ' + json.dumps(raiz) + '\\n# raiz-modo: 0777\\n\\n',
}}
for nombre, texto in casos.items():
    (d / f"proyectos-{{nombre}}.acl").write_text(texto)
pp._recorrer(proy, accion="aplicar")
salida["antes"] = oct(os.stat(raiz).st_mode & 0o7777)
ok, motivo = pp._restaurar_raiz_desde_respaldo(proy)
salida["ok"] = ok
salida["motivo"] = motivo
salida["despues"] = oct(os.stat(raiz).st_mode & 0o7777)
""")
    assert out["ok"] is False, out
    assert out["antes"] == out["despues"] == "0o770", out


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
