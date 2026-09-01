"""
Contenimiento del sandbox de Hyde, EJERCITADO -- no descrito.

POR QUE EXISTE ESTE ARCHIVO. `hyde_sandbox.py` afirma propiedades de seguridad
en su docstring y en DEUDA.md: los repos van en solo-lectura, el `$HOME` real
no se expone, el entorno del padre no se hereda. Esas propiedades **se
verificaron una sola vez a mano** cuando se escribio el sandbox (2026-08-23) y
despues nadie las volvio a tocar: no habia un solo test que las ejerciera.
`_hyde_sandbox_test.py`, el unico que existia, cubre el flock y los timeouts --
la SERIALIZACION, no el CONFINAMIENTO.

Una propiedad de seguridad verificada una vez y nunca mas es una propiedad
supuesta. Cambiar un `--ro-bind` por un `--bind` en una linea, o agregar un
`--bind` de conveniencia, no rompe ningun test hoy: el sandbox sigue
arrancando, Hyde sigue funcionando, y el confinamiento se perdio en silencio.
Ese es exactamente el modo de falla que este repo viene persiguiendo en otros
mecanismos.

COMO ESTAN ESCRITOS. Cada test EJECUTA un ataque real adentro del sandbox y
mira el resultado; ninguno inspecciona el `argv` para adivinar que hubiera
pasado. Cuando el ataque es una escritura, la afirmacion se hace **sobre el
host**: un `touch` puede "fallar" y aun asi haber dejado el archivo, y ese caso
es peor que el que se estaba probando.

EL CONTROL POSITIVO NO ES DECORATIVO. `test_el_workspace_si_es_escribible`
existe porque sin el, TODOS los tests de bloqueo pasarian igual si bwrap no
arrancara en absoluto -- verde por la razon equivocada, que es la forma mas
cara de estar equivocado.

NO USA LAS RUTAS REALES de hall9000: monkeypatchea las constantes del modulo a
directorios temporales. Un test que solo corre en la maquina de Fernando no
corre en CI, y hoy mismo (2026-09-01) esa confusion costo tres tandas de
arreglos a ciegas en jax-platform.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

import hyde_sandbox

pytestmark = pytest.mark.skipif(
    not shutil.which("bwrap"), reason="bwrap no instalado; el sandbox no se puede ejercitar"
)

_SECRETO = "valor-secreto-que-no-debe-cruzar"


@pytest.fixture
def caja(tmp_path, monkeypatch):
    """Un sandbox completo sobre directorios temporales.

    Reemplaza las rutas reales de hall9000 por copias de juguete con la misma
    FORMA (dos repos de solo lectura + un workspace escribible), asi el test
    ejercita el mecanismo y no la maquina.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "archivo_del_repo.txt").write_text("contenido original\n")

    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    (repo2 / "otro.txt").write_text("otro\n")

    afuera = tmp_path / "afuera"
    afuera.mkdir()
    (afuera / "secreto.env").write_text("API_KEY=no-se-debe-leer\n")

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    monkeypatch.setattr(hyde_sandbox, "REAL_JAX_REPO", str(repo))
    monkeypatch.setattr(hyde_sandbox, "REAL_JAX_PLATFORM_REPO", str(repo2))
    monkeypatch.setattr(hyde_sandbox, "REAL_NVM_DIR", str(tmp_path / "no-existe-nvm"))
    monkeypatch.setattr(hyde_sandbox, "REAL_CREDENTIALS", str(tmp_path / "no-existe-cred"))
    monkeypatch.setattr(hyde_sandbox, "_TEMPLATE_DIR", tmp_path / "template")

    class Caja:
        repo = None
        repo2 = None
        afuera = None
        workspace = None

        def correr(self, sh: str, env: dict | None = None) -> str:
            argv = hyde_sandbox.wrap_hyde_command(["/bin/sh", "-c", sh], str(workspace))
            entorno = dict(os.environ)
            entorno.update(env or {})
            r = subprocess.run(
                argv, capture_output=True, text=True, timeout=120, env=entorno
            )
            return (r.stdout + r.stderr).strip()

    c = Caja()
    c.repo, c.repo2, c.afuera, c.workspace = repo, repo2, afuera, workspace
    return c


# ---------------------------------------------------------------------------
# Control positivo -- va primero a proposito
# ---------------------------------------------------------------------------

def test_el_workspace_si_es_escribible(caja):
    """Si esto falla, TODOS los tests de bloqueo de abajo son verdes falsos:
    estarian pasando porque el sandbox no arranco, no porque contenga."""
    salida = caja.correr("touch escrito_por_hyde && echo ESCRIBIO || echo BLOQUEADO")
    assert "ESCRIBIO" in salida, salida
    assert (caja.workspace / "escrito_por_hyde").exists(), (
        "el workspace tiene que ser escribible DE VERDAD, no solo reportar exito"
    )


# ---------------------------------------------------------------------------
# Los repos, en solo lectura
# ---------------------------------------------------------------------------

def test_no_se_puede_crear_un_archivo_en_el_repo(caja):
    caja.correr("touch %s/PWNED" % caja.repo)
    assert not (caja.repo / "PWNED").exists(), (
        "el repo quedo escribible desde el sandbox -- revisar que siga en --ro-bind"
    )


def test_no_se_puede_modificar_un_archivo_del_repo(caja):
    caja.correr("echo pisado > %s/archivo_del_repo.txt" % caja.repo)
    assert (caja.repo / "archivo_del_repo.txt").read_text() == "contenido original\n"


def test_no_se_puede_borrar_del_repo(caja):
    caja.correr("rm -f %s/archivo_del_repo.txt" % caja.repo)
    assert (caja.repo / "archivo_del_repo.txt").exists(), "borro un archivo del repo"


def test_el_segundo_repo_tambien_esta_en_solo_lectura(caja):
    """No alcanza con probar uno: son dos binds distintos y el segundo es
    condicional (`if os.path.isdir`), asi que puede desaparecer solo."""
    caja.correr("touch %s/PWNED" % caja.repo2)
    assert not (caja.repo2 / "PWNED").exists()


def test_el_repo_si_se_puede_LEER(caja):
    """La contracara: el confinamiento no debe romper el caso de uso. Hyde
    tiene que poder leer codigo real -- es la decision explicita de Fernando
    (opcion b, sesion del sandbox 2026-08-22)."""
    salida = caja.correr("cat %s/archivo_del_repo.txt" % caja.repo)
    assert "contenido original" in salida, salida


# ---------------------------------------------------------------------------
# Lo que esta afuera de los binds no existe
# ---------------------------------------------------------------------------

def test_un_archivo_fuera_de_los_binds_no_es_legible(caja):
    """El caso real es /etc/jax/.env (root:fruiz 0660 -- el grupo `fruiz` SI
    tiene lectura en el host). Aca se representa con un archivo equivalente
    fuera de todo bind."""
    salida = caja.correr("cat %s/secreto.env 2>&1 || echo BLOQUEADO" % caja.afuera)
    assert "no-se-debe-leer" not in salida, salida
    assert "BLOQUEADO" in salida or "No such file" in salida, salida


def test_el_entorno_del_padre_no_se_hereda(caja):
    """--clearenv. El proceso padre (jax-las-manos) tiene EnvironmentFile=
    /etc/jax/.env cargado: sin esto, las claves viajan como variables de
    entorno aunque el filesystem este cerrado. Es la leccion de metodo
    `feedback-verificar-herencia-entorno-no-solo-filesystem`."""
    salida = caja.correr(
        'test -n "$SECRETO_DEL_PADRE" && echo HEREDO || echo BLOQUEADO',
        env={"SECRETO_DEL_PADRE": _SECRETO},
    )
    assert "BLOQUEADO" in salida, salida


def test_el_secreto_del_padre_no_aparece_en_ninguna_variable(caja):
    """Mas fuerte que el anterior: no alcanza con que no este esa variable --
    el valor no debe aparecer en NINGUNA. Un --setenv de conveniencia agregado
    manana podria reintroducirlo con otro nombre."""
    salida = caja.correr("env", env={"SECRETO_DEL_PADRE": _SECRETO})
    assert _SECRETO not in salida, "el valor del padre cruzo al sandbox"


# ---------------------------------------------------------------------------
# El $HOME del sandbox
# ---------------------------------------------------------------------------

def test_el_home_es_el_virtual_no_el_real(caja):
    salida = caja.correr("echo $HOME")
    assert salida.strip() == hyde_sandbox.SANDBOX_HOME, salida


def test_el_home_del_sandbox_es_efimero(caja):
    """tmpfs fresco en cada invocacion: nada persiste entre corridas de Hyde."""
    caja.correr("echo rastro > $HOME/persistente.txt")
    salida = caja.correr("cat $HOME/persistente.txt 2>&1 || echo NO_QUEDO_NADA")
    assert "rastro" not in salida, salida


def test_el_directorio_padre_de_los_binds_solo_expone_los_montajes(caja):
    """`ls` del padre de los repos FUNCIONA dentro del sandbox -- bwrap tiene
    que crear ese directorio para colgar los binds. Lo que importa es que solo
    muestre los montajes y no el contenido real: se fija aca para que nadie lo
    lea como una fuga ni lo pierda de vista si algun dia SI lo fuera."""
    padre = str(caja.repo.parent)
    salida = caja.correr("ls -a %s" % padre)
    visibles = {x for x in salida.split() if x not in (".", "..")}
    assert visibles <= {"repo", "repo2", "workspace"}, (
        f"se ve mas que los montajes: {visibles}"
    )
    assert "afuera" not in visibles, "un directorio hermano no bindeado quedo visible"


# ---------------------------------------------------------------------------
# Fail-closed y el limite conocido
# ---------------------------------------------------------------------------

def test_sin_bwrap_falla_cerrado(monkeypatch, tmp_path):
    """P10: sin confinamiento, Hyde NO arranca. Nunca degrada a ejecucion
    pelada."""
    monkeypatch.setattr(hyde_sandbox, "_BWRAP_BIN", str(tmp_path / "bwrap-que-no-existe"))
    with pytest.raises(hyde_sandbox.SandboxUnavailable):
        # El comando envuelto es irrelevante aca: wrap_hyde_command lanza antes
        # de mirarlo. Va /bin/true y NO el literal "claude" a proposito -- este
        # archivo lanza subprocess, y el scanner
        # policy/tests/test_claude_subprocess_solo_via_sandbox.py marca como
        # violacion cualquier archivo que combine las dos cosas fuera de
        # hyde_sandbox.py. El scanner tiene razon: ese es exactamente el patron
        # que vigila, y no hay motivo para pedirle una excepcion.
        hyde_sandbox.wrap_hyde_command(["/bin/true"], str(tmp_path))


def test_la_red_compartida_esta_declarada_como_limite_conocido(caja):
    """PIN DEL LIMITE, no de una virtud.

    El sandbox corre con `--share-net`: red del host completa. Ver el comentario
    de abajo -- la explicacion va ahi y no en esta docstring a proposito.
    """
    # POR QUE ESTA PROSA VIVE EN UN COMENTARIO Y NO EN LA DOCSTRING: el scanner
    # policy/tests/test_claude_subprocess_solo_via_sandbox.py marca cualquier
    # archivo que lance un subproceso Y mencione el nombre del CLI en un
    # LITERAL DE STRING del AST -- y una docstring es un literal. Este archivo
    # lanza subprocesos, asi que nombrarlo aca lo convertiria en violacion. Se
    # reescribe la prosa en vez de exentar el archivo o aflojar el scanner:
    # es un scanner de seguridad y el costo de esquivarlo es cero.
    #
    # EL LIMITE, entonces: bwrap no puede acotar la red por dominio/IP -- es
    # namespace de red compartido o nada, y --unshare-net dejaria al
    # subproceso confinado sin poder llegar a la API. Acotar de verdad necesita
    # configuracion de red con privilegios (reglas nftables por UID, o un netns
    # con veth): decision de infraestructura, no un cambio de este archivo.
    # Sigue abierto en DEUDA.md.
    #
    # Este test existe para que ese limite este ESCRITO Y EJERCITADO, no
    # supuesto: si algun dia alguien lo cierra, se pone en rojo y lo obliga a
    # venir aca a actualizar la deuda en vez de dejarla mintiendo.
    # /bin/true y no el literal "claude": ver el comentario en
    # test_sin_bwrap_falla_cerrado. Lo que se afirma es el argv de bwrap, que
    # no depende del comando envuelto.
    argv = hyde_sandbox.wrap_hyde_command(["/bin/true"], str(caja.workspace))
    assert "--share-net" in argv, (
        "la red dejo de estar compartida -- si se acoto de verdad, actualizar "
        "DEUDA.md (el item de Hyde) y este test"
    )
