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
    # Credencial por defecto de la caja: el token OAuth -- la única vía
    # soportada hoy en producción (decisión de Fernando, 2026-09-27, ver
    # hyde_sandbox.py). Sin esto wrap_hyde_command fallaría cerrado
    # (HydeCredentialUnavailable) antes de construir el argv, y NINGÚN test
    # de este archivo podría ejercitar el sandbox -- son pruebas de
    # confinamiento, no de la credencial en sí. Los tests que sí ejercitan
    # el mecanismo de credencial (más abajo) pisan esto con su propio
    # monkeypatch.
    monkeypatch.setenv(hyde_sandbox.HYDE_OAUTH_TOKEN_ENV, "token-de-prueba-de-la-caja")

    class Caja:
        repo = None
        repo2 = None
        afuera = None
        workspace = None

        def correr(self, sh: str, env: dict | None = None) -> str:
            """Arma el sandbox con wrap_hyde_command y lo ejecuta pasando el
            `env` que esa función devuelve TAL CUAL a subprocess.run --
            exactamente lo que hace el único punto de entrada aprobado en
            producción (ver hyde_sandbox.py) para B-1 (auditoría
            adversarial 2026-09-27): nunca fusionado con el os.environ
            real. Antes esa frontera la ponía `--clearenv` DENTRO del argv
            de bwrap y acá se le pasaba el os.environ completo (más lo que
            pidiera el test); ahora bwrap no usa --clearenv/--setenv en
            absoluto, así que la frontera es el `env` de retorno.

            El parámetro `env` de este método simula variables que YA
            estarían en el os.environ del proceso real que arma el sandbox
            (jaxsvc, con /etc/jax/.env cargado) -- se inyectan
            temporalmente en os.environ antes de llamar a
            wrap_hyde_command (que sí lee os.environ, para
            HYDE_OAUTH_TOKEN_ENV) y se restauran después, para poder
            probar que NO terminan en el `env` que esa función devuelve."""
            previos = {k: os.environ.get(k) for k in (env or {})}
            os.environ.update(env or {})
            try:
                argv, sandbox_env = hyde_sandbox.wrap_hyde_command(
                    ["/bin/sh", "-c", sh], str(workspace)
                )
            finally:
                for k, v in previos.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            r = subprocess.run(
                argv, capture_output=True, text=True, timeout=120, env=sandbox_env
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
    """El proceso padre (jax-las-manos) tiene EnvironmentFile=/etc/jax/.env
    cargado: sin esta frontera, las claves viajan como variables de
    entorno aunque el filesystem este cerrado. Es la leccion de metodo
    `feedback-verificar-herencia-entorno-no-solo-filesystem`. Desde B-1
    (auditoría adversarial 2026-09-27) la frontera ya no es --clearenv
    dentro del argv de bwrap, sino el `env` mínimo que wrap_hyde_command
    devuelve y que Caja.correr pasa TAL CUAL a subprocess.run -- este test
    sigue probando la misma propiedad, con el mecanismo nuevo."""
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
    argv, _env = hyde_sandbox.wrap_hyde_command(["/bin/true"], str(caja.workspace))
    assert "--share-net" in argv, (
        "la red dejo de estar compartida -- si se acoto de verdad, actualizar "
        "DEUDA.md (el item de Hyde) y este test"
    )


# ---------------------------------------------------------------------------
# Credencial de Hyde: CLAUDE_CODE_OAUTH_TOKEN (cuenta Max) o el archivo,
# nunca los dos a la vez ni ninguno -- decisión de Fernando 2026-09-27 (ver
# hyde_sandbox.py). El hallazgo que motiva esto: corriendo como `jaxsvc`
# (desde 2026-09-17) REAL_CREDENTIALS es 600 fruiz:fruiz -- `os.path.isfile`
# daba True (alcanza con poder recorrer directorios) y el archivo se
# montaba igual, pero `jaxsvc` no podía leerlo adentro del sandbox: un bind
# inútil, Hyde sin credencial, en silencio.
#
# CORREGIDO 2026-09-27 (auditoría adversarial, hallazgo B-1 BLOCK sobre el
# primer intento): el token NUNCA va en el argv de bwrap -- un intento
# anterior lo pasaba con `--setenv CLAUDE_CODE_OAUTH_TOKEN <valor>`, y el
# argv completo de un proceso es legible por CUALQUIER usuario del host vía
# /proc/<pid>/cmdline (world-readable por defecto; a diferencia de
# /proc/<pid>/environ, que exige el mismo UID o CAP_SYS_PTRACE) --
# verificado en hall9000: /proc no tiene hidepid, y fruiz/axioma ven con
# `ps` los procesos de jaxsvc. Ahora wrap_hyde_command devuelve `(argv,
# env)`: el token, si existe, va SÓLO en `env`, y ni bwrap usa ya
# --clearenv/--setenv para nada -- ver el docstring de wrap_hyde_command.
# ---------------------------------------------------------------------------

_TOKEN_DE_PRUEBA = "oauth-token-de-prueba-no-es-una-credencial-real"


def test_con_oauth_token_lo_setea_y_no_bindea_el_archivo(caja, tmp_path, monkeypatch, caplog):
    """El token GANA sobre el archivo: si está en el entorno, NO se monta
    REAL_CREDENTIALS aunque exista y sea legible -- no hace falta, y evita
    depender de que este proceso pueda leer un archivo que puede no ser
    suyo."""
    credencial_de_mentira = tmp_path / "credencial-que-no-debe-montarse.json"
    credencial_de_mentira.write_text('{"secreto":"no-se-debe-leer-esto"}\n', encoding="utf-8")
    monkeypatch.setattr(hyde_sandbox, "REAL_CREDENTIALS", str(credencial_de_mentira))
    monkeypatch.setenv(hyde_sandbox.HYDE_OAUTH_TOKEN_ENV, _TOKEN_DE_PRUEBA)

    argv, env = hyde_sandbox.wrap_hyde_command(["/bin/true"], str(caja.workspace))
    assert env.get(hyde_sandbox.HYDE_OAUTH_TOKEN_ENV) == _TOKEN_DE_PRUEBA, env
    assert not any(str(credencial_de_mentira) in a for a in argv), (
        "el archivo de credenciales se montó igual, aunque había token"
    )

    # Y EJERCITADO -- no sólo el argv/env: adentro del sandbox, todo $HOME
    # (que es tmpfs) se vuelca entero, sin nombrar ningún archivo puntual --
    # policy/tests/test_claude_subprocess_solo_via_sandbox.py marca como
    # violación cualquier literal de este archivo que combine un subproceso
    # con el nombre del CLI en un string, y una ruta contra el nombre real
    # del directorio de config lo dispararía igual sin necesidad. Se afirma
    # sobre el volcado completo, en Python, con valores que sí pueden
    # nombrarse (el token y el secreto del archivo que no debe montarse).
    salida = caja.correr('env; echo ---; find "$HOME" -type f -exec cat {} \\;')
    assert f"{hyde_sandbox.HYDE_OAUTH_TOKEN_ENV}={_TOKEN_DE_PRUEBA}" in salida, salida
    assert "no-se-debe-leer-esto" not in salida, salida

    assert not any(_TOKEN_DE_PRUEBA in r.getMessage() for r in caplog.records), (
        "el token apareció en un log -- nunca debe loguearse"
    )


def test_el_token_nunca_aparece_en_el_argv(caja, monkeypatch):
    """B-1 BLOCK (auditoría adversarial 2026-09-27): regresión dedicada del
    hallazgo que rechazó el primer intento. El argv completo de bwrap es
    legible por cualquier usuario del host vía /proc/<pid>/cmdline -- el
    token va SÓLO en el `env` de retorno, jamás en ningún elemento del
    argv, con o sin --setenv de por medio."""
    monkeypatch.setenv(hyde_sandbox.HYDE_OAUTH_TOKEN_ENV, _TOKEN_DE_PRUEBA)
    argv, env = hyde_sandbox.wrap_hyde_command(["/bin/true"], str(caja.workspace))
    assert not any(_TOKEN_DE_PRUEBA in a for a in argv), (
        f"el token apareció en el argv de bwrap: {argv}"
    )
    assert not any("--setenv" in a or "--clearenv" in a for a in argv), (
        "bwrap ya no debería usar --setenv/--clearenv en absoluto"
    )
    assert env[hyde_sandbox.HYDE_OAUTH_TOKEN_ENV] == _TOKEN_DE_PRUEBA


def test_sin_token_bindea_el_archivo_si_es_legible(caja, tmp_path, monkeypatch):
    """Comportamiento de hoy, sin token: el archivo se monta -- pero ahora
    la condición es LEGIBLE (`os.access(R_OK)`), no sólo que exista."""
    monkeypatch.delenv(hyde_sandbox.HYDE_OAUTH_TOKEN_ENV, raising=False)
    credencial_legible = tmp_path / "credencial-legible.json"
    credencial_legible.write_text('{"contenido":"credencial-real-de-mentira"}\n', encoding="utf-8")
    monkeypatch.setattr(hyde_sandbox, "REAL_CREDENTIALS", str(credencial_legible))

    argv, env = hyde_sandbox.wrap_hyde_command(["/bin/true"], str(caja.workspace))
    assert hyde_sandbox.HYDE_OAUTH_TOKEN_ENV not in env, env
    assert str(credencial_legible) in argv, "el archivo legible no se montó"

    # Mismo criterio que en el test anterior: se vuelca $HOME entero (sin
    # nombrar el directorio de config en un literal) y se afirma en Python.
    salida = caja.correr('find "$HOME" -type f -exec cat {} \\;')
    assert "credencial-real-de-mentira" in salida, salida


def test_sin_token_y_archivo_no_legible_no_bindea_y_falla_cerrado(caja, tmp_path, monkeypatch, caplog):
    """El archivo EXISTE (isfile=True) pero no es legible por este proceso
    (chmod 000) -- ni token ni archivo usable: no hay bind posible y
    wrap_hyde_command falla cerrado ANTES de armar el sandbox, con un
    warning que no repite ningún dato del contenido del archivo."""
    monkeypatch.delenv(hyde_sandbox.HYDE_OAUTH_TOKEN_ENV, raising=False)
    credencial_sin_permiso = tmp_path / "credencial-sin-permiso.json"
    credencial_sin_permiso.write_text('{"secreto":"inalcanzable-para-este-proceso"}\n', encoding="utf-8")
    os.chmod(credencial_sin_permiso, 0o000)
    monkeypatch.setattr(hyde_sandbox, "REAL_CREDENTIALS", str(credencial_sin_permiso))
    try:
        assert os.path.isfile(credencial_sin_permiso), "isfile debe seguir viendo el archivo"
        assert not os.access(credencial_sin_permiso, os.R_OK), "el chmod 000 no bloqueó la lectura"

        with caplog.at_level("WARNING", logger="hyde_sandbox"):
            with pytest.raises(hyde_sandbox.HydeCredentialUnavailable):
                hyde_sandbox.wrap_hyde_command(["/bin/true"], str(caja.workspace))

        assert any("credencial" in r.getMessage().lower() for r in caplog.records), (
            "no se logueó ningún warning de credencial faltante"
        )
        assert not any("inalcanzable-para-este-proceso" in r.getMessage() for r in caplog.records)
    finally:
        os.chmod(credencial_sin_permiso, 0o600)  # para que tmp_path se pueda limpiar


def test_ninguna_otra_variable_del_padre_cruza_al_entorno_devuelto(caja, monkeypatch):
    """Regresión: agregar la vía del token no puede convertirse en "lo que
    haya en el entorno". `env` devuelto es SIEMPRE {HOME, PATH, LANG} más
    HYDE_OAUTH_TOKEN_ENV cuando está -- cualquier otra variable del proceso
    padre, aunque parezca secreta, se queda afuera, tanto del `env` como
    del `argv` (que ya no lleva --setenv en absoluto)."""
    monkeypatch.setenv(hyde_sandbox.HYDE_OAUTH_TOKEN_ENV, _TOKEN_DE_PRUEBA)
    monkeypatch.setenv("OTRO_SECRETO_DEL_PADRE_QUE_NO_DEBE_CRUZAR", _SECRETO)

    argv, env = hyde_sandbox.wrap_hyde_command(["/bin/true"], str(caja.workspace))
    assert env == {
        "HOME": hyde_sandbox.SANDBOX_HOME,
        "PATH": hyde_sandbox._SAFE_PATH,
        "LANG": "C.UTF-8",
        hyde_sandbox.HYDE_OAUTH_TOKEN_ENV: _TOKEN_DE_PRUEBA,
    }, env
    assert not any(_SECRETO in v for v in env.values()), "una variable del padre cruzó al env del sandbox"
    assert not any(a == "--setenv" for a in argv), "bwrap ya no debería usar --setenv en absoluto"
    assert not any(_SECRETO in a for a in argv), "una variable del padre cruzó al argv de la jaula"


def test_token_de_solo_espacios_cuenta_como_ausente(caja, tmp_path, monkeypatch):
    """B-3 MINOR (auditoría adversarial 2026-09-27): un token OAuth
    (HYDE_OAUTH_TOKEN_ENV) de sólo espacios (typo en /etc/jax/.env, o una
    variable declarada pero vacía con padding) no cuenta como token
    presente -- cae al archivo, igual que si la variable no existiera."""
    monkeypatch.setenv(hyde_sandbox.HYDE_OAUTH_TOKEN_ENV, "   ")
    credencial_legible = tmp_path / "credencial-legible.json"
    credencial_legible.write_text('{"contenido":"credencial-real-de-mentira"}\n', encoding="utf-8")
    monkeypatch.setattr(hyde_sandbox, "REAL_CREDENTIALS", str(credencial_legible))

    argv, env = hyde_sandbox.wrap_hyde_command(["/bin/true"], str(caja.workspace))
    assert hyde_sandbox.HYDE_OAUTH_TOKEN_ENV not in env, env
    assert str(credencial_legible) in argv, "con token de sólo espacios, debía caer al archivo"


def test_token_de_solo_espacios_y_sin_archivo_falla_cerrado(caja, monkeypatch):
    """Mismo caso que arriba, pero sin archivo de respaldo: un token de
    sólo espacios equivale a ausente, así que sin archivo legible no hay
    ninguna credencial usable -- falla cerrado, no arranca con un token
    OAuth vacío."""
    monkeypatch.setenv(hyde_sandbox.HYDE_OAUTH_TOKEN_ENV, "    ")
    monkeypatch.setattr(hyde_sandbox, "REAL_CREDENTIALS", "/no/existe/de/verdad-en-este-test.json")
    with pytest.raises(hyde_sandbox.HydeCredentialUnavailable):
        hyde_sandbox.wrap_hyde_command(["/bin/true"], str(caja.workspace))
