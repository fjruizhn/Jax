"""El lanzador de jaula del Faro (plan 0.3c, R3): arma y arranca el bwrap de UNA ejecucion.

LO QUE RECIBE LA JAULA (MAJOR-2/3). Por BIND DE ARCHIVO, solo: su socket (`/faro/puerto.sock`, lectura y
escritura), su token (`/faro/token`, solo lectura) y el rele (`/faro/relay/jax/faro/relay.py`, solo lectura; es
solo biblioteca estandar y Python lo importa como paquete de espacios de nombres con `PYTHONPATH=/faro/relay`),
mas `/usr` de solo lectura, `/proc`, `/dev` y un `/tmp` propio. El directorio de sockets (`/run/faro`, 0700 de
`faro`) NO se monta ni se menciona: el argv solo nombra los dos archivos de ESA ejecucion. Sin red, sin
IPC/PID/UTS compartidos, entorno vacio salvo cuatro variables, sesion propia, muere con su padre. El contenido del
token jamas viaja por argv ni por entorno: el rele lo lee del archivo.

COMO SE LLEGA A UN UID PROPIO (medido en hall9000, bubblewrap 0.11.1, 2026-10-02). El uid de la jaula lo pone la
ELEVACION (`JAX_FARO_JAULA_ELEVAR`, que lleva `{uid}`; p. ej. `/usr/bin/sudo -n -u #{uid} --`), no el argv:
  - un bwrap que corre como root NO puede cambiar de uid dentro: el perfil de AppArmor `bwrap//&unpriv_bwrap`
    confina a sus hijos y `setresuid` da EPERM (con `--cap-add ALL` tambien);
  - un bwrap sin privilegios corre con el uid de quien lo lanza, y su namespace de usuario no cambia el uid que
    ve el kernel; por eso `SO_PEERCRED` del Puerto ve ese uid. No se usa `--unshare-user` ni `--uid`.
El uid sale de la `Ejecucion` (`uid_esperado`, fijado por el orquestador y validado por el canal de control) y
aqui se vuelve a validar: no root, no el del servicio, no el del orquestador.

LO QUE ESTE MODULO NO RESUELVE (declarado, no escondido): un bwrap que corre como el uid de la jaula tiene que
abrir las FUENTES de los binds en `/run/faro`, que es 0700 de `faro`, y `sudo` cierra los descriptores heredados
(no se pueden pasar ya abiertos con `--bind-fd`). Quien arranque con un uid distinto necesita abrir esas dos
rutas por la jaula antes de bajar de uid: es el elevador del scope por ejecucion (0.6) o su ayudante (0.10),
una decision de diseno que el plan no cierra. Ver «Desviaciones» del plan.

BITACORA PRIMERO: `jaula_lanzada` se anota antes de arrancar (si no se puede anotar, no se lanza); un fallo de
arranque queda como `jaula_fallo` y se propaga.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import shlex
import stat
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import relay as _relay
from .bitacora import Bitacora
from .config import ConfigFaroInvalida

DESTINO_SOCKET = "/faro/puerto.sock"
DESTINO_TOKEN = "/faro/token"
DESTINO_RELAY_RAIZ = "/faro/relay"
DESTINO_RELAY = "/faro/relay/jax/faro/relay.py"
RELAY_FUENTE = str(Path(_relay.__file__))
ENTORNO_LANZADOR = {"PATH": "/usr/bin:/bin"}
_UID_MAX = 2 ** 32 - 2
_ELEVAR_EJEMPLO = ("/usr/bin/sudo", "-n", "-u", "#{uid}", "--")
_COMANDO_EJEMPLO = ("/usr/bin/python3", "-m", "jax.faro.relay", "--socket", DESTINO_SOCKET, "--token-file", DESTINO_TOKEN)


@dataclass(frozen=True)
class ConfigJaula:
    """`JAX_FARO_BWRAP` (ruta absoluta del binario) y `JAX_FARO_JAULA_ELEVAR` (la elevacion, con `{uid}`) son
    obligatorias: sin ellas no hay lanzador (falla cerrado)."""
    bwrap: Path
    elevar: tuple[str, ...]

    def __post_init__(self) -> None:
        ruta = Path(self.bwrap)
        if not ruta.is_absolute():
            raise ConfigFaroInvalida(f"JAX_FARO_BWRAP tiene que ser una ruta absoluta, no {str(ruta)!r}")
        try:
            st = os.stat(ruta)
        except OSError as exc:
            raise ConfigFaroInvalida(f"JAX_FARO_BWRAP no se puede leer ({type(exc).__name__}): {ruta}") from exc
        if not stat.S_ISREG(st.st_mode) or not os.access(ruta, os.X_OK):
            raise ConfigFaroInvalida(f"JAX_FARO_BWRAP no es un ejecutable: {ruta}")
        if not self.elevar or not all(isinstance(e, str) and e for e in self.elevar):
            raise ConfigFaroInvalida("JAX_FARO_JAULA_ELEVAR esta vacia: sin elevacion la jaula correria con el uid de `faro`")
        if not Path(self.elevar[0]).is_absolute():
            raise ConfigFaroInvalida("el primer elemento de JAX_FARO_JAULA_ELEVAR tiene que ser una ruta absoluta (nada de buscar en PATH)")
        if not any("{uid}" in e for e in self.elevar):
            raise ConfigFaroInvalida(
                "JAX_FARO_JAULA_ELEVAR tiene que llevar `{uid}`: sin el uid de la jaula en la elevacion, bwrap correria con el de `faro`")

    @classmethod
    def desde_entorno(cls, env: Mapping[str, str]) -> "ConfigJaula":
        def pedir(nombre: str) -> str:
            valor = (env.get(nombre) or "").strip()
            if not valor:
                raise ConfigFaroInvalida(f"{nombre} no esta definida: sin ella no hay lanzador de jaula")
            return valor
        try:
            elevar = tuple(shlex.split(pedir("JAX_FARO_JAULA_ELEVAR")))
        except ValueError as exc:
            raise ConfigFaroInvalida("JAX_FARO_JAULA_ELEVAR no se puede interpretar (comillas sin cerrar)") from exc
        return cls(bwrap=Path(pedir("JAX_FARO_BWRAP")), elevar=elevar)


def validar_uid_jaula(uid: object, *, prohibidos: Iterable[int] = ()) -> int:
    """El uid de una jaula: un entero real, que no sea root, ni el del servicio, ni uno de `prohibidos`."""
    if isinstance(uid, bool) or not isinstance(uid, int) or not 1 <= uid <= _UID_MAX:
        raise ConfigFaroInvalida("el uid de la jaula tiene que ser un entero entre 1 y 2**32-2")
    if uid == os.geteuid():
        raise ConfigFaroInvalida(f"el uid de la jaula ({uid}) es el del servicio: cualquier proceso de `faro` entraria como la jaula")
    if uid in set(prohibidos):
        raise ConfigFaroInvalida(f"el uid de la jaula ({uid}) es uno de los reservados (orquestador)")
    return uid


def _ruta(valor: object, nombre: str) -> str:
    if not isinstance(valor, str) or not valor or "\x00" in valor or not valor.startswith("/"):
        raise ConfigFaroInvalida(f"{nombre} tiene que ser una ruta absoluta de texto sin NUL")
    return valor


def _montajes(ruta_socket: str, ruta_token: str, relay: str) -> list[str]:
    return [
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net", "--unshare-cgroup-try",
        "--die-with-parent", "--new-session", "--clearenv",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        "--bind", ruta_socket, DESTINO_SOCKET,
        "--ro-bind", ruta_token, DESTINO_TOKEN,
        "--ro-bind", relay, DESTINO_RELAY,
        "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", "/tmp",
        "--setenv", "PYTHONPATH", DESTINO_RELAY_RAIZ, "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--chdir", "/tmp",
    ]


def argv_montajes(ruta_socket: str, ruta_token: str, *, relay: str | None = None) -> list[str]:
    """Las opciones de bwrap (sin el `--` ni el comando): funcion pura de las DOS rutas de la ejecucion."""
    return _montajes(_ruta(str(ruta_socket), "la ruta del socket"), _ruta(str(ruta_token), "la ruta del token"),
                     _ruta(relay if relay is not None else RELAY_FUENTE, "la ruta del rele"))


def _validar_comando(comando: object) -> list[str]:
    if (not isinstance(comando, Sequence) or isinstance(comando, (str, bytes)) or not comando
            or not all(isinstance(c, str) and "\x00" not in c for c in comando)
            or not comando[0].startswith("/")):
        raise ConfigFaroInvalida("el comando de la jaula tiene que ser una lista de textos cuyo primer elemento es un ejecutable absoluto")
    return list(comando)


class LanzadorJaula:
    def __init__(self, cfg: ConfigJaula, bitacora: Bitacora, *, uids_prohibidos: Iterable[int] = (),
                 crear_proceso: Callable | None = None):
        self._cfg = cfg
        self._bitacora = bitacora
        self._prohibidos = frozenset(uids_prohibidos)
        self._crear_proceso = crear_proceso if crear_proceso is not None else asyncio.create_subprocess_exec

    def argv(self, ejecucion, ruta_socket: str, ruta_token: str, comando: Sequence[str]) -> list[str]:
        """El argv de bwrap (sin la elevacion ni el binario) de ESA ejecucion."""
        validar_uid_jaula(ejecucion.uid_esperado, prohibidos=self._prohibidos)
        return [*argv_montajes(ruta_socket, ruta_token), "--", *_validar_comando(comando)]

    def argv_completo(self, ejecucion, ruta_socket: str, ruta_token: str, comando: Sequence[str]) -> list[str]:
        cuerpo = self.argv(ejecucion, ruta_socket, ruta_token, comando)
        uid = str(ejecucion.uid_esperado)
        return [*(e.replace("{uid}", uid) for e in self._cfg.elevar), str(self._cfg.bwrap), *cuerpo]

    async def lanzar(self, srv, comando: Sequence[str]):
        """Anota `jaula_lanzada` y arranca el proceso (que se devuelve). `srv` es el `ServidorPuerto` de la ejecucion."""
        e = srv.ejecucion
        argv = self.argv_completo(e, str(srv.ruta_socket), str(srv.ruta_token), comando)    # valida ANTES de anotar o arrancar
        await self._bitacora.registrar(
            "jaula_lanzada", run_id=e.run_id, id_correlacion=e.id_correlacion, uid_jaula=e.uid_esperado, motor=e.motor,
            entry_point=e.entry_point, hash_argv=hashlib.sha256("\x00".join(argv).encode()).hexdigest())
        try:
            return await self._crear_proceso(
                *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=dict(ENTORNO_LANZADOR), start_new_session=True)
        except Exception as exc:
            try:
                await self._bitacora.registrar("jaula_fallo", run_id=e.run_id, id_correlacion=e.id_correlacion,
                                               uid_jaula=e.uid_esperado, motivo=type(exc).__name__)
            except Exception:  # fail-closed: el error de arranque es el que sube; solo la anotacion fallo
                pass
            raise


# --------------------------------------------------------------------------- #
# el ejemplo versionado (ops/faro/bwrap-ejemplo.txt), GENERADO por esta funcion #
# --------------------------------------------------------------------------- #

_ENCABEZADO = """\
# ops/faro/bwrap-ejemplo.txt -- GENERADO por jax.faro.jaula.ejemplo(); tests/test_faro_lanzador.py vigila que no se
# desvie del codigo (si cambia el argv, se regenera con:  python -c "from jax.faro.jaula import ejemplo; print(end=ejemplo())").
#
# El argv de bwrap de UNA ejecucion (plan 0.3c). <run_id> y <uid_jaula> los pone el lanzador; la elevacion
# (JAX_FARO_JAULA_ELEVAR) es lo que da a la jaula su uid propio, distinto del de `faro` y del de root.
# La jaula recibe por bind de archivo SOLO su socket, su token y el rele; el directorio /run/faro no se monta.
# Dentro, el cliente MCP lanza:  python -m jax.faro.relay --socket /faro/puerto.sock --token-file /faro/token
#
# PENDIENTE DE DISENO (no resuelto en este paso): quien corre bwrap con otro uid no puede abrir /run/faro (0700 de
# `faro`) para montar esas dos fuentes, y sudo cierra los descriptores heredados (no sirve --bind-fd). El elevador
# (0.6/0.10) tiene que abrirlas antes de bajar de uid.
#
"""


def _lineas(tokens: list[str]) -> list[str]:
    grupos: list[list[str]] = []
    for t in tokens:
        if t.startswith("--") or not grupos:
            grupos.append([t])
        else:
            grupos[-1].append(t)
    return [" ".join(shlex.quote(x) for x in g) for g in grupos]


def ejemplo() -> str:
    elevar = [e.replace("{uid}", "<uid_jaula>") for e in _ELEVAR_EJEMPLO]
    cuerpo = [*_montajes("/run/faro/<run_id>.sock", "/run/faro/<run_id>.token", "<ruta de jax/faro/relay.py>"), "--"]
    lineas = [" ".join(shlex.quote(x) for x in elevar), "/usr/bin/bwrap", *_lineas(cuerpo)]
    return _ENCABEZADO + " \\\n  ".join(lineas) + " \\\n  " + " ".join(_COMANDO_EJEMPLO) + "\n"
