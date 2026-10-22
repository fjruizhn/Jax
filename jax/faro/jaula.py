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

COMO ABRE EL UID DE LA JAULA SUS FUENTES. El bwrap de la jaula corre con SU uid y tiene que abrir las dos fuentes de
sus binds en `/run/faro` (0700 de `faro`) sin poder listarlo: lo permite una ACL por ejecucion que pone el Puerto
(`acl.py`: directorio `--x`, socket `rw-`, token `r--`, solo para ese uid). `sudo` cierra los descriptores
heredados, asi que no se pasan ya abiertos con `--bind-fd`. Probado con bwrap real corriendo como otro uid.

PERFILES (auditoria de 0.3bc, MAJOR-2). Los montajes son un perfil componible: `perfil_rele` es la JAULA MINIMA DEL
RELE (sin red, sin /etc, sin workspace ni home: solo corre el rele contra su Puerto); `perfil_motor` le suma el
workspace en rw, `$HOME` en tmpfs, el paquete en ro y un /etc minimo. La RED de un motor es una decision pendiente
de 0.5/0.6: hoy el perfil de motor solo existe con `red="aislada"`. El uid de la jaula, el rango y el orquestador se
revalidan aqui (`desde_control`).

BITACORA PRIMERO: `jaula_lanzada` se anota antes de arrancar (si no se puede anotar, no se lanza); un fallo de
arranque queda como `jaula_fallo` y se propaga.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shlex
import stat
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import relay as _relay
from .bitacora import Bitacora
from .config import ConfigFaroInvalida

logger = logging.getLogger(__name__)
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


def validar_uid_jaula(uid: object, *, prohibidos: Iterable[int] = (), rango: tuple[int, int] | None = None) -> int:
    """El uid de una jaula: un entero real, que no sea root, ni el del servicio, ni uno de `prohibidos`, y dentro de
    `rango` (el de `JAX_FARO_JAULA_UID_MIN..MAX`) si se conoce."""
    if isinstance(uid, bool) or not isinstance(uid, int) or not 1 <= uid <= _UID_MAX:
        raise ConfigFaroInvalida("el uid de la jaula tiene que ser un entero entre 1 y 2**32-2")
    if rango is not None and not rango[0] <= uid <= rango[1]:
        raise ConfigFaroInvalida(f"el uid de la jaula ({uid}) esta fuera del rango configurado {rango[0]}..{rango[1]}")
    if uid == os.geteuid():
        raise ConfigFaroInvalida(f"el uid de la jaula ({uid}) es el del servicio: cualquier proceso de `faro` entraria como la jaula")
    if uid in set(prohibidos):
        raise ConfigFaroInvalida(f"el uid de la jaula ({uid}) es uno de los reservados (orquestador)")
    return uid


def _ruta(valor: object, nombre: str) -> str:
    if not isinstance(valor, str) or not valor or "\x00" in valor or not valor.startswith("/"):
        raise ConfigFaroInvalida(f"{nombre} tiene que ser una ruta absoluta de texto sin NUL")
    return valor


# /etc minimo del perfil de motor: certificados (ssl), resolucion de nombres, cuentas y nsswitch. Nunca /etc entero.
ETC_MINIMO = ("/etc/ssl", "/etc/ca-certificates", "/etc/resolv.conf", "/etc/hosts", "/etc/passwd", "/etc/group", "/etc/nsswitch.conf")
DESTINO_WORKSPACE = "/work"
DESTINO_PAQUETE = "/faro/paquete"
HOME_MOTOR = "/home/jaula"


def _construir(ruta_socket: str, ruta_token: str, relay: str, motor: dict | None) -> list[str]:
    """Los montajes comunes (aislamiento + los dos archivos de la ejecucion + el rele) y, si `motor` trae el
    workspace y el paquete, lo del perfil de motor. La RED la decide quien llama: ambos perfiles hoy la cierran."""
    home = HOME_MOTOR if motor else "/tmp"
    argv = [
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-net", "--unshare-cgroup-try",
        "--die-with-parent", "--new-session", "--clearenv",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        *(["--tmpfs", HOME_MOTOR] if motor else []),
        "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64",
        *([x for ruta in ETC_MINIMO for x in ("--ro-bind", ruta, ruta)] if motor else []),
        "--bind", ruta_socket, DESTINO_SOCKET,
        "--ro-bind", ruta_token, DESTINO_TOKEN,
        "--ro-bind", relay, DESTINO_RELAY,
        *(["--bind", motor["workspace"], DESTINO_WORKSPACE, "--ro-bind", motor["paquete"], DESTINO_PAQUETE] if motor else []),
        "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "HOME", home,
        "--setenv", "PYTHONPATH", DESTINO_RELAY_RAIZ, "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
        "--chdir", DESTINO_WORKSPACE if motor else "/tmp",
    ]
    return argv


def perfil_rele(ruta_socket: str, ruta_token: str, *, relay: str | None = None) -> list[str]:
    """LA JAULA MINIMA DEL RELE: lo justo para correr `python -m jax.faro.relay` contra SU Puerto. Sin red, sin /etc,
    sin workspace ni home. No puede correr un motor (eso es el perfil de motor). Funcion pura de las dos rutas."""
    return _construir(_ruta(str(ruta_socket), "la ruta del socket"), _ruta(str(ruta_token), "la ruta del token"),
                      _ruta(relay if relay is not None else RELAY_FUENTE, "la ruta del rele"), None)


argv_montajes = perfil_rele          # nombre anterior


def _fuera_del_directorio_de_sockets(ruta: str, ruta_socket: str, nombre: str) -> str:
    ruta = _ruta(ruta, nombre)
    w, d = PurePosixPath(ruta), PurePosixPath(ruta_socket).parent
    if w == d or d in w.parents or w in d.parents:
        raise ConfigFaroInvalida(f"{nombre} ({ruta}) es el directorio de sockets, o esta dentro o lo contiene: la jaula no lo ve")
    return ruta


def perfil_motor(ruta_socket: str, ruta_token: str, *, workspace: str, paquete: str, red: str | None,
                 relay: str | None = None) -> list[str]:
    """EL PERFIL DE MOTOR: el del rele mas lo que un motor necesita (spec §2): el workspace de la ejecucion en
    lectura y escritura (`/work`, tambien el directorio de trabajo), `$HOME` en un tmpfs propio, el paquete del
    ecosistema de solo lectura (`/faro/paquete`) y un `/etc` minimo (ssl, resolv, hosts, passwd, group, nsswitch).

    RED: DECISION PENDIENTE (0.5/0.6). Un motor necesita red; las dos salidas son `--unshare-net` con atajos por
    socket Unix (lo que hace el Puerto) o red compartida filtrada por `skuid`. Hasta decidirlo, el unico valor es
    `red="aislada"` (sin red) y cualquier otro se rechaza: no se improvisa una."""
    if red != "aislada":
        raise ConfigFaroInvalida(
            f"la red del perfil de motor ({red!r}) es una decision pendiente de 0.5/0.6 (atajos por socket Unix o red "
            "filtrada por skuid); hoy solo existe red='aislada' (sin red)")
    sock = _ruta(str(ruta_socket), "la ruta del socket")
    return _construir(sock, _ruta(str(ruta_token), "la ruta del token"), _ruta(relay if relay is not None else RELAY_FUENTE, "la ruta del rele"),
                      {"workspace": _fuera_del_directorio_de_sockets(workspace, sock, "el workspace"),
                       "paquete": _fuera_del_directorio_de_sockets(paquete, sock, "el paquete")})


def _validar_comando(comando: object) -> list[str]:
    if (not isinstance(comando, Sequence) or isinstance(comando, (str, bytes)) or not comando
            or not all(isinstance(c, str) and "\x00" not in c for c in comando)
            or not comando[0].startswith("/")):
        raise ConfigFaroInvalida("el comando de la jaula tiene que ser una lista de textos cuyo primer elemento es un ejecutable absoluto")
    return list(comando)


class LanzadorJaula:
    def __init__(self, cfg: ConfigJaula, bitacora: Bitacora, *, uids_prohibidos: Iterable[int] = (),
                 crear_proceso: Callable | None = None, uid_min: int | None = None, uid_max: int | None = None):
        self._cfg = cfg
        self._bitacora = bitacora
        self._prohibidos = frozenset(uids_prohibidos)
        self._rango = (uid_min, uid_max) if uid_min is not None and uid_max is not None else None
        self._crear_proceso = crear_proceso if crear_proceso is not None else asyncio.create_subprocess_exec
        self._procesos: dict[int, list] = {}          # uid de jaula -> procesos lanzados (para `jaula_viva`)
        self._vigias: set[asyncio.Task] = set()

    @classmethod
    def desde_control(cls, cfg: ConfigJaula, bitacora: Bitacora, control, **kw) -> "LanzadorJaula":
        """Con el rango de uids de jaula y el uid del orquestador del canal de control: el lanzador los REVALIDA."""
        return cls(cfg, bitacora, uids_prohibidos={control.orquestador_uid}, uid_min=control.jaula_uid_min,
                   uid_max=control.jaula_uid_max, **kw)

    def argv(self, ejecucion, ruta_socket: str, ruta_token: str, comando: Sequence[str], *, perfil: str = "rele",
             workspace: str | None = None, paquete: str | None = None, red: str | None = None) -> list[str]:
        """El argv de bwrap (sin la elevacion ni el binario) de ESA ejecucion, con el perfil `rele` (por defecto) o `motor`."""
        validar_uid_jaula(ejecucion.uid_esperado, prohibidos=self._prohibidos, rango=self._rango)
        if perfil == "rele":
            montajes = perfil_rele(ruta_socket, ruta_token)
        elif perfil == "motor":
            if not workspace or not paquete:
                raise ConfigFaroInvalida("el perfil de motor necesita workspace y paquete")
            montajes = perfil_motor(ruta_socket, ruta_token, workspace=workspace, paquete=paquete, red=red)
        else:
            raise ConfigFaroInvalida(f"perfil de jaula desconocido: {perfil!r}")
        return [*montajes, "--", *_validar_comando(comando)]

    def argv_completo(self, ejecucion, ruta_socket: str, ruta_token: str, comando: Sequence[str], **perfil) -> list[str]:
        cuerpo = self.argv(ejecucion, ruta_socket, ruta_token, comando, **perfil)
        uid = str(ejecucion.uid_esperado)
        return [*(e.replace("{uid}", uid) for e in self._cfg.elevar), str(self._cfg.bwrap), *cuerpo]

    def jaula_viva(self, uid: int) -> bool:
        """¿Sigue vivo algun proceso lanzado con ese uid de jaula? (gancho del canal de control: no reasignar un uid vivo)"""
        return any(getattr(p, "returncode", 0) is None for p in self._procesos.get(uid, ()))

    async def esperar_terminos(self) -> None:
        if self._vigias:
            await asyncio.gather(*self._vigias, return_exceptions=True)

    async def _vigilar(self, e, proceso) -> None:
        rc = await proceso.wait()
        try:
            await self._bitacora.registrar("jaula_termino", run_id=e.run_id, id_correlacion=e.id_correlacion,
                                           uid_jaula=e.uid_esperado, rc=rc)
        except Exception:  # fail-closed: la jaula ya termino; solo la anotacion fallo y queda en el log
            logger.exception("no se pudo registrar el termino de la jaula de %s", e.run_id)

    async def lanzar(self, srv, comando: Sequence[str], **perfil):
        """Anota `jaula_lanzada` y arranca el proceso (que se devuelve); cuando termina anota `jaula_termino` con su
        codigo de salida. `srv` es el `ServidorPuerto` de la ejecucion."""
        e = srv.ejecucion
        argv = self.argv_completo(e, str(srv.ruta_socket), str(srv.ruta_token), comando, **perfil)    # valida ANTES de anotar o arrancar
        await self._bitacora.registrar(
            "jaula_lanzada", run_id=e.run_id, id_correlacion=e.id_correlacion, uid_jaula=e.uid_esperado, motor=e.motor,
            entry_point=e.entry_point, hash_argv=hashlib.sha256("\x00".join(argv).encode()).hexdigest())
        try:
            proceso = await self._crear_proceso(
                *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=dict(ENTORNO_LANZADOR), start_new_session=True)
        except Exception as exc:
            try:
                await self._bitacora.registrar("jaula_fallo", run_id=e.run_id, id_correlacion=e.id_correlacion,
                                               uid_jaula=e.uid_esperado, motivo=type(exc).__name__)
            except Exception:  # fail-closed: el error de arranque es el que sube; solo la anotacion fallo
                pass
            raise
        if hasattr(proceso, "wait"):
            self._procesos.setdefault(e.uid_esperado, []).append(proceso)
            tarea = asyncio.create_task(self._vigilar(e, proceso), name=f"faro-jaula-{e.run_id}")
            self._vigias.add(tarea)
            tarea.add_done_callback(self._vigias.discard)
        return proceso


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
# Quien corre bwrap con el uid de la jaula abre sus dos fuentes en /run/faro (0700 de `faro`) gracias a una ACL por
# ejecucion que pone el Puerto (jax/faro/acl.py): directorio --x, socket rw-, token r--, solo para ese uid.
# El perfil de motor deja la RED como decision pendiente de 0.5/0.6 (aqui, aislada: sin red).
# La regla de sudoers que da la elevacion: ops/faro/sudoers-jaula-ejemplo.
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


def _bloque(titulo: str, elevar: list[str], cuerpo: list[str], comando: tuple[str, ...]) -> str:
    lineas = [" ".join(shlex.quote(x) for x in elevar), "/usr/bin/bwrap", *_lineas([*cuerpo, "--"])]
    return f"# --- {titulo} ---\n" + " \\\n  ".join(lineas) + " \\\n  " + " ".join(comando) + "\n"


def ejemplo() -> str:
    """Los dos perfiles: la jaula minima del rele y la de un motor (red aislada: la decision de red es de 0.5/0.6)."""
    elevar = [e.replace("{uid}", "<uid_jaula>") for e in _ELEVAR_EJEMPLO]
    sock, tok, rel = "/run/faro/<run_id>.sock", "/run/faro/<run_id>.token", "<ruta de jax/faro/relay.py>"
    motor = {"workspace": "<workspace de la ejecucion>", "paquete": "<paquete del ecosistema>"}
    return (_ENCABEZADO
            + _bloque("PERFIL DEL RELE (jaula minima: solo el rele contra su Puerto)", elevar, _construir(sock, tok, rel, None), _COMANDO_EJEMPLO)
            + "\n"
            + _bloque("PERFIL DE MOTOR (workspace rw, $HOME tmpfs, paquete ro, /etc minimo; red AISLADA: decision pendiente de 0.5/0.6)",
                      elevar, _construir(sock, tok, rel, motor), ("<el motor>", "<sus argumentos>")))


_RE_USUARIO = re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
_RE_RUTA_SUDOERS = re.compile(r"^/[A-Za-z0-9_./-]+$")
MAX_UIDS_SUDOERS = 4096


def ejemplo_sudoers(usuario: str, bwrap: str, uid_min: int, uid_max: int) -> str:
    """La regla de sudoers de ejemplo: `usuario` puede correr SOLO `bwrap` y SOLO como los uids del rango de jaulas.
    sudoers no tiene rangos numericos, asi que se enumeran (hasta `MAX_UIDS_SUDOERS`); mas alla, un grupo."""
    if not _RE_USUARIO.fullmatch(usuario or ""):
        raise ConfigFaroInvalida("usuario de sudoers invalido")
    if not _RE_RUTA_SUDOERS.fullmatch(bwrap or ""):
        raise ConfigFaroInvalida("el comando de sudoers tiene que ser una ruta absoluta sin comodines ni espacios")
    if isinstance(uid_min, bool) or isinstance(uid_max, bool) or not (isinstance(uid_min, int) and isinstance(uid_max, int)) \
            or not 1 <= uid_min <= uid_max <= _UID_MAX:
        raise ConfigFaroInvalida("el rango de uids de jaula tiene que cumplir 1 <= MIN <= MAX (el rango no incluye al administrador)")
    if uid_max - uid_min + 1 > MAX_UIDS_SUDOERS:
        raise ConfigFaroInvalida(f"mas de {MAX_UIDS_SUDOERS} uids: sudoers no tiene rangos; usa un grupo de cuentas de jaula")
    uids = [f"#{u}" for u in range(uid_min, uid_max + 1)]
    filas = ", \\\n    ".join(", ".join(uids[i:i + 8]) for i in range(0, len(uids), 8))
    return (
        "# ops/faro/sudoers-jaula-ejemplo -- GENERADO por jax.faro.jaula.ejemplo_sudoers(); una prueba vigila que no se desvie.\n"
        "# Va en /etc/sudoers.d/ (validar con `visudo -cf`). El servicio puede correr SOLO bwrap y SOLO como los uids\n"
        "# del rango de jaulas (JAX_FARO_JAULA_UID_MIN..MAX): nunca como el administrador ni como una cuenta real.\n"
        "# sudoers no tiene rangos numericos: se enumeran los uids.\n"
        "#\n"
        "# CUENTAS DE JAULA: sudo resuelve `-u '#<uid>'` contra la base de cuentas y rechaza un uid sin cuenta\n"
        "# (`sudo: unknown user #<uid>`; medido con sudo 1.9.17). Hay que hacer UNA de dos cosas:\n"
        "#   a) crear una cuenta de sistema por cada uid del rango (sin shell ni home), o\n"
        "#   b) descomentar la linea siguiente: con `runas_allow_unknown_id` sudo acepta un uid sin cuenta, y lo que\n"
        "#      limita a quien puede ser el servicio sigue siendo el Runas_Alias de abajo, no la existencia de la cuenta.\n"
        f"# Defaults:{usuario} runas_allow_unknown_id\n"
        f"Runas_Alias JAULAS = {filas}\n"
        f"{usuario} ALL=(JAULAS) NOPASSWD: {bwrap}\n")
