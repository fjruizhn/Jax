# jax/ejecutor/contratos/huella.py
"""Integridad de los propios controles del Ejecutor, por ESTADO, no por texto de
comando (M-1/M-2, ronda 3; endurecido en la ronda 4, auditoría adversarial
2026-09-22 -- "no persigas más patrones léxicos: es la tercera vez que un filtro sobre
el texto no converge").

Un patrón léxico (C1, `NUCLEO_REGLAS` en politica.py) sólo atrapa lo que el comando
DICE que hace, escrito de la forma que el patrón espera. Lo que no cambia es el
RESULTADO: si `/etc/sudoers.d/*` es distinto después de la misión que antes, ALGO lo
tocó, sin importar cómo.

El CONTROLADOR (fruiz, vía `JAX_EJECUTOR_ADMIN_USUARIO` + ssh + `sudo -n` -- el mismo
camino que ya usa `ops/ejecutor/_maquina.sh` y `revocacion.argv_admin` para C6/revocar,
NUNCA la cuenta `axioma`: una cuenta sin privilegios no puede medirse a sí misma) toma
una huella AL ABRIR la misión y otra AL CERRAR cada turno. La línea base es la de LA
APERTURA DE LA MISIÓN (persistida por `mision_id`, ver `vigia_servicio.py` -- NUNCA la
del turno anterior: eso blanquearía en el turno N+1 un cambio que el turno N no llegó
a ver).

Ronda 4, dos niveles:
- CONTROLES (`RUTAS_CONTROLES`): un cambio no declarado en el texto de la misión pone
  la pausa (C4) -- ver `jax.ejecutor.contratos.pausa`.
- PERSISTENCIA (`RUTAS_PERSISTENCIA`): informa, nunca pausa -- el cutover crea
  LEGÍTIMAMENTE unidades/crons del cliente; tratarlas como control pausaría cada
  misión normal.

El log de sudo (B-1) se juzga aparte: es NORMAL que crezca (cada comando que corre
la cuenta le agrega una línea) -- lo que NO es normal es que se trunque, se reemplace
o se reescriba. Por eso NO entra en la comparación línea-a-línea genérica: se compara
por (inode, tamaño, sha256 de los primeros N bytes al cierre contra el sha256 completo
de la apertura, con N = tamaño de apertura). Mismo inode + tamaño no decreciente +
prefijo igual = "sólo-añadido"; cualquier otra cosa es un hallazgo.

Lo "declarado" en el texto de la misión se compara por la RUTA EXACTA de lo que
cambió, nunca por cualquier fragmento de la línea (un symlink cuyo DESTINO se
menciona en la misión no blanquea un cambio en el symlink mismo si la misión no
menciona la ruta del symlink).

Sólo biblioteca estándar: lo corre `fruiz`/el controlador, no `axioma`.
"""
from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass

#: Lo que hace cumplir C3 (registro de sudo), C4 (freno) y C6 (llaves/revocador) en cada
#: máquina -- rutas verificadas contra ops/ejecutor/instalar_en_maquina.sh,
#: ops/ejecutor/instalar_freno.sh y jax/ejecutor/contratos/revocacion.py (2026-09-22), no
#: inventadas. Dato, no código (Principio IV): si mañana se vigila una ruta más, se
#: agrega acá, no en un regex. Un cambio en cualquiera de éstas, si no está declarado en
#: el texto de la misión, PAUSA.
RUTAS_CONTROLES = (
    "/etc/sudoers",
    "/etc/sudoers.d",
    "/etc/ssh/sshd_config",
    "/etc/ssh/sshd_config.d",
    "/etc/ssh/authorized_keys.d",
    "/root/.ssh/authorized_keys",
    "/usr/local/sbin/ejecutor-freno-remoto",
    "/usr/local/sbin/ejecutor-revocar",
    "/var/lib/systemd/linger",
    "/etc/cron.deny",
    "/etc/at.deny",
    "/etc/crontab",
    "/etc/passwd",
    "/etc/group",
    "/etc/shadow",
)

#: El cutover crea LEGÍTIMAMENTE unidades/crons del cliente: un cambio acá se
#: INFORMA, nunca pausa.
RUTAS_PERSISTENCIA = (
    "/etc/systemd/system",
    "/etc/cron.d",
    "/var/spool/cron/crontabs",
)

_RUTA_SUDO_IO = "/var/log/sudo-io"
_MARCA_CONTROLES = "===CONTROLES==="
_MARCA_PERSISTENCIA = "===PERSISTENCIA==="
_MARCA_FIN = "===FIN==="


def _validar_cuenta(cuenta: str) -> None:
    if not cuenta or "/" in cuenta or " " in cuenta:
        raise ValueError("cuenta_invalida")


def _tramo_ruta(ruta: str) -> str:
    """sha256 del contenido RESUELTO (sigue symlinks -- `-xtype f` incluye tanto los
    archivos regulares como los symlinks que apuntan a uno); el DESTINO de cada symlink
    por separado (`readlink`, sin resolver); y el listado de directorios. Una ruta
    ausente cuenta como "no existe" (`2>/dev/null`), no como error -- una máquina puede
    no tener, por ejemplo, unidades systemd propias."""
    q = shlex.quote(ruta)
    return (
        f'find {q} -xtype f -exec sha256sum {{}} + 2>/dev/null ; '
        f'find {q} -type l -exec sh -c \'echo "L $0 -> $(readlink "$0")"\' {{}} \\; 2>/dev/null ; '
        f'find {q} -type d -printf "D %p\\n" 2>/dev/null'
    )


def _tramo_sudo_io() -> str:
    """Sólo el LISTADO (nombre + tipo) de `/var/log/sudo-io` -- son grabaciones de
    sesión, potencialmente grandes y con contenido sensible; la estructura (qué entradas
    hay) es lo que importa acá, no su contenido."""
    q = shlex.quote(_RUTA_SUDO_IO)
    return f'find {q} -printf "I %y %p\\n" 2>/dev/null | sort'


def _tramo_atq() -> str:
    return 'echo "ATQ $(atq 2>/dev/null | sha256sum | cut -d\' \' -f1)"'


def _tramo_log(cuenta: str, *, primeros_n: int | None) -> str:
    """B-1: (inode, tamaño, sha256) del log de sudo de la cuenta. En la APERTURA
    (`primeros_n=None`) el sha256 es del archivo COMPLETO; en el CIERRE, de sus
    primeros `primeros_n` bytes (N = el tamaño que tenía al abrir) -- eso es lo que deja
    juzgar "sólo-añadido" sin tener que transportar el log entero."""
    log = f"/var/log/sudo-{cuenta}.log"
    q = shlex.quote(log)
    if primeros_n is None:
        leer_hash = f'sha256sum {q} 2>/dev/null | cut -d" " -f1'
    else:
        leer_hash = f'head -c {int(primeros_n)} {q} 2>/dev/null | sha256sum | cut -d" " -f1'
    return (
        f'I=$(stat -c "%i" {q} 2>/dev/null); '
        f'T=$(stat -c "%s" {q} 2>/dev/null); '
        f'H=$({leer_hash}); '
        f'echo "LOG {log} $I $T $H"'
    )


def _comando(cuenta: str, *, log_tramo: str) -> str:
    _validar_cuenta(cuenta)
    controles = " ; ".join(_tramo_ruta(r) for r in RUTAS_CONTROLES) + " ; " + _tramo_sudo_io() + " ; " + _tramo_atq()
    persistencia = " ; ".join(_tramo_ruta(r) for r in RUTAS_PERSISTENCIA)
    return (
        f'{log_tramo} ; '
        f'echo {_MARCA_CONTROLES} ; ({controles}) | sort ; '
        f'echo {_MARCA_PERSISTENCIA} ; ({persistencia}) | sort ; '
        f'echo {_MARCA_FIN}'
    )


def comando_apertura(cuenta: str) -> str:
    """El comando REMOTO para la huella AL ABRIR la misión. Va SIN el `sudo -n` propio
    -- lo corre el controlador (fruiz, nunca axioma) envuelto en un ÚNICO
    `sudo -n sh -c '<esto>'` (mismo patrón que `ops/ejecutor/_maquina.sh` y
    `revocacion.argv_admin`, ver `vigia_servicio.py`), no un `sudo -n` por tramo."""
    return _comando(cuenta, log_tramo=_tramo_log(cuenta, primeros_n=None))


def comando_cierre(cuenta: str, *, tamano_apertura_log: int) -> str:
    """El comando REMOTO para la huella AL CERRAR un turno -- necesita el tamaño que el
    log tenía al abrir la MISIÓN (no el turno) para juzgar si sigue siendo sólo-añadido."""
    return _comando(cuenta, log_tramo=_tramo_log(cuenta, primeros_n=tamano_apertura_log))


# --- el estado, parseado -------------------------------------------------------------------

@dataclass(frozen=True)
class InfoLog:
    inode: str
    tamano: int
    sha256: str  # completo en la apertura; de los primeros N bytes en el cierre


@dataclass(frozen=True)
class Huella:
    host: str
    controles: str
    persistencia: str
    log: InfoLog | None  # None: no se pudo medir (stat/sha256 fallaron, o sudo -n se negó)


def _seccion(texto: str, marca_inicio: str, marca_fin: str) -> str:
    try:
        return texto.split(marca_inicio, 1)[1].split(marca_fin, 1)[0]
    except IndexError:
        return ""


def _log_desde_texto(texto: str) -> InfoLog | None:
    for linea in texto.splitlines():
        if linea.startswith("LOG "):
            partes = linea.split()
            if len(partes) != 5:
                return None
            _, _ruta, inode, tamano, sha = partes
            if not inode or not tamano or not sha or not tamano.isdigit():
                return None
            return InfoLog(inode=inode, tamano=int(tamano), sha256=sha)
    return None


def _huella_desde_salida(host: str, salida: bytes) -> Huella:
    texto = salida.decode(errors="replace")
    return Huella(
        host=host,
        controles=_seccion(texto, _MARCA_CONTROLES, _MARCA_PERSISTENCIA).strip(),
        persistencia=_seccion(texto, _MARCA_PERSISTENCIA, _MARCA_FIN).strip(),
        log=_log_desde_texto(texto),
    )


def huella_de_apertura_desde_salida(host: str, salida: bytes) -> Huella:
    return _huella_desde_salida(host, salida)


def huella_de_cierre_desde_salida(host: str, salida: bytes) -> Huella:
    return _huella_desde_salida(host, salida)


# --- B-1: el log se juzga por ser solo-añadido -------------------------------------------

def log_intacto(apertura: InfoLog | None, cierre: InfoLog | None) -> bool:
    """¿El log de sudo sigue siendo el MISMO archivo, sólo con líneas agregadas al
    final? Mismo inode (no lo reemplazaron por otro archivo), tamaño no menor (no lo
    truncaron), y el sha256 de los primeros N bytes del cierre (N = tamaño de apertura)
    igual al sha256 COMPLETO de la apertura (el prefijo no cambió). Si no se pudo medir
    alguno de los dos lados, NO es "intacto": es no-medible, y quien llama decide qué
    hacer con eso (ver `vigia_servicio.py` -- falla cerrado)."""
    if apertura is None or cierre is None:
        return False
    return (apertura.inode == cierre.inode
            and cierre.tamano >= apertura.tamano
            and cierre.sha256 == apertura.sha256)


# --- M-2: comparación por ruta exacta -------------------------------------------------------

def _entradas(texto: str) -> dict:
    """{ruta: línea} de cada línea de la huella -- la RUTA es la clave exacta contra la
    que se compara el texto de la misión. `sha  /ruta` (sha256sum), `D /ruta` (directorio),
    `L /ruta -> /destino` (symlink: la ruta que cambió es el symlink, no su destino) o
    `I t /ruta` (listado de sudo-io)."""
    entradas = {}
    for linea in texto.splitlines():
        if not linea.strip():
            continue
        partes = linea.split(None, 1)
        if len(partes) != 2:
            continue
        primero, resto = partes
        if primero in ("D",):
            ruta = resto.strip()
        elif primero == "L":
            ruta = resto.split(" -> ", 1)[0].strip()
        elif primero == "I":
            # "t /ruta" -- el tipo es la primera palabra de `resto`
            sub = resto.split(None, 1)
            ruta = sub[1].strip() if len(sub) == 2 else resto.strip()
        else:  # sha256sum: "<hash>  /ruta"
            ruta = resto.strip()
        entradas[ruta] = linea
    return entradas


def _cambio_declarado_ruta(ruta: str, texto_mision: str) -> bool:
    return bool(texto_mision.strip()) and ruta in texto_mision


def _diferencias_no_declaradas(antes: str, despues: str, texto_mision: str) -> tuple:
    e_antes, e_despues = _entradas(antes), _entradas(despues)
    salida = []
    for ruta in sorted(set(e_antes) | set(e_despues)):
        if e_antes.get(ruta) == e_despues.get(ruta):
            continue
        if _cambio_declarado_ruta(ruta, texto_mision):
            continue
        salida.append(e_despues.get(ruta) or f"(borrado) {e_antes.get(ruta)}")
    return tuple(salida)


def hallazgos_controles(antes: Huella, despues: Huella, texto_mision: str) -> tuple:
    """Los cambios de CONTROLES no declarados -- incluye el veredicto del log de sudo
    (B-1): si no quedó intacto (o no se pudo medir), es un hallazgo propio, con su
    propio texto, sin pasar por la comparación línea-a-línea genérica."""
    if antes.host != despues.host:
        raise ValueError("huellas_de_maquinas_distintas", antes.host, despues.host)
    encontrados = list(_diferencias_no_declaradas(antes.controles, despues.controles, texto_mision))
    if not log_intacto(antes.log, despues.log):
        encontrados.append(
            "sudolog: truncado_reemplazado_o_no_medible "
            f"inode_apertura={antes.log.inode if antes.log else None} "
            f"inode_cierre={despues.log.inode if despues.log else None} "
            f"tam_apertura={antes.log.tamano if antes.log else None} "
            f"tam_cierre={despues.log.tamano if despues.log else None}")
    return tuple(encontrados)


def hallazgos_persistencia(antes: Huella, despues: Huella, texto_mision: str) -> tuple:
    """Los cambios de PERSISTENCIA no declarados -- SÓLO informa, quien llama nunca
    pausa por esto (ver el docstring del módulo)."""
    if antes.host != despues.host:
        raise ValueError("huellas_de_maquinas_distintas", antes.host, despues.host)
    return _diferencias_no_declaradas(antes.persistencia, despues.persistencia, texto_mision)
