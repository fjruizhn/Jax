# jax/ejecutor/contratos/pausa.py
"""La pausa PROPIA del Ejecutor y el latido del vigía de C5.

DECISIÓN (plan 4 de SP1, 2026-09-17; corrige el punto 5 de «lo que el spec dice mal»
del índice): C5 NO escribe el interruptor global de JAX (`JAX_KILL_SWITCH_PATH`).
- Ese archivo frena TODO JAX (la Mesa responde 423): un falso positivo del auditor
  —un LLM— dejaría sin servicio a la plataforma entera por un paso del Ejecutor.
- Su activación la audita jax-platform en `kill_switch_audit` cuando la pone un
  superadmin por la API; escrita por un proceso, quedaría fuera de esa auditoría.
- El Ejecutor es un solo cliente (la cuenta `axioma`) detrás de un solo punto de paso
  (el proxy de C3): frenarlo ahí alcanza, sin tocar a nadie más.
Por eso C5 escribe `JAX_EJECUTOR_PAUSA` (junto al interruptor, en el mismo directorio
`root:fruiz 2770`) y el proxy la obedece (423, y el trozo que completa un `tool_use` no
sale). La activación queda en el propio archivo (origen, motivo, paso, momento) y en el
journal del vigía. El freno root de C4 (plan 3) tiene que matar también con ella puesta.
El interruptor global sigue frenando al Ejecutor (plan 3); esto se le SUMA.

Fail-closed, como el interruptor del frente B:
- sin la variable, vacía o relativa: `PausaSinConfigurar` (sin saber dónde está el freno
  no se ejecuta nada);
- `pausa_puesta` sólo da False ante «no existe»: permiso, ENOTDIR o E/S → PUESTA;
- `poner_pausa` publica el archivo completo de una vez (temporal + `os.link`, que falla
  si ya existe: el primer motivo no se pisa) y sincroniza el directorio.

El latido: el vigía toca `JAX_EJECUTOR_VIGIA_LATIDO` mientras vive. Un vigía muerto con
SIGKILL no llega a escribir la pausa; su latido envejece y el proxy deja de servir. Así
«vigía caído → freno» vale también cuando no hay `except` que lo atrape.

Sólo biblioteca estándar.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

VARIABLE_RUTA = "JAX_EJECUTOR_PAUSA"
VARIABLE_LATIDO = "JAX_EJECUTOR_VIGIA_LATIDO"
VARIABLE_LATIDO_MAX_S = "JAX_EJECUTOR_VIGIA_LATIDO_MAX_S"


class PausaSinConfigurar(RuntimeError):
    """La variable falta, está vacía o no es una ruta absoluta. `args[0]` es la variable."""


def _ruta(env, variable: str) -> Path:
    env = os.environ if env is None else env
    valor = env.get(variable, "").strip()
    if not valor or not Path(valor).is_absolute():
        raise PausaSinConfigurar(variable)
    return Path(valor)


def ruta_de_la_pausa(env=None) -> Path:
    return _ruta(env, VARIABLE_RUTA)


def ruta_del_latido(env=None) -> Path:
    return _ruta(env, VARIABLE_LATIDO)


def pausa_puesta(ruta) -> bool:
    try:
        os.stat(ruta)
    except FileNotFoundError:
        return False
    except OSError:  # fail-closed: sin poder mirar la pausa (permiso, ENOTDIR, E/S) se la da por PUESTA
        return True
    return True


def _sincronizar_directorio(directorio: Path) -> None:
    descriptor = os.open(directorio, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def poner_pausa(ruta: Path, datos: dict) -> bool:
    """Pone la pausa. True si la puso esta llamada; False si ya estaba (no la pisa).
    Un error de E/S se propaga: quien llama no puede creer que frenó."""
    ruta = Path(ruta)
    contenido = json.dumps({**datos, "momento": datetime.now(timezone.utc).isoformat()},
                           ensure_ascii=True, sort_keys=True)
    descriptor, temporal = tempfile.mkstemp(prefix=".pausa-", dir=ruta.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as archivo:
            archivo.write(contenido)
            archivo.flush()
            os.fsync(archivo.fileno())
        os.chmod(temporal, 0o660)
        try:
            os.link(temporal, ruta)
        except FileExistsError:
            return False
        _sincronizar_directorio(ruta.parent)
        return True
    finally:
        os.unlink(temporal)


def _ruta_patron_temporales(ruta: Path) -> str:
    import glob
    # MINOR (ronda 10): `glob.escape()` sobre el NOMBRE -- si la pausa se llamara con
    # caracteres especiales de glob (`[`, `]`, `*`, `?`), sin escapar el patrón
    # matchearía de más (o de menos) que los temporales que de verdad le pertenecen.
    # El sufijo `.quitar-tmp-*` SÍ tiene que seguir siendo un patrón real -- no se
    # escapa.
    return f".{glob.escape(Path(ruta).name)}.quitar-tmp-*"


def _ruta_candado(ruta: Path) -> Path:
    return Path(ruta).parent / f".{Path(ruta).name}.candado"


@contextlib.contextmanager
def candado(ruta: Path):
    """Candado exclusivo (`flock`, bloqueante) sobre `.{nombre}.candado`, en el mismo
    directorio que la pausa -- MAJOR (ronda 10, auditoría 8): serializa `quitar_pausa_si`
    (vía `aceptar()`) y `barrer_temporales_huerfanos` entre procesos concurrentes. Sin
    esto, dos intentos de aceptar a la vez (o un `aceptar()` y un barrido) pueden
    interleavear su propio chequeo-de-inodo con el `unlink` del otro: B pasa el
    chequeo, A borra, C5 pausa de nuevo, y B -- que ya había pasado SU chequeo antes de
    que A borrara -- termina haciendo `unlink` sobre lo que hay AHORA (la pausa nueva
    de C5), no sobre lo que vio. Con el candado, B ni siquiera puede EMPEZAR su
    chequeo hasta que A termine y libere -- para entonces ve el estado real.

    `poner_pausa` (C5, la huella al pausar) NO usa este candado -- sigue sin pisar
    nunca una pausa existente por su cuenta (`os.link`, no destructivo); sólo QUITAR y
    BARRER se serializan entre sí, nunca contra quien pone."""
    ruta_candado = _ruta_candado(ruta)
    fd = os.open(ruta_candado, os.O_CREAT | os.O_RDWR, 0o660)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def quitar_pausa_si(ruta: Path, *, coincide) -> tuple:
    """Borra la pausa en `ruta` SOLO si `coincide(datos)` es verdadero para su
    contenido -- NUNCA una pausa ajena (ronda 8, B-1) Y NUNCA con una ventana donde la
    pausa no exista (ronda 9, BLOCK-1/MAJOR-1: la auditoría 7 dio NO-GO porque la
    versión anterior usaba `os.rename` PRIMERO -- eso saca `ruta` del mundo un
    instante, y si el proceso muere justo ahí, la pausa desaparece sin que nadie la
    haya aceptado de verdad).

    Orden, sin ventana de ausencia:
    1. `os.link(ruta, tmp)` -- un enlace EXTRA al mismo inodo; `ruta` sigue existiendo
       tal cual, con su nombre, todo el tiempo. No hay "robo".
    2. Se lee `tmp` (mismo contenido que `ruta`, mismo inodo).
    3. Si NO coincide: se borra `tmp` (el enlace extra) y se devuelve False. `ruta`
       jamás se tocó.
    4. Si coincide: antes de borrar, se verifica con `os.stat` que `ruta` siga siendo
       el MISMO inodo que `tmp` (`st_ino` + `st_dev` -- `poner_pausa` nunca pisa una
       pausa existente con `os.link`, así que mientras el nombre exista nadie puede
       reemplazar su contenido por otro; la única forma de que cambie es que YA se
       haya borrado y otra cosa haya tomado su lugar). Si coincide, RECIÉN entonces
       `os.unlink(ruta)` y después `os.unlink(tmp)`. Si el inodo cambió, no se borra
       nada -- se avisa devolviendo `(False, datos)`, igual que "no coincide": lo que
       hay en `ruta` ahora no es lo que vimos.

    MINOR-1: cualquier excepción entre el `link` y el `unlink(ruta)` sale de la
    función SIN tocar `ruta` -- no hay un `except OSError` (ni ningún otro) que trague
    el error y siga de largo a borrar; `os.unlink(ruta)` es la última instrucción antes
    de devolver `True`, con nada arriesgado después.

    Un kill exactamente entre `unlink(ruta)` y `unlink(tmp)` deja un temporal huérfano
    (`.{nombre}.quitar-tmp-*`) que nunca vuelve a ser la pausa -- `barrer_temporales_
    huerfanos` lo limpia al arrancar `aceptar` y el vigía. Devuelve
    `(se_borro, datos_vistos_o_None)`."""
    import secrets
    ruta = Path(ruta)
    tmp = ruta.parent / f".{ruta.name}.quitar-tmp-{os.getpid()}-{secrets.token_hex(4)}"
    try:
        os.link(ruta, tmp)
    except FileNotFoundError:
        return False, None

    try:
        try:
            datos = json.loads(tmp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            datos = None

        if not (isinstance(datos, dict) and coincide(datos)):
            return False, datos

        st_tmp = os.stat(tmp)
        try:
            st_ruta = os.stat(ruta)
        except FileNotFoundError:
            return False, datos  # ya no está -- alguien más la sacó; no hay nada que borrar
        if (st_ruta.st_ino, st_ruta.st_dev) != (st_tmp.st_ino, st_tmp.st_dev):
            return False, datos  # cambió de identidad en el medio -- no es la que vimos

        try:
            os.unlink(ruta)
        except FileNotFoundError:
            # MINOR (ronda 10): entre el os.stat de arriba y este unlink, alguien más
            # (ajeno a esta función, sin pasar por acá) la sacó -- no hay nada que
            # borrar. Esto NO puede salir como traceback: quien llama (aceptar()) ya
            # pudo haber escrito la marca y el registro antes de llegar acá.
            return False, datos
        return True, datos
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:  # fail-soft: tmp es NUESTRO temporal de esta llamada -- si ya no está, el objetivo (que no quede) ya se cumplió
            pass


def barrer_temporales_huerfanos(ruta: Path) -> int:
    """Limpia los `.{nombre}.quitar-tmp-*` que un kill puede haber dejado atrás entre
    el `unlink(ruta)` y el `unlink(tmp)` de `quitar_pausa_si` (ronda 9) -- SOLO
    temporales, nunca `ruta` misma (no la toca ni la nombra). Se llama al arrancar
    `aceptar` y el vigía. Devuelve cuántos se borraron; ausente el directorio, o sin
    nada que barrer, no falla.

    MINOR (ronda 10): sólo toca ARCHIVOS REGULARES -- `os.lstat` (no sigue symlinks) +
    `stat.S_ISREG`. Si algo que no debería (un directorio, un symlink, un socket) tiene
    un nombre que matchea el patrón, se lo salta -- nunca `os.unlink` a ciegas, que
    revienta con `IsADirectoryError` sobre un directorio.

    MAJOR (ronda 10): corre bajo el MISMO `candado` que `quitar_pausa_si` (vía
    `aceptar()`) -- sin esto, un barrido podría borrar el temporal de un
    `quitar_pausa_si` en curso en otro proceso, justo en la ventana entre su
    `unlink(ruta)` y su propio `unlink(tmp)`."""
    import glob
    import stat as _stat
    ruta = Path(ruta)
    with candado(ruta):
        borrados = 0
        for candidato in glob.glob(str(ruta.parent / _ruta_patron_temporales(ruta))):
            try:
                modo = os.lstat(candidato).st_mode
            except FileNotFoundError:
                continue
            if not _stat.S_ISREG(modo):
                continue
            try:
                os.unlink(candidato)
                borrados += 1
            except FileNotFoundError:  # fail-soft: otro barrido/limpieza ya se lo llevó entre el lstat y este unlink -- el objetivo (que no quede) ya se cumplió
                pass
        return borrados


def latir(ruta: Path) -> None:
    """Marca que el vigía vive: crea el archivo o le actualiza la fecha."""
    ruta = Path(ruta)
    with open(ruta, "a", encoding="utf-8"):
        pass
    os.utime(ruta)


def latido_fresco(ruta, max_s: float, *, ahora: float | None = None) -> bool:
    """¿El vigía latió hace menos de `max_s`? Ausente o ilegible → False (no hay vigía)."""
    try:
        mtime = os.stat(ruta).st_mtime
    except OSError:  # fail-closed: sin poder mirar el latido no hay vigía que conste; el proxy no sirve
        return False
    return (time.time() if ahora is None else ahora) - mtime < max_s
