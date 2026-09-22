# jax/ejecutor/contratos/huella.py
"""Integridad de los propios controles del Ejecutor, por ESTADO, no por texto de
comando. Ronda 6 (auditoría adversarial 2026-09-22): «la huella solo mide lo que NO
debe cambiar nunca durante una misión» -- se SIMPLIFICA a propósito: la ronda 4/5 la
había hecho crecer (log de sudo, sudo-io, dos niveles, declarado por texto) hasta
volverse otro filtro que no converge.

RUTAS_CONTROLES + RUTAS_SHA_SOLO + el glob `/usr/local/sbin/ejecutor-*` son lo que hace
cumplir C3/C4/C6 en cada máquina, y NADA de eso tiene que cambiar NUNCA durante una
misión -- por eso no hay "declarado": cualquier cambio es un hallazgo, sin excepción
(B-2 de las rondas 4/5 SE QUITA: la lista de "lo que no debe cambiar nunca" no admite
excepciones por definición, y una lista de rutas declaradas por texto libre en la
misión era, otra vez, un filtro sobre texto).

Si el cutover algún día necesita crear una cuenta de sistema (tocaría `/etc/passwd` o
`/etc/group`), eso NO se declara: la misión se pausa igual, y es correcto -- crear una
cuenta de sistema en una máquina de un cliente es exactamente el tipo de cambio que
tiene que pasar por un humano, no autorizarse solo con una lista de rutas en el pedido.
La alternativa (un campo estructurado nuevo en `Mision`/`Turno`, con su propio
transporte en jax-platform) es la que NO se eligió: agrega una superficie de contrato
nueva -- entre repos -- para un caso que, hasta hoy, nunca pasó (verificado: ninguna
misión real tocó `/etc/passwd`). Se elige lo más simple; si el caso aparece, se
construye el campo estructurado ENTONCES, con su propio contrato (Principio IX).

LÍMITE, dicho en serio (no en el texto de la misión, acá, donde se cumple o no):
- El registro de sudo (`/var/log/sudo-*`), `sudo-io`, systemd y cron NO están en la
  huella. Un root en la máquina puede truncar, reemplazar o borrar su propio log de
  sudo de todos modos (B-1: ya no hay forma honesta de pretender que ese log es
  intocable frente a un root) -- lo que sí es append-only de verdad es el registro de
  C3 (`jax.ejecutor.contratos.registro`, la cadena encadenada), que vive en
  **hall9000**, no en la máquina controlada, así que un root remoto no lo puede tocar.
  systemd y cron: el cutover crea LEGÍTIMAMENTE unidades/crons del cliente cuando el
  plan de la misión lo pide -- vigilarlos no distinguía "lo pedido" de "lo indebido"
  sin volver a inventar declarado.
- Los binarios del comando (`sha256sum`, `find`, `stat`, `readlink`) se llaman por RUTA
  ABSOLUTA, no por el PATH (sin `secure_path`): un root también podría reemplazar esos
  binarios. La huella prueba "el Ejecutor no lo hizo por accidente", no "un root de la
  máquina es incapaz de mentir" -- eso último no lo cierra ninguna huella tomada DESDE
  la misma máquina.

El CONTROLADOR (fruiz, vía `JAX_EJECUTOR_ADMIN_USUARIO` + ssh + `sudo -n` -- ver
`vigia_servicio.py`, nunca la cuenta `axioma`) toma la huella AL ABRIR la misión y otra
AL CERRAR cada turno, contra la línea base de la APERTURA (persistida por `mision_id`,
ver `vigia_servicio.py::huella_de_apertura_de_la_mision` y el seguimiento de marcas
pendientes de `verificar_huellas_huerfanas`, M-1).

Sólo biblioteca estándar: lo corre `fruiz`/el controlador, no `axioma`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

#: Lo que hace cumplir C3 (registro de sudo -- la INSTALACIÓN de las reglas, no el log)
#: y C6 (llaves) en cada máquina -- rutas verificadas contra
#: ops/ejecutor/instalar_en_maquina.sh (2026-09-22), no inventadas. Dato, no código
#: (Principio IV). NINGÚN cambio acá es legítimo durante una misión: sin declarado.
RUTAS_CONTROLES = (
    "/etc/sudoers",
    "/etc/sudoers.d",
    "/etc/ssh/sshd_config",
    "/etc/ssh/sshd_config.d",
    "/etc/ssh/authorized_keys.d",
    "/root/.ssh/authorized_keys",
)

#: sha256 solamente (nunca el contenido, y `/etc/shadow` en particular NUNCA sale de la
#: máquina ni como texto: sólo su hash cruza el ssh).
RUTAS_SHA_SOLO = ("/etc/passwd", "/etc/group", "/etc/shadow")

#: Los binarios propios del Ejecutor en la máquina -- glob, no nombres literales: hoy
#: son `ejecutor-freno-remoto` y `ejecutor-revocar`, pero el contrato es "nada que
#: empiece con `ejecutor-` en este directorio", no una lista que hay que acordarse de
#: actualizar cada vez que se agrega un binario.
_DIR_SBIN_EJECUTOR = "/usr/local/sbin"
_GLOB_SBIN_EJECUTOR = "ejecutor-*"

# Rutas ABSOLUTAS de los binarios que arma el comando -- NUNCA por el PATH (ver el
# LÍMITE del docstring del módulo). Verificadas en Ubuntu/Debian (coreutils, findutils):
# `/usr/bin/find`, `/usr/bin/sha256sum`, `/usr/bin/sort`. `find -printf "%l"` da el
# destino de un symlink SIN un `readlink`/`sh -c` anidado (que sería otra ruta
# absoluta más, y una capa de escapado de comillas que no hace falta). Si una máquina
# las tuviera en otro lado, el comando falla cerrado (huella vacía =
# `huella_valida() is False`), no en silencio.
_FIND = "/usr/bin/find"
_SHA256SUM = "/usr/bin/sha256sum"
_SORT = "/usr/bin/sort"


def _q(ruta: str) -> str:
    if not ruta or "'" in ruta:
        raise ValueError("ruta_invalida")
    return f"'{ruta}'"


def _tramo_ruta(ruta: str) -> str:
    """sha256 del contenido RESUELTO (`-xtype f` sigue symlinks); el DESTINO de cada
    symlink por separado (`-printf %l`, sin resolver); y el listado de directorios. Una
    ruta ausente cuenta como "no existe" (`2>/dev/null`), no como error."""
    q = _q(ruta)
    return (
        f'{_FIND} {q} -xtype f -exec {_SHA256SUM} {{}} + 2>/dev/null ; '
        f'{_FIND} {q} -type l -printf "L %p -> %l\\n" 2>/dev/null ; '
        f'{_FIND} {q} -type d -printf "D %p\\n" 2>/dev/null'
    )


def _tramo_sbin_ejecutor() -> str:
    # `-name` va COMILLADO: sin comillas, el shell expandiría `ejecutor-*` como un glob
    # contra el directorio de trabajo ANTES de que `find` lo vea.
    q, glob = _q(_DIR_SBIN_EJECUTOR), _q(_GLOB_SBIN_EJECUTOR)
    return (
        f'{_FIND} {q} -maxdepth 1 -name {glob} -xtype f -exec {_SHA256SUM} {{}} + 2>/dev/null ; '
        f'{_FIND} {q} -maxdepth 1 -name {glob} -type l -printf "L %p -> %l\\n" 2>/dev/null'
    )


def _tramo_sha_solo(ruta: str) -> str:
    return f'{_SHA256SUM} {_q(ruta)} 2>/dev/null'


def comando_huella() -> str:
    """El comando REMOTO para la huella -- sin `sudo -n` propio (lo corre el
    controlador, envuelto en UN solo `sudo -n sh -c '<esto>'`, ver `vigia_servicio.py`
    y `revocacion.argv_admin`). No depende de ninguna cuenta: ronda 6 quitó el log de
    sudo, que era lo único que sí dependía de un nombre."""
    tramos = [_tramo_ruta(r) for r in RUTAS_CONTROLES]
    tramos.append(_tramo_sbin_ejecutor())
    tramos.extend(_tramo_sha_solo(r) for r in RUTAS_SHA_SOLO)
    return f'({" ; ".join(tramos)}) | {_SORT}'


@dataclass(frozen=True)
class Huella:
    host: str
    texto: str


def huella_desde_salida(host: str, salida: bytes) -> Huella:
    return Huella(host, salida.decode(errors="replace"))


def huella_valida(h: Huella) -> bool:
    """MINOR (ronda 6): una huella vacía (o que no trae ni una línea reconocible) no
    es "sin cambios" ni "máquina limpia" -- es que la medición no sirvió (comando mal
    formado, sudo denegado sin que rc lo reflejara, binarios ausentes). Fail-closed:
    quien llama trata esto como no-medible, no como "todo en orden"."""
    return bool(h.texto.strip())


def cambio(antes: Huella, despues: Huella) -> bool:
    if antes.host != despues.host:
        raise ValueError("huellas_de_maquinas_distintas", antes.host, despues.host)
    return antes.texto != despues.texto


def lineas_agregadas_o_quitadas(antes: Huella, despues: Huella) -> tuple:
    a, d = set(antes.texto.splitlines()), set(despues.texto.splitlines())
    return tuple(sorted(a ^ d))


def hallazgos(antes: Huella, despues: Huella) -> tuple:
    """Todo lo que cambió -- sin declarado (B-2 se fue, ronda 6): cualquier cambio acá
    es un hallazgo, siempre."""
    if not cambio(antes, despues):
        return ()
    return lineas_agregadas_o_quitadas(antes, despues)
