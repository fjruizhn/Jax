# jax/ejecutor/contratos/huella.py
"""Integridad de los propios controles del Ejecutor, por ESTADO, no por texto de
comando. Ronda 6 (auditoría adversarial 2026-09-22): «la huella solo mide lo que NO
debe cambiar nunca durante una misión» -- se SIMPLIFICA a propósito: la ronda 4/5 la
había hecho crecer (log de sudo, sudo-io, dos niveles, declarado por texto) hasta
volverse otro filtro que no converge.

RUTAS_CONTROLES + el glob `/usr/local/sbin/ejecutor-*` son lo que hace cumplir C3/C4/C6
en cada máquina, y NADA de eso tiene que cambiar NUNCA durante una misión -- por eso no
hay "declarado": cualquier cambio es un hallazgo, sin excepción (B-2 de las rondas 4/5
SE QUITA: la lista de "lo que no debe cambiar nunca" no admite excepciones por
definición, y una lista de rutas declaradas por texto libre en la misión era, otra vez,
un filtro sobre texto).

`/etc/passwd`, `/etc/group` y `/etc/shadow` SALIERON de la huella (ronda 7, BLOCK
reproducido en producción, 2026-09-22): la ronda 6 los medía "por las dudas", asumiendo
que ningún cutover real los tocaba -- FALSO. `apt install` crea cuentas de sistema para
la mayoría de los paquetes de servicio (nginx, postfix, mysql, …), y el plugin de correo
de aaPanel hace lo mismo: los dos son administración LEGÍTIMA y COMÚN, no un ataque a
los controles del Ejecutor. Medirlos hacía que CUALQUIER `apt install` normal pausara el
cutover -- y, peor, dejaba el host bloqueado sin salida (ver M-1 más abajo, estados de
la marca). Una cuenta nueva con sudo REAL de todos modos deja rastro en `sudoers.d`, que
sí se mide -- ese es el control que importa, no `/etc/passwd`.

LÍMITE, dicho en serio (no en el texto de la misión, acá, donde se cumple o no):
- El registro de sudo (`/var/log/sudo-*`), `sudo-io`, systemd, cron y
  `/etc/passwd`/`/etc/group`/`/etc/shadow` NO están en la huella. Un root en la máquina
  puede truncar, reemplazar o borrar su propio log de sudo de todos modos (B-1: ya no
  hay forma honesta de pretender que ese log es intocable frente a un root) -- lo que sí
  es append-only de verdad es el registro de C3 (`jax.ejecutor.contratos.registro`, la
  cadena encadenada), que vive en **hall9000**, no en la máquina controlada, así que un
  root remoto no lo puede tocar. systemd, cron y las cuentas de sistema: el cutover los
  crea/toca LEGÍTIMAMENTE cuando el plan de la misión lo pide (instalar un paquete,
  configurar un servicio del cliente) -- vigilarlos no distinguía "lo pedido" de "lo
  indebido" sin volver a inventar declarado, y en la práctica pausaba trabajo normal.
- Los binarios del comando (`sha256sum`, `find`) se llaman por RUTA ABSOLUTA, no por el
  PATH (sin `secure_path`): un root también podría reemplazar esos binarios. La huella
  prueba "el Ejecutor no lo hizo por accidente", no "un root de la máquina es incapaz de
  mentir" -- eso último no lo cierra ninguna huella tomada DESDE la misma máquina.

El CONTROLADOR (fruiz, vía `JAX_EJECUTOR_ADMIN_USUARIO` + ssh + `sudo -n` -- ver
`vigia_servicio.py`, nunca la cuenta `axioma`) toma la huella AL ABRIR la misión y otra
AL CERRAR cada turno, contra la línea base de la APERTURA (persistida por `mision_id`,
ver `vigia_servicio.py::huella_de_apertura_de_la_mision` y el seguimiento de marcas
pendientes de `verificar_huellas_huerfanas`, M-1).

Estados de la marca persistida (ronda 7, M-1 -- el BLOCK reproducido: la ronda 6 sólo
tenía un booleano "pendiente", y una vez `reportada` el host quedaba bloqueado PARA
SIEMPRE, sin salida):
- `ABIERTA`: la línea base se tomó y la comparación de CIERRE nunca se hizo (kill,
  reinicio). Es la ÚNICA que revisa `verificar_huellas_huerfanas` -- una nueva
  comparación puede resolverla (limpia -> `CERRADA`) o confirmarla (sucia -> `REPORTADA`).
- `REPORTADA`: se comparó, salió sucia, ya puso la pausa, con su diff guardado.
  Bloquea CUALQUIER misión nueva en ese host hasta que Fernando la acepte -- por
  `python -m jax.ejecutor.contratos.huella aceptar --host <host> --mision <mision_id>`
  (ver `principal`, más abajo, y `docs/ejecutor-huella-aceptar.md`).
- `CERRADA`: se comparó, salió limpia. No bloquea nada; sigue guardada como línea
  base fija de la misión (para los turnos siguientes), no se vuelve a re-tomar.

Sólo biblioteca estándar (más `jax.ejecutor.contratos.pausa`, que también lo es --
para `quitar_pausa_si`, ronda 8 B-1): lo corre `fruiz`/el controlador, no `axioma`.
"""
from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from jax.ejecutor.contratos import pausa

#: Lo que hace cumplir C3 (registro de sudo -- la INSTALACIÓN de las reglas, no el log)
#: y C6 (llaves) en cada máquina -- rutas verificadas contra
#: ops/ejecutor/instalar_en_maquina.sh (2026-09-22), no inventadas. Dato, no código
#: (Principio IV). NINGÚN cambio acá es legítimo durante una misión: sin declarado.
#:
#: LÍMITE 9 (ronda 2, auditoría adversarial 2026-09-22): esta tupla se queda FIJA a
#: propósito -- el `authorized_keys` del ADMINISTRADOR (dónde vive el acceso
#: privilegiado real, y dónde el arreglo de jax#260 puso la llave del servicio) SÍ entra
#: a la huella, pero por `ruta_authorized_keys_admin(admin_usuario)` / `comando_huella
#: (admin_usuario)` -- necesita saber la cuenta, y RUTAS_CONTROLES no depende de nada.
#:
#: MAJOR-4 (ronda 3): `/etc/ejecutor-huella` -- el directorio donde
#: `instalar_huella_en_maquina.sh` escribe `admin_usuario` (root 0644) para que
#: `ops/ejecutor/ejecutor-huella` sepa a qué cuenta pertenece el `authorized_keys` a
#: medir SIN hardcodear un nombre en el guion. Que ESTE archivo entre a RUTAS_CONTROLES
#: (fija, no depende de la cuenta) cierra el hueco obvio: si alguien lo borra o lo
#: cambia para apuntar a otra cuenta, el tramo derivado (`ruta_authorized_keys_admin`)
#: puede dejar de medir lo que debía -- pero el cambio en ESTE archivo, que sí es fijo,
#: se ve igual, sin declarado, como cualquier otro control.
RUTAS_CONTROLES = (
    "/etc/sudoers",
    "/etc/sudoers.d",
    "/etc/ssh/sshd_config",
    "/etc/ssh/sshd_config.d",
    "/etc/ssh/authorized_keys.d",
    "/root/.ssh/authorized_keys",
    "/etc/ejecutor-huella",
)

#: Dónde vive el archivo de una sola línea con el usuario administrador -- lo escribe
#: `instalar_huella_en_maquina.sh`, lo lee `ops/ejecutor/ejecutor-huella` con `cat`.
RUTA_ADMIN_USUARIO_CONFIG = "/etc/ejecutor-huella/admin_usuario"

#: Los binarios propios del Ejecutor en la máquina -- glob, no nombres literales: hoy
#: son `ejecutor-freno-remoto` y `ejecutor-revocar`, pero el contrato es "nada que
#: empiece con `ejecutor-` en este directorio", no una lista que hay que acordarse de
#: actualizar cada vez que se agrega un binario.
_DIR_SBIN_EJECUTOR = "/usr/local/sbin"
_GLOB_SBIN_EJECUTOR = "ejecutor-*"

#: BLOCK-E (ronda 5, auditoría adversarial 2026-09-22): el sentinela que representa "se
#: barrió el glob de binarios propios" -- no es una ruta real (el `*` no puede aparecer
#: en un nombre de archivo real), así que nunca colisiona con un archivo de verdad
#: encontrado por el barrido. Antes el glob quedaba FUERA de `huella_valida()` por
#: diseño ("su desaparición total no invalida nada", LÍMITE documentado más abajo) --
#: eso también dejaba pasar un `find` que fallara en silencio sobre este barrido
#: exactamente igual que en cualquier directorio de `RUTAS_CONTROLES`.
RUTA_GLOB_SBIN_EJECUTOR = f"{_DIR_SBIN_EJECUTOR}/{_GLOB_SBIN_EJECUTOR}"

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


# RONDA 4 (auditoría adversarial 2026-09-22): medido el 2026-09-22 (Fernando): el
# `authorized_keys` del ROOT SÍ EXISTE (archivo vacío, 0600, dueño root) en hall9000,
# atemai Y prod -- sólo FALTA en `bridge`. El texto anterior de esta ronda decía lo
# contrario ("no existe en NINGUNA de las tres") -- ERA FALSO; corregido en ronda 5
# (MAJOR-G) sin tocar el arreglo, que sigue haciendo falta: bridge SÍ está sano con esa
# ruta ausente, y la ronda 3 bloqueaba el Ejecutor ahí igual que si hubiera fallado la
# medición. La versión de ronda 2/3 (`find ... 2>/dev/null`, sin más) daba CERO líneas
# tanto si la ruta no existía COMO si no se pudo medir (permiso denegado, `find` roto) --
# indistinguibles, y `huella_valida(rutas=...)` trataba las dos como inválidas,
# bloqueando el Ejecutor en una máquina SANA. Mirror EXACTO (sin f-string: es texto de
# shell con sus propias llaves y `$`, interpolarlo habría sido un baño de escapes) de
# `tramo()`/`tramo_sbin_ejecutor()` en `ops/ejecutor/ejecutor-huella` --
# `tests/test_ejecutor_huella_sh.py` compara la SALIDA de este texto contra la del
# script real, no sólo su forma.
#
# BLOCK-E (ronda 5, auditoría adversarial 2026-09-22): el caso ARCHIVO ya tenía guarda
# (`sha256sum` sin salida -> `E ... sha256sum_fallo`); el caso DIRECTORIO no la tenía --
# los tres `find` iban con `2>/dev/null` y SIN mirar su `$?`, así que un `find` que
# fallara a mitad de camino (un subdirectorio sin permiso, por ejemplo) hacía
# desaparecer TODAS las líneas de contenido en silencio, y el guion de todos modos
# imprimía `D <ruta>` -- la huella pasaba por buena sin haber podido enumerar el
# contenido real. Ahora el resultado de los tres `find` se junta en un archivo aparte
# y su `$?` se mira ANTES de decidir si el estado es `D` (los tres a cero) o `E ...
# find_fallo` (cualquiera de los tres no-cero) -- comprobado por el auditor con bwrap.
#
# MINOR (ronda 6, auditoría adversarial 2026-09-22): `cat`/`rm` iban SIN ruta absoluta
# -- inconsistente con find/sha256sum/stat/mktemp, que sí la tienen (mismo LÍMITE del
# docstring del módulo). Corregido acá y en `ops/ejecutor/ejecutor-huella`.
#
# MINOR, el mutante que sobrevivía (ronda 6): el `if false` en el chequeo de `$?`
# ($rc1/$rc2/$rc3) sólo se probaba contra EL GUION REAL (`ops/ejecutor/ejecutor-huella`,
# `tests/test_ejecutor_huella_sh.py`) -- ESTE texto, el que de verdad corre por ssh en
# producción, podía tener el mismo mutante sin que ningún test lo viera, porque el test
# de sincronía (MAJOR-7) sólo compara SALIDAS sobre árboles SANOS: con un `find` que no
# falla en ninguno de los dos lados, la salida es idéntica con o sin el chequeo de
# `$?`. `tests/test_ejecutor_huella_sh.py` ahora agrega un árbol con un directorio
# ILEGIBLE a esa misma comparación -- si CUALQUIERA de los dos lados (guion o este
# texto) perdiera el chequeo, las dos salidas dejarían de coincidir ahí.
_CUERPO_TRAMO_SH = r'''tramo() {
  ruta="$1"
  err="$(/usr/bin/mktemp)"
  tipo="$(/usr/bin/stat -c '%F' "$ruta" 2>"$err")"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    if grep -q "No such file or directory" "$err"; then
      echo "A $ruta"
    else
      motivo="$(tr '\n' ' ' < "$err" | tr -s ' ')"
      echo "E $ruta ${motivo:-motivo_desconocido}"
    fi
    /usr/bin/rm -f "$err"
    return 0
  fi
  /usr/bin/rm -f "$err"
  case "$tipo" in
    "regular file"|"regular empty file")
      linea_hash="$(/usr/bin/sha256sum "$ruta" 2>/dev/null)"
      if [ -n "$linea_hash" ]; then
        echo "$linea_hash"
      else
        echo "E $ruta sha256sum_fallo"
      fi
      ;;
    directory)
      salida="$(/usr/bin/mktemp)"
      errd="$(/usr/bin/mktemp)"
      /usr/bin/find "$ruta" -mindepth 1 -xtype f -exec /usr/bin/sha256sum {} + >"$salida" 2>>"$errd"
      rc1=$?
      /usr/bin/find "$ruta" -mindepth 1 -type l -printf "L %p -> %l\n" >>"$salida" 2>>"$errd"
      rc2=$?
      /usr/bin/find "$ruta" -mindepth 1 -type d -printf "D %p\n" >>"$salida" 2>>"$errd"
      rc3=$?
      if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ] || [ "$rc3" -ne 0 ]; then
        echo "E $ruta find_fallo"
      else
        echo "D $ruta"
        /usr/bin/cat "$salida"
      fi
      /usr/bin/rm -f "$salida" "$errd"
      ;;
    *)
      echo "E $ruta tipo_no_esperado:$tipo"
      ;;
  esac
}'''

# BLOCK-E, punto 2 (ronda 5): el glob de `/usr/local/sbin/ejecutor-*` tenía el MISMO
# defecto que el caso directorio de `tramo()` -- dos `find` con `2>/dev/null` sin mirar
# `$?` -- y además NO estaba representado en `huella_valida(rutas=...)` en absoluto (su
# desaparición total no invalidaba nada, por diseño). Ahora reporta su propio estado
# (`D`/`E ... find_fallo`) bajo el sentinela `RUTA_GLOB_SBIN_EJECUTOR`
# (`/usr/local/sbin/ejecutor-*`, un `*` literal que nunca puede colisionar con un
# archivo real) -- y `RUTAS_DECLARADAS_POR_DEFAULT` lo exige como cualquier otra ruta.
_CUERPO_TRAMO_SBIN_SH = r'''tramo_sbin_ejecutor() {
  base="$1"
  patron="$2"
  salida="$(/usr/bin/mktemp)"
  errd="$(/usr/bin/mktemp)"
  /usr/bin/find "$base" -maxdepth 1 -name "$patron" -xtype f -exec /usr/bin/sha256sum {} + >"$salida" 2>>"$errd"
  rc1=$?
  /usr/bin/find "$base" -maxdepth 1 -name "$patron" -type l -printf "L %p -> %l\n" >>"$salida" 2>>"$errd"
  rc2=$?
  if [ "$rc1" -ne 0 ] || [ "$rc2" -ne 0 ]; then
    echo "E $base/$patron find_fallo"
  else
    echo "D $base/$patron"
    /usr/bin/cat "$salida"
  fi
  /usr/bin/rm -f "$salida" "$errd"
}'''


def _tramo_ruta(ruta: str) -> str:
    """UNA llamada a la función `tramo()` (ver `_CUERPO_TRAMO_SH`) -- reemplaza el
    `find ... 2>/dev/null` de rondas 2/3, que no distinguía "no existe" de "no se pudo
    medir". `_q`: la MISMA función de escapado que ya usaba esto, sin cambios."""
    return f"tramo {_q(ruta)}"


def _tramo_sbin_ejecutor() -> str:
    # BLOCK-E (ronda 5): ya NO arma el `find ... ; find ...` a mano acá -- llama a
    # `tramo_sbin_ejecutor()` (ver `_CUERPO_TRAMO_SBIN_SH`), que sí mira el `$?` de los
    # dos `find` antes de decidir el estado.
    return f"tramo_sbin_ejecutor {_q(_DIR_SBIN_EJECUTOR)} {_q(_GLOB_SBIN_EJECUTOR)}"


def ruta_authorized_keys_admin(admin_usuario: str) -> str:
    """LÍMITE que ronda 2 cierra (auditoría adversarial 2026-09-22, punto 9): el
    `authorized_keys` del ADMINISTRADOR (`JAX_EJECUTOR_ADMIN_USUARIO`, hoy `fruiz`) es
    donde vive el acceso privilegiado real -- y donde este mismo arreglo pone la llave
    del servicio (`instalar_huella_en_maquina.sh`). No medirlo dejaría el propio cambio
    que este commit hace invisible a la huella.

    MAJOR-5 (ronda 3, auditoría adversarial 2026-09-22): esto asumía `/home/<admin>` --
    CORREGIDO: sale de `pwd.getpwnam(admin_usuario).pw_dir`, el passwd REAL de esta
    máquina, no una convención. Dato verificado por Fernando esa noche: en `bridge`
    (Ubuntu 24.04 + Hestia) `/home/fruiz` SÍ existe -- la sospecha de un layout tipo
    macOS era falsa -- pero igual se lee del passwd: `ejecutor-huella` (el script
    remoto) hace lo mismo con `getent passwd`, y esta función es la que un test de
    sincronía (MAJOR-7) compara contra la salida real del script EN ESTA MISMA
    máquina -- los dos tienen que resolver el mismo passwd para que la comparación
    signifique algo. `KeyError` (cuenta inexistente) se traduce a `ValueError`, fail-
    closed, igual que el resto de las validaciones de este módulo.

    MINOR (ronda 5, auditoría adversarial 2026-09-22): dicho en serio, sin dejarlo
    implícito -- `pwd.getpwnam` resuelve el passwd de la máquina donde CORRE ESTE
    PROCESO PYTHON (hall9000, siempre: esta función sólo se usa acá y en el instalador,
    nunca en una remota), NO el de la máquina remota cuya huella se está armando. Que
    hoy coincida con el passwd de las tres remotas (medido: mismo UID/GID/home para
    `fruiz` en hall9000, atemai, bridge y prod) es un HECHO de HOY, no una garantía del
    código -- si una remota cambiara el home de esa cuenta, esta función seguiría
    devolviendo la ruta de HALL9000, no la real de esa remota. Por eso el camino en
    PRODUCCIÓN (`ops/ejecutor/ejecutor-huella::tramo_admin()`) nunca llama a esta
    función: resuelve `getent passwd` EN LA PROPIA REMOTA, en el momento. Esta función
    sólo sirve para: (a) el test de sincronía (MAJOR-7, ya citado arriba, que compara
    contra el script real EN ESTE MISMO host) y (b) `vigia_servicio.py::_principal`,
    que pasa su resultado como `rutas_extra` a `huella_valida()` -- ahí la comparación
    es de TEXTO contra lo que devolvió la remota, así que si algún día un passwd
    diverge, el síntoma sería un `huella_valida` que rechaza todo en esa máquina (fail-
    closed), no un dato mal medido en silencio."""
    if not admin_usuario or "/" in admin_usuario or admin_usuario.strip() != admin_usuario:
        raise ValueError("admin_usuario_invalido")
    try:
        home = pwd.getpwnam(admin_usuario).pw_dir
    except KeyError:
        raise ValueError("admin_usuario_sin_passwd") from None
    if not home or not home.startswith("/"):
        raise ValueError("admin_usuario_sin_home")
    return f"{home}/.ssh/authorized_keys"


def comando_huella(admin_usuario: str) -> str:
    """El texto de lo que había que medir -- YA NO se manda por ssh (ver el arreglo del
    bug de producción, jax#260, 2026-09-22, en el docstring del módulo y en
    `argv_huella_servicio`, más abajo): antes se envolvía en UN `sudo -n sh -c '<esto>'`
    armado por `revocacion.argv_admin` como el ADMINISTRADOR; ahora la misma lógica
    (RUTAS_CONTROLES + el authorized_keys del administrador + el glob de
    `/usr/local/sbin/ejecutor-*`, mismos binarios por ruta absoluta) vive, ESTÁTICA, en
    `ops/ejecutor/ejecutor-huella` -- el comando forzado de la llave PROPIA del
    servicio. Esta función sigue acá como la definición en Python de QUÉ se mide (dato,
    no código, Principio IV); `tests/test_ejecutor_huella_sh.py` verifica -- comparando
    SALIDAS sobre el mismo árbol de prueba, ronda 2 MAJOR-7, no sólo listas de rutas --
    que el script real mide exactamente lo mismo.

    Ronda 2 (LÍMITE 9): ya NO tiene firma vacía -- `admin_usuario` hace falta para
    `ruta_authorized_keys_admin`. `test_comando_huella_no_pide_una_cuenta` (ronda 6)
    queda retirado a propósito: la premisa que probaba ("no depende de ninguna
    cuenta") dejó de ser cierta el día que la huella tuvo que empezar a vigilar SU
    PROPIA llave de acceso, que vive en el `authorized_keys` de una cuenta concreta.

    Ronda 4: la función `tramo()` (definida UNA vez, `_CUERPO_TRAMO_SH`) va ANTES del
    grupo que la invoca -- `{ tramo r1 ; tramo r2 ; ... ; <glob de sbin> ; } | sort`.
    El texto que resuelve `ruta_authorized_keys_admin(admin_usuario)` -- no un `cat`
    del archivo de config ni un `getent` propios -- porque acá Python YA sabe la
    cuenta; `ops/ejecutor/ejecutor-huella::tramo_admin()` hace ese trabajo de más
    (leer el archivo, resolver con `getent`) para el camino real, donde nadie le pasa
    la cuenta por argv.

    Ronda 5 (BLOCK-E): el glob de sbin también pasa por una función DEFINIDA UNA vez
    (`_CUERPO_TRAMO_SBIN_SH`/`tramo_sbin_ejecutor()`), igual que `tramo()` -- las DOS
    definiciones van antes del grupo que las invoca.

    Ronda 5 (hallazgo propio, al agregar la línea del glob de sbin a la comparación):
    esta función NUNCA había puesto `export LC_ALL=C` en su propio texto -- sólo el
    script real lo hace (ronda 2, MINOR). Con un único `sort` de pocas líneas, casi
    siempre coincidía por casualidad con el locale de la sesión que corría el test;
    agregar la línea del glob lo hizo divergir de verdad (medido: bajo `en_US.UTF-8`,
    `sort` intercala una línea `D ...` en una posición distinta que bajo `C`, y
    `test_el_guion_y_comando_huella_dan_la_misma_salida_sobre_el_mismo_arbol` -- que YA
    existía -- lo detectó). El texto de esta función ahora fija `LC_ALL=C` también,
    igual que el script."""
    llamadas = [_tramo_ruta(r) for r in RUTAS_CONTROLES]
    llamadas.append(_tramo_ruta(ruta_authorized_keys_admin(admin_usuario)))
    cuerpo = " ; ".join(llamadas) + " ; " + _tramo_sbin_ejecutor()
    return f'export LC_ALL=C\n{_CUERPO_TRAMO_SH}\n{_CUERPO_TRAMO_SBIN_SH}\n{{ {cuerpo} ; }} | {_SORT}'


# --- EL CAMINO REMOTO: la llave PROPIA DEL SERVICIO, no la personal del administrador ---
#
# Bug de producción (jax#260, 2026-09-22, medido por Fernando): `vigia_servicio.py` lo
# lanza `jax-platform` como SUBPROCESO -- y desde el 2026-09-17 (decisión de Fernando,
# cuenta de servicio) `jax-platform.service` corre como `jaxsvc`
# (`/etc/systemd/system/jax-platform.service.d/cuenta-de-servicio.conf: User=jaxsvc`),
# NO como `fruiz`. Varios docstrings de este árbol (este módulo, `vigia_servicio.py`,
# `ops/ejecutor/instalar_vigia.sh`) seguían afirmando "hereda la identidad de fruiz" --
# ERA FALSO, corregido en esta ronda. `jaxsvc` no puede leer `~fruiz/.ssh/*` (600, dueño
# `fruiz`), así que `revocacion.argv_admin` (sin `-i`, resolución de identidad por
# DEFAULT de ssh) no encontraba ninguna llave utilizable: `vigia_no_latio=true rc=2`
# medido en producción, el Ejecutor bloqueado por completo.
#
# El arreglo: una llave PROPIA del servicio (`JAX_EJECUTOR_HUELLA_LLAVE`, bajo
# `/etc/jax/controlador/` -- ese directorio YA es `jaxsvc:jaxsvc 700`, igual que
# `JAX_EJECUTOR_CONTROLADOR_LLAVE`, pero un PAR DISTINTO: ese es para `axioma@127.0.0.1`
# local; éste, para el ADMINISTRADOR en cada remota) autorizada en cada máquina con
# comando forzado hacia `ejecutor-huella` (`ops/ejecutor/ejecutor-huella` +
# `ops/ejecutor/instalar_huella_en_maquina.sh`) -- mismo patrón que ya usa C4
# (`ejecutor-freno-remoto`, `JAX_EJECUTOR_FRENO_LLAVE`). El comando forzado (`restrict`,
# sin pty, sin reenvíos, `from=` acotado a hall9000) limita lo que esa llave puede hacer
# aunque quien la lea quisiera abrir una shell con ella. `IdentitiesOnly=yes` hace
# cumplir que ssh NUNCA ofrezca otra llave ni caiga a un agente -- sin eso, un fallo de
# la llave del servicio podría hacer que ssh probara silenciosamente la personal del
# administrador (si por algún accidente de entorno estuviera al alcance), que es
# exactamente el defecto que este arreglo cierra.
VARIABLE_HUELLA_LLAVE = "JAX_EJECUTOR_HUELLA_LLAVE"
VARIABLE_HUELLA_KNOWN_HOSTS = "JAX_EJECUTOR_HUELLA_KNOWN_HOSTS"


def argv_huella_servicio(h, *, llave: Path, known_hosts: Path, admin_usuario: str, tope_s: float) -> list[str]:
    """El ssh REAL para tomar la huella -- reemplaza `revocacion.argv_admin` +
    `comando_huella()` en el camino en vivo (ver el bloque de arriba). `-i llave` es la
    llave PROPIA DEL SERVICIO (jaxsvc puede leerla) -- `IdentitiesOnly=yes` hace que ssh
    NUNCA ofrezca otra. Se conecta como `admin_usuario` (la MISMA cuenta que
    `JAX_EJECUTOR_ADMIN_USUARIO`, `fruiz`) -- lo que cambia es la CREDENCIAL, no de qué
    cuenta es huésped: el comando forzado del lado remoto es lo que acota qué puede
    hacer esa llave. `UserKnownHostsFile` dedicado (`known_hosts`): `jaxsvc` no comparte
    el `$HOME/.ssh/known_hosts` de `fruiz` ni de `axioma`. `-F /dev/null` (ronda 2,
    MINOR): ignora CUALQUIER `~/.ssh/config`/`/etc/ssh/ssh_config` del proceso que
    invoca -- sin esto, un `Host` con `ProxyJump`/`IdentityFile`/`User` para ese mismo
    nombre o IP (heredado, a mano, o por accidente) podría pisar `-i`/`IdentitiesOnly`
    en silencio; con `-F /dev/null` sólo cuentan las opciones que este comando pasa
    explícitamente. El comando remoto que se manda (`"ejecutor-huella"`) es cosmético --
    el `command=` forzado en la remota lo reemplaza siempre -- pero deja algo legible en
    el log de sshd sobre qué se pidió."""
    return ["ssh", "-F", "/dev/null", "-i", str(llave), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={known_hosts}",
            "-o", f"ConnectTimeout={int(tope_s)}", "-p", str(h.puerto), f"{admin_usuario}@{h.ip}",
            "ejecutor-huella"]


# --- BLOCK-2/MAJOR-3 (ronda 2): la línea de authorized_keys del ADMINISTRADOR remoto, --
# generada por PYTHON y TESTEABLE -- antes la armaba `instalar_huella_en_maquina.sh` en
# una variable de bash sin ningún test. `actualizar_authorized_keys_admin` CONVERGE: si
# ya hay una línea marcada (`MARCA_HUELLA_SERVICIO`), la REEMPLAZA -- por ejemplo si
# `origen_ip` cambió porque hall9000 cambió de IP -- nunca la duplica ni dos entradas
# compiten por el mismo comando forzado.

MARCA_HUELLA_SERVICIO = "ejecutor-huella-servicio"
#: MAJOR-5 (ronda 2): el sudoers acotado en la remota exige EXACTAMENTE este comando
#: SIN argumentos -- `ejecutor-huella ""` en el sudoers.d, no sólo `ejecutor-huella` a
#: secas. Verificado empíricamente (no supuesto, Principio I) en un contenedor Ubuntu
#: 24.04 limpio, sudo 1.9.17p2, 2026-09-22: una regla NOPASSWD sin argumentos en el
#: sudoers ACEPTA cualquier argumento (`sudo -n cmd hostil` → rc=0); sólo agregando la
#: cadena vacía (`cmd ""`) sudo exige que la invocación NO tenga argumentos (`sudo -n
#: cmd hostil` → rechazado, pide contraseña). Sin la comilla vacía, "sin argumentos: no
#: hay superficie de ataque" habría sido una afirmación falsa sobre el propio sudoers.
_COMANDO_FORZADO_HUELLA = "sudo -n /usr/local/sbin/ejecutor-huella"


#: MAJOR-6 (ronda 3, auditoría adversarial 2026-09-22): mismo patrón que
#: `revocacion._TIPO_DE_LLAVE` -- una LISTA BLANCA de tipos reales de llave ssh, no
#: "sin caracteres raros". Antes `linea_authorized_keys_servicio` sólo validaba
#: `clave`/`origen_ip`; un `tipo` con un salto de línea colaba una SEGUNDA línea en el
#: authorized_keys sin `command=`/`restrict` -- una llave de acceso completo.
_TIPO_DE_LLAVE = re.compile(r"^(ssh-(ed25519|rsa|dss)|ecdsa-sha2-nistp\d+|sk-(ssh-ed25519|ecdsa-sha2-nistp256)@openssh\.com)$")


def _validar_ip_literal(origen_ip: str) -> str:
    """MAJOR-6 (ronda 3): `from=` de ssh admite PATRONES (glob) -- `from="*"` autoriza
    CUALQUIER origen, que es exactamente lo que este control existe para impedir. Se
    exige una IP LITERAL (v4 o v6), nunca un patrón -- `ipaddress.ip_address` rechaza
    `*`, `?`, rangos y cualquier otra cosa que no sea una dirección exacta."""
    import ipaddress

    if not isinstance(origen_ip, str) or not origen_ip or any(c.isspace() for c in origen_ip) or '"' in origen_ip:
        raise ValueError("origen_ip_invalido")
    try:
        ipaddress.ip_address(origen_ip)
    except ValueError:
        raise ValueError("origen_ip_invalido") from None
    return origen_ip


def linea_authorized_keys_servicio(tipo: str, clave: str, *, origen_ip: str) -> str:
    """La línea que `instalar_huella_en_maquina.sh` agrega al `authorized_keys` del
    administrador remoto -- `command=` forzado, `restrict` (sin pty/reenvíos/agente/
    variables de entorno del cliente) y `from=` acotado al origen (hall9000). BLOCK-2:
    si falta `command=`, `restrict` o `from=`, esto ya no protege nada -- por eso hay
    tests que exigen los tres literalmente presentes y un mutante que los borra.

    MAJOR-6 (ronda 3): `tipo` valida contra una lista blanca real (`_TIPO_DE_LLAVE`,
    mismo patrón que `revocacion.py`) -- un salto de línea en `tipo` (o en `clave`)
    podía colar una SEGUNDA línea de `authorized_keys` sin `command=`/`restrict`
    delante, una llave de acceso completo disfrazada de este arreglo. `clave` rechaza
    CUALQUIER whitespace (no sólo espacio: `\\n`, `\\t`, `\\r`) y comillas. `origen_ip`
    tiene que ser una IP LITERAL -- `from="*"` autoriza cualquier origen."""
    if not _TIPO_DE_LLAVE.match(tipo or ""):
        raise ValueError("tipo_invalido")
    if not clave or any(c.isspace() for c in clave) or "'" in clave or '"' in clave:
        raise ValueError("llave_invalida")
    origen_ip = _validar_ip_literal(origen_ip)
    return f'command="{_COMANDO_FORZADO_HUELLA}",restrict,from="{origen_ip}" {tipo} {clave} {MARCA_HUELLA_SERVICIO}'


def actualizar_authorized_keys_admin(actuales: str, *, tipo: str, clave: str, origen_ip: str) -> str:
    """MAJOR-3: converge de verdad. Si ya hay una línea marcada
    `MARCA_HUELLA_SERVICIO` en `actuales`, la QUITA y pone la nueva (con el `origen_ip`
    -- o `tipo`/`clave`, si algún día rota -- actual) al final; si no hay ninguna, la
    agrega. El resto del archivo (otras llaves del administrador, líneas propias)
    queda intacto y en el mismo orden. Correr esto dos veces con los MISMOS argumentos
    da el MISMO resultado (idempotente); con un `origen_ip` distinto, REEMPLAZA la
    línea vieja en vez de duplicarla (convergente)."""
    nueva = linea_authorized_keys_servicio(tipo, clave, origen_ip=origen_ip)
    lineas = [l for l in actuales.splitlines() if not l.rstrip().endswith(f" {MARCA_HUELLA_SERVICIO}")]
    lineas.append(nueva)
    return "\n".join(lineas) + "\n"


@dataclass(frozen=True)
class Huella:
    host: str
    texto: str


def huella_desde_salida(host: str, salida: bytes) -> Huella:
    return Huella(host, salida.decode(errors="replace"))


#: MAJOR-6 (ronda 2): 64 hex + dos espacios -- exactamente lo que imprime `sha256sum`.
_LINEA_CON_HASH = re.compile(r"^[0-9a-f]{64}  ")

#: Los CUATRO estados que `tramo()` puede reportar para una ruta declarada (ronda 4,
#: auditoría adversarial 2026-09-22; hecho corregido en ronda 5, MAJOR-G -- ver el
#: comentario sobre `_CUERPO_TRAMO_SH` más arriba): `HASH`/`D` (medida, existe) y
#: `AUSENTE` (medida, confirmada que NO existe) son estados VÁLIDOS -- medido el
#: 2026-09-22: `/root/.ssh/authorized_keys` SÍ existe (vacío, 0600, root) en hall9000,
#: atemai y prod; sólo FALTA en `bridge`, y esa es la configuración SANA de bridge, no
#: una medición rota ahí. `ERROR` (no se pudo medir -- permiso denegado, tipo
#: inesperado, `sha256sum`/`find` roto) es el ÚNICO inválido.
_HASH, _D, _AUSENTE, _ERROR = "HASH", "D", "AUSENTE", "ERROR"

#: Las rutas que `huella_valida` exige ver representadas -- por default, las FIJAS
#: (RUTAS_CONTROLES) MÁS el glob de binarios de `/usr/local/sbin` (BLOCK-E, ronda 5:
#: antes NO entraba -- "su desaparición total no invalida nada" -- lo que también
#: dejaba pasar un `find` roto sobre ese barrido sin que nadie lo notara; ahora
#: `tramo_sbin_ejecutor()` reporta su propio estado bajo el sentinela
#: `RUTA_GLOB_SBIN_EJECUTOR`, que se valida como cualquier otra ruta). El
#: authorized_keys del administrador NO entra por default porque es host/cuenta-
#: dependiente (`ruta_authorized_keys_admin`) -- MAJOR-C (ronda 4): un llamador que
#: conoce la cuenta (`vigia_servicio.py`) tiene que agregarla explícitamente a `rutas=`.
RUTAS_DECLARADAS_POR_DEFAULT = RUTAS_CONTROLES + (RUTA_GLOB_SBIN_EJECUTOR,)


def _clasificar_linea(linea: str) -> tuple[str, str] | None:
    """`(ruta, estado)` de una línea de huella, o `None` si no tiene forma reconocible.
    `E <ruta> <motivo>`: la ruta es el PRIMER token después de `E ` -- nunca puede
    tener espacios (viene de `_q`/de una ruta de archivo real), a diferencia del
    motivo, que sí puede traerlos."""
    if _LINEA_CON_HASH.match(linea):
        return linea[66:], _HASH
    if linea.startswith("D "):
        return linea[2:], _D
    if linea.startswith("A "):
        return linea[2:], _AUSENTE
    if linea.startswith("E "):
        return linea[2:].split(" ", 1)[0], _ERROR
    if linea.startswith("L "):
        return linea[2:].split(" -> ", 1)[0], "L"
    return None


def _estado_de_ruta_declarada(lineas: list[str], ruta: str) -> str | None:
    """El estado de la línea que representa EXACTAMENTE `ruta` (nunca una línea de
    CONTENIDO por debajo de ella, que usa la MISMA forma de hash/D/L pero para una
    ruta más larga). Si ninguna línea representa `ruta` en sí -- ni siquiera un
    `E` -- pero SÍ hay contenido reportado POR DEBAJO de ella (una ruta que es
    directorio siempre trae su propia línea `D` de todos modos desde ronda 4 -- esto
    queda como red de contención para cualquier llamador que pase una ruta que sólo
    aparezca como padre de contenido), eso cuenta como medido -- si no hay NADA, `None`
    (ni medido, ni declarado ausente: la medición de esa ruta ni siquiera corrió).

    MAJOR-H (ronda 5, auditoría adversarial 2026-09-22): un destino de symlink (o un
    nombre de archivo) puede contener un salto de línea DE VERDAD -- `find -printf`
    lo imprime tal cual, sin escapar nada -- y eso puede partir una línea en dos,
    forjando una SEGUNDA línea que dice ser la MISMA `ruta` exacta que la línea
    genuina. Quedarse con la PRIMERA por orden (el criterio de rondas 2-4) deja que la
    forjada gane si ordena antes que la real -- por ejemplo una `A <ruta>` forjada
    tapando una `D <ruta>` genuina, o al revés. Dos (o más) líneas que dicen ser
    EXACTAMENTE la misma ruta declarada NUNCA es un estado sano -- ninguna corrida
    legítima de `tramo()`/`tramo_sbin_ejecutor()` emite dos líneas de estado para la
    misma ruta de nivel superior -- así que ante una colisión así, la ruta se reporta
    `_ERROR` (inválida), sin intentar adivinar cuál de las dos "es la de verdad"."""
    coincidencias = []
    con_contenido_debajo = False
    for linea in lineas:
        clasificada = _clasificar_linea(linea)
        if clasificada is None:
            continue
        r, estado = clasificada
        if r == ruta:
            coincidencias.append(estado)
        elif r.startswith(ruta + "/"):
            con_contenido_debajo = True
    if len(coincidencias) > 1:
        return _ERROR
    if coincidencias:
        return coincidencias[0]
    return _D if con_contenido_debajo else None


def huella_valida(h: Huella, *, rutas: tuple | None = None) -> bool:
    """MINOR (ronda 6): una huella vacía (o que no trae ni una línea reconocible) no
    es "sin cambios" ni "máquina limpia" -- es que la medición no sirvió (comando mal
    formado, sudo denegado sin que rc lo reflejara, binarios ausentes). Fail-closed:
    quien llama trata esto como no-medible, no como "todo en orden".

    MAJOR-6 (ronda 2, auditoría adversarial 2026-09-22): "no vacía" NO ALCANZABA. Si
    `sha256sum` faltara en la remota, el tramo de esa ruta ahora reporta explícitamente
    `E <ruta> sha256sum_fallo` (ver `tramo()`/`_CUERPO_TRAMO_SH`) -- así que el chequeo
    de abajo, por-ruta, ya lo cubre cuando se pasa `rutas=`; para el caso genérico
    (`rutas=None`) se conserva "al menos un hash en algún lado" como red de contención.

    RONDA 4 (BLOCK reproducido en atemai y prod): "cada ruta declarada tiene que
    aparecer" (MINOR, ronda 3) NO distinguía "esta ruta no existe" (sano) de "no se
    pudo medir" (roto) -- las dos daban CERO líneas para esa ruta con el `find`
    anterior, y esta función las trataba igual: inválida. Eso bloqueaba el Ejecutor en
    la única máquina donde esa ruta de verdad está ausente (medido, ronda 5, MAJOR-G:
    `bridge` -- hall9000, atemai y prod SÍ la tienen, vacía). Ahora exige, para CADA
    ruta de `rutas`, que su estado sea `HASH`, `D` o `AUSENTE` -- `ERROR` (o ausencia
    total de la línea, o MÁS DE UNA línea reclamando la misma ruta -- MAJOR-H, ronda 5,
    ver `_estado_de_ruta_declarada`) invalida la huella entera. Que una ruta pase de
    `AUSENTE` a existir (o al revés) sigue siendo un cambio de TEXTO real (`A <ruta>`
    desaparece, aparece un hash/`D`) -- `cambio()`/`hallazgos()` lo ven igual que
    cualquier otro, sin tocar nada acá.

    `rutas=None` (el default) SALTA el chequeo por-ruta -- lo pide `vigia_servicio.py`
    explícitamente en sus CUATRO llamadas reales (con `RUTAS_DECLARADAS_POR_DEFAULT`
    más el authorized_keys del administrador, MAJOR-C); dejarlo opcional evita que
    esta función necesite adivinar qué se declaró cuando quien llama no lo sabe (por
    ejemplo, pruebas o usos genéricos de una sola línea)."""
    texto = h.texto.strip()
    if not texto:
        return False
    lineas = texto.splitlines()
    if not any(_LINEA_CON_HASH.match(linea) for linea in lineas):
        return False
    if rutas is None:
        return True
    return all(_estado_de_ruta_declarada(lineas, ruta) in (_HASH, _D, _AUSENTE) for ruta in rutas)


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


# --- M-1 (ronda 7): estados de la marca persistida, y su aceptación --------------------

ABIERTA = "abierta"
REPORTADA = "reportada"
CERRADA = "cerrada"

_MISION_ID_VALIDA = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class MisionIdInvalido(ValueError):
    """`args[0]` es el valor recibido."""


def ruta_huella(misiones, mision_id: str, host: str) -> Path:
    if not _MISION_ID_VALIDA.match(mision_id):
        raise MisionIdInvalido(mision_id)
    if not host or "/" in host or host.strip() != host:
        raise ValueError("host_invalido")
    return Path(misiones) / mision_id / "huella" / f"{host}.json"


@dataclass(frozen=True)
class Marca:
    huella: Huella
    estado: str  # ABIERTA | REPORTADA | CERRADA
    diff: tuple = field(default_factory=tuple)
    aceptada_por: str | None = None
    aceptada_en: str | None = None


def marca_a_json(m: Marca) -> dict:
    return {"host": m.huella.host, "texto": m.huella.texto, "estado": m.estado,
            "diff": list(m.diff), "aceptada_por": m.aceptada_por, "aceptada_en": m.aceptada_en}


def marca_desde_json(d: dict) -> Marca:
    return Marca(huella=Huella(host=d["host"], texto=d["texto"]), estado=d["estado"],
                diff=tuple(d.get("diff") or ()), aceptada_por=d.get("aceptada_por"),
                aceptada_en=d.get("aceptada_en"))


def escribir_marca(ruta: Path, m: Marca) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".tmp")
    tmp.write_text(json.dumps(marca_a_json(m)), encoding="utf-8")
    os.replace(tmp, ruta)


def leer_marca(ruta: Path) -> Marca:
    return marca_desde_json(json.loads(Path(ruta).read_text(encoding="utf-8")))


# --- la CLI de aceptación: `python -m jax.ejecutor.contratos.huella aceptar ...` --------
#
# Corre `sudo -u jaxsvc` (nunca `fruiz` a secas, nunca `axioma`, nunca `root` a secas --
# ver `docs/ejecutor-huella-aceptar.md` para el porqué completo: el registro de C3 y
# `JAX_EJECUTOR_MISIONES` son de `jaxsvc`). CORREGIDO (bug de producción, jax#260,
# 2026-09-22): este comentario decía "corre como FRUIZ" -- ERA FALSO, y por eso
# `_tomar_huella_actual` tenía el MISMO defecto que `vigia_servicio.py`: armaba el ssh
# con `revocacion.argv_admin` (sin `-i`, resolución de identidad por default), que
# `jaxsvc` no puede satisfacer con la llave personal de `fruiz`.

async def _tomar_huella_actual(host_nombre: str, *, politica_ruta: Path, admin_usuario: str,
                               huella_llave: Path, huella_known_hosts: Path,
                               tope_s: float = 30) -> Huella:
    """Toma la huella de AHORA MISMO contra `host_nombre`, leyendo su `ip`/`puerto` de
    la política exportada -- mismo camino que `vigia_servicio._tomar_huella`:
    `argv_huella_servicio` (la llave PROPIA del servicio, NUNCA la personal de
    `admin_usuario` -- ver el bloque de arriba). Import diferido: evita un ciclo con
    `vigia_servicio` (que ya importa `huella`) y a `politica`, que `huella.py` no
    necesita para nada más que esto."""
    from jax.ejecutor.contratos import politica as P
    from jax.ejecutor.contratos import vigia_servicio as V

    doc = json.loads(Path(politica_ruta).read_bytes())
    hosts = {h.nombre: h for h in P.validar(doc).hosts}
    h = hosts.get(host_nombre)
    if h is None:
        raise ValueError("host_desconocido", host_nombre)
    argv = argv_huella_servicio(h, llave=huella_llave, known_hosts=huella_known_hosts,
                                admin_usuario=admin_usuario, tope_s=tope_s)
    return await V.correr_huella_por_ssh(argv, host_nombre, tope_s=tope_s)


def _registrar_aceptacion(registro_ruta: Path, *, host: str, mision_id: str, aceptado_por: str,
                          diff: tuple, sin_medir: bool, motivo: str | None = None,
                          identidad_declarada_por: str = "proceso") -> int:
    """Deja constancia de la aceptación en el registro append-only de C3 (el mismo que
    ya audita cada paso del cerebro -- `jax.ejecutor.contratos.registro`, cadena
    encadenada en hall9000).

    M-1 (ronda 8): `/var/log/jax-ejecutor/registro.jsonl` es de `jaxsvc`, con ACL
    `fruiz:r--` -- fruiz SÓLO puede LEERLO, no escribir ahí (verificado en producción,
    `getfacl`, 2026-09-22: `user::rw-`, `user:fruiz:r--`, `other::---`; y el
    DIRECTORIO, `/var/log/jax-ejecutor/`, también sólo le da `fruiz:r-x` -- ni
    siquiera puede CREAR un archivo nuevo ahí). Corre esta CLI con
    `sudo -u jaxsvc python -m jax.ejecutor.contratos.huella aceptar ...` (no
    `sudo python -m ...`: eso escribiría como root, y la marca de la huella -- que SÍ
    es escribible por fruiz vía la ACL de `JAX_EJECUTOR_MISIONES` -- quedaría con un
    dueño distinto al resto del árbol si el mismo proceso la toca de paso).

    M-2 (ronda 9, MAJOR-2): `aceptado_por` sale de `SUDO_UID`, y el evento lo dice tal
    cual -- `identidad_declarada_por` queda como `"sudo"` o `"proceso"` (ver
    `_resolver_identidad_invocante`). No es "identidad verificada": es lo que el
    entorno DECLARÓ. El control real de quién pudo llegar hasta acá son los permisos
    de este mismo registro y de `JAX_EJECUTOR_MISIONES` (ambos `jaxsvc`) más quién
    tiene sudo real hacia `jaxsvc` en esta máquina."""
    from jax.ejecutor.contratos.registro import Registro

    reg = Registro(registro_ruta)
    try:
        return reg.anotar({"evento": "huella_aceptada", "host": host, "mision_id": mision_id,
                           "aceptado_por": aceptado_por, "sin_medir": sin_medir, "diff_aceptado": list(diff),
                           "motivo": motivo, "identidad_declarada_por": identidad_declarada_por})
    finally:
        reg.cerrar()


async def aceptar(*, misiones: Path, mision_id: str, host: str, aceptado_por: str, sin_medir: bool = False,
                  motivo: str | None = None, identidad_declarada_por: str = "proceso",
                  politica_ruta: Path | None = None, admin_usuario: str | None = None,
                  huella_llave: Path | None = None, huella_known_hosts: Path | None = None,
                  registro_ruta: Path | None = None, pausa_ruta: Path | None = None,
                  tomar_huella_actual=None, registrar=_registrar_aceptacion,
                  ahora=None, salida=print) -> int:
    """El camino de aceptación explícita (M-1, ronda 7): muestra el diff que pausó,
    toma una línea base NUEVA (salvo `sin_medir`: una máquina que ya no existe o no
    responde -- se acepta sin comparar, registrado como tal), dice quién y cuándo en
    el registro de C3, deja la marca de nuevo `ABIERTA` con la nueva base, y BORRA la
    pausa GLOBAL del Ejecutor (`pausa_ruta`, `JAX_EJECUTOR_PAUSA`) -- sin esto no
    alcanza: `arranque.exigir_contratos` (C5) sigue viendo esa pausa puesta y rechaza
    CUALQUIER misión nueva, aunque la marca de la huella ya esté `ABIERTA` de nuevo; el
    pausa.json es un archivo APARTE del que la huella no sabe nada. Devuelve 0 si
    aceptó, 2 si no encontró la marca o si no hay nada que aceptar.

    B-2 (ronda 8): sólo se acepta una marca `REPORTADA` -- es la única con un diff de
    verdad que aceptar. `--sin-medir` es la ÚNICA excepción, y TAMBIÉN exige
    `REPORTADA` o `ABIERTA` (una máquina que se cayó a mitad de turno, nunca llegó a
    compararse, y de verdad ya no responde): con `CERRADA` no hay nada pendiente, y
    `sin_medir` tampoco hace nada ahí. `motivo` (MINOR, ronda 8) es obligatorio con
    `sin_medir=True` -- se registra en el evento de C3, para que quede escrito POR QUÉ
    se aceptó sin poder confirmar nada.

    `tomar_huella_actual`/`registrar` inyectables (tests): por default,
    `tomar_huella_actual` es `_tomar_huella_actual` (ssh real con la llave PROPIA del
    servicio -- necesita `politica_ruta`/`admin_usuario`/`huella_llave`/
    `huella_known_hosts`; ver el arreglo del bug de producción jax#260, 2026-09-22, en
    el docstring de `argv_huella_servicio`) y `registrar` es `_registrar_aceptacion`
    (escribe en el registro real de C3, necesita `registro_ruta`).

    Barrido (ronda 9): al arrancar, limpia los temporales huérfanos que un kill puede
    haber dejado de una corrida anterior de `quitar_pausa_si` (`.{nombre}.quitar-tmp-*`
    -- nunca la pausa misma, ver `pausa.barrer_temporales_huerfanos`)."""
    from datetime import datetime, timezone

    if pausa_ruta is not None:
        pausa.barrer_temporales_huerfanos(pausa_ruta)

    if sin_medir and not (motivo and motivo.strip()):
        salida("codigo=motivo_obligatorio detalle=\"--sin-medir exige --motivo <texto>\"")
        return 2

    tomar = tomar_huella_actual or (
        lambda h: _tomar_huella_actual(h, politica_ruta=politica_ruta, admin_usuario=admin_usuario,
                                       huella_llave=huella_llave, huella_known_hosts=huella_known_hosts))

    ruta = ruta_huella(misiones, mision_id, host)

    async def _cuerpo() -> int:
        # MAJOR-K (ronda 7, auditoría adversarial 2026-09-22): la marca se leía UNA
        # sola vez, AFUERA de este candado (capturada como una variable ya fija que
        # `_cuerpo()` sólo heredaba por clausura) -- pese a que este mismo comentario
        # (desde ronda 10, auditoría 8) ya decía "TODO esto ... corre bajo el
        # candado", sin que fuera cierto para la lectura misma. Con DOS `aceptar()`
        # concurrentes sobre la MISMA marca REPORTADA: los dos leían `marca.estado ==
        # REPORTADA` ANTES de que ninguno tuviera el candado; el primero en
        # adquirirlo acepta y reescribe la marca como `ABIERTA`, lo suelta, y el
        # segundo -- que YA había decidido "es REPORTADA" con SU lectura vieja --
        # adquiere el candado y sigue igual, escribiendo una SEGUNDA línea base (con
        # SU PROPIA remedición, que puede ya no coincidir con la que el primero
        # aceptó) y un segundo evento de C3, absorbiendo en silencio cualquier cambio
        # real ocurrido entre medio sin reportarlo nunca. Ahora la lectura (y la
        # validación de estado) son lo PRIMERO que pasa DENTRO de `_cuerpo()` -- bajo
        # el candado, si hay uno -- así que el segundo `aceptar()` ve la marca YA
        # `ABIERTA` (la que el primero dejó) y la rechaza. Sin `pausa_ruta` (más
        # abajo, `_cuerpo()` corre SIN candado): no hay contra qué serializar, mismo
        # comportamiento de siempre.
        try:
            marca = leer_marca(ruta)
        except (OSError, ValueError, KeyError):
            salida(f"codigo=huella_no_encontrada host={host} mision_id={mision_id}")
            return 2

        estados_aceptables = (REPORTADA, ABIERTA) if sin_medir else (REPORTADA,)
        if marca.estado not in estados_aceptables:
            salida(f"codigo=huella_no_reportada host={host} mision_id={mision_id} estado={marca.estado}")
            return 2

        salida(f"--- diff de {host} ({mision_id}), estado={marca.estado} ---")
        for linea in marca.diff:
            salida(linea)
        salida("--- fin diff ---")

        if sin_medir:
            nueva_huella = marca.huella
        else:
            nueva_huella = await tomar(host)
            # MAJOR-I (ronda 6, auditoría adversarial 2026-09-22): esto NO llamaba a
            # `huella_valida()` -- una remedición rota (una ruta con `E ... find_fallo`,
            # sudo denegado a mitad de camino, lo que sea) se tomaba igual como la
            # NUEVA línea base, con `estado=ABIERTA` y la pausa BORRADA -- exactamente
            # lo que este mecanismo existe para impedir. `sin_medir` queda AFUERA de
            # este chequeo a propósito: ahí `nueva_huella` es la vieja marca ya
            # aceptada como tal por Fernando (`motivo` obligatorio, ver arriba), no una
            # medición nueva que pueda salir rota.
            rutas_exigidas = RUTAS_DECLARADAS_POR_DEFAULT
            if admin_usuario is not None:
                rutas_exigidas = rutas_exigidas + (ruta_authorized_keys_admin(admin_usuario),)
            if not huella_valida(nueva_huella, rutas=rutas_exigidas):
                salida(f"codigo=medicion_no_valida host={host} mision_id={mision_id} "
                      "detalle=\"la remedicion no esta completa -- no se acepta, la pausa sigue puesta\"")
                return 2

        momento = ahora() if ahora is not None else datetime.now(timezone.utc).isoformat()
        registrar(registro_ruta, host=host, mision_id=mision_id, aceptado_por=aceptado_por,
                 diff=marca.diff, sin_medir=sin_medir, motivo=motivo,
                 identidad_declarada_por=identidad_declarada_por)
        escribir_marca(ruta, Marca(huella=nueva_huella, estado=ABIERTA, aceptada_por=aceptado_por,
                                   aceptada_en=momento))
        if pausa_ruta is not None:
            # B-1 (ronda 8): NUNCA levantar una pausa ajena -- sólo la de ESTA huella
            # (origen=huella, mismo host, misma mision_id). Si C4/C5 pausaron por su
            # cuenta (o la huella de OTRO host/misión), se deja intacta: la huella ya
            # se aceptó, pero el Ejecutor sigue pausado por lo que sea que puso esa
            # otra pausa -- avisa con `codigo=pausa_de_otro_origen`, no falla.
            #
            # MINOR (ronda 10): para acá la marca y el registro YA se escribieron -- la
            # aceptación de la huella en sí es un hecho válido, independiente de que
            # este paso (un efecto colateral de conveniencia, no la parte autoritativa)
            # salga bien. Por eso el orden es éste -- marca/registro primero, pausa
            # después -- y por eso cualquier excepción de acá NUNCA puede escapar como
            # traceback crudo dejando la marca ya escrita y el mensaje final sin decir
            # nada: se atrapa, se reporta con un código claro, y `aceptar` sigue hasta
            # el final igual.
            try:
                borro, vista = pausa.quitar_pausa_si(
                    pausa_ruta, coincide=lambda d: (d.get("origen") == "huella" and d.get("host") == host
                                                    and d.get("mision_id") == mision_id))
            except Exception as exc:  # fail-soft: la huella YA se aceptó (marca+registro escritos); este paso es un efecto colateral, y se avisa con pausa_no_verificable -- no se traga en silencio
                salida(f"codigo=pausa_no_verificable tipo={type(exc).__name__} detalle=\"{exc}\"")
            else:
                # Nunca mudo (ronda 10): SIEMPRE dice qué pasó con la pausa, en los
                # tres casos -- se borró la propia, sigue puesta y es de otro origen, o
                # no había ninguna que quitar.
                if borro:
                    salida(f"codigo=pausa_propia_borrada host={host} mision_id={mision_id}")
                elif vista is not None:
                    salida(f"codigo=pausa_de_otro_origen origen={vista.get('origen')} "
                          f"motivo={vista.get('motivo')} host_de_la_pausa={vista.get('host')}")
                else:
                    salida("codigo=sin_pausa_que_quitar")
        salida(f"huella_aceptada=true host={host} mision_id={mision_id} sin_medir={sin_medir}")
        return 0

    if pausa_ruta is not None:
        with pausa.candado(pausa_ruta):
            return await _cuerpo()
    return await _cuerpo()


def _resolver_identidad_invocante(env) -> tuple[int, str, str]:
    """M-2 (ronda 8; corregido ronda 9 tras la auditoría 7, MAJOR-2): esto NO verifica
    identidad -- lee lo que `SUDO_UID` DECLARA. `SUDO_UID` es el uid de quien invocó
    `sudo` en la invocación MÁS EXTERNA que tocó este proceso, resuelto con `pwd` para
    confirmar que ese uid corresponde a ALGÚN usuario real del sistema (no basura ni un
    uid inexistente) -- pero esa confirmación es sobre el NÚMERO, no sobre la PERSONA:
    `pwd.getpwuid` no prueba que quien tecleó `sudo` sea de verdad el dueño de ese uid,
    sólo que el uid declarado existe. Cuando no hay `SUDO_UID` (corre sin `sudo`), cae a
    `os.getuid()` -- el uid real del proceso. Nunca se usan los strings
    `SUDO_USER`/`USER` directamente -- cualquiera puede exportarlos a mano.

    Devuelve `(uid, nombre, declarada_por)` -- `declarada_por` es `"sudo"` cuando el
    uid salió de `SUDO_UID`, o `"proceso"` cuando salió de `os.getuid()`. Esa etiqueta
    se registra TAL CUAL en el evento de C3 (`_registrar_aceptacion`): el registro dice
    "declarado por sudo", nunca "identidad verificada" -- no lo es.

    EL CONTROL DE VERDAD no es esta función: es quién puede escribir el registro y la
    marca (permisos de `/var/log/jax-ejecutor/` y de `JAX_EJECUTOR_MISIONES`, ambos de
    `jaxsvc`) MÁS quién tiene sudo real hacia `jaxsvc` en esta máquina -- hoy en
    hall9000, sólo `fruiz` (a `axioma` se le quitó el sudo ahí la noche del
    2026-09-22; verificado con `sudo -l -U axioma` → no permitido). Esta función sólo
    decide qué NOMBRE queda escrito junto a una acción que YA requirió ese acceso real
    para llegar hasta acá.

    LÍMITE, sin cerrar: cualquiera con una regla `ALL` (puede correr como CUALQUIER
    usuario, no sólo `jaxsvc`) puede encadenar `sudo -u <alguien> sudo -u jaxsvc ...`
    y hacer que `SUDO_UID` declare el uid de `<alguien>` en vez del propio -- no hace
    falta ser root, alcanza con esa regla. No hay forma de detectar eso desde acá."""
    valor = str(env.get("SUDO_UID", "")).strip()
    if valor.isdigit():
        try:
            uid = int(valor)
            return uid, pwd.getpwuid(uid).pw_name, "sudo"
        except (KeyError, OverflowError, ValueError):  # fail-soft: SUDO_UID inválido o inexistente -- cae al os.getuid() real del proceso, el fallback documentado
            pass
    uid = os.getuid()
    return uid, pwd.getpwuid(uid).pw_name, "proceso"


def principal(argv: list[str]) -> int:
    """`python -m jax.ejecutor.contratos.huella aceptar --host <host> --mision <id>
    [--sin-medir --motivo "<texto>"]` -- lee `JAX_EJECUTOR_MISIONES`,
    `JAX_EJECUTOR_ADMIN_USUARIO`, `JAX_EJECUTOR_REGISTRO`, `JAX_EJECUTOR_PAUSA` (se
    borra al aceptar, y SÓLO si es la pausa de ESTA huella -- ver B-1 en `aceptar()`),
    `JAX_EJECUTOR_CUENTA` (la cuenta del Ejecutor, `axioma`), la política exportada
    (`JAX_EJECUTOR_POLITICA`, misma que lee `cuenta_axioma.cuenta_desde_entorno`) y,
    desde el arreglo del bug de producción (jax#260, 2026-09-22),
    `JAX_EJECUTOR_HUELLA_LLAVE`/`JAX_EJECUTOR_HUELLA_KNOWN_HOSTS` (la llave PROPIA del
    servicio para volver a medir -- ver `argv_huella_servicio`) del entorno. Las dos
    últimas se exigen SIEMPRE, aunque la corrida termine usando `--sin-medir`: mismo
    criterio que ya regía para `politica_ruta`/`admin_usuario`, que tampoco hacían falta
    para ese camino y de todos modos se piden por adelantado -- fail-closed sobre
    configuración incompleta, no sobre si esta corrida en particular los va a usar.

    M-1 (ronda 8): el registro de C3 (`/var/log/jax-ejecutor/registro.jsonl`) y el
    árbol de misiones (`JAX_EJECUTOR_MISIONES`) son de `jaxsvc` -- `fruiz` sólo tiene
    lectura sobre el primero. Se corre así:

        sudo -u jaxsvc python -m jax.ejecutor.contratos.huella aceptar \\
            --host <host> --mision <id>

    (NO `sudo python -m ...`: eso escribe como root, no como el dueño real del árbol.)

    EL CONTROL DE VERDAD (ronda 9, MAJOR-2, tras la auditoría 7): no es un chequeo de
    identidad dentro de este código -- son los permisos del registro y de
    `JAX_EJECUTOR_MISIONES` (ambos `jaxsvc`) MÁS quién tiene sudo real hacia `jaxsvc`.
    HECHO (verificado 2026-09-22, la misma noche): a `axioma` se le quitó el sudo en
    hall9000 (`sudo -l -U axioma` → no permitido; las máquinas remotas lo conservan) --
    así que hoy, en hall9000, `aceptar` sólo lo puede correr de punta a punta quien
    tenga sudo hacia `jaxsvc`, y eso hoy es sólo `fruiz`. `SUDO_UID` (ver
    `_resolver_identidad_invocante`) decide el NOMBRE que queda escrito como
    `aceptado_por` -- pero eso es un DATO DECLARADO por el entorno, no una identidad
    verificada por este proceso; el registro lo etiqueta `identidad_declarada_por`.

    El rechazo de más abajo (si el uid invocante es el de `JAX_EJECUTOR_CUENTA`) es
    protección contra el ERROR ACCIDENTAL -- alguien corriendo esto sin darse cuenta
    de qué cuenta es -- no una barrera anti-suplantación: quien de verdad tiene sudo
    hacia `jaxsvc` puede declarar cualquier `SUDO_UID` que quiera (ver el LÍMITE en
    `_resolver_identidad_invocante`)."""
    import argparse
    import asyncio
    import os as _os

    p = argparse.ArgumentParser(prog="python -m jax.ejecutor.contratos.huella")
    sub = p.add_subparsers(dest="comando", required=True)
    ac = sub.add_parser("aceptar", help="Acepta una huella REPORTADA y desbloquea el host.")
    ac.add_argument("--host", required=True)
    ac.add_argument("--mision", required=True, dest="mision_id")
    ac.add_argument("--sin-medir", action="store_true",
                    help="La máquina ya no existe o no responde: acepta sin volver a medir.")
    ac.add_argument("--motivo", default=None,
                    help="Obligatorio con --sin-medir: por qué se acepta sin remedir. Queda en el registro.")
    args = p.parse_args(argv)

    env = _os.environ
    try:
        cuenta_ejecutor = env["JAX_EJECUTOR_CUENTA"]
        uid_invocante, aceptado_por, declarada_por = _resolver_identidad_invocante(env)
        try:
            uid_axioma = pwd.getpwnam(cuenta_ejecutor).pw_uid
        except KeyError:
            uid_axioma = None
        if uid_axioma is not None and uid_invocante == uid_axioma:
            # Freno del error accidental (no anti-suplantación, ver docstring arriba).
            print(f"codigo=axioma_no_puede_aceptar_su_propia_huella cuenta={cuenta_ejecutor}", file=sys.stderr)
            return 2

        from jax.ejecutor.contratos import pausa as _pausa
        return asyncio.run(aceptar(
            misiones=Path(env["JAX_EJECUTOR_MISIONES"]), mision_id=args.mision_id, host=args.host,
            politica_ruta=Path(env["JAX_EJECUTOR_POLITICA"]), admin_usuario=env["JAX_EJECUTOR_ADMIN_USUARIO"],
            huella_llave=Path(env["JAX_EJECUTOR_HUELLA_LLAVE"]),
            huella_known_hosts=Path(env["JAX_EJECUTOR_HUELLA_KNOWN_HOSTS"]),
            registro_ruta=Path(env["JAX_EJECUTOR_REGISTRO"]), pausa_ruta=_pausa.ruta_de_la_pausa(env),
            aceptado_por=aceptado_por, sin_medir=args.sin_medir, motivo=args.motivo,
            identidad_declarada_por=declarada_por))
    except KeyError as exc:
        print(f"codigo=sin_configurar variable={exc.args[0]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
