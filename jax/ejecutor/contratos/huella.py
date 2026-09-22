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
RUTAS_CONTROLES = (
    "/etc/sudoers",
    "/etc/sudoers.d",
    "/etc/ssh/sshd_config",
    "/etc/ssh/sshd_config.d",
    "/etc/ssh/authorized_keys.d",
    "/root/.ssh/authorized_keys",
)

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


def comando_huella() -> str:
    """El comando REMOTO para la huella -- sin `sudo -n` propio (lo corre el
    controlador, envuelto en UN solo `sudo -n sh -c '<esto>'`, ver `vigia_servicio.py`
    y `revocacion.argv_admin`). No depende de ninguna cuenta: ronda 6 quitó el log de
    sudo, que era lo único que sí dependía de un nombre."""
    tramos = [_tramo_ruta(r) for r in RUTAS_CONTROLES]
    tramos.append(_tramo_sbin_ejecutor())
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
# Corre como FRUIZ (nunca axioma -- mismo criterio que toda la toma de huella). Ver
# docs/ejecutor-huella-aceptar.md para el procedimiento completo y por qué existe.

async def _tomar_huella_actual(host_nombre: str, *, politica_ruta: Path, admin_usuario: str,
                               tope_s: float = 30) -> Huella:
    """Toma la huella de AHORA MISMO contra `host_nombre`, leyendo su `ip`/`puerto` de
    la política exportada (mismo camino que `vigia_servicio._tomar_huella` --
    `revocacion.argv_admin` + un solo `sudo -n sh -c`). Import diferido: evita un ciclo
    con `vigia_servicio` (que ya importa `huella`) y a `politica`/`revocacion`, que
    `huella.py` no necesita para nada más que esto."""
    import shlex

    from jax.ejecutor.contratos import politica as P
    from jax.ejecutor.contratos import revocacion
    from jax.ejecutor.contratos import vigia_servicio as V

    doc = json.loads(Path(politica_ruta).read_bytes())
    hosts = {h.nombre: h for h in P.validar(doc).hosts}
    h = hosts.get(host_nombre)
    if h is None:
        raise ValueError("host_desconocido", host_nombre)
    argv = revocacion.argv_admin(h, admin_usuario, f"sudo -n sh -c {shlex.quote(comando_huella())}")
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
    `tomar_huella_actual` es `_tomar_huella_actual` (ssh real, necesita
    `politica_ruta`/`admin_usuario`) y `registrar` es `_registrar_aceptacion` (escribe
    en el registro real de C3, necesita `registro_ruta`).

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
        lambda h: _tomar_huella_actual(h, politica_ruta=politica_ruta, admin_usuario=admin_usuario))

    ruta = ruta_huella(misiones, mision_id, host)
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

    momento = ahora() if ahora is not None else datetime.now(timezone.utc).isoformat()
    registrar(registro_ruta, host=host, mision_id=mision_id, aceptado_por=aceptado_por,
             diff=marca.diff, sin_medir=sin_medir, motivo=motivo,
             identidad_declarada_por=identidad_declarada_por)
    escribir_marca(ruta, Marca(huella=nueva_huella, estado=ABIERTA, aceptada_por=aceptado_por,
                               aceptada_en=momento))
    if pausa_ruta is not None:
        # B-1 (ronda 8): NUNCA levantar una pausa ajena -- sólo la de ESTA huella
        # (origen=huella, mismo host, misma mision_id). Si C4/C5 pausaron por su
        # cuenta (o la huella de OTRO host/misión), se deja intacta: la huella ya se
        # aceptó, pero el Ejecutor sigue pausado por lo que sea que puso esa otra
        # pausa -- avisa con `codigo=pausa_de_otro_origen`, no falla.
        borro, vista = pausa.quitar_pausa_si(
            pausa_ruta, coincide=lambda d: (d.get("origen") == "huella" and d.get("host") == host
                                            and d.get("mision_id") == mision_id))
        if not borro and vista is not None:
            salida(f"codigo=pausa_de_otro_origen origen={vista.get('origen')} "
                  f"motivo={vista.get('motivo')} host_de_la_pausa={vista.get('host')}")
    salida(f"huella_aceptada=true host={host} mision_id={mision_id} sin_medir={sin_medir}")
    return 0


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
        except (KeyError, OverflowError, ValueError):
            pass
    uid = os.getuid()
    return uid, pwd.getpwuid(uid).pw_name, "proceso"


def principal(argv: list[str]) -> int:
    """`python -m jax.ejecutor.contratos.huella aceptar --host <host> --mision <id>
    [--sin-medir --motivo "<texto>"]` -- lee `JAX_EJECUTOR_MISIONES`,
    `JAX_EJECUTOR_ADMIN_USUARIO`, `JAX_EJECUTOR_REGISTRO`, `JAX_EJECUTOR_PAUSA` (se
    borra al aceptar, y SÓLO si es la pausa de ESTA huella -- ver B-1 en `aceptar()`),
    `JAX_EJECUTOR_CUENTA` (la cuenta del Ejecutor, `axioma`) y la política exportada
    (`JAX_EJECUTOR_POLITICA`, misma que lee `cuenta_axioma.cuenta_desde_entorno`) del
    entorno.

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
            registro_ruta=Path(env["JAX_EJECUTOR_REGISTRO"]), pausa_ruta=_pausa.ruta_de_la_pausa(env),
            aceptado_por=aceptado_por, sin_medir=args.sin_medir, motivo=args.motivo,
            identidad_declarada_por=declarada_por))
    except KeyError as exc:
        print(f"codigo=sin_configurar variable={exc.args[0]}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(principal(sys.argv[1:]))
