# jax/ejecutor/contratos/huella.py
"""Integridad de los propios controles del Ejecutor, por ESTADO, no por texto de
comando (M-1/M-2, ronda 3, auditoría adversarial 2026-09-22 -- "no persigas más
patrones léxicos: es la tercera vez que un filtro sobre el texto no converge").

Un patrón léxico (C1, `NUCLEO_REGLAS` en politica.py) sólo atrapa lo que el comando
DICE que hace, escrito de la forma que el patrón espera -- un envoltorio nuevo, un
alias, `python3 -c`, `sed` en vez de un editor, todo lo cambia y el patrón deja de
aplicar. Lo que no cambia es el RESULTADO: si `/etc/sudoers.d/*` es distinto después
de la misión que antes, ALGO lo tocó, sin importar cómo.

El controlador (fruiz, desde hall9000 -- el mismo que ya entra por ssh a las máquinas
para C6) toma una HUELLA (sha256 + listado + atributos) de los archivos que hacen
cumplir C3/C4/C6 en cada máquina de la misión con sudo, AL ARRANCAR y AL CERRAR. Si
difieren y el cambio no está en el plan declarado de la misión (`cambio_declarado`,
más abajo -- por ahora, mención literal en el texto de la misión: el mismo criterio
con el que ya se juzga `fuera_de_mision` en auditor_instrucciones.md), es un hallazgo;
si la misión sigue viva, se pone la pausa (C4) -- ver `jax.ejecutor.contratos.pausa`.

Identidad: "prohibido tocarlos, y se detecta al cierre" -- nunca "el gancho lo
impide": el gancho (léxico) no puede prometer eso, y esto es lo que sí lo prueba.

Sólo biblioteca estándar: lo corre `fruiz`/el controlador, no `axioma`.
"""
from __future__ import annotations

import hashlib
import shlex
from dataclasses import dataclass

#: Lo que hace cumplir C3 (registro de sudo), C4 (freno) y C6 (llaves/revocador) en
#: cada máquina -- rutas verificadas contra ops/ejecutor/instalar_en_maquina.sh y
#: jax/ejecutor/contratos/revocacion.py (2026-09-22), no inventadas. Dato, no código
#: (Principio IV): si mañana se vigila una ruta más, se agrega acá, no en un regex.
RUTAS_A_VIGILAR = (
    "/etc/sudoers",
    "/etc/sudoers.d",
    "/etc/ssh/authorized_keys.d",
    "/root/.ssh/authorized_keys",
    "/etc/cron.d",
    "/var/spool/cron/crontabs",
    "/etc/systemd/system",
)


def comando_huella(cuenta: str) -> str:
    """El comando REMOTO (root, vía `sudo -n`; lo corre el controlador, nunca axioma)
    que junta sha256 + listado de cada ruta de RUTAS_A_VIGILAR -- recursivo, ordenado,
    con la ruta en la salida (`sha256sum` ya la imprime) así un archivo AGREGADO o
    BORRADO también se nota, no sólo uno modificado -- y tamaño+inode+atributos
    (`lsattr`) del log de sudo de la cuenta. Una ruta ausente no es un error: cuenta
    como "no existe" en la huella (una máquina puede no tener unidades systemd
    propias, por ejemplo) -- por eso el `2>/dev/null || true` en cada tramo."""
    if not cuenta or "/" in cuenta or " " in cuenta:
        raise ValueError("cuenta_invalida")
    q = shlex.quote
    log = f"/var/log/sudo-{cuenta}.log"
    tramos = []
    for ruta in RUTAS_A_VIGILAR:
        tramos.append(
            f'(sudo -n find {q(ruta)} -type f -exec sha256sum {{}} + 2>/dev/null; '
            f'sudo -n find {q(ruta)} \\( -type l -o -type d \\) '
            f'-printf "%y %p -> %l\\n" 2>/dev/null) | sort'
        )
    tramos.append(f'sudo -n stat -c "log %s %i" {q(log)} 2>/dev/null || echo "log ausente"')
    tramos.append(f'sudo -n lsattr {q(log)} 2>/dev/null || echo "lsattr ausente"')
    return " ; ".join(tramos)


@dataclass(frozen=True)
class Huella:
    host: str
    texto: str
    sha256: str


def huella_desde_salida(host: str, salida: bytes) -> Huella:
    texto = salida.decode(errors="replace")
    return Huella(host, texto, hashlib.sha256(texto.encode()).hexdigest())


def cambio(antes: Huella, despues: Huella) -> bool:
    """¿Algo cambió? Compara el hash del TEXTO COMPLETO -- una línea de más, una de
    menos, o un sha256 de un archivo distinto, todo cuenta. `antes.host` y
    `despues.host` tienen que ser la MISMA máquina: comparar huellas de máquinas
    distintas no dice nada."""
    if antes.host != despues.host:
        raise ValueError("huellas_de_maquinas_distintas", antes.host, despues.host)
    return antes.sha256 != despues.sha256


def lineas_agregadas_o_quitadas(antes: Huella, despues: Huella) -> tuple[str, ...]:
    """Para el mensaje del hallazgo: qué LÍNEAS son distintas (agregadas o quitadas),
    no el texto entero -- más legible que volcar la huella completa. No pretende ser
    un diff con contexto; es lo mínimo para que el hallazgo diga QUÉ cambió."""
    a, d = set(antes.texto.splitlines()), set(despues.texto.splitlines())
    return tuple(sorted(a ^ d))


def cambio_declarado(linea: str, texto_mision: str) -> bool:
    """¿El texto de la misión menciona lo que cambió? Mismo criterio, deliberadamente,
    que ya usa el auditor de C5 para juzgar `fuera_de_mision` (auditor_instrucciones.md):
    una mención literal en el texto que Fernando/la plataforma declaró para ESTA
    misión. No es un parser de "plan estructurado" -- ese formato no existe hoy en
    `Mision`/`Turno` (texto libre) -- pero SÍ es verificable y no se puede fingir sin
    que quede escrito en la misión misma."""
    return bool(texto_mision.strip()) and any(
        palabra in texto_mision for palabra in _fragmentos_relevantes(linea))


def _fragmentos_relevantes(linea: str) -> tuple[str, ...]:
    """Las rutas absolutas que aparecen en la línea de la huella (la primera columna de
    `sha256sum`, o la ruta del listado) -- son lo único con valor de búsqueda; el
    hash/tamaño/inode no van a aparecer nunca en el texto de una misión."""
    return tuple(p for p in linea.split() if p.startswith("/"))


def hallazgos(antes: Huella, despues: Huella, texto_mision: str) -> tuple[str, ...]:
    """Las líneas cambiadas que NO están declaradas en la misión -- vacío si no hay
    ninguna (nada cambió, o todo lo que cambió está declarado)."""
    if not cambio(antes, despues):
        return ()
    return tuple(l for l in lineas_agregadas_o_quitadas(antes, despues) if not cambio_declarado(l, texto_mision))
