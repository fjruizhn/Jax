#!/usr/bin/env python3
"""ops/rutas_de_produccion_verificador.py -- la lógica de
`ops/rutas-de-produccion.sh --verificar`, en un módulo Python importable y
probado (auditoría de escalón 3, PR jax#277, MAJOR-3: el guion original en
bash fallaba ABIERTO en 6 casos reales -- clave en alcance ausente, valor
vacío, valor entre comillas, espacios junto al `=`, un symlink que resuelve
a `/home/fruiz/jax`, y `JAX_WORKSPACE_DIR` pasando en silencio sin ninguna
excepción declarada).

`ops/rutas-de-produccion.sh` es un envoltorio fino: lee `/etc/jax/.env` con
`sudo -n cat`, le pasa el TEXTO a este módulo (nunca hace a este módulo leer
la ruta por su cuenta -- `ENV_FILE` no es configurable por entorno, a
propósito, para que nadie pueda desviar la verificación de producción con
una variable de entorno puesta por error), y usa `resolver_como_jaxsvc` /
`jaxsvc_puede` (subprocesos `sudo -n -u jaxsvc realpath -e` / `test -r|-w`)
para las partes que de verdad necesitan tocar el filesystem como esa cuenta
de servicio.

Diseño fail-closed en cada punto donde el guion viejo fallaba abierto:

- **Ausente / duplicada / vacía**: cada clave de `KEYS_EN_ALCANCE` tiene que
  aparecer EXACTAMENTE una vez en el archivo y con un valor no vacío tras
  normalizar. Cero apariciones, dos o más, o un valor vacío -- las tres son
  FALLA, nunca "no se dice nada".
- **Comillas / espacios junto al `=`**: `_LINEA` exige `NOMBRE=valor` sin
  espacio entre el nombre y el `=` (una línea `JAX_BIN = /x` no matchea esa
  forma exacta y la clave queda AUSENTE -- exactamente el caso de arriba, no
  un valor mal leído en silencio). `normalizar_valor` sólo acepta un valor
  SIN comillas y sin espacio en ningún extremo, o un valor COMPLETO entre
  comillas dobles o simples (sin comillas embebidas, sin escapes) -- una
  mezcla ambigua (`"algo` sin cerrar, comillas mixtas, espacio antes o
  después de un valor sin comillas) levanta `ValorAmbiguo`: no se adivina lo
  que systemd habría hecho, se rechaza.
- **Symlink hacia el checkout de trabajo**: las tres claves en alcance se
  comparan por su ruta REAL (`realpath -e`, corrido como `jaxsvc` -- la
  cuenta que de verdad las usa), no por el string crudo del `.env`. Un
  symlink que apunte a `/home/fruiz/jax/...` se atrapa aunque el valor
  literal del `.env` diga otra cosa.
- **Lista de PERMITIDOS, no de prohibidos**: para las tres claves en
  alcance, la ruta real tiene que empezar con `/srv/jax-prod/`,
  `/srv/jax-data/` o `/var/log/jax/` -- cualquier otra cosa es FALLA, así
  sea `/home/fruiz/jax` o cualquier lugar que a nadie se le ocurrió excluir
  todavía. Antes era una lista de PROHIBIDOS (`/home/fruiz/jax/`,
  `/home/fruiz/.local/`): cualquier ruta que no coincidiera pasaba, aunque
  fuera otra ruta igual de mala.
- **`JAX_WORKSPACE_DIR` como excepción DECLARADA, no como suerte de
  string**: el chequeo genérico (para el resto de las claves de ruta, las
  que no están en alcance) es ahora contra `/home/*` completo, no sólo
  `/home/fruiz/jax/` -- una red más ancha. `JAX_WORKSPACE_DIR` vive bajo
  `/home/fruiz/jax-workspace` A PROPÓSITO (decisión de diseño de la tarea
  original: no es checkout de código ni dato de producción) y está en
  `EXCEPCIONES_FASE_A` con el motivo escrito, igual que
  `JAX_MISSIONS_DIR`/`JAX_BIN`. Antes "pasaba" solo porque su valor no
  empezaba exactamente con `/home/fruiz/jax/` (con barra) -- un accidente
  de substring, no una decisión verificada.

Sin encoding: se asume UTF-8, igual que el resto del repo.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
#  Constantes -- mismo criterio de "clave de ruta" en todo el repo: sólo los
#  sufijos de abajo se leen o imprimen (nunca credenciales, tokens, hosts).
# ---------------------------------------------------------------------------

SUFIJOS_DE_RUTA = ("_PATH", "_DIR", "_BASE", "_BIN", "_FILE", "_LOG")

#: Las tres claves que ESTE cambio mueve (ver docs/runbooks/rutas-de-produccion.md).
KEYS_EN_ALCANCE = ("JAX_CONFIG_PATH", "JAX_AUDIT_LOG_PATH", "JAX_REPO_BASE")

#: Lista de PERMITIDOS (no de prohibidos) para las claves en alcance -- ver
#: docstring del módulo.
PERMITIDOS_EN_ALCANCE = ("/srv/jax-prod/", "/srv/jax-data/", "/var/log/jax/")

#: Claves de ruta EXCLUIDAS del chequeo genérico "nada bajo /home/*", con
#: motivo escrito -- nunca por omisión (mismo criterio que EXCEPCIONES en
#: policy/tests/test_archivos_de_test_wireados_en_ci.py).
EXCEPCIONES_FASE_A: dict[str, str] = {
    "JAX_MISSIONS_DIR": (
        "consumidor real (jax-platform/backend/api/command.py: "
        "MISSIONS_DIR = ruta_absoluta_requerida(\"JAX_MISSIONS_DIR\")); no se "
        "mueve en este cambio -- mover el checkout que ejecuta cada misión es "
        "una decisión de arquitectura mayor, pendiente de que Fernando la "
        "tome. Ver docs/runbooks/rutas-de-produccion.md."
    ),
    "JAX_BIN": (
        "consumidor real (jax-platform/backend/api/command.py, invocado como "
        "$JAX_BIN --task <mission_file>); el lanzador (~/.local/bin/jax) hace "
        "cd $HOME/jax y usa su propio .venv -- moverlo decide DESDE QUÉ "
        "CHECKOUT corre cada misión, no sólo una ruta de config/log. "
        "Pendiente, decisión de Fernando."
    ),
    "JAX_WORKSPACE_DIR": (
        "decisión de diseño de la tarea original (HECHOS, 2026-09-25): vive "
        "bajo /home/fruiz/jax-workspace A PROPÓSITO -- no es el checkout de "
        "código de un agente ni un dato de producción, es un directorio de "
        "trabajo aparte. Se queda."
    ),
}

#: Claves con sufijo de ruta que NO son una ruta de sistema de archivos --
#: plantillas de URL de frontend. `realpath`/`test -r` sobre esto no tiene
#: sentido; se excluyen de las dos fases.
NO_ES_RUTA_DE_FILESYSTEM: dict[str, str] = {
    "JAX_PIPELINE_DETAIL_PATH": "plantilla de ruta de frontend (/historial/{pipeline_id}), no existe en disco.",
    "PIPELINE_DETAIL_PATH": "misma plantilla, alias sin prefijo JAX_.",
}

# NOMBRE=valor, SIN espacio entre el nombre y el "=" -- ver docstring del
# módulo: una línea con espacio ahí (`JAX_BIN = /x`) no matchea, y la clave
# queda como "ausente" en vez de leerse mal en silencio.
_LINEA = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def es_clave_de_ruta(clave: str) -> bool:
    return clave.endswith(SUFIJOS_DE_RUTA)


def parsear_env(texto: str) -> dict[str, list[str]]:
    """clave -> lista de valores CRUDOS (sin normalizar) en el orden en que
    aparecen. Una clave con 2+ entradas es una clave DUPLICADA -- se detecta
    acá (longitud de la lista), no se resuelve "el último gana" como haría
    systemd: para las claves en alcance eso es ambiguo a propósito y
    `clasificar_alcance` lo rechaza.

    Comentario = línea cuyo primer carácter no blanco es "#". Líneas en
    blanco se ignoran. Sin continuación de línea ni escapes de systemd más
    allá de lo que `normalizar_valor` entiende explícitamente -- ver ahí."""
    resultado: dict[str, list[str]] = {}
    for linea in texto.splitlines():
        despojada = linea.strip()
        if not despojada or despojada.startswith("#"):
            continue
        m = _LINEA.match(linea)
        if not m:
            continue
        clave, crudo = m.group(1), m.group(2)
        resultado.setdefault(clave, []).append(crudo)
    return resultado


class ValorAmbiguo(ValueError):
    """El valor crudo no es, sin ambigüedad, ni un valor sin comillas y sin
    espacios en los extremos, ni un valor completo entre comillas parejas."""


_SIN_COMILLAS_SIN_ESPACIOS = re.compile(r"""^[^\s'"]+$""")
_ENTRE_COMILLAS_DOBLES = re.compile(r'^"([^"]*)"$')
_ENTRE_COMILLAS_SIMPLES = re.compile(r"^'([^']*)'$")


def normalizar_valor(crudo: str) -> str:
    """Ver docstring del módulo (sección de comillas/espacios). Puede estar
    vacío (`""`, `''` o cadena vacía literal) -- eso lo decide el llamador,
    no esta función."""
    if crudo == "":
        return ""
    m = _ENTRE_COMILLAS_DOBLES.match(crudo)
    if m:
        return m.group(1)
    m = _ENTRE_COMILLAS_SIMPLES.match(crudo)
    if m:
        return m.group(1)
    if _SIN_COMILLAS_SIN_ESPACIOS.match(crudo):
        return crudo
    raise ValorAmbiguo(
        f"valor {crudo!r} no es interpretable sin ambigüedad (¿comillas a "
        "medias, espacio suelto, comillas mezcladas?)"
    )


@dataclass
class Hallazgo:
    clave: str
    motivo: str


@dataclass
class ResultadoFaseA:
    problemas: list[Hallazgo] = field(default_factory=list)
    revisadas: int = 0

    @property
    def ok(self) -> bool:
        return not self.problemas


def _valor_unico_normalizado(clave: str, crudos: list[str]) -> tuple[str | None, str | None]:
    """(valor, None) si la clave aparece exactamente una vez con un valor
    normalizable y no vacío. (None, motivo) en cualquier otro caso."""
    if not crudos:
        return None, "clave ausente en /etc/jax/.env"
    if len(crudos) > 1:
        return None, f"clave duplicada ({len(crudos)} apariciones) -- ambiguo, cuál vale?"
    try:
        valor = normalizar_valor(crudos[0])
    except ValorAmbiguo as exc:
        return None, str(exc)
    if not valor:
        return None, "valor vacío"
    return valor, None


def verificar_fase_a(
    entorno: dict[str, list[str]],
    resolver: Callable[[str], str | None],
) -> ResultadoFaseA:
    """`resolver(ruta_cruda) -> ruta_real | None`. En producción,
    `resolver_como_jaxsvc` (sudo -n -u jaxsvc realpath -e). En tests, un
    dict de mentira -- así un symlink se prueba sin tocar el filesystem.
    `None` significa "no se pudo resolver" (no existe, sin permiso): eso
    también es un problema de Fase A, se reporta como tal.

    Dos chequeos:
      (1) Las tres claves de KEYS_EN_ALCANCE: exactamente una vez, valor no
          vacío, ruta REAL dentro de PERMITIDOS_EN_ALCANCE.
      (2) El resto de las claves de ruta presentes: ruta REAL fuera de
          /home/* -- salvo las de EXCEPCIONES_FASE_A y las de
          NO_ES_RUTA_DE_FILESYSTEM (no se resuelven ni se chequean, no son
          rutas de filesystem)."""
    resultado = ResultadoFaseA()

    for clave in KEYS_EN_ALCANCE:
        resultado.revisadas += 1
        valor, motivo = _valor_unico_normalizado(clave, entorno.get(clave, []))
        if motivo:
            resultado.problemas.append(Hallazgo(clave, motivo))
            continue
        real = resolver(valor)
        if real is None:
            resultado.problemas.append(
                Hallazgo(clave, f"{valor}: no se pudo resolver la ruta real (¿no existe? ¿sin permiso?)"))
            continue
        if not any(real.startswith(p) for p in PERMITIDOS_EN_ALCANCE):
            resultado.problemas.append(
                Hallazgo(clave, f"{valor} (real: {real}) no está bajo ninguno de {PERMITIDOS_EN_ALCANCE}"))

    for clave, crudos in entorno.items():
        if clave in KEYS_EN_ALCANCE or clave in EXCEPCIONES_FASE_A or clave in NO_ES_RUTA_DE_FILESYSTEM:
            continue
        if not es_clave_de_ruta(clave):
            continue
        if len(crudos) != 1:
            continue  # ambigüedad de una clave fuera de alcance no es lo que esta fase audita
        try:
            valor = normalizar_valor(crudos[0])
        except ValorAmbiguo:
            continue
        if not valor:
            continue
        resultado.revisadas += 1
        real = resolver(valor) or valor
        if real.startswith("/home/") or real.startswith("/home"):
            # startswith("/home") a secas cubre tambien el caso limite "/home" pelado
            if real == "/home" or real.startswith("/home/"):
                resultado.problemas.append(
                    Hallazgo(clave, f"{valor} (real: {real}) está bajo /home/ -- sin excepción declarada"))

    return resultado


# ---------------------------------------------------------------------------
#  Fase B -- jaxsvc puede leer (y, para dos claves, escribir) las rutas en
#  alcance. Subprocesos reales aislados en funciones chicas para que los
#  tests puedan inyectar un `ejecutar` de mentira sin sudo.
# ---------------------------------------------------------------------------

EjecutarSudo = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _ejecutar_sudo_real(argv: list[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(argv, capture_output=True, text=True, timeout=15)


def resolver_como_jaxsvc(ruta: str, ejecutar: EjecutarSudo = _ejecutar_sudo_real) -> str | None:
    """`realpath -e` (existe Y resuelve symlinks) corrido como jaxsvc -- la
    cuenta real que usa estas rutas, no fruiz ni root. `None` si no existe o
    sin permiso."""
    r = ejecutar(["sudo", "-n", "-u", "jaxsvc", "realpath", "-e", "--", ruta])
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def jaxsvc_puede_leer(ruta: str, ejecutar: EjecutarSudo = _ejecutar_sudo_real) -> bool:
    return ejecutar(["sudo", "-n", "-u", "jaxsvc", "test", "-r", "--", ruta]).returncode == 0


def jaxsvc_puede_escribir(ruta: str, ejecutar: EjecutarSudo = _ejecutar_sudo_real) -> bool:
    return ejecutar(["sudo", "-n", "-u", "jaxsvc", "test", "-w", "--", ruta]).returncode == 0


@dataclass
class ResultadoFaseB:
    problemas: list[Hallazgo] = field(default_factory=list)


def verificar_fase_b(
    entorno: dict[str, list[str]],
    ejecutar: EjecutarSudo = _ejecutar_sudo_real,
) -> ResultadoFaseB:
    """Sólo corre sobre las tres claves en alcance, y sólo si Fase A ya las
    encontró con un valor único y no vacío (si Fase A ya las marcó ausentes/
    duplicadas/vacías/ambiguas, Fase B no tiene nada que probar)."""
    resultado = ResultadoFaseB()
    for clave in KEYS_EN_ALCANCE:
        valor, motivo = _valor_unico_normalizado(clave, entorno.get(clave, []))
        if motivo:
            continue  # ya lo reportó Fase A
        if not jaxsvc_puede_leer(valor, ejecutar):
            resultado.problemas.append(Hallazgo(clave, f"{valor}: jaxsvc no puede LEER esta ruta"))
        if clave == "JAX_AUDIT_LOG_PATH" and not jaxsvc_puede_escribir(valor, ejecutar):
            resultado.problemas.append(Hallazgo(clave, f"{valor}: jaxsvc no puede ESCRIBIR el log de auditoría"))
        if clave == "JAX_REPO_BASE":
            documentos = valor.rstrip("/") + "/documents"
            if not jaxsvc_puede_escribir(documentos, ejecutar):
                resultado.problemas.append(Hallazgo(clave, f"jaxsvc no puede ESCRIBIR en {documentos}"))
    return resultado


# ---------------------------------------------------------------------------
#  CLI -- envoltura fina. `ops/rutas-de-produccion.sh --verificar` hace
#  `sudo -n cat /etc/jax/.env` y le pasa el TEXTO a este módulo por stdin;
#  la ruta del archivo nunca es configurable por variable de entorno.
# ---------------------------------------------------------------------------

def verificar(texto_env: str) -> tuple[bool, str]:
    """(ok, reporte). `ok=False` con CUALQUIER problema de Fase A o Fase B."""
    entorno = parsear_env(texto_env)
    fase_a = verificar_fase_a(entorno, resolver_como_jaxsvc)
    fase_b = verificar_fase_b(entorno, _ejecutar_sudo_real) if fase_a.ok else ResultadoFaseB()

    lineas: list[str] = []
    if fase_a.problemas:
        lineas.append("FASE A -- claves de ruta con problemas:")
        for h in fase_a.problemas:
            lineas.append(f"  {h.clave}: {h.motivo}")
    if fase_b.problemas:
        lineas.append("FASE B -- jaxsvc no tiene el acceso que necesita:")
        for h in fase_b.problemas:
            lineas.append(f"  {h.clave}: {h.motivo}")

    if fase_a.problemas or fase_b.problemas:
        return False, "\n".join(lineas)
    return True, (
        f"rutas-de-produccion: todas las rutas de producción en alcance están "
        f"fuera de /home y jaxsvc tiene el acceso que necesita "
        f"({fase_a.revisadas} claves de ruta revisadas)"
    )


def _main(argv: list[str]) -> int:
    import sys

    if len(argv) != 2 or argv[1] != "--verificar":
        print(f"uso: {argv[0]} --verificar", file=sys.stderr)
        return 2

    r = _ejecutar_sudo_real(["sudo", "-n", "test", "-r", "/etc/jax/.env"])
    if r.returncode != 0:
        print("rutas-de-produccion: sudo -n no puede leer /etc/jax/.env "
              "(falta la credencial cacheada de sudo -n?)", file=sys.stderr)
        return 1

    leido = _ejecutar_sudo_real(["sudo", "-n", "cat", "/etc/jax/.env"])
    if leido.returncode != 0:
        print("rutas-de-produccion: sudo -n cat /etc/jax/.env falló", file=sys.stderr)
        return 1

    ok, reporte = verificar(leido.stdout)
    print(reporte, file=(sys.stdout if ok else sys.stderr))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv))
