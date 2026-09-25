#!/usr/bin/env python3
"""ops/rutas_de_produccion_verificador.py -- la lógica de
`ops/rutas-de-produccion.sh --verificar`, en un módulo Python importable y
probado.

Historia de las dos rondas de auditoría de escalón 3 sobre PR jax#277:

**Ronda 1 (MAJOR-3):** el guion original en bash fallaba ABIERTO en 6 casos
reales -- clave en alcance ausente, valor vacío, valor entre comillas,
espacios junto al `=`, un symlink que resuelve a `/home/fruiz/jax`, y
`JAX_WORKSPACE_DIR` pasando en silencio sin ninguna excepción declarada.
Se reescribió acá, en Python, con una lista de PERMITIDOS (no de
prohibidos) y excepciones declaradas con motivo.

**Ronda 2 (MAJOR-A):** el parser de la ronda 1 seguía fallando abierto:
`_LINEA.match` corría sobre la línea SIN recortar y, cuando una línea no
calzaba el patrón estricto, la ignoraba EN SILENCIO en vez de reportarla.
Dos casos reales que systemd SÍ aplica (y este módulo antes no veía en
absoluto):

  - `  JAX_CONFIG_PATH=/home/fruiz/jax/...` (indentada): el `^` del regex
    exige que la clave empiece en la posición 0 -- con espacio antes, la
    línea entera se descartaba como si no existiera, y si ANTES había una
    línea limpia con el valor correcto, ese valor correcto "ganaba" en la
    lectura de este módulo mientras que systemd, que SÍ interpreta la
    indentada como una asignación válida (gana la última), terminaba
    cargando la ruta mala.
  - `JAX_AUDIT_LOG_PATH = /home/fruiz/jax/...` (espacio junto al `=`):
    mismo problema -- la línea no calzaba, se ignoraba, y si coexistía con
    una línea limpia anterior, este módulo reportaba "todo bien" mientras
    la variable de entorno REAL podía terminar siendo otra.

El arreglo: `parsear_env` ya NO ignora nada. Toda línea no vacía y no
comentario que no calce `^NOMBRE=valor$` exacto (sin espacio antes del
nombre, sin espacio antes del `=`) es un ERROR DE PARSEO, con su número de
línea, y CUALQUIER error de parseo -- esté cerca de una clave en alcance o
no -- tira abajo el resultado completo de `verificar()`: si este módulo no
puede leer el archivo con confianza, no certifica nada. Líneas que terminan
que terminan en una barra invertida (continuación de línea de systemd, que este módulo no reproduce) se
tratan igual, como error de parseo.

Además, ronda 2 agrega una fase de VERDAD EFECTIVA (Fase C): en vez de
confiar sólo en la lectura estática del archivo, lee `/proc/<MainPID>/environ`
de `jax-las-manos` y `jax-platform` de verdad (sudo -n) y compara contra la
MISMA lista de permitidos -- lo que el kernel dice que el proceso tiene
cargado, no lo que este módulo interpretó que debería tener.

Y la lista de permitidos pasa de ser compartida (Fase A ronda 1: cualquiera
de las 3 claves podía apuntar a cualquiera de los 3 destinos) a ser POR
CLAVE: `JAX_CONFIG_PATH` sólo bajo `/srv/jax-prod/jax/config/`,
`JAX_AUDIT_LOG_PATH` sólo bajo `/var/log/jax/las_manos/`, `JAX_REPO_BASE`
sólo exactamente `/srv/jax-data/repo`.

`ops/rutas-de-produccion.sh` es un envoltorio fino: lee `/etc/jax/.env` con
`sudo -n cat`, le pasa el TEXTO a este módulo por stdin (`ENV_FILE` no es
configurable por entorno, a propósito).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable

# ---------------------------------------------------------------------------
#  Constantes
# ---------------------------------------------------------------------------

SUFIJOS_DE_RUTA = ("_PATH", "_DIR", "_BASE", "_BIN", "_FILE", "_LOG")

#: Las tres claves que ESTE cambio mueve (ver docs/runbooks/rutas-de-produccion.md).
KEYS_EN_ALCANCE = ("JAX_CONFIG_PATH", "JAX_AUDIT_LOG_PATH", "JAX_REPO_BASE")

#: Lista de PERMITIDOS, POR CLAVE (ronda 2, MINOR): antes era una lista
#: compartida entre las tres claves -- JAX_CONFIG_PATH podía "pasar" apuntando
#: a /srv/jax-data (el destino de JAX_REPO_BASE) sin que nada lo objetara.
#: Cada clave tiene ahora exactamente el destino que le corresponde.
PERMITIDOS_POR_CLAVE: dict[str, tuple[str, ...]] = {
    "JAX_CONFIG_PATH": ("/srv/jax-prod/jax/config/",),
    "JAX_AUDIT_LOG_PATH": ("/var/log/jax/las_manos/",),
    "JAX_REPO_BASE": ("/srv/jax-data/repo",),
}

#: Claves de ruta EXCLUIDAS del chequeo genérico "nada bajo /home/*", con
#: motivo escrito -- nunca por omisión.
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
    "JAX_KILL_SWITCH_PATH": (
        "existencia OPCIONAL a propósito, no un error -- verificado 2026-09-25: "
        "es el archivo del freno de emergencia (interruptor.py): PRESENTE "
        "significa 'JAX pausado', AUSENTE significa 'operando normal'. "
        "`realpath -e` (exige existencia) fallando es el estado SANO por "
        "defecto -- tratar eso como una violación de ruta sería fail-open al "
        "revés: alarmar en el caso normal. Ronda 2 de la auditoría de "
        "escalón 3 (jax#277) volvió el chequeo genérico estricto ante "
        "'realpath fallido', y ESE endurecimiento sacó a la luz este caso: "
        "antes pasaba de largo por accidente (el chequeo viejo ignoraba en "
        "silencio una ruta que no resolvía), no porque estuviera bien "
        "declarado. Corre igual bajo /etc/jax/, nunca bajo /home/."
    ),
}

#: Claves con sufijo de ruta que NO son una ruta de sistema de archivos --
#: plantillas de URL de frontend. `realpath`/`test -r` sobre esto no tiene
#: sentido; se excluyen de las dos fases.
NO_ES_RUTA_DE_FILESYSTEM: dict[str, str] = {
    "JAX_PIPELINE_DETAIL_PATH": "plantilla de ruta de frontend (/historial/{pipeline_id}), no existe en disco.",
    "PIPELINE_DETAIL_PATH": "misma plantilla, alias sin prefijo JAX_.",
}

#: Los dos servicios de producción cuyo entorno VIVO se audita en Fase C.
SERVICIOS = ("jax-las-manos", "jax-platform")

# NOMBRE=valor -- SIN espacio antes del nombre (systemd NO ignora la
# indentación, la aplica; este módulo, al no poder confiar en reproducir esa
# semántica exacta, prefiere marcar la línea como error de parseo antes que
# adivinar) y SIN espacio entre el nombre y el "=".
_LINEA = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def es_clave_de_ruta(clave: str) -> bool:
    return clave.endswith(SUFIJOS_DE_RUTA)


@dataclass
class Hallazgo:
    clave: str
    motivo: str


def parsear_env(texto: str) -> tuple[dict[str, list[str]], list[Hallazgo]]:
    """(clave -> lista de valores CRUDOS en orden de aparición, errores de
    parseo con número de línea). YA NO IGNORA NADA EN SILENCIO (ronda 2,
    MAJOR-A): toda línea no vacía y no comentario que no calce
    `^NOMBRE=valor$` exacto, o cuyo valor termine en `\\` (continuación de
    línea de systemd que este módulo no reproduce), es un error de parseo.

    Comentario = línea cuyo primer carácter no blanco es "#". Una línea
    INDENTADA nunca es un comentario válido para este chequeo aunque
    empiece con espacios y luego "#": si `despojada` empieza con "#" se
    trata iguialmente como comentario (systemd sí permite comentarios
    indentados) -- lo que NO se permite es una asignación indentada."""
    entorno: dict[str, list[str]] = {}
    errores: list[Hallazgo] = []
    for numero, linea in enumerate(texto.splitlines(), start=1):
        despojada = linea.strip()
        if not despojada or despojada.startswith("#"):
            continue
        m = _LINEA.match(linea)
        if not m:
            errores.append(Hallazgo(
                f"línea {numero}",
                f"no calza NOMBRE=valor exacto (¿indentada? ¿espacio junto al \"=\"?): {linea!r}",
            ))
            continue
        clave, crudo = m.group(1), m.group(2)
        if crudo.endswith("\\"):
            errores.append(Hallazgo(
                f"línea {numero}",
                f"termina en \"\\\" (continuación de línea de systemd, no soportada por este parser): {linea!r}",
            ))
            continue
        entorno.setdefault(clave, []).append(crudo)
    return entorno, errores


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


def _bajo_prefijo(ruta: str, prefijo: str) -> bool:
    """Chequeo de prefijo con límite de path: `/srv/jax-data/repo` SÍ cubre
    `/srv/jax-data/repo` y `/srv/jax-data/repo/documents`, pero NO
    `/srv/jax-data/repo-otro-nombre` -- un `str.startswith` pelado sí lo
    dejaría pasar, que es exactamente el tipo de fail-open que este módulo
    entero existe para cerrar."""
    prefijo = prefijo.rstrip("/")
    return ruta == prefijo or ruta.startswith(prefijo + "/")


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
    errores_de_parseo: list[Hallazgo] | None = None,
) -> ResultadoFaseA:
    """`resolver(ruta_cruda) -> ruta_real | None`. En producción,
    `resolver_como_jaxsvc` (sudo -n -u jaxsvc realpath -e). En tests, un
    dict de mentira.

    Tres chequeos:
      (0) Cualquier error de parseo (línea indentada, espacio junto al "=",
          continuación de línea) tira abajo el resultado ENTERO -- no
          importa si está lejos de las claves en alcance: si el parseo no
          es confiable, nada de lo que sigue lo es tampoco.
      (1) Las tres claves de KEYS_EN_ALCANCE: exactamente una vez, valor no
          vacío, ruta REAL dentro de SU PROPIO PERMITIDOS_POR_CLAVE (ronda
          2: ya no una lista compartida).
      (2) El resto de las claves de ruta presentes: ruta REAL fuera de
          /home/* -- salvo las de EXCEPCIONES_FASE_A y las de
          NO_ES_RUTA_DE_FILESYSTEM. Duplicadas, valores ambiguos o rutas que
          no resuelven TAMBIÉN son FALLA acá (ronda 2, MINOR): antes se
          saltaban en silencio con `continue`, dejando un hueco fail-open
          para cualquier clave de ruta futura que no fuera una de las tres
          en alcance."""
    resultado = ResultadoFaseA()
    resultado.problemas.extend(errores_de_parseo or [])

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
        permitidos = PERMITIDOS_POR_CLAVE[clave]
        if not any(_bajo_prefijo(real, p) for p in permitidos):
            resultado.problemas.append(
                Hallazgo(clave, f"{valor} (real: {real}) no está bajo ninguno de {permitidos}"))

    for clave, crudos in entorno.items():
        if clave in KEYS_EN_ALCANCE or clave in EXCEPCIONES_FASE_A or clave in NO_ES_RUTA_DE_FILESYSTEM:
            continue
        if not es_clave_de_ruta(clave):
            continue
        resultado.revisadas += 1
        valor, motivo = _valor_unico_normalizado(clave, crudos)
        if motivo:
            resultado.problemas.append(Hallazgo(clave, motivo))
            continue
        real = resolver(valor)
        if real is None:
            resultado.problemas.append(
                Hallazgo(clave, f"{valor}: no se pudo resolver la ruta real (¿no existe? ¿sin permiso?)"))
            continue
        if _bajo_prefijo(real, "/home"):
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
#  Fase C (ronda 2, MAJOR-A) -- verdad EFECTIVA: lo que el kernel dice que
#  los procesos reales tienen cargado en su entorno, no lo que este módulo
#  interpretó que /etc/jax/.env dice. Independiente de Fase A/B -- corre
#  siempre, aporta evidencia aparte.
# ---------------------------------------------------------------------------

def obtener_main_pid(servicio: str, ejecutar: EjecutarSudo = _ejecutar_sudo_real) -> str | None:
    r = ejecutar(["systemctl", "show", "-p", "MainPID", "--value", servicio])
    if r.returncode != 0:
        return None
    valor = r.stdout.strip()
    if not valor or valor == "0":
        return None
    return valor


def leer_environ_del_proceso(pid: str, ejecutar: EjecutarSudo = _ejecutar_sudo_real) -> dict[str, str] | None:
    """`/proc/<pid>/environ`: pares NOMBRE=valor separados por NUL, sin
    ningún tipo de citado de shell -- lo que systemd cargó de VERDAD, sin
    pasar por la interpretación de este módulo."""
    r = ejecutar(["sudo", "-n", "cat", f"/proc/{pid}/environ"])
    if r.returncode != 0:
        return None
    entorno: dict[str, str] = {}
    for par in r.stdout.split("\x00"):
        if not par or "=" not in par:
            continue
        clave, _, valor = par.partition("=")
        entorno[clave] = valor
    return entorno


@dataclass
class ResultadoFaseC:
    problemas: list[Hallazgo] = field(default_factory=list)
    servicios_revisados: list[str] = field(default_factory=list)
    servicios_sin_pid: list[str] = field(default_factory=list)


def verificar_fase_c(
    resolver: Callable[[str], str | None],
    obtener_pid: Callable[[str], str | None] = obtener_main_pid,
    leer_environ: Callable[[str], dict[str, str] | None] = leer_environ_del_proceso,
) -> ResultadoFaseC:
    resultado = ResultadoFaseC()
    for servicio in SERVICIOS:
        pid = obtener_pid(servicio)
        if pid is None:
            resultado.servicios_sin_pid.append(servicio)
            continue
        resultado.servicios_revisados.append(servicio)
        entorno_vivo = leer_environ(pid)
        if entorno_vivo is None:
            resultado.problemas.append(
                Hallazgo(servicio, f"no se pudo leer /proc/{pid}/environ (sudo -n sin permiso?)"))
            continue
        for clave in KEYS_EN_ALCANCE:
            valor_vivo = entorno_vivo.get(clave)
            if valor_vivo is None:
                resultado.problemas.append(Hallazgo(f"{servicio}:{clave}", "no está en el entorno vivo del proceso"))
                continue
            real = resolver(valor_vivo)
            if real is None:
                resultado.problemas.append(
                    Hallazgo(f"{servicio}:{clave}", f"{valor_vivo}: no se pudo resolver la ruta real"))
                continue
            permitidos = PERMITIDOS_POR_CLAVE[clave]
            if not any(_bajo_prefijo(real, p) for p in permitidos):
                resultado.problemas.append(
                    Hallazgo(f"{servicio}:{clave}", f"{valor_vivo} (real: {real}) no está bajo ninguno de {permitidos}"))
    return resultado


# ---------------------------------------------------------------------------
#  CLI -- envoltura fina.
# ---------------------------------------------------------------------------

def verificar(texto_env: str) -> tuple[bool, str]:
    """(ok, reporte). `ok=False` con CUALQUIER problema de Fase A, Fase B o
    Fase C (errores de parseo incluidos, vía Fase A)."""
    entorno, errores_de_parseo = parsear_env(texto_env)
    fase_a = verificar_fase_a(entorno, resolver_como_jaxsvc, errores_de_parseo)
    fase_b = verificar_fase_b(entorno, _ejecutar_sudo_real) if fase_a.ok else ResultadoFaseB()
    fase_c = verificar_fase_c(resolver_como_jaxsvc)

    lineas: list[str] = []
    if fase_a.problemas:
        lineas.append("FASE A -- claves de ruta con problemas (incluye errores de parseo):")
        for h in fase_a.problemas:
            lineas.append(f"  {h.clave}: {h.motivo}")
    if fase_b.problemas:
        lineas.append("FASE B -- jaxsvc no tiene el acceso que necesita:")
        for h in fase_b.problemas:
            lineas.append(f"  {h.clave}: {h.motivo}")
    if fase_c.problemas:
        lineas.append("FASE C -- el entorno VIVO de un proceso no coincide con lo permitido:")
        for h in fase_c.problemas:
            lineas.append(f"  {h.clave}: {h.motivo}")
    if fase_c.servicios_sin_pid:
        lineas.append(
            "FASE C -- aviso: no se pudo obtener MainPID (¿servicio inactivo?) de: "
            + ", ".join(fase_c.servicios_sin_pid))

    hay_problemas = bool(fase_a.problemas or fase_b.problemas or fase_c.problemas)
    if hay_problemas:
        return False, "\n".join(lineas)

    resumen = (
        f"rutas-de-produccion: todas las rutas de producción en alcance están "
        f"fuera de /home, jaxsvc tiene el acceso que necesita, y el entorno "
        f"VIVO de {len(fase_c.servicios_revisados)} servicio(s) coincide "
        f"({fase_a.revisadas} claves de ruta revisadas)"
    )
    if fase_c.servicios_sin_pid:
        resumen += "\n" + lineas[-1]  # el aviso de servicios sin PID, igual en verde
    return True, resumen


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
