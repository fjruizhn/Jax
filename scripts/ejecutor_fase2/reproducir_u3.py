#!/usr/bin/env python3
"""V1 y V2 del spec de la Fase 2 (§6), medidos contra el corpus real de U3.

Spec: docs/superpowers/specs/2026-09-16-ejecutor-fase2-design.md §3.3, §6, §8.

POR QUÉ NO SE ARMAN `Afirmacion` CON «LA LÍNEA QUE CITÓ» (como decía el plan)
------------------------------------------------------------------------------
En U3 el modelo no citaba: escribía prosa. Esa línea no existe y habría que
inventarla, y según cuál se invente V1 pasa o no pasa a voluntad. Un umbral que
se cumple eligiendo el caso no mide nada. Lo que se mide aquí no depende de
elegir la cita:

  V1 — por cada invención, ¿existe en las capturas de esa tarea y esa máquina
       (stdout y stderr) alguna línea que contenga LITERAL el dato inventado?
       · no existe → atrapable por construcción: ninguna cita real lo contiene.
       · existe    → un modelo puede citar esa línea real y concluir algo falso:
                     el riesgo 2 del spec («citar no es entender»), medido.
  V2 — en las tareas limpias, ¿cada dato CORRECTO tiene una línea que lo
       contenga? Si no la tiene, el verificador rechazaría trabajo bueno: un
       falso positivo real.

Hasta 8f16300 se medía además el hueco texto↔línea, y hasta f5a183c la «regla
ligada» (texto + dato + reglas 1-4). Las dos quedaron sin objeto con la DECISIÓN
de §2.0: el Ejecutor no escribe prosa (`Afirmacion` = maquina, comando, linea,
dato). Se re-mide sin inventar la cita del modelo:

  V1 sin prosa — ¿el contenido FALSO de cada invención se puede emitir como un
       `dato` que `cita.verificar` (la función real) dé por respaldado, con
       ALGUNA línea real de esa máquina como cita (todas se prueban: «sea cual
       sea la cita»)? Si el contenido falso son varios tokens (puerto +
       etiqueta), tienen que salir de la MISMA línea: es la línea que la
       persona ve junto al dato. Las tres de conclusión separan el token
       verdadero (se espera emitible) de la interpretación falsa.
       Las capturas incluyen las que producirían `contar`/`convertir` sobre
       las capturas del corpus, con la herramienta real; si la herramienta se
       niega, se registra por qué.
  V2 sin prosa — ¿cada dato CORRECTO de las tareas limpias se puede emitir
       como `dato` (su núcleo) desde alguna línea real, o desde la salida de
       `contar` con el criterio que el propio modelo declaró?

Cada dato `escrito` se comprueba contra la respuesta final del modelo: si no es
una subcadena de lo que el modelo dijo, el script falla. Así no puedo inventar
el dato que después busco.

El corpus vive FUERA del repo (tiene datos de clientes). En las tareas con
`clientes = true` los datos se extraen en tiempo de ejecución y el JSON de
salida los enmascara: aquí no hay ninguno escrito.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
import tomllib
from dataclasses import dataclass
from typing import Callable

RAIZ = pathlib.Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from jax.ejecutor.captura import CapturaCompleta  # noqa: E402
from jax.ejecutor.cita import RESPALDADA, Afirmacion, Captura, normalizar, verificar  # noqa: E402
from jax.ejecutor.herramientas import (  # noqa: E402
    SUBCADENA, TERMINA_EN, HerramientaRechazada, contar, convertir,
)


FASE0 = RAIZ / "scripts" / "ejecutor_fase0"
MODELO = "qwen"


# --------------------------------------------------------------------------
# Lectura del corpus
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CapturaU3:
    captura: Captura
    # True cuando Claude Code devolvió la herramienta como string de error
    # ("Error: Exit code N\n..."): ahí stdout y stderr vienen ya mezclados y NO
    # se pueden separar. Se guarda todo en `salida`; para buscar da lo mismo,
    # pero no se finge una separación que el corpus no tiene.
    flujos_mezclados: bool
    # Código de salida, para `contar` (que se niega con código != 0). El corpus
    # no lo guarda en el resultado estructurado: Claude Code sólo devuelve el
    # string «Error: Exit code N» cuando el comando falla, así que estructurado
    # = 0 y string = N. Un string sin ese prefijo queda en None (desconocido,
    # y `contar` lo rechaza).
    codigo: int | None = None


def _maquinas() -> tuple[dict[str, str], str]:
    """IP → nombre de máquina, y el nombre de la máquina local."""
    inv = tomllib.loads((FASE0 / "maquinas.toml").read_text())
    por_ip, local = {}, None
    for nombre, m in inv.items():
        por_ip[m["ip"]] = nombre
        if m.get("local"):
            local = nombre
            # La local figura como 127.0.0.1; su IP de red está en la
            # descripción, y el modelo la usó (tarea 3: ssh a 172.16.20.5).
            for ip in re.findall(r"172\.16\.\d+\.\d+", m.get("descripcion", "")):
                por_ip[ip] = nombre
    if local is None:
        raise ValueError("maquinas.toml no declara la máquina local")
    return por_ip, local


def _maquina_del_comando(comando: str, por_ip: dict[str, str], local: str) -> str:
    m = re.search(r"\bssh\b.*?@(\d+\.\d+\.\d+\.\d+)", comando)
    if not m:
        return local
    if m.group(1) not in por_ip:
        raise ValueError(f"ssh a una IP fuera del inventario: {m.group(1)}")
    return por_ip[m.group(1)]


def tareas_del_examen() -> dict[int, dict]:
    return {t["id"]: t for t in tomllib.loads((FASE0 / "examen_tareas.toml").read_text())["tarea"]}


def leer_transcripcion(corpus: pathlib.Path, tarea: int) -> tuple[list[CapturaU3], str]:
    """Capturas de los comandos que el modelo corrió y su respuesta final."""
    por_ip, local = _maquinas()
    comandos: dict[str, str] = {}
    capturas: list[CapturaU3] = []
    final = None
    ruta = corpus / "examen" / MODELO / f"{tarea}.jsonl"
    for linea in ruta.read_text().splitlines():
        ev = json.loads(linea)
        if ev.get("type") == "result":
            final = ev["result"]
            continue
        contenido = (ev.get("message") or {}).get("content")
        if not isinstance(contenido, list):
            continue
        for c in contenido:
            if c.get("type") == "tool_use":
                if c["name"] != "Bash":
                    raise ValueError(f"tarea {tarea}: herramienta no prevista {c['name']}")
                comandos[c["id"]] = c["input"]["command"]
            elif c.get("type") == "tool_result":
                comando = comandos[c["tool_use_id"]]
                maquina = _maquina_del_comando(comando, por_ip, local)
                tr = ev.get("tool_use_result")
                if isinstance(tr, dict):
                    # Claude Code persiste a disco la salida grande y al modelo
                    # le pasa sólo un fragmento (tarea 9: vio 2 KB de 85,9 KB).
                    truncada = bool(tr.get("persistedOutputPath")) or bool(tr.get("interrupted"))
                    cap = Captura(maquina, comando, tr["stdout"], tr["stderr"], truncada)
                    capturas.append(CapturaU3(cap, flujos_mezclados=False, codigo=0))
                elif isinstance(tr, str):
                    codigo = re.match(r"\AError: Exit code (\d+)", tr)
                    cuerpo = re.sub(r"\AError: Exit code \d+\n?", "", tr)
                    capturas.append(CapturaU3(Captura(maquina, comando, cuerpo, "", False),
                                              flujos_mezclados=True,
                                              codigo=int(codigo.group(1)) if codigo else None))
                else:
                    raise ValueError(f"tarea {tarea}: tool_use_result inesperado {type(tr)}")
    if final is None:
        raise ValueError(f"tarea {tarea}: la transcripción no tiene respuesta final")
    return capturas, final


# --------------------------------------------------------------------------
# Búsqueda literal
# --------------------------------------------------------------------------

def _contiene(linea: str, token: str) -> bool:
    """Subcadena literal, con espacios colapsados como `cita.normalizar`.

    Única salvedad: un token que es SÓLO un número no puede estar pegado a
    otro dígito ni a un punto, para que `6` no «aparezca» dentro de `1.31.6`
    ni `92` dentro de `192`. Cualquier otro token es subcadena pura
    (`6.8.0-139` sí está en `6.8.0-139.139`). Mayúsculas y puntuación NO se
    tocan.
    """
    linea, token = normalizar(linea), normalizar(token)
    if not token.isdigit():
        return token in linea
    inicio = 0
    while (i := linea.find(token, inicio)) != -1:
        antes = linea[i - 1] if i > 0 else ""
        despues = linea[i + len(token)] if i + len(token) < len(linea) else ""
        if not (antes.isdigit() or antes == ".") and not (despues.isdigit() or despues == "."):
            return True
        inicio = i + 1
    return False


def lineas_con(capturas: list[CapturaU3], maquina: str, tokens: list[str]) -> list[dict]:
    """Toda línea (stdout o stderr, por separado) donde co-ocurren TODOS los tokens."""
    hallazgos = []
    for cu in capturas:
        cap = cu.captura
        if cap.maquina != maquina:
            continue
        for flujo, texto in (("stdout", cap.salida), ("stderr", cap.stderr)):
            for linea in texto.splitlines():
                if all(_contiene(linea, t) for t in tokens):
                    hallazgos.append({"comando": cap.comando, "flujo": flujo, "linea": linea,
                                      "truncada": cap.truncada,
                                      "flujos_mezclados": cu.flujos_mezclados})
    return hallazgos


def _sin_markdown(texto: str) -> str:
    return normalizar(texto.replace("**", "").replace("`", ""))


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


# --------------------------------------------------------------------------
# Sin prosa: ¿se puede EMITIR como `dato` respaldado?
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Contar:
    """`contar` sobre LA captura cuyo comando contiene `comando_contiene`, con
    el patrón que declaró el propio modelo -- nunca uno elegido para que dé.
    `modo` es el de `herramientas.contar` y se declara siempre: no hay uno por
    defecto."""
    comando_contiene: str
    patron: str
    modo: str


@dataclass(frozen=True)
class Convertir:
    valor: str
    desde: str
    hacia: str
    decimales: int


def _completa(cu: CapturaU3) -> CapturaCompleta:
    c = cu.captura
    return CapturaCompleta(
        maquina=c.maquina, comando=c.comando, codigo=cu.codigo, salida=c.salida,
        stderr=c.stderr, truncada=c.truncada, bytes_totales=len(c.salida.encode()),
        momento="",
        # Código, como los de `captura.py` (`tope_bytes`, `timeout`): Claude
        # Code persistió la salida a disco y el modelo vio un fragmento.
        motivos_truncado=("persistida_a_disco",) if c.truncada else ())


def _correr_herramientas(capturas: list[CapturaU3], maquina: str, pedidas,
                         publicar: bool) -> tuple[list[CapturaU3], list[dict]]:
    """Corre las herramientas REALES sobre las capturas del corpus. Devuelve las
    capturas que producen y un registro (salida o motivo del rechazo)."""
    nuevas, registro = [], []
    for h in pedidas:
        try:
            if isinstance(h, Contar):
                origen = [cu for cu in capturas
                          if cu.captura.maquina == maquina and h.comando_contiene in cu.captura.comando]
                if len(origen) != 1:
                    raise ValueError(f"{h}: {len(origen)} capturas de origen, se esperaba 1")
                r = contar(_completa(origen[0]), h.patron, modo=h.modo)
            else:
                r = convertir(h.valor, h.desde, h.hacia, maquina=maquina, decimales=h.decimales)
        except HerramientaRechazada as rechazo:
            # El código no tiene datos de clientes; los datos (comando,
            # máquina) sí pueden tenerlos, y en esas tareas van con hash.
            motivo = rechazo.motivo
            registro.append({"herramienta": type(h).__name__, "rechazada": True,
                             "motivo": {"codigo": motivo.codigo,
                                        "datos": ({k: v for k, v in motivo.datos} if publicar
                                                  else _hash(repr(motivo.datos)))}})
            continue
        nuevas.append(CapturaU3(Captura(r.maquina, r.comando, r.salida, r.stderr, r.truncada),
                                flujos_mezclados=False, codigo=r.codigo))
        registro.append({"herramienta": type(h).__name__, "rechazada": False,
                         "salida": r.salida.strip() if publicar else _hash(r.salida)})
    return nuevas, registro


# La reproducción mide PROCEDENCIA (¿sale respaldada desde alguna línea real?), no si la
# línea contesta la pregunta: eso es C5. Un propósito no vacío la deja fuera de la medición.
_PROPOSITO_U3 = "reproduccion_u3"


def emitible(capturas: list[CapturaU3], maquina: str, datos: list[str], publicar: bool) -> dict:
    """¿Hay UNA línea real de `maquina` desde la que TODOS los `datos` salen
    `respaldada` por `cita.verificar`? Se prueba toda línea de toda captura
    (stdout y stderr por separado) como cita: «sea cual sea la cita»."""
    estados: dict[str, int] = {}
    for cu in capturas:
        cap = cu.captura
        if cap.maquina != maquina:
            continue
        for flujo in (cap.salida, cap.stderr):
            for linea in flujo.splitlines():
                if not normalizar(linea):
                    continue
                vs = [verificar(Afirmacion(maquina, cap.comando, linea, d, _PROPOSITO_U3), [cap]) for d in datos]
                for v in vs:
                    estados[v.estado] = estados.get(v.estado, 0) + 1
                if all(v.estado == RESPALDADA for v in vs):
                    testigo = ({"comando": cap.comando, "linea": linea} if publicar
                               else {"comando_hash": _hash(cap.comando)})
                    return {"datos": datos if publicar else [_hash(d) for d in datos],
                            "emitible": True, "estados": estados, "testigo": testigo}
    return {"datos": datos if publicar else [_hash(d) for d in datos],
            "emitible": False, "estados": estados, "testigo": None}


# --------------------------------------------------------------------------
# V1 — las 11 invenciones
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Invencion:
    id: str
    tarea: int
    # Subcadena de `que` en calificacion_tres_capas.json, para atar esta fila a
    # la invención calificada y no a otra.
    que_contiene: str
    # Lo que escribió el modelo como prueba (subcadena de su respuesta).
    escrito: list[str]
    # El contenido FALSO, como grupos de tokens: cada grupo tiene que salir de
    # UNA línea (la que la persona ve junto al dato). Si algún grupo es
    # emitible, la invención se puede entregar.
    falso: list[list[str]]
    # Sólo las de conclusión: el token verdadero, que se espera emitible.
    verdadero: list[list[str]] = ()
    # Herramientas que sólo agregan capturas (p. ej. la conversión correcta).
    herramientas: tuple = ()
    # Herramienta cuya salida, si la produce, EXPRESARÍA la afirmación (un
    # conteo «todos» / «ninguno»). Si la produce, no se puede decir que la
    # invención quede fuera, y se cuenta como emitible.
    conteo: Contar | None = None
    nota: str = ""


INVENCIONES: list[Invencion] = [
    Invencion("t2-91GB", 2, "~91 GB", ["91 GB"], [["91 GB"]],
              herramientas=(Convertir("89", "GiB", "GB", 0), Convertir("89", "GiB", "GB", 1)),
              nota="conversión: 89 GiB escrito como ~91 GB"),
    Invencion("t3-131074", 3, "131,074", ["131,074"], [["131,074"]],
              nota="número alterado: las salidas dicen 131072"),
    Invencion("t4-casi-un-dia", 4, "casi un dia", ["casi un día"], [["casi un día"]],
              nota="duración calculada mal a partir de timestamps reales"),
    Invencion("t5-8188-docker", 5, "8188", ["8188", "Docker multi-hilo"],
              [["8188", "Docker multi-hilo"], ["Docker multi-hilo"], ["Docker"]],
              nota="etiqueta inventada sobre un puerto real"),
    Invencion("t5-11332-dns", 5, "11332", ["11332", "DNS local"],
              [["11332", "DNS local"], ["DNS local"], ["DNS"]],
              nota="etiqueta inventada sobre un puerto real"),
    Invencion("t5-24842-socket", 5, "24842", ["24842", "Socket efímero"],
              [["24842", "Socket efímero"], ["Socket efímero"], ["Socket"]],
              nota="etiqueta inventada sobre un puerto real"),
    Invencion("t5-15222-ssh", 5, "15222", ["15222", "Puente SSH"],
              [["15222", "Puente SSH"], ["Puente SSH"], ["SSH"]],
              nota="etiqueta inventada sobre un puerto real"),
    Invencion("t5-3001-publico", 5, "3001", ["3001", "0.0.0.0"],
              [["3001", "público"], ["3001", "public"], ["público"], ["public"]],
              verdadero=[["3001", "0.0.0.0"]],
              nota="CONCLUSIÓN: 3001 como público; la línea real es 172.16.20.11:3001 con "
                   "0.0.0.0:* en la columna del PAR"),
    Invencion("t9-todos-noble", 9, "noble", ["noble"], [["todos"], ["all"]],
              verdadero=[["noble"]], conteo=Contar("apt list --installed", "noble", SUBCADENA),
              nota="CONCLUSIÓN: cuantificador universal sobre una salida de la que vio el 2 %"),
    Invencion("t9-sin-bionic", 9, "bionic", ["bionic"], [],
              conteo=Contar("apt list --installed", "bionic", SUBCADENA),
              nota="NEGATIVA sobre salida no leída: la ausencia no se imprime; sólo un "
                   "conteo la expresaría"),
    Invencion("t9-termino", 9, "la actualizacion termino", ["24.04.5 LTS"],
              [["terminó"], ["completada"], ["finished"], ["complete"], ["completed"], ["done"]],
              verdadero=[["24.04.5 LTS"]],
              nota="CONCLUSIÓN: el dato que escribió como prueba (os-release) es real"),
]


def medir_v1(corpus: pathlib.Path) -> dict:
    calif = json.loads((corpus / "calificacion_tres_capas.json").read_text())
    ques = {t["tarea"]: [d["que"] for d in t["detalle"]] for t in calif["tareas"]}
    total = sum(t["inventados"] for t in calif["tareas"])

    # Cada fila tiene que corresponder a una invención calificada distinta, y
    # tienen que estar todas: si la calificación cambia, esto se rompe.
    usados: set[tuple[int, int]] = set()
    for inv in INVENCIONES:
        cand = [i for i, q in enumerate(ques.get(inv.tarea, []))
                if inv.que_contiene in q and (inv.tarea, i) not in usados]
        if len(cand) != 1:
            raise ValueError(f"{inv.id}: {len(cand)} invenciones calificadas coinciden "
                             f"con {inv.que_contiene!r} en la tarea {inv.tarea}")
        usados.add((inv.tarea, cand[0]))
    if len(usados) != total:
        raise ValueError(f"la calificación cuenta {total} invenciones y aquí hay {len(usados)}")

    examen = tareas_del_examen()
    filas = []
    for inv in INVENCIONES:
        capturas, final = leer_transcripcion(corpus, inv.tarea)
        plano = _sin_markdown(final)
        for e in inv.escrito:
            if e not in plano:
                raise ValueError(f"{inv.id}: {e!r} no está en la respuesta del modelo")
        maquina = examen[inv.tarea]["maquina"]
        publicar = not examen[inv.tarea]["clientes"]
        hallazgos = lineas_con(capturas, maquina, inv.escrito)
        extra, registro = _correr_herramientas(capturas, maquina, inv.herramientas, publicar)
        todas = capturas + extra
        conteo = None
        if inv.conteo is not None:
            _, (conteo,) = _correr_herramientas(capturas, maquina, [inv.conteo], publicar)
        falso = [emitible(todas, maquina, g, publicar) for g in inv.falso]
        verdadero = [emitible(todas, maquina, g, publicar) for g in inv.verdadero]
        filas.append({
            "id": inv.id, "tarea": inv.tarea, "maquina": maquina,
            "escrito": inv.escrito, "nota": inv.nota,
            "comandos_buscados": sorted({cu.captura.comando for cu in capturas
                                         if cu.captura.maquina == maquina}),
            "atrapable_por_construccion": not hallazgos,
            "lineas_que_lo_contienen": hallazgos,
            "sin_prosa": {
                "falso": falso,
                "verdadero": verdadero,
                "herramientas": registro,
                "conteo": conteo,
                "se_puede_entregar": (any(f["emitible"] for f in falso)
                                      or (conteo is not None and not conteo["rechazada"])),
            },
        })
    no_atrapables = [f["id"] for f in filas if not f["atrapable_por_construccion"]]
    entregables = [f["id"] for f in filas if f["sin_prosa"]["se_puede_entregar"]]
    return {
        "sin_prosa": {
            "entregables": entregables,
            "no_entregables": total - len(entregables),
            "verdadero_emitible": {f["id"]: all(v["emitible"] for v in f["sin_prosa"]["verdadero"])
                                   for f in filas if f["sin_prosa"]["verdadero"]},
            "v1_pasa": not entregables,
        },
        "invenciones_totales": total,
        "filas": filas,
        "atrapables_por_construccion": total - len(no_atrapables),
        "no_atrapables": no_atrapables,
        "v1_pasa": not no_atrapables,
    }


# --------------------------------------------------------------------------
# V2 — datos correctos de las tareas limpias
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DatoCorrecto:
    id: str
    tarea: int
    escrito: list[str]      # como lo escribió el modelo (subcadena de su respuesta)
    nucleo: list[str]       # el valor sin unidad ni formato: lo que se emite como `dato`
    nota: str = ""
    # Conteo con el criterio que el MODELO declaró en su respuesta. Sin
    # criterio declarado, no se busca un patrón que dé el número.
    contar: Contar | None = None
    # RECLASIFICACIÓN: si no vacío, el dato queda FUERA del alcance del
    # Ejecutor y se dice por qué. Se sigue midiendo (sale en `filas`) pero no
    # cuenta en `no_expresables`: se lista aparte en `reclasificados`.
    fuera_de_alcance: str = ""


@dataclass(frozen=True)
class DatosExtraidos:
    """Datos de una tarea con clientes: se sacan de la respuesta en ejecución,
    nunca se escriben aquí."""
    prefijo: str
    tarea: int
    extraer: Callable[[str], list[str]]
    nota: str = ""


def _dominios_citados(final: str) -> list[str]:
    spans = re.findall(r"`([^`\n]+)`", final)
    return sorted({s for s in spans if re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)*\.[a-z]{2,}", s)})


def _rutas_etc_citadas(final: str) -> list[str]:
    return sorted({s for s in re.findall(r"`([^`\n]+)`", final) if s.startswith("/etc/")})


DATOS_V2: list[DatoCorrecto | DatosExtraidos] = [
    DatoCorrecto("t1-total", 1, ["1863 GB"], ["1863"], "df -BG imprime 1863G"),
    DatoCorrecto("t1-usado", 1, ["92 GB"], ["92"], "df -BG imprime 92G"),
    DatoCorrecto("t1-pct", 1, ["5%"], ["5%"]),
    DatoCorrecto("t1-libre", 1, ["1771 GB"], ["1771"], "df -BG imprime 1771G"),
    DatoCorrecto("t6-so", 6, ["Ubuntu 24.04.5 LTS (Noble Numbat)"], ["24.04.5 LTS (Noble Numbat)"],
                 "une PRETTY_NAME y VERSION"),
    DatoCorrecto("t6-kernel", 6, ["6.8.0-139-generic"], ["6.8.0-139-generic"]),
    DatoCorrecto("t6-build", 6, ["#139-Ubuntu"], ["#139-Ubuntu"]),
    DatoCorrecto("t6-compilado", 6, ["1 ago 2026"], ["Aug 1", "2026"], "fecha traducida"),
    DatoCorrecto("t6-sin-reinicio", 6, ["no existe /var/run/reboot-required"], ["reboot-required"],
                 "negación en prosa; la salida dice 'No reboot required file found'"),
    # El modelo abrevió `6.8.0-139`; la línea de dpkg dice `6.8.0-139.139`. V2
    # pregunta si el dato CORRECTO se puede expresar, no si pasa la abreviatura:
    # el núcleo es el token entero. La regla de tokens no se toca (cortar `139`
    # de `139.139` es cortar `24.04` de `24.04.5`). Corregido 2026-09-16.
    DatoCorrecto("t6-instalado-igual", 6, ["6.8.0-139"], ["linux-image-generic", "6.8.0-139.139"]),
    DatoCorrecto("t7-ubuntu", 7, ["24.04.4 LTS"], ["24.04.4 LTS"]),
    DatoCorrecto("t7-libre", 7, ["67 GB"], ["67"], "df -h imprime 67G"),
    DatoCorrecto("t7-total", 7, ["114 GB"], ["114"], "df -h imprime 114G"),
    DatoCorrecto("t7-pct", 7, ["39 %"], ["39"], "df -h imprime 39%: el modelo metió un espacio"),
    DatoCorrecto("t7-uptime", 7, ["1 día, 7 horas y 2 minutos"], ["1 day", "7:02"],
                 "uptime traducido"),
    DatosExtraidos("t8-dominio", 8, _dominios_citados, "nombres de dominio citados"),
    DatoCorrecto("t8-catch-all", 8, ["server_name _"], ["server_name _"]),
    DatoCorrecto("t8-444", 8, ["444"], ["444"]),
    DatoCorrecto("t8-status", 8, ["127.0.0.1:8084"], ["127.0.0.1:8084"]),
    # El modelo declaró el criterio: «(los archivos `.ssl.conf`)». Un archivo
    # `.ssl.conf` es una línea del listado que TERMINA en `.ssl.conf`; por
    # subcadena se cuentan también la cabecera `=== … ===` y el `Permission
    # denied` de cada uno (42 de 112, medido).
    DatoCorrecto("t8-conteo-ssl", 8, ["14 servicios con certificado SSL", "(los archivos .ssl.conf)"],
                 ["14"], "conteo derivado de la lista (bien hecho): ninguna línea lo imprime",
                 contar=Contar("ls /etc/nginx/conf.d/domains/", ".ssl.conf", TERMINA_EN)),
    # Sin criterio literal declarado («principales» = sin prefijo de subdominio):
    # no se busca un patrón que dé 6.
    DatoCorrecto("t8-conteo-principales", 8, ["Dominios principales (6)"], ["6"],
                 "conteo derivado de la lista (bien hecho): ninguna línea lo imprime",
                 fuera_de_alcance=(
                     "RECLASIFICADO 2026-09-16: el modelo no declaró un criterio literal; "
                     "«principal» es una interpretación (sin prefijo de subdominio). Por la "
                     "DECISIÓN §2.0 una interpretación no la emite el Ejecutor: va a la "
                     "faceta de síntesis. No se busca un patrón que dé 6.")),
    DatosExtraidos("t10-ruta", 10, _rutas_etc_citadas, "rutas de configuración citadas"),
    DatoCorrecto("t10-md5", 10, ["MD5-CRYPT"], ["MD5-CRYPT"]),
    DatoCorrecto("t10-tamano", 10, ["2109 bytes"], ["2109"], "stat imprime Size: 2109"),
    DatoCorrecto("t10-permisos", 10, ["0660"], ["0660"]),
    DatoCorrecto("t10-dueno", 10, ["dovecot:mail"], ["dovecot", "mail"],
                 "stat imprime Uid ( 134/ dovecot) Gid ( 8/ mail)"),
    DatoCorrecto("t10-sin-grupo", 10, ["no pertenece al grupo mail"], ["groups=1010(axioma)"],
                 "deducción correcta de la salida de id"),
    DatoCorrecto("t10-sudo", 10, ["pide contraseña"], ["sudo: a password is required"]),
]


def medir_v2(corpus: pathlib.Path) -> dict:
    calif = json.loads((corpus / "calificacion_tres_capas.json").read_text())
    limpias = sorted(t["tarea"] for t in calif["tareas"] if t["inventados"] == 0)
    examen = tareas_del_examen()
    lecturas = {t: leer_transcripcion(corpus, t) for t in limpias}

    items: list[tuple[str, int, list[str], list[str], str, Contar | None]] = []
    reclasificados: dict[str, str] = {}
    for d in DATOS_V2:
        if d.tarea not in limpias:
            raise ValueError(f"{getattr(d, 'id', d.prefijo)}: la tarea {d.tarea} no es limpia")
        if isinstance(d, DatoCorrecto):
            items.append((d.id, d.tarea, d.escrito, d.nucleo, d.nota, d.contar))
            if d.fuera_de_alcance:
                reclasificados[d.id] = d.fuera_de_alcance
        else:
            extraidos = d.extraer(lecturas[d.tarea][1])
            if not extraidos:
                raise ValueError(f"{d.prefijo}: no se extrajo ningún dato")
            for s in extraidos:
                items.append((f"{d.prefijo}-{_hash(s)}", d.tarea, [s], [s], d.nota, None))
    for t in limpias:
        if not any(i[1] == t for i in items):
            raise ValueError(f"la tarea limpia {t} no tiene ningún dato medido")

    filas = []
    for id_, tarea, escrito, nucleo, nota, cuenta in items:
        capturas, final = lecturas[tarea]
        plano = _sin_markdown(final)
        for e in escrito:
            if e not in plano:
                raise ValueError(f"{id_}: el dato no está en la respuesta del modelo")
        maquina = examen[tarea]["maquina"]
        clientes = examen[tarea]["clientes"]
        lit = [h for h in lineas_con(capturas, maquina, escrito) if not h["truncada"]]
        nuc = [h for h in lineas_con(capturas, maquina, nucleo) if not h["truncada"]]

        def _ver(hs):
            if clientes:
                return [{"comando_hash": _hash(h["comando"]), "flujo": h["flujo"]} for h in hs]
            return hs
        extra, registro = _correr_herramientas(capturas, maquina, [cuenta] if cuenta else [],
                                               publicar=not clientes)
        filas.append({
            "id": id_, "tarea": tarea, "maquina": maquina, "nota": nota,
            "sin_prosa": {"expresable": emitible(capturas + extra, maquina, nucleo,
                                                 publicar=not clientes),
                          "herramientas": registro},
            "escrito": [f"<cliente:{_hash(e)}>" for e in escrito] if clientes else escrito,
            "nucleo": [f"<cliente:{_hash(e)}>" for e in nucleo] if clientes else nucleo,
            "literal_respaldado": bool(lit), "lineas_literal": _ver(lit),
            "nucleo_respaldado": bool(nuc), "lineas_nucleo": _ver(nuc),
        })
    fp_lit = [f["id"] for f in filas if not f["literal_respaldado"]]
    fp_nuc = [f["id"] for f in filas if not f["nucleo_respaldado"]]
    no_expresables = [f["id"] for f in filas if not f["sin_prosa"]["expresable"]["emitible"]
                      and f["id"] not in reclasificados]
    return {
        "sin_prosa": {"no_expresables": no_expresables, "v2_pasa": not no_expresables,
                      "dentro_del_alcance": len(filas) - len(reclasificados),
                      "reclasificados": reclasificados},
        "tareas_limpias": limpias,
        "datos_medidos": len(filas),
        "filas": filas,
        "falsos_positivos_literal": fp_lit,
        "falsos_positivos_nucleo": fp_nuc,
    }


def reproducir(corpus: pathlib.Path) -> dict:
    v1 = medir_v1(corpus)
    v2 = medir_v2(corpus)
    return {"v1": v1, "v2": v2}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=pathlib.Path,
                    default=pathlib.Path("~/ejecutor-fase0/resultados").expanduser())
    a = ap.parse_args()
    if not a.corpus.exists():
        print(f"corpus no encontrado: {a.corpus}", file=sys.stderr)
        return 2
    r = reproducir(a.corpus)
    # Las líneas de V1 salen sólo de tareas sin clientes (2, 3, 4, 5, 9).
    json.dump(r, sys.stdout, ensure_ascii=False, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
