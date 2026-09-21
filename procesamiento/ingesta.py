"""Ingesta: el original entra a `fuente/` y su extracto queda en `procesado/`.

Dos invariantes:
  1. `fuente/` es inmutable. El original entra y no se toca mas. Dos
     archivos DISTINTOS con el mismo nombre no pueden pisarse -- ni en
     secuencia ni bajo concurrencia (C-3) -- y un symlink plantado ahi
     jamas se sigue, ni para leer ni para escribir (C-1).
  2. `procesado/` es DESECHABLE: se borra entero y se reconstruye. Nada que
     no sea reconstruible vive ahi. Un cache del que no te podes fiar para
     borrarlo no es un cache: es una segunda base de datos.

El cache se indexa por `sha256` del original, pero la huella SOLA no
alcanza para decidir si un acierto de cache es valido -- ver
`_ficha_de_cache_valida` (I-1 a I-5).

=====================================================================
Ronda de arreglo 1 (2026-09-21, task-8-hallazgos.md -- 3 Criticos, 8
Importantes, veredicto spec NO, sin ratificar nada):
=====================================================================

- **C-1 (incumplimiento de spec §4):** el draft resolvia el destino con
  `candidato.exists()`, que SIGUE symlinks -- un enlace roto en `fuente/`
  hacia que la copia escribiera *a traves* del enlace, fuera del
  workspace, con contenido de un cliente. El spec manda reusar
  `tool_authority.resolve_jailed_path` (ya probado con casos
  adversariales) en vez de escribir codigo de seguridad nuevo. Acá se
  reusa para DOS cosas: (a) `trabajo` se valida contra el jail antes de
  tocar el filesystem (cierra I-7 de paso), y (b) cada nombre candidato en
  `fuente/` se valida igual. Ademas, defensa en profundidad: la apertura
  real usaba `O_NOFOLLOW` (nunca sigue un symlink al escribir, exista o no
  su destino) -- por si el jail se equivocara. *(Superado en ronda 2: la
  escritura ya no usa `os.open` -- ver más abajo.)*

  `resolve_jailed_path` exige un path RELATIVO (es el contrato de los
  tool_calls, que nunca traen rutas absolutas). La ingesta SI trabaja con
  rutas absolutas (`trabajo` es una carpeta real del filesystem), asi que
  `_resolver_bajo_jail` hace la traduccion: resuelve la forma canonica,
  la expresa relativa a `WORKSPACE_ROOT`, y RECIEN AHI delega el chequeo
  de fondo -- no se reescribe la logica de "sigue symlinks y compara
  contra el jail", esa sigue viviendo en un solo lugar.

- **C-2 (el defecto que este PR vino a cerrar, reabierto por
  concurrencia):** el temporal `<huella>.parcial` era determinista y
  COMPARTIDO -- dos ingestas del mismo archivo entrelazadas dejaban un
  `procesado/` con parte de las salidas de una y parte de la otra, con
  una ficha que afirmaba que estaban todas. Y como el segundo pase daba
  acierto de cache, nunca se curaba. Ahora el temporal es
  `<huella>.<pid>-<uuid4>.parcial` -- unico por LLAMADA, nunca
  compartido -- y se borra en un `finally` (cierra I-8 de paso: no quedan
  temporales huerfanos si algo revienta a mitad de camino).

- **C-3 (mismo defecto de nombres del draft original, reabierto por
  concurrencia):** entre "decidir el nombre" y "copiar" habia una ventana
  -- dos ingestas concurrentes de archivos DISTINTOS con el mismo nombre
  podian terminar con un solo archivo en `fuente/` y una ficha que
  MENTIA sobre cual era. La creacion se hizo ATOMICA con
  `O_CREAT|O_EXCL|O_NOFOLLOW` *(ronda 2: reemplazado por `os.link()`, ver
  más abajo)*. La comprobacion "no existe" y la creacion son la MISMA
  operacion del kernel -- no hay ventana que otro escritor pueda
  colarse. Si el nombre ya esta ocupado (por otro contenido, o por un
  symlink que jamas se sigue), se prueba el siguiente nombre candidato;
  el que ya estaba ahi no se toca nunca.

- **I-1 (el mas revelador -- la prueba de que era un accidente, no una
  decision, es que la mutacion "no cachear cuando no hubo salidas" pasaba
  los 8 tests en verde):** antes, CUALQUIER estado se cacheaba para
  siempre -- un `error` por "tesseract no instalado" quedaba condenado
  aunque se instalara tesseract un minuto despues. Regla nueva: `ok` y
  `parcial` se cachean, `error` y `sin_extractor` NO. Un error no es un
  extracto reconstruible, es la AUSENCIA de uno, y cachear una ausencia
  como si fuera trabajo hecho es lo que la volvia permanente. La ficha
  se escribe igual (queda el registro de que se intento y por que
  fallo); lo que cambia es que leerla cuenta como fallo de cache y se
  reintenta.

- **I-2 (invalidacion por version del extractor):** cambiar el
  extractor (o su version) no invalidaba una ficha ya escrita -- los
  Criticos de las rondas anteriores de este mismo proyecto NO alcanzaban
  a un archivo ya ingerido. Al leer, se compara `extractor_version`
  contra lo que el extractor VIGENTE reporta hoy (consulta barata: leer
  un `__version__` ya importado, no releer el archivo) -- si no
  coincide, fallo de cache.

- **I-3 (la clave ignoraba la extension, que es lo que la compuerta usa
  para rutear):** dos archivos con los mismos bytes y distinta extension
  compartian entrada de cache. La ficha ahora guarda la extension con la
  que se ruteo (`detalle["_extension_ingesta"]`) y el acierto de cache la
  valida contra `destino.suffix`.

- **I-4 (ficha corrupta = excepcion para siempre):** una `ficha.json`
  truncada a 0 bytes (corte de luz tipico) hacia `ValueError` en CADA
  pase. Ahora una ficha ilegible o invalida es, simplemente, fallo de
  cache -- nunca una excepcion hacia el llamador.

- **I-5 (la completitud del cache era inverificable POR DISEÑO):** sin
  esto, C-2/I-4 son reparables SOLO por casualidad. La ficha ahora lista
  sus salidas (`detalle["_salidas_ingesta"]`) y el acierto de cache
  verifica que TODAS esten presentes en disco -- borrar un `.csv` de
  `procesado/` ahora se nota y se regenera.

- **I-6 (nombre largo + colision -> OSError crudo):** el tronco del
  nombre se recorta ANTES de agregarle el sufijo de huella, para que el
  resultado siempre quepa dentro del limite tipico de un filesystem
  (255).

=====================================================================
Ronda de arreglo 2 (2026-09-21, task-8-hallazgos-r2.md -- lo que la ronda
1 trajo de nuevo, y lo que destapo):
=====================================================================

- **A-1/A-2 (mismo modelo de amenaza que C-1, ahora por el camino de leer
  en vez de escribir):** `O_NOFOLLOW` cerro el symlink, pero el camino de
  comparacion de huellas volvia a "abrir lo que haya" -- un FIFO plantado
  en `fuente/` colgaba `sha256_de` PARA SIEMPRE (sin timeout); un
  directorio con el mismo nombre reventaba con `IsADirectoryError`, en
  cada pase. Arreglo unico para los dos: solo se le calcula la huella a
  un archivo REGULAR (`Path.is_file()`) -- cierra tambien socket y
  dispositivo sin enumerarlos.

- **A-3 (el reintento acotado en 10 no aguantaba lo que decia aguantar):**
  medido con hilos reales, SIN jitter, a 16 hilos fallaba 21 de 25 veces
  con un `OSError(39)` crudo -- reintentos INMEDIATOS se resincronizan
  entre si. Arreglo: (1) jitter con backoff entre reintentos; (2) un error
  de dominio (`ReemplazoProcesadoAgotado`) en vez del `OSError` crudo si
  se agotan; (3) medido de nuevo despues del arreglo -- el numero esta en
  task-8-report.md, no es una suposicion.

  **Hallazgo PROPIO durante esa misma medicion, mas alla de lo que pedia
  A-3** (no esta en task-8-hallazgos-r2.md): a mas hilos escribiendo el
  MISMO archivo, un lector que pierde la carrera de `O_CREAT|O_EXCL`
  podia ver el archivo GANADOR a mitad de escribir -- `sha256_de` lee lo
  que hay EN ESE INSTANTE, calcula una huella que no coincide (esta
  incompleto), y lo trata como "contenido distinto" cuando es el MISMO
  archivo en vuelo. `_asegurar_en_fuente` se reescribio: el contenido se
  escribe COMPLETO a un temporal propio de la llamada primero, y solo se
  PUBLICA con `os.link()` (hardlink) -- atomico igual que
  `O_CREAT|O_EXCL` (verificado empiricamente: falla con `FileExistsError`
  ante CUALQUIER cosa que ya exista en ese nombre -- symlink valido, roto,
  o archivo regular -- sin seguirlo jamas), pero sin la ventana de lectura
  a medio escribir, porque el nombre final nunca es el que se escribe.
  Este cambio estructural tambien vuelve moot dos de los seis hallazgos
  menores de A-5 (ver abajo): ya no hay ningun `os.open` con `O_NOFOLLOW`
  al que quitarle la bandera, ni una escritura directa al nombre final
  cuyo fallo a mitad de camino haya que limpiar con un `unlink` -- las
  dos preocupaciones quedaron cerradas por construccion, no por un test.

- **A-4 (el agujero estaba en el test, no en el codigo):** `test_I3_...`
  solo cubria el orden `.xlsx -> .docx`. Al reves (`.docx` primero)
  reaparecia el mismo defecto. Parametrizado en los DOS ordenes.

- **A-5 (seis mutaciones menores):** tres se atacaron con test propio
  (temporal huerfano sin limpiar en `finally`; `ficha.sha256 != huella`
  neutralizado; `.lower()` de la extension quitado). Las otras tres --
  `O_NOFOLLOW`, el `unlink` de limpieza, y el jail sobre
  `destino_literal` -- se investigaron a fondo, con evidencia empirica
  pegada en task-8-report.md: las dos primeras dejaron de existir como
  codigo ejecutable con la reescritura de arriba: la tercera
  (`_resolver_bajo_jail(destino_literal)`) se mantiene como defensa en
  profundidad DELIBERADA aunque se demostro que, con la construccion
  ACTUAL de `_nombre_candidato` (un solo componente de ruta, y
  `fuente_abs` siempre un nivel por debajo de `trabajo_abs`), un ".."
  como unico candidato posible de escape SOLO puede llegar hasta
  `trabajo_abs` -- que ya esta, por definicion, dentro del jail. No se
  fabrico un test que simulara una vulnerabilidad que hoy no existe; se
  deja documentado el porque, para que si `_nombre_candidato` cambia
  algun dia (mas de un componente de ruta, u otro nivel de anidamiento),
  quede claro que ESTA es la linea que hay que revisar primero.
"""
from __future__ import annotations

import errno
import os
import random
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from motor_registry import tool_authority

from procesamiento import compuerta
from procesamiento.extractores import excel, ocr, pdf, word
from procesamiento.ficha import Ficha, sha256_de


class IngestaError(Exception):
    """Error de dominio de la ingesta. Nunca se deja escapar un `OSError`
    crudo del filesystem hacia el llamador sin contexto -- ver A-3,
    task-8-hallazgos-r2.md."""


class ReemplazoProcesadoAgotado(IngestaError):
    """Se agotaron los reintentos para reemplazar `procesado/<huella>/`:
    otro escritor de la MISMA huella siguió ganando la carrera entre el
    `rmtree` y el `rename` (ver el comentario en `ingerir`). No es
    corrupción -- el invariante de C-2 no se reabre -- es contención
    sostenida bajo concurrencia real más alta de la medida."""

# 255: limite tipico de NAME_MAX en filesystems Linux (ext4 y la mayoria).
# Se trata como caracteres, no bytes, por simplicidad -- los casos de uso
# reales (nombres subidos por un cliente) son mayormente ASCII; un nombre
# con caracteres multibyte que se pase del limite en BYTES pero no en
# caracteres queda fuera de este recorte (I-6 pide "que quepa", no un
# calculo exacto de bytes de filesystem).
_LIMITE_NOMBRE = 255

# Cuántas veces se reintenta el reemplazo final de `procesado/<huella>/`
# cuando otro escritor de la MISMA huella gana la carrera entre nuestro
# `rmtree` y nuestro `rename` (ver el comentario en `ingerir`).
#
# A-3 (task-8-hallazgos-r2.md): medido con hilos reales, SIN jitter, el
# reintento en 10 no aguantaba lo que decía aguantar -- a 16 hilos fallaba
# 21 de 25 veces con un OSError(39) crudo. La causa: reintentos
# INMEDIATOS se resincronizan (todos los perdedores de una ronda vuelven
# a chocar en la siguiente, a la vez). Con jitter, el número medido
# después del arreglo está en task-8-report.md -- "debería aguantar" es
# una suposición, así que se mide y se escribe.
_MAX_INTENTOS_REEMPLAZO = 10

# Tope del sleep aleatorio entre reintentos, que CRECE con el número de
# intento (backoff simple: intento 1 sortea en [0, _JITTER_MAXIMO_SEGUNDOS],
# intento 2 en [0, 2*_JITTER_MAXIMO_SEGUNDOS], etc.) -- desincroniza a los
# escritores que arrancaron casi juntos sin convertir esto en una espera
# larga: en el caso normal (sin contención) el jitter nunca se ejecuta,
# porque el primer intento siempre corre sin sleep.
_JITTER_MAXIMO_SEGUNDOS = 0.02

# D-2 (task-9-brief.md, 2026-09-21): cuántas veces se reintenta un fallo
# PERMANENTE (`error`/`sin_extractor`) antes de dejar de pagar el costo
# completo del extractor en cada llamada. Medido: un PDF de CamScanner con
# páginas de 1836x2376 pt (9x el área normal a 300 DPI) agota el timeout de
# `pdftoppm` (~300s) y, como I-1 nunca cachea un `error` (a propósito -- un
# fallo transitorio tiene que poder curarse), ese documento volvía a costar
# 300s en CADA ingesta, sin techo. El ruling conserva la curación y le pone
# techo: tres intentos -- no uno, porque el primero puede fallar por algo de
# veras pasajero --, y el tercero es el último mientras no cambie la versión
# del extractor (ver `_estado_de_error_cacheado`, que es la señal de que la
# causa pudo haberse arreglado).
_MAX_INTENTOS_ERROR = 3

# nombre del extractor (Resultado.extractor / Ficha.extractor) -> modulo
# que sabe reportar SU version vigente -- consulta barata (un __version__
# ya importado, o un `--version` de proceso, nunca una re-extraccion) para
# I-2: invalidar el cache cuando el extractor cambio de version.
_MODULOS_POR_EXTRACTOR = {
    excel.EXTRACTOR: excel,
    ocr.EXTRACTOR: ocr,
    pdf.EXTRACTOR: pdf,
    word.EXTRACTOR: word,
}


def _version_vigente(nombre_extractor: str) -> str | None:
    modulo = _MODULOS_POR_EXTRACTOR.get(nombre_extractor)
    if modulo is None:
        return None
    try:
        return modulo._version()
    except Exception:  # fail-soft: modulo._version() puede fallar (import roto, atributo ausente); se devuelve None y el llamador lo trata como version desconocida, forzando reextraccion en vez de confiar en una cache que no puede validar
        return None


def _ahora() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _resolver_bajo_jail(ruta: Path) -> Path:
    """Valida `ruta` contra el MISMO jail que `tool_authority` usa para
    tool_calls (`WORKSPACE_ROOT` + `resolve_jailed_path`) -- no se escribe
    un jail nuevo (spec §4, C-1). `resolve_jailed_path` exige un path
    RELATIVO; acá se hace la traducción antes de delegarle el chequeo de
    fondo (sigue symlinks vía `Path.resolve()`, compara contra
    `WORKSPACE_ROOT`) -- esa lógica sigue viviendo en un solo lugar."""
    resuelto = Path(ruta).resolve(strict=False)
    try:
        relativo = resuelto.relative_to(tool_authority.WORKSPACE_ROOT)
    except ValueError:
        raise ValueError(
            f"ingesta: la ruta escapa del workspace (directo o vía symlink): "
            f"resuelve a '{resuelto}', fuera de '{tool_authority.WORKSPACE_ROOT}'"
        ) from None

    if str(relativo) == ".":
        return tool_authority.WORKSPACE_ROOT

    validado, razon = tool_authority.resolve_jailed_path(str(relativo), [])
    if validado is None:
        raise ValueError(f"ingesta: {razon}")
    return validado


def ruta_procesado(trabajo: Path, huella: str) -> Path:
    trabajo_abs = _resolver_bajo_jail(Path(trabajo))
    return trabajo_abs / "procesado" / huella


def _resolver_subruta_de_fuente(fuente_abs: Path, subruta: str | Path | None) -> Path:
    """Resuelve la subcarpeta OPCIONAL de `fuente/` donde se conserva la
    estructura de origen (camino de entrada más simple, 2026-09-21,
    `scripts/procesar_archivos.py`: "un archivo que estaba en 'Estados
    Financieros/EEFF.pdf' tiene que quedar en
    'fuente/estados-financieros/EEFF.pdf', no suelto en la raíz").

    Pasa por el MISMO jail que el resto de esta pieza (`_resolver_bajo_jail`,
    spec §4, C-1) -- una subruta también es una ruta que el llamador
    controla. Y ADEMÁS, acá, se exige que el resultado quede DENTRO de
    `fuente_abs`: el jail global sólo garantiza "adentro de
    WORKSPACE_ROOT", así que una subruta con '..' que se quedara dentro del
    workspace pero saliera de `fuente/` (hacia `procesado/`, por ejemplo)
    pasaría el jail global sin problema y rompería igual el invariante de
    ESTA carpeta puntual -- `fuente/` inmutable, cada cosa en su lugar."""
    if not subruta:
        return fuente_abs
    resuelto = _resolver_bajo_jail(fuente_abs / Path(subruta))
    if resuelto != fuente_abs and fuente_abs not in resuelto.parents:
        raise ValueError(
            f"ingesta: la subruta '{subruta}' resuelve fuera de 'fuente/' "
            f"({resuelto})"
        )
    return resuelto


def _nombre_candidato(origen: Path, sufijo_extra: str = "") -> str:
    """I-6: el TRONCO se recorta antes de agregar el sufijo de huella, así
    el resultado siempre cabe en el límite del filesystem -- nunca un
    `OSError` crudo por un nombre demasiado largo."""
    extension = origen.suffix
    tronco = origen.stem
    nombre = f"{tronco}{sufijo_extra}{extension}"
    if len(nombre) <= _LIMITE_NOMBRE:
        return nombre
    recorte = max(1, _LIMITE_NOMBRE - len(sufijo_extra) - len(extension))
    return f"{tronco[:recorte]}{sufijo_extra}{extension}"


def _candidatos_de_nombre(origen: Path, huella: str) -> list[str]:
    return [
        _nombre_candidato(origen),
        _nombre_candidato(origen, f"-{huella[:8]}"),
        _nombre_candidato(origen, f"-{huella}"),
    ]


def _asegurar_en_fuente(origen: Path, fuente_abs: Path, huella: str) -> Path:
    """Copia `origen` a `fuente_abs` bajo un nombre LIBRE. Nunca pisa un
    archivo existente de contenido distinto, y nunca sigue un symlink --
    ni para leerlo (para decidir si "ya está") ni para escribir a través
    de él (C-1).

    Hallazgo PROPIO (no está en task-8-hallazgos-r2.md; lo destapó medir
    A-3 con más hilos, tal como pedía el ruling): con `O_CREAT|O_EXCL`
    escribiendo DIRECTO al nombre final, un lector concurrente que pierde
    la carrera puede ver el archivo a MITAD de escribir -- `sha256_de` lo
    lee TAL COMO ESTÁ en ese instante, calcula una huella que no coincide
    (porque está incompleto), y lo trata como "contenido distinto" cuando
    en realidad es EL MISMO archivo, todavía en vuelo. Con suficientes
    hilos esto cascadea por los tres nombres candidatos y agota los
    reintentos con un `ValueError` que no debería existir (medido: 1-2 de
    30 corridas a partir de 6 hilos, antes de este arreglo).

    Arreglo: el contenido se escribe COMPLETO a un temporal PROPIO de esta
    llamada primero, y sólo se PUBLICA con `os.link()` (hardlink) --
    atómico igual que `O_CREAT|O_EXCL` (falla con `FileExistsError` si el
    nombre ya existe, EN CUALQUIER FORMA -- symlink válido, roto, o
    archivo regular -- sin seguirlo jamás, verificado empíricamente antes
    de este cambio), pero con la ventaja de que cuando `link()` tiene
    éxito el contenido YA está completo: ningún lector puede ver un
    archivo a medio escribir bajo el nombre final, porque el nombre final
    nunca es el que se escribe."""
    temporal = fuente_abs / f".tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with open(temporal, "wb") as escritura, open(origen, "rb") as lectura:
            shutil.copyfileobj(lectura, escritura)

        candidatos = _candidatos_de_nombre(origen, huella)
        for nombre_candidato in candidatos:
            destino_literal = fuente_abs / nombre_candidato

            if destino_literal.is_symlink():
                # Nunca se sigue, nunca se pisa, nunca se lee a través de
                # él: se trata como una colisión de nombre más y se
                # prueba el siguiente candidato. El symlink queda intacto.
                continue

            # Defensa en profundidad (spec §4): el mismo jail que valida
            # `trabajo`, ahora sobre el destino final -- por si
            # `origen.name` compusiera algo que terminara escapando.
            _resolver_bajo_jail(destino_literal)

            try:
                os.link(temporal, destino_literal)
            except FileExistsError:
                # Colisión: el nombre ya está ocupado. Si mientras tanto
                # se volvió un symlink (TOCTOU), tampoco se sigue.
                if destino_literal.is_symlink():
                    continue
                # A-1/A-2 (task-8-hallazgos-r2.md): el MISMO modelo de
                # amenaza que C-1 -- algo plantado en `fuente/` -- pero
                # por el camino de comparar huellas en vez de escribir.
                # Un FIFO cuelga `sha256_de` PARA SIEMPRE (abre y
                # bloquea, sin timeout); un directorio revienta con
                # `IsADirectoryError`. Sólo se le calcula la huella a un
                # archivo REGULAR -- esto cierra los dos Y cualquier otro
                # tipo raro (socket, dispositivo) sin tener que
                # enumerarlos uno por uno.
                if not destino_literal.is_file():
                    continue
                if sha256_de(destino_literal) == huella:
                    return destino_literal  # mismo contenido -- ya está
                continue  # contenido distinto -- siguiente candidato
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    continue  # se volvió symlink justo antes del link() (TOCTOU)
                raise
            return destino_literal

        raise ValueError(
            f"ingesta: no se pudo asegurar un nombre libre en 'fuente/' para "
            f"'{origen.name}' tras {len(candidatos)} intentos (colisiones de "
            "contenido o symlinks en cada nombre candidato)"
        )
    finally:
        # El temporal es descartable en cuanto se publicó (o se decidió
        # no usarlo): `link()` exitoso deja el contenido vivo bajo el
        # nombre final -- son dos nombres para el mismo inodo, borrar uno
        # no afecta al otro.
        temporal.unlink(missing_ok=True)


def _ficha_de_cache_valida(carpeta: Path, huella: str, extension_actual: str) -> Ficha | None:
    """Lee la ficha cacheada en `carpeta` y decide si es un acierto de
    caché VÁLIDO. `None` si hay que reextraer -- ficha ilegible (I-4),
    estado no reconstruible (I-1), extractor/versión desactualizados
    (I-2), extensión distinta a la que rutea hoy (I-3), o falta alguna
    salida listada (I-5). Nunca lanza -- un fallo de lectura es, siempre,
    un fallo de caché, no una excepción hacia el llamador."""
    ficha_json = carpeta / "ficha.json"
    if not ficha_json.is_file():
        return None
    try:
        ficha = Ficha.desde_json(ficha_json.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None  # I-4

    if ficha.sha256 != huella:
        return None  # defensivo: no debería pasar, la carpeta ya está indexada por huella

    if ficha.estado not in {"ok", "parcial"}:
        return None  # I-1: un error/sin_extractor no es un extracto reconstruible

    version_vigente = _version_vigente(ficha.extractor)
    if version_vigente is None or ficha.extractor_version != version_vigente:
        return None  # I-2

    if ficha.detalle.get("_extension_ingesta") != extension_actual:
        return None  # I-3

    salidas_listadas = ficha.detalle.get("_salidas_ingesta")
    if not isinstance(salidas_listadas, list):
        return None  # ficha sin inventario de salidas -- no hay forma de verificar completitud
    for nombre_salida in salidas_listadas:
        if not (carpeta / nombre_salida).is_file():
            return None  # I-5

    return ficha


def _estado_de_error_cacheado(
    carpeta: Path, huella: str, extension_actual: str
) -> tuple[Ficha | None, int]:
    """D-2: lee la ficha cacheada en `carpeta`, si la hay, y la interpreta
    como un intento previo FALLIDO (`error` o `sin_extractor` -- el ruling
    es explícito: los dos se comportan igual acá). Devuelve `(None, 0)`
    cuando no hay una ficha de fallo UTILIZABLE para contar intentos: no
    existe, está corrupta (I-4), la huella o la extensión no coinciden
    (defensivo / I-3), el estado no es de fallo, o el extractor VIGENTE
    cambió de versión desde que se escribió -- en ese caso la cuenta
    arranca de cero, porque un cambio de versión es la señal de que la
    causa del fallo pudo haberse arreglado (misma lógica de I-2, aplicada
    ahora también al conteo de intentos, no sólo al acierto de caché).

    CORRECCIÓN (ronda P10, 2026-09-21, hallazgo reportado sin tocar --
    ruling del coordinador): `_version_vigente` devolviendo `None` significa
    "no se pudo determinar la versión vigente", NO "la versión cambió". Son
    cosas distintas y antes se trataban igual: un extractor cuya consulta de
    versión falla de forma PERSISTENTE reiniciaba la cuenta en cada ingesta
    y el tope de D-2 dejaba de existir en silencio -- exactamente el defecto
    de los 300s por llamada que D-2 vino a cerrar. Ahora sólo una versión
    CONOCIDA y DISTINTA de la guardada reinicia la cuenta; un `None` deja el
    conteo tal cual estaba, como si la consulta de versión no se hubiera
    podido hacer (que es, literalmente, lo que pasó).

    Si hay una ficha de fallo UTILIZABLE, devuelve `(ficha, intentos)` con
    el número de intentos ya gastados. Nunca lanza -- un fallo de lectura
    acá es, igual que en `_ficha_de_cache_valida`, un fallo de caché, no
    una excepción hacia el llamador."""
    ficha_json = carpeta / "ficha.json"
    if not ficha_json.is_file():
        return None, 0
    try:
        ficha = Ficha.desde_json(ficha_json.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None, 0  # I-4

    if ficha.sha256 != huella:
        return None, 0  # defensivo: no debería pasar, la carpeta ya está indexada por huella

    if ficha.estado not in {"error", "sin_extractor"}:
        return None, 0  # no es un fallo -- este camino no aplica

    if ficha.detalle.get("_extension_ingesta") != extension_actual:
        return None, 0  # I-3 aplicado también acá

    version_vigente = _version_vigente(ficha.extractor)
    if version_vigente is not None and ficha.extractor_version != version_vigente:
        return None, 0  # I-2: el extractor cambió -- la cuenta arranca de cero
    # version_vigente is None: "no sé" -- se sigue de largo y se cuenta la
    # ficha como utilizable, con los intentos que ya tenía.

    intentos = ficha.detalle.get("_intentos")
    if not isinstance(intentos, int) or intentos < 1:
        # Ficha de fallo escrita ANTES de este arreglo (o corrupta en este
        # campo puntual): no tiene `_intentos` -- cuenta como un primer
        # intento ya gastado, nunca como cero (que reiniciaría el tope
        # sin motivo).
        intentos = 1
    return ficha, intentos


def _resumen_parcial(detalle: dict) -> str:
    """Frase CORTA -- la reserva arquitectónica de final-hallazgos.md pide
    explícitamente "qué falta y dónde está el detalle, no las 141 cifras".
    Mira las claves YA conocidas que los cuatro extractores escriben en
    `parcial` (nunca inventa una nueva taxonomía); si ninguna aplica, cae
    en `detalle["razon"]` cuando existe, y si tampoco, un aviso genérico --
    nunca se queda muda."""
    partes: list[str] = []

    paginas_sin_texto = detalle.get("paginas_sin_texto")
    if paginas_sin_texto:
        total_paginas = detalle.get("paginas")
        de_total = f" de {total_paginas}" if total_paginas else ""
        lista = ",".join(str(p) for p in paginas_sin_texto)
        partes.append(f"{len(paginas_sin_texto)}{de_total} paginas sin texto: {lista}")

    paginas_con_dudas = detalle.get("paginas_con_dudas")
    if paginas_con_dudas:
        partes.append(f"{len(paginas_con_dudas)} paginas con dudas")

    palabras_dudosas = detalle.get("palabras_dudosas")
    if palabras_dudosas:
        partes.append(f"{len(palabras_dudosas)} palabras/cifras dudosas")

    cuadros = detalle.get("cuadros_de_texto_omitidos")
    if cuadros:
        partes.append(f"{cuadros} cuadros de texto omitidos")

    notas = detalle.get("notas_al_pie_omitidas")
    if notas:
        partes.append(f"{notas} notas al pie omitidas")

    formulas = detalle.get("formulas_sin_valor")
    if isinstance(formulas, dict) and formulas.get("total"):
        partes.append(f"{formulas['total']} formulas sin valor en cache")

    no_tabulares = detalle.get("no_tabulares")
    if no_tabulares:
        partes.append(f"{len(no_tabulares)} hoja(s) no tabular(es) omitidas")

    # MINOR 3 (final-hallazgos.md, adenda 2026-09-21): la pérdida MÁS
    # GRAVE que puede tener un libro -- una hoja entera que no se pudo
    # extraer -- era justo la única que este resumen no nombraba. `excel.py`
    # declara `fallidas` (hoja + excepción) cuando `_hoja_a_csv` revienta, y
    # `hojas_extraidas < hojas` es la señal binaria de que algo se quedó
    # afuera. Sin esto, con una hoja perdida el aviso decía el genérico
    # "extracto parcial -- ver ficha.json" -- nada de qué faltaba.
    fallidas = detalle.get("fallidas")
    if fallidas:
        partes.append(
            f"{len(fallidas)} hoja(s) perdida(s) por error: "
            + "; ".join(str(f) for f in fallidas)
        )
    else:
        hojas_totales = detalle.get("hojas")
        hojas_extraidas = detalle.get("hojas_extraidas")
        if (
            isinstance(hojas_totales, int)
            and isinstance(hojas_extraidas, int)
            and hojas_extraidas < hojas_totales
            and not no_tabulares
        ):
            # Red de seguridad genérica: el hueco no lo explica ni
            # `fallidas` ni `no_tabulares` (las dos causas conocidas) --
            # no debería dispararse hoy, pero si `excel.py` gana una
            # tercera razón para no extraer una hoja, esto sigue nombrando
            # el hueco en vez de quedarse callado.
            partes.append(
                f"{hojas_totales - hojas_extraidas} de {hojas_totales} hojas "
                "no se extrajeron"
            )

    if not partes:
        razon = detalle.get("razon")
        if razon:
            partes.append(str(razon))

    return "; ".join(partes) if partes else "extracto parcial -- ver ficha.json"


def _encabezado_parcial(huella: str, detalle: dict) -> str:
    """La reserva arquitectónica de final-hallazgos.md: "Toda la honestidad
    de este sistema vive en `ficha.json`, y nadie la lee" -- un modelo que
    hace `file_read` sobre `procesado/<huella>/texto.txt` (o `.md`, o un
    `.csv`) recibía un extracto `parcial` SIN ningún aviso: la declaración
    de qué faltaba vivía sólo en la ficha, que nadie pide. Este encabezado
    va DENTRO del archivo que el modelo lee -- el único lugar donde sirve
    -- y sólo se antepone cuando el estado NO es 'ok' (el caso feliz no
    paga ruido; y `error`/`sin_extractor` nunca traen salidas que decorar,
    por invariante de `Resultado`)."""
    resumen = _resumen_parcial(detalle)
    return (
        f"<!-- EXTRACTO PARCIAL · {resumen} -- "
        f"ficha completa: procesado/{huella}/ficha.json -->"
    )


def ingerir(origen: Path, trabajo: Path, *, subruta: str | Path | None = None) -> Ficha:
    """`subruta` (opcional): subcarpeta de `fuente/` donde se conserva la
    subcarpeta de origen -- ej. `"estados-financieros"` para un archivo que
    venía de una carpeta `Estados Financieros/` (ver
    `scripts/procesar_archivos.py`). `None` o `""` es la raíz de `fuente/`,
    el comportamiento de siempre. `procesado/` NO se ve afectado: sigue
    indexado por sha256, plano -- sólo cambia dónde vive la copia en
    `fuente/`."""
    origen = Path(origen)
    trabajo_abs = _resolver_bajo_jail(Path(trabajo))  # C-1/I-7
    fuente_raiz = trabajo_abs / "fuente"
    fuente_abs = _resolver_subruta_de_fuente(fuente_raiz, subruta)
    fuente_abs.mkdir(parents=True, exist_ok=True)

    huella = sha256_de(origen)
    extension_actual = origen.suffix.lower()
    destino = _asegurar_en_fuente(origen, fuente_abs, huella)  # C-1/C-3

    carpeta = ruta_procesado(trabajo_abs, huella)
    ficha_cacheada = _ficha_de_cache_valida(carpeta, huella, extension_actual)
    if ficha_cacheada is not None:
        # Cache vivo y VERIFICADO -- cero trabajo.
        return ficha_cacheada

    # D-2: un fallo PERMANENTE (`error`/`sin_extractor`) tiene techo --
    # tres intentos, y el tercero es el último mientras no cambie la
    # versión del extractor. Sin esto, un documento que SIEMPRE falla paga
    # el costo completo del extractor en CADA ingesta (medido: 300s por
    # llamada con un PDF patológico), porque I-1 nunca cachea un `error` a
    # propósito -- un fallo transitorio tiene que poder curarse.
    ficha_error_previa, intentos_previos = _estado_de_error_cacheado(
        carpeta, huella, extension_actual
    )
    if ficha_error_previa is not None and intentos_previos >= _MAX_INTENTOS_ERROR:
        return ficha_error_previa

    resultado = compuerta.extraer(destino)

    # Temporal único POR LLAMADA (C-2): dos ingestas del mismo archivo,
    # incluso entrelazadas, nunca comparten directorio de escritura -- no
    # hay ventana en la que una pise las salidas de la otra. Se limpia en
    # `finally`: si algo revienta a mitad de camino, no queda un huérfano
    # (cierra I-8).
    temporal = carpeta.with_name(f"{carpeta.name}.{os.getpid()}-{uuid.uuid4().hex}.parcial")
    try:
        temporal.mkdir(parents=True)

        # Reserva arquitectónica (final-hallazgos.md, ronda de cierre): un
        # extracto `parcial` lleva el aviso DENTRO del propio archivo que
        # el modelo lee -- la ficha (donde vivía la declaración completa)
        # no es lo que un `file_read` sobre `procesado/<huella>/` entrega.
        #
        # MAJOR 1 (final-hallazgos.md, adenda 2026-09-21, ruling de
        # Fernando sobre su propio ruling anterior): ese "dentro" asumía
        # texto/markdown -- pero `excel.py` emite CSV, y un CSV no tiene
        # sintaxis de comentario. Un `<!-- EXTRACTO PARCIAL ... -->`
        # antepuesto a un CSV NO es un comentario: es UNA FILA MÁS.
        # `csv.DictReader` la toma como nombre de columna y manda todo lo
        # demás a la clave `None` -- reproducido con un `.xlsx` con una
        # fórmula sin caché (condición de 'parcial' frecuentísima en
        # libros financieros): arreglamos la honestidad rompiendo el dato,
        # justo en los documentos que ya venían declarados dañados.
        #
        # Por formato: `.md`/`.txt` siguen llevando el aviso DENTRO (ahí
        # es texto libre, funciona). Un `.csv` queda INTACTO -- byte a
        # byte lo que el extractor produjo, nunca decorado -- y el aviso
        # va a un archivo HERMANO `AVISO.txt` en la misma carpeta
        # `procesado/<huella>/`. Un modelo que lee sólo el CSV no ve el
        # aviso -- se acepta a propósito: un CSV corrupto es peor que un
        # aviso que hay que ir a buscar, y la carpeta lo muestra al
        # listarla (y queda en `_salidas_ingesta`, así que I-5 lo cubre:
        # si se borra, el próximo acierto de caché falla y se regenera).
        encabezado = (
            _encabezado_parcial(huella, resultado.detalle)
            if resultado.estado == "parcial"
            else None
        )
        hubo_csv_sin_encabezado = False
        total_extracto_bytes = 0
        for nombre, contenido in resultado.salidas.items():
            if encabezado is not None:
                if nombre.lower().endswith(".csv"):
                    hubo_csv_sin_encabezado = True
                else:
                    contenido = f"{encabezado}\n\n{contenido}"
            (temporal / nombre).write_text(contenido, encoding="utf8")
            # MINOR (final-hallazgos.md, adenda): contado DESPUÉS de
            # decorar -- antes se sumaba `resultado.salidas.values()` sin
            # transformar, subestimando el tamaño real del archivo que
            # queda en disco cuando el encabezado se antepone.
            total_extracto_bytes += len(contenido.encode("utf8"))

        salidas_ingesta = sorted(resultado.salidas.keys())
        if encabezado is not None and hubo_csv_sin_encabezado:
            (temporal / "AVISO.txt").write_text(encabezado, encoding="utf8")
            total_extracto_bytes += len(encabezado.encode("utf8"))
            salidas_ingesta = sorted(salidas_ingesta + ["AVISO.txt"])

        detalle = {
            **dict(resultado.detalle),
            # I-3/I-5: lo que hace falta para verificar un acierto de
            # caché sin tener que confiar ciegamente en él.
            "_extension_ingesta": extension_actual,
            "_salidas_ingesta": salidas_ingesta,
        }
        if resultado.estado in {"error", "sin_extractor"}:
            # D-2: registra CUÁNTOS intentos lleva este fallo -- es lo que
            # `_estado_de_error_cacheado` lee en la próxima ingesta para
            # decidir si reintenta o si ya agotó el tope.
            detalle["_intentos"] = intentos_previos + 1

        if resultado.salidas:
            # I-6 (final-hallazgos.md, ronda de cierre): un extracto que en
            # total supera el tope de lectura de un tool_call
            # (`tool_authority.MAX_READ_BYTES`) es ilegible para quien lo
            # consume vía `file_read`, aunque el extracto esté COMPLETO y
            # sea CORRECTO -- no cambia el estado (nada se perdió), pero
            # tampoco puede quedar invisible (medido: xlsx-04, 159.077 B
            # de original -> 17.861.532 B de extracto, 'ok', 89x el tope --
            # y el criterio §7.B.2 no lo ve porque sólo evalúa documentos
            # cuyo ORIGINAL no cabía). `total_extracto_bytes` ya es el
            # tamaño REAL post-decoración (ver el conteo arriba).
            if total_extracto_bytes > tool_authority.MAX_READ_BYTES:
                detalle["excede_tope_lectura"] = True
                detalle["excede_tope_lectura_bytes"] = {
                    "extracto": total_extracto_bytes,
                    "original": origen.stat().st_size,
                    "tope": tool_authority.MAX_READ_BYTES,
                }

        ficha = Ficha(
            sha256=huella,
            origen=str(destino.relative_to(trabajo_abs)),
            extractor=resultado.extractor,
            extractor_version=resultado.version,
            fecha=_ahora(),
            estado=resultado.estado,
            detalle=detalle,
        )
        (temporal / "ficha.json").write_text(ficha.a_json(), encoding="utf8")

        # Escritura atómica por carpeta: primero temporal (único, ver
        # arriba), después rename. Si algo se cae a la mitad, no queda un
        # `procesado/` a medias que parezca completo -- `carpeta` (el
        # nombre final) sólo aparece de un solo golpe.
        #
        # Hallazgo propio durante la verificación de C-2 (no estaba en
        # task-8-hallazgos.md, lo destapó un test de concurrencia REAL
        # fallando ~50% de las veces): el `rmtree` y el `rename` son DOS
        # operaciones separadas -- entre una y otra, otro escritor de la
        # MISMA huella puede colarse y volver a crear `carpeta` con su
        # propio rename exitoso. El nuestro entonces revienta con
        # `OSError(ENOTEMPTY)` porque `rename()` no pisa un directorio NO
        # vacío en POSIX. Como la extracción es determinista (misma
        # huella -> mismo contenido, sólo cambia `fecha`), es seguro
        # reintentar: eventualmente hay una ventana en la que nadie más
        # está escribiendo, y CUALQUIERA de los dos resultados que gane
        # es válido -- lo que no puede pasar es una excepción sin
        # sentido hacia el llamador por una carrera entre dos escritores
        # legítimos.
        for intento in range(_MAX_INTENTOS_REEMPLAZO):
            if intento > 0:
                # A-3: jitter con backoff. Reintentos INMEDIATOS se
                # resincronizan -- todos los perdedores de una ronda
                # vuelven a chocar juntos en la siguiente. El primer
                # intento SIEMPRE corre sin sleep (caso normal, sin
                # contención, cero costo).
                time.sleep(random.uniform(0, _JITTER_MAXIMO_SEGUNDOS * intento))
            if carpeta.exists():
                shutil.rmtree(carpeta, ignore_errors=True)
            try:
                temporal.rename(carpeta)
                break
            except OSError as exc:
                if exc.errno != errno.ENOTEMPTY:
                    raise
                if intento == _MAX_INTENTOS_REEMPLAZO - 1:
                    # A-3: nunca un OSError(39) crudo hacia el llamador --
                    # un error de dominio con contexto (qué huella, cuántos
                    # intentos) es lo mínimo que alguien necesita para
                    # decidir si reintentar más arriba tiene sentido.
                    raise ReemplazoProcesadoAgotado(
                        f"ingesta: no se pudo reemplazar '{carpeta}' tras "
                        f"{_MAX_INTENTOS_REEMPLAZO} intentos -- contención "
                        f"sostenida con otro escritor de la misma huella "
                        f"({huella})"
                    ) from exc
    finally:
        if temporal.exists():
            shutil.rmtree(temporal, ignore_errors=True)

    return ficha
