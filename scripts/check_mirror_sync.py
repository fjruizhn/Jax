#!/usr/bin/env python3
"""
Compara las FAMILIAS DE ESPEJOS entre jax y jax-platform.

Un "espejo" es un modulo replicado a proposito en varios codebases -- el patron
declarado de "sin paquete compartido", cada repo con su conector minimo. Es una
decision consciente, y tiene un costo conocido: un arreglo hecho en una copia y
no en la otra queda invisible. **Ese costo ya se cobro TRES veces en 2026**
(facet_resolver._db_conn sin el guard fail-closed; los cuatro sitios del default
a la instancia muerta 3306; y el propio credential_resolver._db_conn), siempre
con la misma forma: se cierra en un repo y sobrevive en el otro.

Este script existia solo para `facet_resolver` (como
`scripts/check_facet_resolver_sync.py`, renombrado el 2026-09-01). Se
generalizo en vez de copiarse: una segunda copia del comparador seria un espejo
mas, con el mismo defecto que viene a detectar. Las familias se declaran como
DATOS al final del archivo; agregar una es agregar una entrada.

Compara solo los simbolos que DEBEN ser identicos. Lo que legitimamente difiere
por diseno -- el import del conector local, y lo que este declarado con el
marcador -- no se compara o no se reporta.

NO arregla nada, diagnostico puro -- mismo espiritu que find_unread_columns.py.

Uso:
  python3 scripts/check_mirror_sync.py

Variables de entorno:
  JAX_PLATFORM_REPO_ROOT   raiz del checkout de jax-platform
                           (default: ~/jax-platform)

Exit code 0 si todo esta sincronizado, 1 si hay drift real, 2 si falta un
archivo declarado (fail-closed: un espejo que no se puede leer NO se saltea en
silencio -- eso dejaria el checker en verde sin haber comparado nada, que es
exactamente el modo de falla que este repo viene persiguiendo).

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass
from pathlib import Path

JAX_ROOT = Path(__file__).resolve().parent.parent
JAX_PLATFORM_ROOT = Path(
    os.environ.get("JAX_PLATFORM_REPO_ROOT", Path.home() / "jax-platform")
)

# Marcador que DECLARA una divergencia como deliberada. Vive en el codigo,
# junto a la divergencia, y no en una lista dentro de este script: una lista
# aparte se desincroniza igual que el codigo que pretende vigilar.
#
# POR QUE EXISTE: hasta 2026-09-01 este checker gritaba TRES veces y dos eran
# falsas (ResolvedFacet y _query_facet divergen a proposito por
# max_tokens_param). El drift REAL -- _db_conn sin el guard fail-closed contra
# el default a la instancia muerta 3306 -- quedaba escondido entre el ruido, y
# el script ademas no corria en ningun workflow. Un detector que no distingue
# lo esperado de lo anomalo entrena a ignorarlo.
MARCADOR = "DIVERGENCIA DELIBERADA"


@dataclass(frozen=True)
class Familia:
    """Una familia de espejos: un canonico y las copias que deben seguirlo."""

    nombre: str
    canonico: Path
    espejos: tuple[tuple[str, Path], ...]  # (etiqueta legible, ruta)
    compartidos: tuple[str, ...]
    nota: str = ""


def _bloque_declarativo(lineas: list[str], node: ast.AST) -> str:
    """Texto donde se busca el MARCADOR para un simbolo dado.

    NO es lo mismo que el segmento que se compara. Para una funcion o una clase
    el marcador va en el docstring, que SI es parte del segmento -- pero una
    CONSTANTE DE MODULO no tiene docstring, y los comentarios no estan en el
    AST: `ast.get_source_segment` de un `Assign` devuelve la sentencia pelada.

    Medido el 2026-09-01, probando el checker rompiendolo: una divergencia
    declarada sobre una constante -- con el marcador en el comentario de arriba
    Y en el de la misma linea -- se reportaba igual como DRIFT. O sea que el
    mecanismo de declaracion NO EXISTIA para constantes, justo la clase de
    simbolo que se agrego a la comparacion ese mismo dia (FACET_SEAL_PATH).

    La ventana va desde el bloque contiguo de comentarios inmediatamente
    anterior al nodo hasta el final de su ultima linea COMPLETA, asi que cubre
    las dos formas naturales de escribirlo.
    """
    inicio = node.lineno - 1
    while inicio > 0 and lineas[inicio - 1].lstrip().startswith("#"):
        inicio -= 1
    fin = getattr(node, "end_lineno", node.lineno)
    return "\n".join(lineas[inicio:fin])


def _extract(path: Path, compartidos: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    """{nombre: (segmento_a_comparar, texto_donde_buscar_el_marcador)}."""
    source = path.read_text()
    lineas = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    out = {}
    for node in tree.body:
        name = getattr(node, "name", None)
        # Constantes de modulo (`NOMBRE = ...`): no tienen `.name`, asi que
        # hasta 2026-09-01 este checker no las miraba nunca. Se agregaron por
        # FACET_SEAL_PATH -- dos espejos apuntando a sellos distintos dejarian
        # todo lo demas identico y la invalidacion no cruzaria. Solo asignacion
        # simple a UN nombre: un desempaquetado no es lo que se quiere comparar
        # y no vale la pena adivinarlo.
        if name is None and isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
        if name in compartidos:
            out[name] = (
                ast.get_source_segment(source, node),
                _bloque_declarativo(lineas, node),
            )
    return out


def revisar(familia: Familia) -> tuple[list[str], list[str], list[str]]:
    """Devuelve (drift, declaradas, faltantes) para una familia."""
    canonico = _extract(familia.canonico, familia.compartidos)
    drift, declaradas, faltantes = [], [], []

    for etiqueta, ruta in familia.espejos:
        espejo = _extract(ruta, familia.compartidos)
        for name in familia.compartidos:
            if name not in canonico or name not in espejo:
                faltantes.append(f"{name} ({etiqueta})")
                continue
            segmento_can, decl_can = canonico[name]
            segmento_esp, decl_esp = espejo[name]
            if segmento_can == segmento_esp:
                continue
            # Divergencia declarada EN EL CODIGO, en cualquiera de las copias.
            # Se busca en el bloque declarativo, no en el segmento: ver
            # _bloque_declarativo() -- una constante no tiene docstring.
            if MARCADOR in decl_can or MARCADOR in decl_esp:
                declaradas.append(f"{name} ({etiqueta})")
            else:
                drift.append(f"{name} ({etiqueta})")
    return drift, declaradas, faltantes


# ---------------------------------------------------------------------------
# Las familias. Agregar una es agregar una entrada aca.
# ---------------------------------------------------------------------------

FAMILIAS = (
    Familia(
        nombre="facet_resolver",
        canonico=JAX_ROOT / "jax" / "core" / "facet_resolver.py",
        espejos=(
            # las_manos/facet_resolver.py es un SYMLINK a jax/core (Bloque 2,
            # 2026-08-21): comparar hoy es un no-op. Se incluye igual, y a
            # proposito: el dia que alguien lo reemplace por una copia real, la
            # copia entra a la comparacion sola. La afirmacion "dos archivos,
            # tres procesos" deja de ser un comentario que nadie verifica.
            ("las_manos", JAX_ROOT / "las_manos" / "facet_resolver.py"),
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "facet_resolver.py"),
        ),
        # load_facet_registry queda afuera a proposito: exclusivo de jax (REPL).
        compartidos=(
            "FacetUnavailableError",
            "ResolvedFacet",
            "_CacheEntry",
            "_db_conn",
            "_query_facet",
            "resolve_facet",
            # Sello de invalidacion cross-proceso (Q3, 2026-09-01). El mecanismo
            # solo funciona si los tres procesos miran EL MISMO archivo y lo
            # interpretan igual. FACET_SEAL_PATH en particular: si los espejos
            # apuntaran a sellos distintos, todo lo demas coincidiria byte a
            # byte y la invalidacion no cruzaria -- drift invisible dentro del
            # propio mecanismo construido para cerrar un punto ciego.
            "FACET_SEAL_PATH",
            "_seal_mtime",
            "_tocar_sello",
            "_entrada_sellada",
            "invalidate_facet_cache",
        ),
        nota="jax-platform tiene copia real aparte (repo distinto, no se puede "
             "symlinkear entre repos y sobrevivir un clone fresco).",
    ),
    Familia(
        nombre="crypto_secrets",
        canonico=JAX_ROOT / "jax" / "core" / "crypto_secrets.py",
        espejos=(
            # las_manos/crypto_secrets.py es SYMLINK a jax/core desde el
            # 2026-09-16 (E-10): comparar ahí es un no-op, a propósito, como
            # facet_resolver. Hasta ese día eran tres archivos reales.
            ("las_manos", JAX_ROOT / "las_manos" / "crypto_secrets.py"),
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "crypto_secrets.py"),
        ),
        # EXCLUIDOS A PROPOSITO, mismo criterio que load_facet_registry:
        # `encrypt_secret` y `decrypt_db_secret` existen SOLO en jax-platform y
        # no son drift. La razon ya estaba escrita en el docstring de la copia
        # de jax: jax-platform es el lado que CIFRA (sync bidireccional
        # BD->.env) y ademas lee `user_api_keys`, tabla suya; los procesos de
        # JAX (worker, las_manos, REPL) solo necesitan DESCIFRAR. Meterlos en
        # la comparacion pondria la familia en rojo permanente por una
        # asimetria de diseno, que es justo lo que el marcador vino a evitar.
        #
        # PROVIDER_ENV_KEYS entra a proposito: es la lista de secretos que se
        # descifran en memoria, y su drift no se ve hasta que una key nueva
        # llega cifrada al proceso que no la tiene en la lista.
        compartidos=(
            "PROVIDER_ENV_KEYS",
            "_get_fernet",
            "decrypt_secret",
            "decrypt_provider_keys_in_env",
        ),
        nota="Dos archivos reales (jax/core y jax-platform) + symlink en las_manos/ "
             "desde 2026-09-16 (E-10). jax-platform tiene ademas encrypt_secret y "
             "decrypt_db_secret, excluidos por diseno (es el lado que cifra).",
    ),
    Familia(
        nombre="credential_resolver",
        canonico=JAX_ROOT / "jax" / "core" / "credential_resolver.py",
        espejos=(
            # las_manos/credential_resolver.py es SYMLINK a jax/core desde el
            # 2026-09-16 (E-11). Hasta ese día era un TERCER archivo real que
            # podía driftear dentro de jax. El test de no-fail-open escanea el
            # symlink por las dos rutas (no deduplica por resolve()): inocuo,
            # las dos muestran la misma marca.
            ("las_manos", JAX_ROOT / "las_manos" / "credential_resolver.py"),
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "credential_resolver.py"),
        ),
        # Los 10 simbolos de nivel superior, medidos identicos en los tres
        # archivos el 2026-09-01. La unica diferencia real es el import de
        # crypto_secrets (`from jax.core.crypto_secrets` vs `from
        # crypto_secrets`), que es un ImportFrom y no un simbolo nombrado: no
        # entra a la comparacion y no necesita marcador.
        #
        # _PROVIDER_ENV_KEY_MAP importa especialmente: mapea proveedor -> env
        # var de fallback. Si un espejo tuviera un mapa distinto, un proceso
        # leeria la credencial de OTRA variable de entorno y el sintoma seria
        # "esa faceta no funciona en Jacobs pero si en Mesa web".
        compartidos=(
            "logger",
            "CREDENTIAL_CACHE_TTL_SECONDS",
            "CREDENTIAL_STALE_MAX_SECONDS",
            "_PROVIDER_ENV_KEY_MAP",
            "CredentialUnavailableError",
            "_CacheEntry",
            "_db_conn",
            "_query_active_credential",
            "resolve_credential",
            "resolve_credential_instrumented",
        ),
        nota="Dos archivos reales (jax/core y jax-platform) + symlink en las_manos/ "
             "desde 2026-09-16 (E-11). El canónico de jax importa bare primero y cae "
             "a jax.core; los ImportFrom no se comparan.",
    ),
    Familia(
        nombre="db_connect_config",
        canonico=JAX_ROOT / "jax" / "core" / "db_connect_config.py",
        espejos=(
            # Misma forma que facet_resolver: las_manos/db_connect_config.py
            # es un SYMLINK a jax/core (Tarea 2b, ronda de arreglo 2,
            # 2026-09-14), asi que compararlo hoy es un no-op -- se incluye
            # igual, a proposito, por la misma razon que facet_resolver: si
            # algun dia deja de ser symlink, la copia real entra a la
            # comparacion sola.
            ("las_manos", JAX_ROOT / "las_manos" / "db_connect_config.py"),
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "db_connect_config.py"),
        ),
        # Un solo simbolo compartido: el helper entero es esa funcion (mas el
        # docstring de modulo, que no entra a la comparacion -- ast.parse no
        # lo trata como un nodo con `.name` ni como un Assign de un nombre).
        compartidos=(
            "db_connect_timeout_seconds",
        ),
        nota="jax-platform recibe la copia verbatim de jax/core/db_connect_config.py "
             "(Tarea 2b, ronda de arreglo 2, 2026-09-14) -- sin symlink cruzado entre "
             "repos posible, mismo criterio que las demas familias.",
    ),
    Familia(
        nombre="contrato_dispatch",
        canonico=JAX_ROOT / "jax" / "core" / "contrato_dispatch.py",
        espejos=(
            # Symlink a jax/core (PR-K), como facet_resolver: no-op hoy, a
            # proposito.
            ("las_manos", JAX_ROOT / "las_manos" / "contrato_dispatch.py"),
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "contrato_dispatch.py"),
        ),
        # El contrato PURO: los validadores y la regla de que transporte lo
        # exige. La lectura de la fila y el armado del body por transporte son
        # de cada repo (jax limita tambien Ollama; la Mesa web no).
        compartidos=(
            "logger",
            "TRANSPORTS_CON_CONTRATO_DE_DISPATCH",
            "ModelDispatchConfigError",
            "_MAX_TOKENS_PARAM_NAMES",
            "_MAX_OUTPUT_TOKENS_TOPE_COLUMNA",
            "_max_tokens_field",
            "_max_output_tokens_value",
            "faltantes_del_contrato",
            "errores_del_contrato",
        ),
        nota="Canonico de hecho: jax-platform backend/contrato_dispatch.py (PR-J, "
             "tope de columna y errores_del_contrato de PR-L); jax copia el bloque "
             "verbatim (PR-K rondas 2 y 3, 2026-09-14). ORDEN DE MERGE: PR-L antes "
             "que PR-K -- contra un jax-platform sin PR-L esta familia da drift. Mismo "
             "costo que las demas familias: un arreglo en una copia y no en la "
             "otra aparece aca como drift.",
    ),
    Familia(
        nombre="cola_uso",
        canonico=JAX_ROOT / "jax" / "core" / "cola_uso.py",
        espejos=(
            # Symlink a jax/core (Task 7, 2026-09-15), como facet_resolver y
            # db_connect_config: comparar hoy es un no-op y se incluye igual,
            # por la misma razon que las demas familias.
            ("las_manos", JAX_ROOT / "las_manos" / "cola_uso.py"),
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "uso" / "cola.py"),
        ),
        # TODOS los simbolos de nivel superior del archivo, sin excepciones: el
        # modulo ES el contrato (el formato del archivo del respaldo), no una
        # pieza con partes propias de cada repo. Lo unico que diverge es el
        # docstring de MODULO, que no es un nodo con `.name` ni un Assign y por
        # eso no entra a la comparacion. `tests/test_cola_uso_escritores.py`
        # exige que esta tupla cubra el archivo entero: una familia declarada a
        # medias deja simbolos sin vigilar y el checker da verde sin mirarlos.
        #
        # Importan especialmente CAMPOS, CAMPOS_OBLIGATORIOS, ORIGENES, SUFIJO
        # y DIRECTORIO_POR_DEFECTO: son el formato en disco. Un drift ahi no se
        # ve como un error -- se ve como una cola que crece y nunca drena,
        # porque el que deposita y el que inserta dejaron de hablar el mismo
        # idioma.
        compartidos=(
            "logger",
            "VARIABLE_DIRECTORIO",
            "DIRECTORIO_POR_DEFECTO",
            "VARIABLE_MAX_FILAS",
            "MAX_FILAS_POR_DEFECTO",
            "SUFIJO",
            "SUFIJO_TEMPORAL",
            "SUBDIRECTORIO_CORRUPTOS",
            "CAMPOS",
            "CAMPOS_OBLIGATORIOS",
            "ORIGENES",
            "_lock",
            "_medicion",
            "_estado",
            "estadisticas",
            "reset_estado",
            "_anotar_error",
            "_nueva_medicion",
            "_publicar_profundidad",
            "_ruta_configurada",
            "directorio_del_respaldo",
            "max_filas",
            "_id_seguro",
            "_ahora_iso",
            "_normalizar",
            "_motivo_de_corrupcion",
            "_listar",
            "_contar_barato",
            "_contar",
            "_escribir_atomico",
            "_hacer_lugar",
            "_cuarentena",
            "_leer_lote",
            "_borrar",
            "encolar",
            "leer_pendientes",
            "quitar",
            "contar_pendientes",
        ),
        nota="Canonico de hecho: jax-platform backend/uso/cola.py (Task 1). jax "
             "lleva copia verbatim (Task 7, 2026-09-15). ORDEN DE MERGE: la "
             "plataforma primero -- contra un jax-platform sin backend/uso/cola.py "
             "este checker sale con exit 2 (archivo declarado que falta). jax SOLO "
             "deposita (encolar); leer_pendientes/quitar son de la plataforma, que "
             "es la duena de la tabla y la unica con migraciones.",
    ),
    Familia(
        nombre="interruptor",
        canonico=JAX_ROOT / "jax" / "core" / "interruptor.py",
        espejos=(
            # Symlink a jax/core (frente B, 2026-09-16), como cola_uso: comparar
            # hoy es un no-op y se incluye igual.
            ("las_manos", JAX_ROOT / "las_manos" / "interruptor.py"),
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "interruptor.py"),
        ),
        # El que ESCRIBE el freno (la plataforma) y los que lo LEEN (LAS MANOS,
        # Jacobs, el REPL) tienen que hablar del mismo archivo con la misma
        # semantica: solo ENOENT es suelto. Un drift aca no se ve como error: se
        # ve como un boton que dice "detenido" mientras las manos siguen.
        # `InterruptorActivado`, `INTERVALO_DE_SONDEO` y `correr_con_interruptor`
        # quedan afuera: son solo de jax (asyncio del REPL y de Jacobs).
        # La ruta heredada (Task H, 2026-09-17, requisito del controlador
        # principal): la constante, el estado del aviso, `pausa_presente` y
        # `_heredada_activa` tambien son compartidos. Si la plataforma dejara de
        # mirarla, reportaria "suelto" con LAS MANOS frenadas por la ruta vieja.
        compartidos=(
            "VARIABLE_RUTA",
            "InterruptorSinConfigurar",
            "ruta_del_interruptor",
            "RUTA_HEREDADA",
            "_heredada_avisada",
            "pausa_presente",
            "_heredada_activa",
            "interruptor_activo",
            "_sincronizar_directorio",
            "escribir_pausa",
            "borrar_pausa",
        ),
    ),
    Familia(
        nombre="router_keywords",
        canonico=JAX_ROOT / "jax" / "core" / "router.py",
        espejos=(
            # La Mesa web copia las keywords del auto-ruteo (A-22 de la auditoria
            # de sobre-ingenieria, 2026-09-16). NO puede importarlas: jax.core.router
            # arrastra contrato_dispatch -> `from facet_resolver import _db_conn`, que
            # dentro del backend resuelve al facet_resolver de la plataforma, y ~/jax
            # no existe en el runner (verificado por terceros). Hasta hoy la copia
            # podia divergir sin que nada avisara: los 10 sets y el desempate estaban
            # identicos por AST y ningun checker los miraba.
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "api" / "chat.py"),
        ),
        # Los 10 sets, el orden de desempate y el mapa faceta -> (keywords, fuertes).
        # _sin_tildes y el scoring quedan afuera: el de la Mesa tiene logs y
        # docstring propios. Si un dia divergen por diseno, se declara con el marcador.
        compartidos=(
            "KIMI_KW", "KIMI_STRONG",
            "HIPATIA_KW", "HIPATIA_STRONG",
            "JEKYLL_KW", "JEKYLL_STRONG",
            "THOT_KW", "THOT_STRONG",
            "ADA_KW", "ADA_STRONG",
            "_TIEBREAK",
            "_KW_SETS",
        ),
        nota="Copia en jax-platform backend/api/chat.py (nombres alineados 2026-09-16). "
             "ORDEN DE MERGE: la plataforma primero -- contra un jax-platform con los "
             "nombres viejos (_KIMI_KW...) esta familia da 'falta' en los 12 simbolos.",
    ),
    Familia(
        nombre="tope_pipelines",
        canonico=JAX_ROOT / "jacobs" / "policy.py",
        espejos=(
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "ajustes.py"),
        ),
        compartidos=("MAX_PARALLEL_PIPELINES",),
        nota="Frente C (2026-09-16): el ajuste max_pipelines de la plataforma es "
             "una cuota POR TENANT acotada por el candado GLOBAL de Jacobs (cuenta "
             "todos los pending/running). Si este tope cambia en Jacobs y no en la "
             "plataforma, Admin ofrece un valor que Jacobs rechaza con 422. Canonico: "
             "jacobs/policy.py. ORDEN DE MERGE: la plataforma primero (este job "
             "clona jax-platform master).",
    ),
    Familia(
        nombre="config_entorno",
        canonico=JAX_ROOT / "jax" / "core" / "config_entorno.py",
        espejos=(
            # Symlink a jax/core (E-21), como facet_resolver: no-op hoy, a
            # proposito.
            ("las_manos", JAX_ROOT / "las_manos" / "config_entorno.py"),
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "config_entorno.py"),
        ),
        # TODOS los simbolos del archivo: el modulo entero es la regla de que
        # vale como configuracion de servicio. tests/test_config_entorno.py
        # exige que esta tupla cubra el archivo.
        compartidos=(
            "EntornoInvalido",
            "_valor",
            "url_requerida",
            "ruta_absoluta_requerida",
        ),
        nota="Revision final del frente E (2026-09-16): jax-platform valida "
             "JAX_OLLAMA_URL al arrancar con una copia verbatim, no importando "
             "jax (api/chat.py pone en sys.path el checkout de produccion de jax, "
             "que puede ir detras). ORDEN DE MERGE: jax-platform primero -- contra "
             "un jax-platform sin backend/config_entorno.py este checker sale con "
             "exit 2.",
    ),
    Familia(
        nombre="prioridad",
        canonico=JAX_ROOT / "jax" / "ejecutor" / "prioridad.py",
        espejos=(
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "ejecutor" / "prioridad.py"),
        ),
        # TODOS los simbolos del archivo, constantes incluidas: el modulo ES el
        # protocolo de locks entre la Mesa (jax-platform) y el proxy del Ejecutor
        # (jax). Si una copia cambia un nombre de fichero, el paso del sondeo o la
        # forma de abrir el lock, la prioridad se pierde EN SILENCIO: la Mesa no
        # espera nada y el Ejecutor deja de verla. tests/test_ejecutor_prioridad_espejo.py
        # exige que esta tupla cubra el archivo. Limite conocido del comparador: los
        # decoradores no son parte del segmento (@contextmanager); una copia sin el
        # decorador rompe sus propios tests en jax-platform, no pasa callada.
        compartidos=(
            "ESPERA_AGOTADA",
            "_PASO_MESA_S",
            "_PASO_EJECUTOR_S",
            "EsperaAgotada",
            "_abrir",
            "carril_mesa",
            "hay_mesa_esperando",
            "carril_ejecutor",
            "_soltar",
            "_Toma",
            "_esperar_mesa",
            "_intentar_ejecutor",
            "carril_mesa_async",
            "carril_ejecutor_async",
        ),
        nota="SP3 del Ejecutor (2026-09-17): la Mesa toma carril_mesa_async desde "
             "jax-platform backend/ejecutor/prioridad.py, copia verbatim (el unico "
             "ImportFrom distinto es el de Motivo, ver familia `motivo`). ORDEN DE "
             "MERGE: jax-platform primero -- contra un jax-platform sin el archivo "
             "este checker sale con exit 2.",
    ),
    Familia(
        nombre="motivo",
        canonico=JAX_ROOT / "jax" / "ejecutor" / "cita.py",
        espejos=(
            ("jax-platform", JAX_PLATFORM_ROOT / "backend" / "ejecutor" / "motivo.py"),
        ),
        # Solo Motivo: cita.py es el verificador del Ejecutor entero, y la Mesa no
        # necesita nada mas. motivo.py de jax-platform no tiene otro simbolo (lo
        # exige su propio test). @dataclass(frozen=True) no entra al segmento: si
        # la copia pierde el frozen, lo atrapa el test de jax-platform.
        compartidos=("Motivo",),
        nota="SP3 del Ejecutor (2026-09-17): lo importa la copia de prioridad.py "
             "en jax-platform. ORDEN DE MERGE: jax-platform primero.",
    ),
)


def main() -> int:
    faltan_archivos = []
    for familia in FAMILIAS:
        for ruta in (familia.canonico, *(r for _, r in familia.espejos)):
            if not ruta.exists():
                faltan_archivos.append(f"{familia.nombre}: {ruta}")

    if faltan_archivos:
        print("ERROR: falta al menos un archivo declarado:", file=sys.stderr)
        for f in faltan_archivos:
            print(f"  {f}", file=sys.stderr)
        print(
            "Seteá JAX_PLATFORM_REPO_ROOT si el checkout de jax-platform vive "
            "en otra ruta. NO se saltea en silencio: un espejo que no se puede "
            "leer dejaria este checker en verde sin haber comparado nada.",
            file=sys.stderr,
        )
        return 2

    hubo_drift = False
    for familia in FAMILIAS:
        drift, declaradas, faltantes = revisar(familia)
        print(f"[{familia.nombre}]")
        for name in declaradas:
            print(f"  DECLARADA: '{name}' diverge a proposito (marcador '{MARCADOR}')")
        for name in faltantes:
            print(f"  DRIFT: '{name}' falta en una de las copias")
        for name in drift:
            print(f"  DRIFT: '{name}' difiere del canonico SIN declararlo")
        if drift or faltantes:
            hubo_drift = True
        else:
            print(f"  sincronizado ({len(declaradas)} divergencia(s) declarada(s))")
        print()

    if hubo_drift:
        print("DIVERGENCIA SIN DECLARAR -- revisar cual lado tiene el fix real y")
        print("portarlo al otro a mano (no hay symlink cruzado entre repos posible).")
        print("Si la divergencia es DELIBERADA, declarala poniendo")
        print(f"'{MARCADOR}' en el docstring o comentario de ese simbolo, con la razon.")
        return 1

    print("Todas las familias de espejos, sincronizadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
