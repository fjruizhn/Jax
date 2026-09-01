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
            # Tres archivos reales otra vez -- ninguno es symlink, medido el
            # 2026-09-01. Misma forma que credential_resolver: dos copias
            # dentro de jax y una en jax-platform.
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
        nota="TRES archivos reales. jax-platform tiene ademas encrypt_secret y "
             "decrypt_db_secret, excluidos por diseno (es el lado que cifra).",
    ),
    Familia(
        nombre="credential_resolver",
        canonico=JAX_ROOT / "jax" / "core" / "credential_resolver.py",
        espejos=(
            # OJO -- aca la forma NO es la de facet_resolver, y esto se midio
            # antes de escribirlo (2026-09-01): `las_manos/credential_resolver.py`
            # NO es un symlink, es un TERCER ARCHIVO REAL. O sea que esta
            # familia puede driftear DENTRO del propio repo jax, sin cruzar
            # repos -- una copia mas suelta que la de facet_resolver, y hasta
            # hoy nadie la comparaba con nada.
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
        nota="TRES archivos reales, no dos: las_manos/ tiene copia propia, no "
             "symlink. Verificado 2026-09-01.",
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
