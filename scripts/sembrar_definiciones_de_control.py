#!/usr/bin/env python3
"""Siembra en `jax_evidence.control_definitions` las definiciones EMPAQUETADAS.

POR QUÉ EXISTE (medido en vivo el 2026-09-22, desplegando jax#260 en
hall9000): la tabla `control_definitions` quedaba en CERO filas después de
aplicar la migración, porque en producción nadie la sembraba -- el único
escritor era `MariaDBEvidenceStore.__persist_control_definition`, privado, y
sólo lo llamaba la fixture `_b8_runtime_identity_fixture` de
`tests/policy/test_execution_mariadb_integration.py`. Consecuencia real:
`readonly_status_snapshot` exige que la definición empaquetada tenga una fila
EXACTA en la base (`mariadb_store.py`, `capture()`), no la encuentra, y
levanta `EvidenceArtifactIntegrityError("snapshot definition mismatch")`. El
`except Exception` de `jaxctl/runtime.py::control_status` lo convierte en
`{"classification":"UNAVAILABLE"}` con exit 2 -- o sea, el paso 9 del runbook
`docs/runbooks/implementation-identity.md`, que exige `"verdict":"SUPPORTED"`
como único GO, era INALCANZABLE en un despliegue nuevo, y el mensaje no decía
por qué.

QUÉ SIEMBRA: los controles que enumera `policy/enforcement_evidence/controls/v1.json`
-- el catálogo empaquetado, no una lista escrita a mano acá (una segunda copia
se desincroniza en cuanto alguien agregue un control, y el síntoma sería otra
vez `UNAVAILABLE` sin causa visible).

QUÉ NO PUEDE HACER: inventar una definición. Los bytes salen SIEMPRE de
`load_control_definition` -- la única fuente que `require_trusted_definition`
acepta -- y la escritura entra al contexto `_fixed_composition_write()`, el
mismo guardia que usa el ciclo de vida fijo para la identidad y los
artefactos. El store no expone ningún escritor público: tener una referencia
al store no alcanza para persistir.

SI LA BASE YA TIENE OTRA DEFINICIÓN PARA EL MISMO CONTROL (deriva entre el
código desplegado y la base), el error REAL es de MariaDB, medido el
2026-09-23 contra una base con una fila vieja sembrada a mano:

    pymysql.err.IntegrityError: (1062, "Duplicate entry
    'CTL.B6.GOVERNED_DISPATCH-1' for key 'control_id'")

y NO `EvidenceArtifactIntegrityError("definition collision")`, como decía una
versión anterior de este docstring: el `SELECT ... FOR UPDATE` busca por
`control_definition_hash`, así que una definición distinta tiene otro hash, no
matchea, y el `INSERT` choca contra `UNIQUE(control_id, control_version)`. La
rama de "definition collision" sólo es alcanzable con el MISMO hash y distinto
payload, o sea una colisión de SHA-256. En cualquiera de los dos casos: la
tabla es append-only (`definitions_no_update`/`definitions_no_delete`), esto
NO se arregla pisando la fila -- hay que resolver la deriva.

SALIDA EN VIVO, no al final: cada control se imprime APENAS se siembra. Si el
script muere a mitad (p.ej. el 1062 de arriba en el 5º control), los 4
anteriores ya están commiteados -- cada persist commitea su propia conexión --
y tienen que verse en pantalla. Una versión anterior imprimía recién al
terminar y dejaba la base parcialmente sembrada con CERO salida sobre qué
había quedado adentro.

IDEMPOTENTE: correrlo dos veces deja las mismas filas (el store inserta sólo
si falta, y compara si está). Se corre en CADA despliegue, igual que el
manifiesto -- no "una vez y listo": un control nuevo en el catálogo necesita
su fila antes de que alguien lo consulte.

CÓMO SE CORRE EN PRODUCCIÓN (ver el runbook, paso 5.5): con el intérprete del
venv del servicio y las variables del PROCESO VIVO, nunca sourceando
/etc/jax/.env -- `tests/test_env_se_lee_con_sudo.py` marca esa forma en rojo.

    sudo -u jaxsvc bash -c '
      cd /srv/jax-prod/jax &&
      MAINPID=$(systemctl show -p MainPID --value jax-las-manos) &&
      while IFS= read -r -d "" e; do
        case "$e" in JAX_DB_HOST=*|JAX_DB_PORT=*|JAX_DB_USER=*|JAX_DB_PASSWORD=*) export "$e";; esac
      done < "/proc/$MAINPID/environ" &&
      PYTHONPATH=/srv/jax-prod/jax /srv/jax-prod/jax/las_manos/.venv/bin/python3 \
        scripts/sembrar_definiciones_de_control.py'

Salida: una línea por control (`sembrado` o `ya estaba`) y un resumen. Nunca
imprime credenciales.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: El catálogo empaquetado, fuente única de QUÉ controles existen.
RUTA_CATALOGO = Path(__file__).resolve().parents[1] / "policy/enforcement_evidence/controls/v1.json"


#: La única versión que el catálogo empaquetado describe hoy. `controls/v1.json`
#: lista IDs pelados, sin versión; `ControlDefinition.__post_init__` rechaza
#: cualquier versión != 1, así que asumir 1 acá no inventa nada. El día que
#: exista un v2, el catálogo tiene que decirlo y esta constante deja de valer:
#: `test_el_catalogo_declara_la_version_que_este_script_asume` se pone rojo.
VERSION_DEL_CATALOGO = 1


def controles_empaquetados(ruta_catalogo: Path = RUTA_CATALOGO) -> tuple[tuple[str, int], ...]:
    """Los pares (control_id, control_version) del catálogo, en orden de archivo."""
    catalogo = json.loads(ruta_catalogo.read_text(encoding="utf-8"))
    controles = catalogo.get("controls")
    if not isinstance(controles, list) or not controles:
        raise RuntimeError(f"catálogo de controles vacío o ilegible: {ruta_catalogo}")
    if catalogo.get("schema_version") != "1.0":
        raise RuntimeError(
            f"catálogo con schema_version inesperada ({catalogo.get('schema_version')!r}): "
            "revisar si trae versiones de control distintas de "
            f"{VERSION_DEL_CATALOGO} antes de sembrar")
    return tuple((control_id, VERSION_DEL_CATALOGO) for control_id in controles)


def _conexion():
    """Las credenciales salen del entorno, nunca de un argumento ni de un archivo."""
    import pymysql
    faltantes = [name for name in ("JAX_DB_HOST", "JAX_DB_PORT", "JAX_DB_USER", "JAX_DB_PASSWORD")
                 if not os.environ.get(name)]
    if faltantes:
        raise RuntimeError("faltan variables de entorno de MariaDB: " + ", ".join(faltantes))
    return pymysql.connect(
        host=os.environ["JAX_DB_HOST"], port=int(os.environ["JAX_DB_PORT"]),
        user=os.environ["JAX_DB_USER"], password=os.environ["JAX_DB_PASSWORD"],
        database=os.environ.get("JAX_DB_NAME", "jax_memory"),
        charset="utf8mb4", autocommit=False, connect_timeout=5)


def sembrar(store, controles, *, emitir=print) -> list[tuple[str, int, bool]]:
    """Siembra cada control y devuelve (control_id, control_version, ya_estaba).

    `controles` son pares (control_id, control_version): la versión viaja
    EXPLÍCITA hasta el store. El catálogo de hoy es todo v1, pero pasar sólo
    el ID dejaría que un v2 futuro se ignorara en silencio -- y el síntoma
    sería otra vez `UNAVAILABLE` sin causa visible, que es justo el defecto
    que este script corrige.

    `ya_estaba` se mide ANTES de escribir con `load_control_definition` del
    store (que levanta `EvidenceBlobMissingError` si no hay fila): sin eso, un
    despliegue que no sembró nada se leería igual que uno que sembró todo.
    NO es una garantía transaccional -- entre la lectura y la escritura no hay
    lock -- así que es una SEÑAL para el operador, no un contrato.

    La escritura entra a `_fixed_composition_write()`: el store no tiene
    escritores públicos y `__persist_control_definition` exige ese contexto.
    """
    from policy.enforcement_evidence.errors import EvidenceBlobMissingError
    from policy.enforcement_evidence.control_registry import load_control_definition
    from policy.enforcement_evidence.evidence_store import _fixed_composition_write

    resultado = []
    for control_id, control_version in controles:
        try:
            store.load_control_definition(control_id, control_version)
            ya_estaba = True
        except EvidenceBlobMissingError:
            ya_estaba = False
        with _fixed_composition_write():
            store._MariaDBEvidenceStore__persist_control_definition(
                load_control_definition(control_id, control_version))
        # En vivo, no al final: un fallo a mitad deja las anteriores
        # commiteadas y el operador tiene que saber cuáles.
        emitir(f"  {'ya estaba' if ya_estaba else 'sembrado '}  {control_id} v{control_version}")
        resultado.append((control_id, control_version, ya_estaba))
    return resultado


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalogo", type=Path, default=RUTA_CATALOGO,
                        help="catálogo de controles empaquetado (por defecto, el del repo)")
    args = parser.parse_args(argv)

    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore

    controles = controles_empaquetados(args.catalogo)
    store = MariaDBEvidenceStore(_conexion)
    filas = sembrar(store, controles)
    nuevos = sum(1 for *_resto, ya_estaba in filas if not ya_estaba)
    print(f"{len(filas)} definiciones empaquetadas presentes ({nuevos} sembradas ahora).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- salida legible, no traceback crudo
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
