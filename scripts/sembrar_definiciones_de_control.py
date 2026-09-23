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

QUÉ NO PUEDE HACER: inventar una definición. Sólo pasa IDs a
`persist_packaged_control_definition`, que carga los bytes de
`load_control_definition` -- la única fuente que `require_trusted_definition`
acepta. Si la fila ya existe con OTROS bytes, el store levanta
`EvidenceArtifactIntegrityError("definition collision")` y este script sale
!= 0: una colisión es deriva entre el código desplegado y la base, no algo
que se resuelva pisando la fila.

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


def controles_empaquetados(ruta_catalogo: Path = RUTA_CATALOGO) -> tuple[str, ...]:
    """Los IDs del catálogo empaquetado, en su orden de archivo."""
    catalogo = json.loads(ruta_catalogo.read_text(encoding="utf-8"))
    controles = catalogo.get("controls")
    if not isinstance(controles, list) or not controles:
        raise RuntimeError(f"catálogo de controles vacío o ilegible: {ruta_catalogo}")
    return tuple(controles)


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


def sembrar(store, control_ids) -> list[tuple[str, bool]]:
    """Siembra cada control y devuelve (control_id, ya_estaba) por control.

    `ya_estaba` se mide ANTES de escribir, con `load_control_definition` del
    store (que falla con `EvidenceBlobMissingError` si no hay fila): sin eso,
    el resumen no podría distinguir "lo sembré yo" de "ya estaba", y un
    despliegue que no sembró nada se leería igual que uno que sembró todo.
    """
    from policy.enforcement_evidence.errors import EvidenceBlobMissingError
    resultado = []
    for control_id in control_ids:
        try:
            store.load_control_definition(control_id)
            ya_estaba = True
        except EvidenceBlobMissingError:
            ya_estaba = False
        store.persist_packaged_control_definition(control_id)
        resultado.append((control_id, ya_estaba))
    return resultado


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalogo", type=Path, default=RUTA_CATALOGO,
                        help="catálogo de controles empaquetado (por defecto, el del repo)")
    args = parser.parse_args(argv)

    from policy.enforcement_evidence.mariadb_store import MariaDBEvidenceStore

    control_ids = controles_empaquetados(args.catalogo)
    store = MariaDBEvidenceStore(_conexion)
    filas = sembrar(store, control_ids)
    for control_id, ya_estaba in filas:
        print(f"  {'ya estaba' if ya_estaba else 'sembrado '}  {control_id}")
    nuevos = sum(1 for _cid, ya_estaba in filas if not ya_estaba)
    print(f"{len(filas)} definiciones empaquetadas presentes ({nuevos} sembradas ahora).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- salida legible, no traceback crudo
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
