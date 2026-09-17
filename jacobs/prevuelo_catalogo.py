"""Jacobs — lectura del catálogo para el pre-vuelo (spec 2026-09-17 §4.2-§4.5).

Una conexión por pre-vuelo, no por paso ni por lector: jacobs/prevuelo.py toma
UNA del pool de lectura (store.conexion_de_lectura, Task 15b) y la pasa a
`resolver_motores` y a `leer_catalogo`, que no abren ni cierran conexiones.
Se lee: facetas HTTP con su binding, filas de `model` (contrato y precio),
capability.min_output_tokens, credenciales ACTIVAS y el último evento de
salud de nivel proveedor. Los motores se resuelven con MotorCatalog +
MotorPolicy (la misma regla que despacha).

Credencial (§4.3): se lee la TABLA, no el resolver. resolve_credential_
instrumented cae a .env aunque la credencial esté revocada y resolve_facet
puede servir una copia de hasta 300 s; el pre-vuelo no hereda ninguno de los
dos (su retiro es del frente E / B1.4).

Sin caché, a propósito: son consultas por PK o índice sobre tablas de decenas
de filas, y el pre-vuelo existe para ver la verdad de AHORA. Medido en la Task
15b (task-15b-report.md): lo que costaba por pedido era abrir dos conexiones,
no las consultas; el catálogo de motores en memoria ahorraría ~0,3 ms más de
CPU pero exigiría invalidación entre procesos, y no se hace.

En honor al Prof. Raúl Jacobs.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from jacobs.facet_health import ultimo_evento_de_proveedor
from jacobs.models import Step


@dataclass(frozen=True)
class FilaFaceta:
    key: str
    transport: str
    persona: str | None
    provider_id: str
    base_url: str | None
    model_id: str


@dataclass(frozen=True)
class FilaModelo:
    max_tokens_param: str | None
    max_output_tokens: int | None
    precio_in: Decimal | None
    precio_out: Decimal | None


@dataclass(frozen=True)
class MotorResuelto:
    clave: str
    transporte: str
    provider_id: str
    base_url: str | None
    modelo: str
    max_tokens: int
    tiene_herramientas: bool
    output_schema: str


@dataclass(frozen=True)
class Catalogo:
    facetas: dict[str, FilaFaceta]
    modelos: dict[tuple[str, str], FilaModelo]
    min_output_tokens: dict[str, int]
    proveedores_con_credencial: frozenset[str]
    salud: dict[str, tuple[float, str]]


def _ph(n: int) -> str:
    return ",".join(["%s"] * n)


def sql_facetas(n: int) -> str:
    return (
        "SELECT f.`key`, f.transport, f.persona, b.provider_id, p.base_url, m.model_id "
        "FROM facet f "
        "JOIN facet_binding b ON b.facet_key = f.`key` AND b.role = 'primary' "
        "JOIN provider p ON p.id = b.provider_id "
        "JOIN model m ON m.id = b.model_ref "
        f"WHERE f.`key` IN ({_ph(n)}) AND f.status = 'active'"
    )


def sql_modelos(n: int) -> str:
    return (
        "SELECT provider_id, model_id, max_tokens_param, max_output_tokens, "
        "price_input_per_1m_usd, price_output_per_1m_usd FROM model WHERE "
        + " OR ".join(["(provider_id = %s AND model_id = %s)"] * n)
    )


def sql_min_output_tokens(n: int) -> str:
    return f"SELECT `key`, min_output_tokens FROM capability WHERE `key` IN ({_ph(n)})"


def sql_credenciales(n: int) -> str:
    return (
        "SELECT DISTINCT provider_id FROM credential "
        f"WHERE provider_id IN ({_ph(n)}) AND state = 'active'"
    )


def _decimal(valor) -> Decimal | None:
    return None if valor is None else Decimal(valor)


async def resolver_motores(pasos: list[Step], *, conexion) -> dict[int, MotorResuelto | None]:
    """Por step_index, el motor que el Motor Registry usaría (None si ninguno).
    Lee el catálogo de motores por `conexion` (la del pre-vuelo) y no la cierra."""
    if not pasos:
        return {}
    from motor_registry.catalog import MotorCatalog
    from motor_registry.policy import MotorPolicy

    catalogo = await MotorCatalog.from_db(conexion=conexion)
    politica = MotorPolicy(catalogo)
    salida: dict[int, MotorResuelto | None] = {}
    for paso in pasos:
        clave = politica.motor_que_despacharia(paso.motor, paso.capability)
        entrada = catalogo.get_motor(clave) if clave else None
        cap = catalogo.get_capability(paso.capability)
        salida[paso.step_index] = None if entrada is None else MotorResuelto(
            clave=clave,
            transporte=entrada.transport,
            provider_id=entrada.provider_id,
            base_url=entrada.api_url or None,
            modelo=entrada.model,
            max_tokens=entrada.max_tokens,
            tiene_herramientas=entrada.has_tool_access,
            output_schema=cap.output_schema if cap else "",
        )
    return salida


async def leer_catalogo(
    *,
    conexion,
    facetas: set[str],
    motores: list[MotorResuelto],
    capabilities: set[str],
    ahora: float,
) -> Catalogo:
    """Todo por `conexion` (la del pre-vuelo); no la abre ni la cierra."""
    async with conexion.cursor() as cur:
        filas_facetas: dict[str, FilaFaceta] = {}
        if facetas:
            claves = sorted(facetas)
            await cur.execute(sql_facetas(len(claves)), claves)
            for key, transport, persona, provider_id, base_url, model_id in await cur.fetchall():
                filas_facetas[key] = FilaFaceta(key, transport, persona, provider_id, base_url, model_id)

        pares = sorted(
            {(f.provider_id, f.model_id) for f in filas_facetas.values()}
            | {(m.provider_id, m.modelo) for m in motores}
        )
        modelos: dict[tuple[str, str], FilaModelo] = {}
        if pares:
            await cur.execute(sql_modelos(len(pares)), [x for par in pares for x in par])
            for provider_id, model_id, param, max_out, p_in, p_out in await cur.fetchall():
                modelos[(provider_id, model_id)] = FilaModelo(
                    param, None if max_out is None else int(max_out), _decimal(p_in), _decimal(p_out))

        minimos: dict[str, int] = {}
        if capabilities:
            caps = sorted(capabilities)
            await cur.execute(sql_min_output_tokens(len(caps)), caps)
            minimos = {key: int(valor) for key, valor in await cur.fetchall()}

        proveedores = sorted({par[0] for par in pares})
        con_credencial: frozenset[str] = frozenset()
        if proveedores:
            await cur.execute(sql_credenciales(len(proveedores)), proveedores)
            con_credencial = frozenset(r[0] for r in await cur.fetchall())

        salud = await ultimo_evento_de_proveedor(
            cur, set(filas_facetas) | {m.clave for m in motores}, ahora,
        )
    return Catalogo(filas_facetas, modelos, minimos, con_credencial, salud)
