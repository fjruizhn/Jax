"""Catalogo cerrado de recursos que admiten tope (R-4, decision de Fernando 2026-10-06).

Solo las clases de ACTOS y DINERO llevan tope: monto de dinero, cantidad de
actos externos, frecuencia, duracion y tokens/costo. NUNCA conexiones,
concurrencia, workers, hilos, procesos ni agentes (D-4): no existen en el
catalogo, y fuera del catalogo no hay tope — una lista negra por palabras nunca
cierra (`multiagente.lanzados`, `subprocesos`, `parallel.calls`, `conns.db`...);
un catalogo cerrado si.

Este modulo es una HOJA (solo stdlib): lo consumen el runtime de topes
(`jax/faro/topes.py`) y el schema de reglas (`policy/rule_authority/schema.py`),
y el espejo JSON se mantiene identico por prueba. Fuera del catalogo se rechaza
en schema y en runtime, por igual.
"""
from __future__ import annotations

CATALOGO_TOPES: dict[str, tuple[str, ...]] = {
    "monto_dinero": ("hnl", "usd"),
    "actos_externos": ("mensajes", "correos", "publicaciones", "compras", "pagos"),
    "frecuencia": ("por_hora", "por_dia"),
    "duracion": ("segundos",),
    "tokens_costo": ("tokens", "usd"),
}

CLASES_RECURSO_CON_TOPE: tuple[str, ...] = tuple(sorted(CATALOGO_TOPES))


def es_recurso_de_catalogo(recurso: object) -> bool:
    """True solo si ``recurso`` es ``<clase>.<subid>`` con la clase declarada y el
    subid en el catalogo de ESA clase. Todo lo demas —infraestructura, agentes,
    variantes de idioma, nombres parecidos— queda fuera."""
    if not isinstance(recurso, str):
        return False
    clase, sep, subid = recurso.partition(".")
    if not sep or subid.count("."):
        return False
    return subid in CATALOGO_TOPES.get(clase, ())


def recursos_del_catalogo() -> tuple[str, ...]:
    """La lista plana ``clase.subid`` (para el espejo JSON y las pruebas de
    identidad entre JSON, Python y runtime)."""
    return tuple(sorted(f"{clase}.{subid}" for clase, subids in CATALOGO_TOPES.items()
                        for subid in subids))
