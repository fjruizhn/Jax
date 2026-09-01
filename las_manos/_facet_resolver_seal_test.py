"""
Sello de invalidacion cross-proceso del `_cache` de facet_resolver (Q3).

POR QUE EXISTE ESTE ARCHIVO. `facet_resolver._cache` vive en TRES procesos
--jax-platform (Mesa web), las_manos/Jacobs y el REPL-- y hasta 2026-09-01
solo uno se invalidaba al rebindear. Los otros dos despachaban contra el
binding VIEJO hasta FACET_CACHE_TTL_SECONDS (30 s), y la sonda por rebinding
no lo podia ver: sondea justamente el unico camino invalidado. El sello lo
cierra --el escritor toca un archivo, cada resolve_facet compara su mtime
contra el instante en que cacheo-- pero solo si la comparacion se hace con el
reloj correcto.

EL MODO DE FALLA SILENCIOSO QUE ESTOS TESTS EXISTEN PARA ATRAPAR:
`_CacheEntry.fetched_at` es `time.monotonic()`, que NO es comparable con un
`st_mtime` (reloj de pared, epoch). Comparar el mtime contra `fetched_at` no
da un error: da un veredicto CONSTANTE --"siempre invalidar" si el origen de
monotonic es chico (en Linux, el boot), "nunca invalidar" si fuera mayor que
epoch-- y en los dos casos el sello no invalida nada util. El bug del cache
queda IGUAL, pero con codigo nuevo que aparenta resolverlo, y ningun test que
solo mire "rebindeo y veo el modelo nuevo" lo detecta: ese caso pasa por
casualidad en una de las dos direcciones.

Por eso la comparacion queda clavada desde LOS DOS LADOS, con el origen de
`time.monotonic()` forzado a valores incompatibles con epoch en cada
direccion:
  - test_sello_viejo_no_invalida_con_monotonic_chico     (atrapa "siempre invalidar")
  - test_sello_nuevo_invalida_con_monotonic_de_origen_futuro (atrapa "nunca invalidar")
Ninguno de los dos pasa si alguien compara `st_mtime` contra `fetched_at`.

Sin DB a proposito: se parchea `_query_facet`, que es el unico punto que
toca MariaDB. Lo que se prueba es la politica de cache, no el I/O.

POR QUE VIVE EN las_manos/ Y NO EN jax/core/: `las_manos/facet_resolver.py` es
un SYMLINK a `jax/core/facet_resolver.py` -- un solo archivo, dos rutas de
import, tres procesos. Importandolo con PYTHONPATH=las_manos este archivo
ejercita exactamente el camino de Jacobs, que es uno de los dos procesos que
hasta hoy no se enteraban de un rebind. El REPL importa el MISMO archivo por
`jax.core.facet_resolver`; el gemelo de estos tests para el tercer proceso vive
en jax-platform (backend/tests/test_facet_resolver_seal.py), y el job
`facet-resolver-sync` valida que los dos espejos no diverjan.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

import facet_resolver as fr


# ---------------------------------------------------------------------------
# Arnes
# ---------------------------------------------------------------------------

@pytest.fixture
def sello(tmp_path, monkeypatch):
    """Aisla el sello en tmp y deja el `_cache` del modulo limpio.

    `_cache` es estado de modulo compartido por todo el proceso de pytest:
    sin este limpiado, el orden de los tests cambiaria el resultado.
    """
    ruta = tmp_path / "facet-cache-seal"
    monkeypatch.setattr(fr, "FACET_SEAL_PATH", str(ruta))
    fr._cache.clear()
    yield ruta
    fr._cache.clear()


def _stub_query(monkeypatch, valores):
    """Reemplaza `_query_facet` por una secuencia de valores conocidos.

    Devuelve la lista de llamadas: su LARGO es la afirmacion real de estos
    tests --cuantas veces se fue a la DB-- porque es lo unico que distingue
    "sirvio del cache" de "re-consulto".
    """
    llamadas = []

    async def _fake(facet_key):
        llamadas.append(facet_key)
        return valores[min(len(llamadas) - 1, len(valores) - 1)]

    monkeypatch.setattr(fr, "_query_facet", _fake)
    return llamadas


def _resolver(facet_key="primary"):
    return asyncio.run(fr.resolve_facet(facet_key))


def _sellar_en(ruta, cuando: float) -> None:
    """Escribe el sello con un mtime EXACTO, en vez de 'ahora'.

    Explicito y no `touch`: estos tests dependen de la relacion de orden
    entre el mtime y el instante cacheado, y un 'ahora' con la granularidad
    del filesystem la volveria una carrera.
    """
    ruta.write_bytes(b"")
    os.utime(ruta, (cuando, cuando))


# ---------------------------------------------------------------------------
# 1. El mecanismo
# ---------------------------------------------------------------------------

def test_sello_nuevo_invalida_la_entrada_cacheada(sello, monkeypatch):
    """Lo que Jacobs y el REPL no podian hacer hasta hoy."""
    llamadas = _stub_query(monkeypatch, ["modelo-viejo", "modelo-nuevo"])

    assert _resolver() == "modelo-viejo"
    assert _resolver() == "modelo-viejo", "dentro del TTL debe servir del cache"
    assert len(llamadas) == 1

    _sellar_en(sello, time.time() + 5)

    assert _resolver() == "modelo-nuevo", (
        "con el sello mas nuevo que la entrada, el cache debe descartarse"
    )
    assert len(llamadas) == 2


def test_entrada_sellada_se_descarta_no_se_sirve_como_stale(sello, monkeypatch):
    """El sello DESCARTA la entrada; no la deja disponible para el camino de
    `serving_stale`. Servir un valor que un escritor declaro superado seria
    peor que declarar estado degradado --es exactamente el binding viejo que
    esta ronda vino a matar."""
    _stub_query(monkeypatch, ["modelo-viejo"])
    assert _resolver() == "modelo-viejo"

    _sellar_en(sello, time.time() + 5)

    async def _revienta(facet_key):
        raise RuntimeError("DB caida")

    monkeypatch.setattr(fr, "_query_facet", _revienta)

    with pytest.raises(fr.FacetUnavailableError):
        _resolver()


# ---------------------------------------------------------------------------
# 2. El modo de falla silencioso: monotonic vs mtime, clavado de los dos lados
# ---------------------------------------------------------------------------

def test_sello_viejo_no_invalida_con_monotonic_chico(sello, monkeypatch):
    """ATRAPA la comparacion `st_mtime > fetched_at` (monotonic).

    Con `monotonic` en un valor de uptime realista (1234.5) y un sello mas
    VIEJO que la entrada, el codigo correcto sirve del cache. El codigo que
    compara contra monotonic ve `1.7e9 > 1234.5` --True-- e invalida SIEMPRE,
    con lo cual el cache deja de existir: una query a la DB por request en
    los tres procesos, regresion de rendimiento silenciosa.
    """
    monkeypatch.setattr(time, "monotonic", lambda: 1234.5)
    llamadas = _stub_query(monkeypatch, ["modelo-viejo", "modelo-nuevo"])

    _sellar_en(sello, time.time() - 3600)  # sello anterior a la entrada

    assert _resolver() == "modelo-viejo"
    assert _resolver() == "modelo-viejo"
    assert len(llamadas) == 1, (
        "un sello VIEJO no invalida nada -- si invalido, la comparacion se "
        "esta haciendo contra time.monotonic() y no contra el reloj de pared"
    )


def test_sello_nuevo_invalida_con_monotonic_de_origen_futuro(sello, monkeypatch):
    """ATRAPA la misma comparacion en la direccion contraria.

    Con el origen de `monotonic` por ENCIMA de epoch, el codigo que compara
    contra monotonic ve `mtime > fetched_at` como False para siempre y NO
    invalida nunca: el sello se vuelve decorativo y el bug de los tres
    procesos queda intacto, con codigo nuevo encima.
    """
    falso_monotonic = time.time() + 10 ** 6
    monkeypatch.setattr(time, "monotonic", lambda: falso_monotonic)
    llamadas = _stub_query(monkeypatch, ["modelo-viejo", "modelo-nuevo"])

    assert _resolver() == "modelo-viejo"

    _sellar_en(sello, time.time() + 5)

    assert _resolver() == "modelo-nuevo", (
        "el sello es mas nuevo en RELOJ DE PARED -- si no invalido, la "
        "comparacion se esta haciendo contra time.monotonic()"
    )
    assert len(llamadas) == 2


def test_cache_entry_guarda_los_dos_relojes(sello, monkeypatch):
    """Los dos relojes se guardan y NO son intercambiables: `fetched_at` mide
    el TTL (monotonic, inmune a saltos de NTP), `fetched_at_wall` se compara
    con el mtime (epoch). Este test fija que existen los dos y que cada uno
    lleva lo suyo, para que nadie 'simplifique' dejando uno solo."""
    _stub_query(monkeypatch, ["modelo-viejo"])
    _resolver()

    entrada = fr._cache["primary"]
    assert "fetched_at" in fr._CacheEntry.__slots__
    assert "fetched_at_wall" in fr._CacheEntry.__slots__
    assert abs(entrada.fetched_at - time.monotonic()) < 5, "fetched_at no es monotonic"
    assert abs(entrada.fetched_at_wall - time.time()) < 5, "fetched_at_wall no es epoch"
    assert entrada.fetched_at_wall > 1_000_000_000, (
        "fetched_at_wall tiene que ser epoch para poder compararse con un mtime"
    )


# ---------------------------------------------------------------------------
# 3. Modo de falla declarado: sin sello / sello ilegible -> TTL, no "sin cache"
# ---------------------------------------------------------------------------

def test_sin_sello_degrada_al_ttl_no_a_sin_cache(sello, monkeypatch):
    """El sello NO existe (caso del dia 1, y de cualquier proceso recien
    arrancado antes del primer rebind). Comportamiento exigido: el de hoy
    --cache con TTL-- y no "invalidar siempre por las dudas", que convertiria
    un archivo faltante en una query por request."""
    assert not sello.exists()
    llamadas = _stub_query(monkeypatch, ["modelo-viejo"])

    _resolver()
    _resolver()
    _resolver()

    assert len(llamadas) == 1, "sin sello, el cache tiene que seguir cacheando"


def test_sello_ilegible_degrada_al_ttl_no_a_sin_cache(sello, monkeypatch, tmp_path):
    """El sello existe pero `os.stat` falla (permisos). Mismo techo: TTL."""
    if os.geteuid() == 0:
        pytest.skip("como root ningun permiso bloquea el stat; el caso no es reproducible")

    jaula = tmp_path / "jaula"
    jaula.mkdir()
    oculto = jaula / "facet-cache-seal"
    _sellar_en(oculto, time.time() + 5)  # mas nuevo: invalidaria SI se pudiera leer
    monkeypatch.setattr(fr, "FACET_SEAL_PATH", str(oculto))
    os.chmod(jaula, 0o000)
    try:
        llamadas = _stub_query(monkeypatch, ["modelo-viejo", "modelo-nuevo"])
        assert _resolver() == "modelo-viejo"
        assert _resolver() == "modelo-viejo"
        assert len(llamadas) == 1, (
            "un sello ilegible es 'sin senal de invalidacion', no 'invalidar'"
        )
    finally:
        os.chmod(jaula, 0o700)


def test_sin_sello_el_ttl_sigue_gobernando(sello, monkeypatch):
    """El sello ADELANTA una invalidacion que el TTL iba a hacer igual; no lo
    reemplaza. Pasado el TTL, se re-consulta aunque no haya sello."""
    reloj = {"t": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: reloj["t"])
    llamadas = _stub_query(monkeypatch, ["modelo-viejo", "modelo-nuevo"])

    assert _resolver() == "modelo-viejo"
    reloj["t"] = fr.FACET_CACHE_TTL_SECONDS + 1
    assert _resolver() == "modelo-nuevo"
    assert len(llamadas) == 2


# ---------------------------------------------------------------------------
# 4. El escritor
# ---------------------------------------------------------------------------

def test_invalidate_facet_cache_toca_el_sello(sello, monkeypatch):
    """Sin esto el sello nunca cambia y el mecanismo entero es decorativo."""
    _stub_query(monkeypatch, ["modelo-viejo"])
    _resolver()
    assert not sello.exists()

    assert fr.invalidate_facet_cache("primary") is True, "habia entrada local"

    assert sello.exists(), "el escritor tiene que crear el sello si no existe"
    assert abs(os.stat(sello).st_mtime - time.time()) < 60


def test_invalidate_facet_cache_lo_ven_los_otros_procesos(sello, monkeypatch):
    """La prueba de que el sello cruza procesos, simulada dentro de uno: una
    entrada cacheada ANTES del invalidate --como la que tienen Jacobs y el
    REPL, que no comparten este `_cache`-- queda invalidada por el sello."""
    _stub_query(monkeypatch, ["modelo-viejo"])
    _resolver()
    entrada_de_otro_proceso = fr._cache["primary"]

    fr.invalidate_facet_cache("primary")

    assert fr._entrada_sellada(entrada_de_otro_proceso), (
        "una entrada cacheada antes del invalidate tiene que quedar sellada"
    )


def test_invalidate_no_revienta_si_el_sello_no_se_puede_escribir(sello, monkeypatch, tmp_path):
    """Fail-soft explicito del ESCRITOR: no se aborta la aprobacion de un
    binding porque no se pudo escribir un archivo de cache. Los otros
    procesos se quedan con el TTL, que es el techo de hoy."""
    if os.geteuid() == 0:
        pytest.skip("como root ningun permiso bloquea la escritura; el caso no es reproducible")

    jaula = tmp_path / "solo-lectura"
    jaula.mkdir()
    os.chmod(jaula, 0o500)
    monkeypatch.setattr(fr, "FACET_SEAL_PATH", str(jaula / "facet-cache-seal"))
    try:
        assert fr.invalidate_facet_cache("inexistente") is False
        assert fr._tocar_sello() is False, "tiene que reportar que no escribio"
    finally:
        os.chmod(jaula, 0o700)
