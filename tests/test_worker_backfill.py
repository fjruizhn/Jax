"""El worker de memoria recalcula los embeddings en ceros en CADA corrida.

Por que en cada corrida y no solo cuando hay conversaciones: `run_once`
volvia temprano con "No hay conversaciones pendientes" -- que es el caso
normal de casi todas las corridas (medido 2026-09-11 04:31). Un paso puesto
despues de ese `return` no correria casi nunca, y las filas en ceros (ver
jax/memory/db.py::backfill_zero_embeddings) seguirian perdidas.

Y por que un fallo del recalculo no frena la extraccion: son dos trabajos
independientes. Ollama caido -- la causa tipica de las filas en ceros -- no
tiene por que impedir destilar facts con el extractor, que es otro proveedor.

Tests puros: MemoryDB se reemplaza por un doble, no hay DB ni red.
"""
import asyncio

from jax.memory import worker


class _DBFalsa:
    def __init__(self, backfill_falla: bool = False):
        self.backfill_falla = backfill_falla
        self.pool = object()
        self.tablas: list[str] = []
        self.pidio_conversaciones = False
        self.cerrada = False

    async def connect(self, **_kwargs):
        return True

    async def backfill_zero_embeddings(self, tabla, limit):
        self.tablas.append(tabla)
        if self.backfill_falla:
            raise RuntimeError("Ollama no responde")
        return {"pendientes": 0, "reparadas": 0, "fallidas": 0}

    async def get_unprocessed_conversations(self, limit):
        self.pidio_conversaciones = True
        return []

    async def close(self):
        self.cerrada = True


def _correr(monkeypatch, db: _DBFalsa):
    monkeypatch.setattr(worker, "MemoryDB", lambda: db)
    class Jobs:
        def __init__(self,*args,**kwargs): pass
        async def pending(self,limit):
            return await db.get_unprocessed_conversations(limit)
    monkeypatch.setattr(worker,"ExtractionJobs",Jobs)
    monkeypatch.setenv("JAX_DB_HOST", "db-de-test")
    asyncio.run(worker.run_once())


def test_extractor_no_duplica_el_trabajo_de_embeddings(monkeypatch):
    db = _DBFalsa()
    _correr(monkeypatch, db)

    assert db.tablas == []
    assert db.pidio_conversaciones
    assert db.cerrada


def test_extractor_sigue_sin_lanzar_embeddings_si_existe_un_fallo_pendiente(monkeypatch):
    db = _DBFalsa(backfill_falla=True)
    _correr(monkeypatch, db)

    # La unidad de embeddings es el único dueño de esa clase de trabajo.
    assert db.tablas == []
    assert db.pidio_conversaciones, "el fallo del recalculo freno la extraccion"
    assert db.cerrada
