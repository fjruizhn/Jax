"""Modelo, dimensión y columna de embeddings: configuración, no código.

2026-09-12: se migra de `nomic-embed-text` (768) a `bge-m3` (1024). Medido
sobre datos reales de jax_memory: en `facts` recall@1 10/12 contra 5/12; en
`messages`, 11/14 contra 5/14. Hasta hoy el modelo, la dimensión y la columna
estaban escritos en el código (`db.py`: "nomic-embed-text", EMBEDDING_DIM =
768, `embedding` en 11 sentencias): cambiar de modelo era editar código y
desplegar a ciegas, y volver atrás, lo mismo.

Ahora salen de JAX_MEMORY_EMBED_MODEL / _DIM / _COLUMN, con los valores de
hoy por defecto: desplegar el código no cambia nada; el corte es de
configuración, y volver atrás también.

Sin DB ni red: el borde de red (httpx) y el pool se simulan.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import patch

from jax.memory import embedding_config as cfgmod
from jax.memory import db as dbmod


class CargarConfigTest(unittest.TestCase):
    def test_sin_variables_quedan_los_valores_de_hoy(self):
        c = cfgmod.cargar({})
        self.assertEqual((c.model, c.dim, c.column), ("nomic-embed-text", 768, "embedding"))

    def test_lee_modelo_dimension_y_columna_del_entorno(self):
        c = cfgmod.cargar({
            "JAX_MEMORY_EMBED_MODEL": "bge-m3",
            "JAX_MEMORY_EMBED_DIM": "1024",
            "JAX_MEMORY_EMBED_COLUMN": "embedding_bge_m3",
        })
        self.assertEqual((c.model, c.dim, c.column), ("bge-m3", 1024, "embedding_bge_m3"))

    def test_rechaza_una_columna_que_no_es_un_identificador(self):
        # La columna se interpola en SQL: nada que no sea un identificador simple.
        for mala in ("embedding; DROP TABLE messages", "emb-edding", "1embedding", "", "Embedding"):
            with self.subTest(columna=mala), self.assertRaises(ValueError):
                cfgmod.cargar({"JAX_MEMORY_EMBED_COLUMN": mala})

    def test_rechaza_una_dimension_invalida(self):
        for mala in ("0", "-5", "mil", "1024.5"):
            with self.subTest(dim=mala), self.assertRaises(ValueError):
                cfgmod.cargar({"JAX_MEMORY_EMBED_DIM": mala})

    def test_vector_cero_tiene_la_dimension_pedida(self):
        self.assertEqual(len(json.loads(cfgmod.zero_vector_text(1024))), 1024)


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class GetEmbeddingTest(unittest.TestCase):
    def _llamar(self, cfg, respuesta):
        enviado = {}

        async def fake_post(client_self, url, json=None, **kw):
            enviado["url"], enviado["json"] = url, json
            return _Resp(respuesta)

        with patch.object(dbmod, "EMBED", cfg), patch("httpx.AsyncClient.post", fake_post):
            resultado = asyncio.run(dbmod.MemoryDB().get_embedding("hola"))
        return resultado, enviado

    def test_pide_el_modelo_configurado_por_api_embed(self):
        cfg = cfgmod.cargar({"JAX_MEMORY_EMBED_MODEL": "bge-m3", "JAX_MEMORY_EMBED_DIM": "3"})
        resultado, enviado = self._llamar(cfg, {"embeddings": [[0.1, 0.2, 0.3]]})
        self.assertTrue(enviado["url"].endswith("/api/embed"), enviado["url"])
        self.assertEqual(enviado["json"]["model"], "bge-m3")
        self.assertEqual(resultado, [0.1, 0.2, 0.3])

    def test_descarta_un_embedding_de_otra_dimension(self):
        # Un modelo que devuelve 768 con la columna en 1024 no puede escribirse.
        cfg = cfgmod.cargar({"JAX_MEMORY_EMBED_MODEL": "bge-m3", "JAX_MEMORY_EMBED_DIM": "1024"})
        resultado, _ = self._llamar(cfg, {"embeddings": [[0.1] * 768]})
        self.assertIsNone(resultado)


class _Cursor:
    def __init__(self, log, filas):
        self.log, self.filas, self.rowcount = log, filas, 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, sql, args=None):
        self.log.append(sql)

    async def fetchall(self):
        return self.filas


class _Conn:
    def __init__(self, log, filas):
        self.log, self.filas = log, filas

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def cursor(self, *a):
        return _Cursor(self.log, self.filas)


class _Pool:
    def __init__(self, filas):
        self.log, self.filas = [], filas

    def acquire(self):
        return _Conn(self.log, self.filas)


class BackfillUsaLaColumnaConfiguradaTest(unittest.TestCase):
    def test_el_backfill_lee_y_escribe_la_columna_configurada(self):
        cfg = cfgmod.cargar({"JAX_MEMORY_EMBED_MODEL": "bge-m3", "JAX_MEMORY_EMBED_DIM": "3",
                             "JAX_MEMORY_EMBED_COLUMN": "embedding_bge_m3"})
        m = dbmod.MemoryDB()
        m.pool = _Pool([(7, "texto")])

        async def fake_emb(texto):
            return [0.1, 0.2, 0.3]

        with patch.object(dbmod, "EMBED", cfg), patch.object(m, "get_embedding", fake_emb):
            asyncio.run(m.backfill_zero_embeddings("messages", limit=5))

        sql = "\n".join(m.pool.log)
        self.assertIn("embedding_bge_m3", sql)
        self.assertNotRegex(sql, r"\bembedding\b(?!_)", "quedó la columna vieja escrita a mano")
        # El vector cero del guard tiene la dimensión configurada, no 768.
        self.assertIn("[0,0,0]", sql)


if __name__ == "__main__":
    unittest.main(verbosity=2)
