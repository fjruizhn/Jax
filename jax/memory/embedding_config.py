"""Modelo, dimensión y columna de los embeddings de memoria: configuración.

POR QUE EXISTE (2026-09-12). Se migra de `nomic-embed-text` (768 dims) a
`bge-m3` (1024). Medido sobre jax_memory real: en `facts` recall@1 10/12
contra 5/12; en `messages`, 11/14 contra 5/14. Hasta hoy los tres valores
estaban escritos en `db.py` (el modelo en get_embedding, EMBEDDING_DIM = 768 y
la columna `embedding` en once sentencias): cambiar de modelo era editar código
y desplegar, y volver atrás, lo mismo.

Los valores por defecto son los de HOY: desplegar este código no cambia nada.
El corte es de configuración (/etc/jax/.env, que leen los cuatro servicios) y
volver atrás también. Ver scripts/migrar_embeddings.py.

La columna se interpola en SQL: se valida como identificador simple al leerla.
Una configuración inválida falla al importar -- ruidoso a propósito: escribir
vectores de 768 en una columna de 1024 (o al revés) no tiene arreglo silencioso.

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_DIM = 768
DEFAULT_COLUMN = "embedding"

_IDENTIFICADOR = re.compile(r"^[a-z_][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str
    dim: int
    column: str


def cargar(env: Mapping[str, str]) -> EmbeddingConfig:
    model = (env.get("JAX_MEMORY_EMBED_MODEL") or DEFAULT_MODEL).strip()
    dim_txt = (env.get("JAX_MEMORY_EMBED_DIM") or str(DEFAULT_DIM)).strip()
    column = env.get("JAX_MEMORY_EMBED_COLUMN", DEFAULT_COLUMN)
    if not dim_txt.isdigit() or int(dim_txt) <= 0:
        raise ValueError(f"JAX_MEMORY_EMBED_DIM={dim_txt!r}: tiene que ser un entero positivo")
    if not _IDENTIFICADOR.match(column or ""):
        raise ValueError(f"JAX_MEMORY_EMBED_COLUMN={column!r}: tiene que ser un identificador "
                         f"simple (minúsculas, dígitos, _), se interpola en SQL")
    if not model:
        raise ValueError("JAX_MEMORY_EMBED_MODEL vacío")
    return EmbeddingConfig(model=model, dim=int(dim_txt), column=column)


def zero_vector_text(dim: int) -> str:
    """Literal del vector cero de `dim` dimensiones (default de las columnas y
    guard del backfill)."""
    return "[" + ",".join(["0"] * dim) + "]"


CONFIG = cargar(os.environ)
