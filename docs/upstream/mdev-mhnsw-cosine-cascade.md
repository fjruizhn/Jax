# Reporte a MariaDB upstream — MHNSW + cosine + ON DELETE CASCADE

**Estado: PREPARADO, NO ENVIADO.** Hace falta una cuenta de
[jira.mariadb.org](https://jira.mariadb.org) — no se crea ni se usa una a nombre
de Fernando sin que él lo decida.

**Cómo enviarlo:** entrar a jira.mariadb.org → *Create* → Project **MDEV** →
Issue Type **Bug** → pegar el título y el cuerpo de abajo → adjuntar
`mdev-mhnsw-cosine-cascade.sql`. Componente: *Vector search*. Al enviarlo,
anotar acá el número de MDEV.

**Búsqueda previa (2026-09-20):** no aparece ningún reporte que combine índice
vectorial MHNSW con borrado en cascada. Lo más cercano es **MDEV-36758**
(LeakSanitizer en `MHNSW_Share::alloc_node`), que es otra cosa.

---

## Título

    MHNSW: rows become unreachable through a cosine vector index after ON DELETE CASCADE

## Cuerpo (en inglés, listo para pegar)

---

After a `FOREIGN KEY ... ON DELETE CASCADE` deletion, a MHNSW vector index with
`DISTANCE='cosine'` starts returning **fewer rows than actually exist**, and does
not recover. The rows are present and readable — only the indexed path misses
them. No error is raised.

**Version:** `12.3.3-MariaDB-ubu2404` (mariadb.org binary distribution), InnoDB.
`@@mhnsw_default_m = 6`, `@@mhnsw_ef_search = 20`, `@@mhnsw_max_cache_size = 16777216`.

### Reproduction

Attached: `mdev-mhnsw-cosine-cascade.sql`. Self-contained — `VECTOR(4)` and the
built-in `seq_1_to_N` engine, no client code.

The shape: 300 rows under parent 1 → `DELETE FROM parent WHERE id = 1` (cascade)
→ insert 25 fresh rows under parent 2 → ask for the 10 nearest.

```
live_rows            25
through the index     7
IGNORE INDEX         10
```

### What narrows it down

Same data, same query, only one variable changed each time:

| Index definition | Deletion | index / scan |
|---|---|---|
| `DISTANCE='cosine' M='16'` | **CASCADE** | **7 of 10** |
| `DISTANCE='cosine' M='16'` | direct `DELETE FROM child` | 10 of 10 |
| `DISTANCE='cosine'` (default M) | **CASCADE** | **0 of 10** |
| `DISTANCE='cosine'` (default M) | direct | 10 of 10 |
| `M='16'` (default distance) | CASCADE | 10 of 10 |
| `DISTANCE='euclidean' M='16'` | CASCADE | 10 of 10 |
| default index | CASCADE | 10 of 10 |

So it takes **both** `DISTANCE='cosine'` **and** a cascade deletion. `M` only
changes how bad it is: with the default `M` the index returned **zero** rows.

**One deleted parent row is enough** — this is not about bulk deletion.

### Reliability

Intermittent but frequent, and the asymmetry is stable:

- `cosine`, `M='16'`, CASCADE: **8 of 8 runs reproduced** (7/10 five times, 0/10 three times).
- `cosine`, default `M`, CASCADE: **7 of 8 runs reproduced**; one run returned 10/10.
- Control, direct `DELETE` instead of cascade: **8 of 8 runs clean** (10/10).

### Workaround, and what does not work

Rebuilding the index restores correct results, with no data loss:

```sql
ALTER TABLE child DROP INDEX idx_v;
ALTER TABLE child ADD VECTOR INDEX idx_v (v) `DISTANCE`='cosine' `M`='16';
```

`OPTIMIZE TABLE` does **not** fix it. Inserting more rows does not fix it either
(measured: 0 of 61 still missing after adding 60 rows).

> When rebuilding, the original index options must be repeated. A bare
> `ADD VECTOR INDEX idx_v (v)` silently recreates it with the default distance
> function and `M`, which changes query results without any error.

### Why this matters beyond search quality

In our use, this index backs a nearest-neighbour lookup used for
**deduplication**. A dedup that returns nothing reads it as "this is new", so a
silently degraded index does not cause missing results — it causes **duplicate
writes**. The failure is silent, and the first remedy anyone reaches for
(`OPTIMIZE TABLE`) reports success without fixing anything.

---

## Nuestro lado

Detectado el 2026-09-20 persiguiendo dos tests inestables. Producción **no está
afectada**: no existe ningún camino que borre conversaciones. El detector, la
reparación y el runbook están en **jax#239**
(`scripts/revisar_indice_vectorial.py`,
`docs/runbooks/indice-vectorial-envenenado.md`).
