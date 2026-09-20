# El índice vectorial miente — detectar y reparar

**Estado:** VERDAD OPERACIONAL, medida el 2026-09-20 en hall9000 contra
MariaDB 12.3.3 (`mariadb-12-3-jax`, 127.0.0.1:3308).
**Producción al escribir esto: SANA.** `messages` (1.638 filas) y `facts` (116)
devuelven 10/10 por el índice y por scan completo.

---

## 1 · Qué pasa

Un índice HNSW puede quedar devolviendo **menos filas de las que hay**, sin
error y para siempre. Medido:

```
300 filas → borrar → insertar 25 → pedir las 10 más cercanas
  por el índice ............ 1 de 10
  con IGNORE INDEX ......... 10 de 10
```

Insertar más filas **no** lo arregla. `OPTIMIZE TABLE` **tampoco**.

### El disparador es la CASCADA, no el volumen

Bisecado con las mismas filas y la misma consulta:

| Borrado | Resultado |
|---|---|
| `DELETE FROM conversations` (cascada a `messages`) | índice 1 / scan 10 — **ROTO** |
| `DELETE FROM messages` (directo) | índice 10 / scan 10 — sano |

Y basta **una sola** conversación. No hace falta vaciar nada.

### La ceguera es PARCIAL

Devolvió **1 de 10**, no 0. Por eso **un detector que pregunte «¿devolvió
vacío?» da verde con el índice roto**. Hay que comparar el conteo por el índice
contra el mismo conteo con `IGNORE INDEX`.

---

## 2 · Por qué importa más de lo que parece

El que se queda ciego no es sólo la búsqueda: es **`_find_nearest_fact`, el
dedup de la memoria**. Para un dedup, «no encontré nada parecido» significa
«esto es nuevo».

Así que un índice envenenado no hace que la memoria encuentre menos: hace que
**DUPLIQUE**, en silencio y para siempre. Es corrupción, no degradación.

---

## 3 · Nos afecta hoy?

**No.** Verificado el 2026-09-20: **no existe ningún camino que borre
conversaciones** en `jax` ni en `jax-platform` — ni SQL directo, ni función, ni
endpoint. Por eso producción está sana.

> ⚠️ **La trampa está en el futuro.** El día que se agregue un «borrar chat»
> —una función de lo más normal— el índice de `messages` empieza a mentir al
> primer uso, y nadie se entera. Quien toque eso, lee este runbook primero.

---

## 4 · Cómo se revisa

```bash
set -a; . <(sudo -n cat /etc/jax/.env); set +a
cd /srv/jax-prod/jax
PYTHONPATH=.:las_manos .venv/bin/python scripts/revisar_indice_vectorial.py
```

Salida sana:

```
jax_memory:
  facts.idx_embedding_bge_m3: sano (indice 10 / scan 10 de una muestra de 10, 116 filas)
  messages.idx_embedding_bge_m3: sano (indice 10 / scan 10 de una muestra de 10, 1638 filas)
```

Códigos: `0` todo sano · `1` alguno miente · `2` no se pudo revisar.

Los índices se descubren por `INDEX_TYPE='VECTOR'`, no por el nombre de la
columna: el día que la columna cambie —ya pasó al migrar a bge-m3— esto los
sigue encontrando sin que nadie lo actualice.

---

## 5 · Cómo se repara

**Se repara sin perder datos.** Medido: de 1 resultado a 10 de 10, con las
filas intactas. No hay que exportar nada ni recalcular embeddings.

```bash
PYTHONPATH=.:las_manos .venv/bin/python scripts/revisar_indice_vectorial.py --reparar-de-verdad
```

Por omisión **no repara**: mira e informa. Reparar es DDL y toma un lock sobre
la tabla, así que se escribe entero y a propósito.

### Antes de reparar en producción

1. **Parar el servicio que escribe** (`jax-las-manos`, `jax-platform`) o esperar
   una ventana sin tráfico. Es DDL: bloquea la tabla como cualquier migración.
2. **Respaldo** del volcado de `jax_memory`, como cualquier cambio de esquema.
3. Reparar, y **volver a revisar** — el propio script lo hace y avisa si sigue roto.

### Lo que NO hay que hacer

- **`OPTIMIZE TABLE` no sirve.** Medido: no repara. Es el remedio que cualquiera
  prueba primero, y falla en silencio.
- **No reconstruir el índice a mano con `ADD VECTOR INDEX (col)` pelado.** El
  índice de producción es
  ``VECTOR KEY `idx_embedding_bge_m3` (`embedding_bge_m3`) `DISTANCE`='cosine' `M`='16'``.
  Recrearlo sin esas opciones lo deja con **otra función de distancia y otro M**:
  la búsqueda «funciona» y devuelve resultados distintos, sin un solo error.
  El script copia la definición exacta del `SHOW CREATE TABLE` antes de borrarla,
  y hay un test que lo defiende (`test_la_reparacion_no_degrada_la_definicion_del_indice`,
  visto en rojo contra la versión ingenua).

---

## 6 · Qué NO se hizo, y por qué

- **No se le puso un centinela automático al `/health` ni un timer.** Con cero
  caminos de borrado, un vigilante diario vigilaría algo que no puede pasar
  todavía. Cuando exista el primer borrado de conversación, se enciende — y ahí
  el lugar es junto a `/health` o a las migraciones, con el último resultado
  **cacheado**, nunca una consulta vectorial por petición.
- **No se cambió el `ON DELETE CASCADE`.** El esquema está bien; el defecto es
  del motor.
- **El reporte a MariaDB upstream está PREPARADO, no enviado** — hace falta una
  cuenta de jira.mariadb.org. El texto y una reproducción en **SQL puro**
  (`VECTOR(4)`, sin cliente) están en `docs/upstream/`. Ahí también está lo que
  acotó la causa: hacen falta **`DISTANCE='cosine'` Y la cascada**; con el `M`
  por omisión la ceguera es **total** (0 de 10), no parcial.

---

## 7 · Cómo se reproduce (por si hay que volver a medirlo)

`tests/test_indice_vectorial.py` lo reproduce entero, en tablas propias —
nunca sobre `messages`, porque envenenar la base de test dejaría mintiendo a
todo lo que corra después. Lo que importa de esas tablas y no es decorado:

* la FK con `ON DELETE CASCADE` — es el disparador;
* el índice con `DISTANCE='cosine'` y `M='16'`, igual que producción. **Con los
  valores por omisión el defecto no se reproduce**, y el test daría verde sobre
  un índice que no es el nuestro.

```bash
set -a; . <(sudo -n cat /etc/jax/.env); set +a
JAX_DB_NAME=jax_memory_test PYTHONPATH=.:las_manos \
  .venv/bin/python -m pytest tests/test_indice_vectorial.py -v
```

---

*En memoria de Jairo Urbina.*
