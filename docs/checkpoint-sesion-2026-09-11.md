# Checkpoint de sesión — 2026-09-11

Brief para retomar JAX en otra máquina sin perder contexto. Complementa
`CONTEXT.md` (memoria viva) y `DEUDA.md` (lista canónica de deuda, **la fuente de
verdad del estado**). Si este documento y `DEUDA.md` difieren, gana `DEUDA.md`, y
si `DEUDA.md` difiere del servidor, gana el servidor (§7 de `CONTEXT.md`).

---

## 0. Cómo retomar (3 pasos)

1. **Restaurar la memoria de Claude Code** en la máquina nueva:
   `~/biblioteca-memoria-20260911-053224.tar.gz` (en hall9000, 69 archivos,
   chmod 600) → extraer en `~/.claude/projects/<proyecto>/` (queda
   `memory/MEMORY.md`). `<proyecto>` depende del directorio desde donde se
   abre la sesión (en hall9000 era `-home-fruiz`).
2. **Leer, en este orden:** este checkpoint → `DEUDA.md` (secciones "Bloquea
   trabajo" y "Anotado, no bloquea", más "Cerrado — kimi y la memoria vector
   cero") → `CONTEXT.md` §7 (método).
3. **Medir antes de trabajar** cualquier ítem (§7 de `CONTEXT.md`): el estado de
   abajo es de 2026-09-11 ~05:30 y envejece.

---

## 1. Estado verificado al cierre (2026-09-11)

| Cosa | Valor |
|---|---|
| `jax` master | `76dda0d` (+ este checkpoint) — árbol limpio salvo `.claude/settings.json` (cambio local ajeno a esta sesión, no commiteado) |
| `jax-platform` master | `0784d5f` — árbol limpio |
| PRs abiertos | 0 en los dos repos |
| Servicios | `jax-platform` reiniciado 02:56 con el código de hoy; `jax-las-manos` sin reiniciar (no cambió nada suyo); worker de memoria por timer, toma el código nuevo en cada pasada |
| Facetas | las 6 sondeables en `ok` tras el reinicio, **kimi incluida** (alerta recuperada sola 02:41:34) |
| Memoria vectorial | 0 filas en vector cero en `messages` y `facts` |
| Backups | corrida real 05:16 con el script nuevo: `✓ Prune diferido en r2`, `restic check` sin errores |

---

## 2. Lo que se hizo hoy

| PR | Qué |
|---|---|
| jax-platform#49 | **kimi en el chat de la Mesa web.** `facet.transport='motor_registry'` era una etiqueta sin lector; alineada con ada (`http_openai_compat` + gate `authorize-facet`). Test de clase: todo facet sembrado es despachable. `CANARY_SWEEP_TIMEOUT_SECONDS` 900→1080 |
| jax#116 + jax-platform#50 | **500 del chat en todo el scope individual.** Embeddings en vector cero → `VEC_DISTANCE_COSINE` da NaN (None o 0.0 según el camino; `IS NULL` no lo ve). Exclusión en SQL + filtro de finitas + consumidor fail-soft |
| jax#118 | **El worker de memoria reintenta embeddings en ceros** en cada pasada (antes del `return` temprano) |
| jax#120 | **18 de 26 archivos de `tests/` sin CI** → jobs `tests-puros` (62) y `jacobs-gobernanza-db` (14). `test_jacobs_timeout_by_capability` estaba roto desde el 09-01. **`jax_memory_schema.sql` regenerado** desde producción + `scripts/check_memory_schema_drift.py` |
| jax#117, #119, #121 | Documentación en `DEUDA.md` de todo lo anterior, con evidencia en vivo |

**Fuera de git (hecho en hall9000, registrado en `DEUDA.md`):**

- `/opt/backup-scripts/backup-hall9000.sh`: prune diferido en R2
  (`--max-unused unlimited --max-repack-size 0`, `timeout 900`) y comentario del
  lock corregido (son 7 días, no 10; sin prune restic nunca borra). Respaldos:
  `.bak-20260911-comentario-lock`, `.bak-20260911-prune-r2`. Ese directorio no
  está en git.
- Prune real en R2: 52,5 MiB, restauración verificada por md5.
- **Backfill** de las 24 filas en vector cero del 2026-06-09 (backup por id en
  `~/backups/`, restauración probada).
- **Rotación de la contraseña de `user_id=2`** (04:50:10). La contraseña ya fue
  entregada por Fernando y el archivo que la contenía se borró con `shred`.

---

## 3. Espera GO de Fernando

**Vaciada el 2026-09-11 por la tarde: Fernando dio GO y el ítem se cerró.**
Ver el cierre en `DEUDA.md` (jax#123 + jax-platform#51, desplegado 18:39 CST) —
y en particular **el límite de su evidencia**: el control con el código viejo
no reprodujo hoy el 1/20.000, así que el mecanismo del flake de CI queda como
hipótesis plausible, no re-confirmada. Texto original del pedido de GO:

- **Sello de `facet_resolver` puede perder una invalidación** (`DEUDA.md`,
  control 2026-09-18). `os.utime(p, None)` deja un `mtime` detrás de
  `time.time()`: 1/20.000 en el filesystem real del sello, 0/20.000 con
  `os.utime(p, (t, t))`. Arreglo de una línea, pero espejado en los dos repos
  y desplegarlo reinicia `jax-platform` y `jax-las-manos`. Causó un rojo
  intermitente de `facet-resolver-seal` en CI.

## 4. Pendiente de Fernando (acción suya, no técnica)

- **Borrar el respaldo del hash anterior de `user_id=2`**
  (`~/security-audit-2026-09/user_id-2-hash-previo-20260911-045010.txt`) una
  vez que la persona confirme que entra. Es lo único que permite revertir.
- Ticket a GitHub Support (texto listo, sin enviar).
- Modo SSL de Cloudflare de los dos dominios del origen interno (decisión).
- Sacar `jax_users` del dump nocturno (ronda propia).

## 5. Abiertos en `DEUDA.md` con fecha

| Ítem | Control |
|---|---|
| ~~Sello de `facet_resolver`~~ **CERRADO 2026-09-11** (jax#123 + jax-platform#51) | — |
| 3 tests de LAS MANOS en proceso sin CI: `config.toml` fija el audit log en una ruta absoluta de hall9000 — **correrlos a mano escribe en el audit log real** | 2026-09-25 |
| Validador de gobernanza lee un catálogo que el Bloque 3 vació (Bloquea trabajo) | — |
| Merge sin revisión en `master` (decidido: no hasta que haya incidente) | — |
| Pendientes de SP4 (brazo negativo, `GROUNDING_UNAVAILABLE`) | al retomar SP4 |

---

## 6. Lecciones de método de esta sesión

1. **Verificar con un turno real, no solo con la sonda.** kimi se habría
   declarado cerrada con el chat caído para todo el scope individual: el 500
   apareció recién con el "hola" de Fernando.
2. **El NaN engaña a toda medición ingenua.** Filas en vector cero: medir con
   `VEC_DISTANCE_EUCLIDEAN(embedding, <cero>) = 0`, nunca con `IS NULL` ni con la
   distancia coseno. Una medición así afirmó "cero vectores cero" con 24
   existiendo.
3. **Reproducir el runner, no el local.** Un venv limpio con el `pip install`
   exacto del job encontró dos veces una dependencia faltante (`cryptography`,
   el `PYTHONPATH` de `las_manos/`) que en hall9000 no se ve.
4. **Medir en el mismo filesystem que el objeto real.** La primera medición del
   sello se hizo en `/tmp` (tmpfs), dio 0 y se leyó como "hipótesis refutada";
   en xfs, donde vive el sello, el defecto aparece.
5. **Una fecha derivada de una sola regla es una suposición.** La "purga de R2 del
   09-08" salió de sumar el lock de 7 días sin mirar la retención (6 meses) ni el
   prune (inexistente): nunca iba a ocurrir.

En memoria de Jairo Urbina.
