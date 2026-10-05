# Deployment and rollback boundaries
## Purpose
Deploy or recover verified builds.
## Scope
Service deployment and implementation identity.
## Preconditions
Verified build/manifest and approved change.
## Authority impact
Deployment integrity; no policy amendment.
## Safe procedure
Deploy through supported platform procedure and preserve identity bindings.
## Verification
Check startup, identity verification, and diagnostics.
## Fail-closed condition
Manifest drift or unavailable trusted source: stop.
## Recovery / escalation
Redeploy verified prior build or escalate.
## Prohibited actions
Do not reuse stale identity for changed tree.

---

# Marcha atrás concreta de `/srv/jax-prod/jax`

Este procedimiento no concede permiso de producción: lo ejecuta una persona con el GO de Fernando o con su ventana abierta, y antes de cada paso que no se deshace solo se confirma que sigue vigente.

Estado verificado el 2026-10-05 (cámbialo por lo que mida `systemctl` el día que lo uses; es una VERDAD OPERACIONAL que caduca):

- `/srv/jax-prod/jax` es propiedad de `jaxsvc:jaxsvc`, está en `master` y limpio (HEAD `5f4f5e0`).
- Corren desde ese checkout `jax-ejecutor-proxy`, `jax-las-manos` (WorkingDirectory `.../las_manos`) y las cinco unidades `jax-memory-*`. `jax-platform` corre desde **otro** checkout (`/srv/jax-prod/jax-platform`) pero lee código de este, y su arranque aplica la migración 005 desde él (ver «Esquema»).
- Cada unidad con `ExecStartPre=/usr/local/sbin/jax-checkout-de-produccion-sano.sh /srv/jax-prod/jax` se **niega a arrancar** si el checkout no está en la rama `master` o tiene cualquier cambio sin comitear (`git status --porcelain` no vacío, también archivos sin seguimiento). Es el freno que impide desplegar código sin revisar por un reinicio automático (2026-09-20). **Toda la marcha atrás tiene que dejar el checkout en `master` y limpio.**

## 0. Antes de tocar nada

1. Fija el SHA destino (`<sha>`: el que se anotó en el paso 1 de `docs/runbooks/memoria-b9-reactivacion.md`; si no se anotó, no se improvisa: se escala).
2. Registra el estado actual en un archivo fechado fuera de `/` y de `~` (por ejemplo bajo `/srv/almacen`):
   - `sudo git -c safe.directory=/srv/jax-prod/jax -C /srv/jax-prod/jax rev-parse HEAD` y `... status --porcelain`;
   - estado de **todos** los timers: `systemctl list-unit-files 'jax-*.timer'` y `systemctl list-timers 'jax-*' --all`;
   - copia de las unidades y drop-ins instalados: `sudo cp -a` de `/etc/systemd/system/jax-*.service`, `jax-*.timer` y `jax-*.service.d/` a una carpeta con fecha. Esta copia es la fuente del paso 3.

## 1. Detener las unidades

Orden: timers primero (para que nada nuevo arranque), después se **espera** a las unidades `oneshot` de memoria y, solo entonces, se paran las tres de larga vida.

```bash
sudo systemctl stop jax-memory-worker.timer jax-memory-embedding.timer jax-memory-synthesis.timer \
  jax-memory-lifecycle.timer jax-memory-vector-health.timer jax-revisar-indice-vectorial.timer
# esperar a que ninguna corrida esté en curso (TimeoutStartSec=15min como tope):
systemctl is-active jax-memory-worker.service jax-memory-embedding.service jax-memory-synthesis.service \
  jax-memory-lifecycle.service jax-memory-vector-health.service   # todas `inactive` o `failed`, nunca `activating`
sudo systemctl stop jax-platform.service jax-las-manos.service jax-ejecutor-proxy.service
systemctl is-active jax-platform.service jax-las-manos.service jax-ejecutor-proxy.service   # `inactive`
```

`systemctl stop` sobre una unidad `oneshot` **no espera** a que termine su corrida: la termina con SIGTERM. Un worker a mitad de un commit queda `UNKNOWN` (lo resuelve el siguiente reclamo leyendo los marcadores de commit; ver `docs/runbooks/memoria-cola-atascada.md`), así que solo se corta una corrida que pasó de `TimeoutStartSec`, y se anota. Parar los timers antes evita que arranque una corrida nueva sobre el árbol que se va a cambiar.

## 2. Volver el código: `reset --hard` en `master`

```bash
sudo -u jaxsvc git -C /srv/jax-prod/jax reset --hard <sha>
```

**Nunca `git checkout <sha>`**: deja HEAD suelto (`rev-parse --abbrev-ref HEAD` da `HEAD`, no `master`) y el freno `jax-checkout-de-produccion-sano.sh` impide arrancar todas las unidades de arriba. `reset --hard` mueve la rama `master` local y la deja limpia. Comprueba, como `jaxsvc`:

```bash
sudo -u jaxsvc git -C /srv/jax-prod/jax rev-parse --abbrev-ref HEAD   # master
sudo -u jaxsvc git -C /srv/jax-prod/jax rev-parse HEAD                # <sha> completo
sudo -u jaxsvc git -C /srv/jax-prod/jax status --porcelain            # vacío
sudo -u jaxsvc /usr/local/sbin/jax-checkout-de-produccion-sano.sh /srv/jax-prod/jax   # rc 0
```

Si `status --porcelain` no sale vacío, se **mira** qué es (archivos sin seguimiento que dejó el despliegue, `__pycache__` sin ignorar) antes de borrarlo; no se corre `git clean -fdx` a ciegas. La rama local queda por detrás de `origin/master`: nadie hace `git pull` aquí sin querer volver a desplegar lo retirado; el siguiente despliegue es otra decisión con su propio GO.

## 3. Restaurar las unidades

Las unidades y drop-ins que el despliegue cambió vuelven a lo que había: se copian de la carpeta fechada del paso 0 (`sudo install -m 644 ...`), no del repo, porque el repo en el SHA anterior puede no tener los drop-ins de producción (se versionaron después). Lo que el despliegue **agregó** y antes no existía (un drop-in nuevo, una unidad nueva) se retira: compara `diff -r` entre la carpeta fechada y `/etc/systemd/system/jax-*`. Cada diferencia se explica antes de tocarla.

## 4. Recargar systemd

```bash
sudo systemctl daemon-reload
systemctl show jax-ejecutor-proxy jax-las-manos jax-memory-worker -p FragmentPath -p DropInPaths --no-pager
```

Los `DropInPaths` tienen que coincidir con los de la copia del paso 0.

## 5. Reiniciar, en este orden

```bash
sudo systemctl restart jax-ejecutor-proxy.service
sudo systemctl restart jax-las-manos.service
sudo systemctl restart jax-platform.service
```

El orden sale de las unidades: `jax-platform` declara `Wants=`/`After=jax-las-manos.service`. Después de cada `restart`, `systemctl is-active` y `journalctl -u <unidad> -n 30 --no-pager`: la línea `jax-checkout-de-produccion-sano: ... limpio y en master (<sha corto>)` confirma el freno y el SHA. Si una unidad no arranca, se **para** y se lee el journal; no se «arregla» el freno.

`jax-platform` ejecuta al arrancar `run_migrations()`, que a su vez aplica 003+005 si el código que ahora lee de `/srv/jax-prod/jax` las contiene. Con un SHA que todavía no las tiene, no pasa nada; con uno que sí, las reaplica (idempotentes).

## 6. Devolver los timers a su estado

Solo los que estaban habilitados en el registro del paso 0, y solo los de esa lista: `sudo systemctl enable --now <timer>` para los que estaban `enabled`; los `disabled` (en 2026-10-05, los de worker y synthesis) se quedan como están. Verifica con `systemctl list-timers 'jax-*' --all` que los próximos disparos coinciden con los del registro.

## Esquema: cuándo se baja y cuándo no

**La marcha atrás de código no baja el esquema por sí sola**, y casi nunca debe. Las migraciones 001-013 son aditivas (tablas, columnas, índices) y se aplican aparte del despliegue y antes del código que las usa.

**No hace falta bajar el esquema** cuando se vuelve a un código que ignora lo nuevo: las tablas `memory_extraction_job_events` (007), `embedding_generation_attempts` (009), los índices de 010-013 y las columnas añadidas no estorban a un código anterior que no las nombra. Verificado en `jax/`, `jacobs/` y `las_manos/` (2026-10-05): ningún `FORCE INDEX`/`USE INDEX` nombra `idx_conversation` ni `idx_embedding_generation_revision`. Dejar el esquema donde está es lo reversible: bajarlo, no.

**Sí hay que mirarlo (y posiblemente bajarlo)** en estos casos, siempre con respaldo restaurable antes (Principio VI):

1. **La migración misma es el defecto.** 011, 012 y 013 bloquean escrituras 1-3 s (`ALGORITHM=COPY`); si eso es lo que causó el incidente, la bajada es la «Marcha atrás de 007–013» de `jax/memory/b9_migrations/README.md`, en orden inverso (013, 012, 011, 010, 009, 008, 007). Bajar 007 borra la auditoría del re-encolado y bajar 009 los contadores de intentos: exportarlos antes si importan.
2. **El código anterior no soporta lo nuevo.** La UNIQUE de 008 `(revision_id, embedding_space_id)` rechaza duplicados: si el código al que se vuelve puede insertar dos generaciones para el mismo par, fallará con `IntegrityError`. **No verificado** para cada SHA: se lee el `embedding_worker` de ese SHA antes de decidir. Para bajar 008 primero se restaura el índice de 010 (la FK lo necesita).
3. **Migración 005 (proyectos).** La reaplica `jax-platform` en **cada arranque** si el código que lee contiene `apply_project_authority_migration`. Bajarla (`scripts/b9_revertir_005.py`, dry-run por defecto, `--aplicar` no es transaccional) solo sirve **después** de volver a un código que ya no la llama; hacerlo antes y reiniciar `jax-platform` la vuelve a crear. Falla cerrado si quedan proyectos ARCHIVED/HIDDEN/DISABLED o `project_documents` con filas.
4. **001-004 y 006** (esquema base, autoridad de proyecto, `tenant_id` en bindings y revisiones con su trigger y FK, tablas de jobs) **no tienen bajada escrita**. No se inventa una: se restaura del respaldo con restauración probada. El código del worker actual usa 007 y 009 en cada corrida; el anterior no las conoce y las ignora.

Regla corta: **código atrás, esquema donde está**, salvo que la migración sea la causa o el código anterior choque con ella. Ante la duda se escala a Fernando; no se baja a ojo.

## Verificación final

`systemctl is-active` de las tres unidades, el journal con la línea del freno y el SHA correcto, `systemctl list-timers` contra el registro, y `memory_extraction_jobs`: `python -m jax.memory.worker --atascados` no debe mostrar nada nuevo atribuible a la marcha atrás.
