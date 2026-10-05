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
- Corren desde ese checkout `jax-ejecutor-proxy`, `jax-las-manos` (WorkingDirectory `.../las_manos`) y las cinco unidades `jax-memory-*`. `jax-platform` corre desde **otro** checkout (`/srv/jax-prod/jax-platform`) pero lee código de este, y su arranque aplica B9 001-002 y la autoridad de proyecto (003 y 005) desde él, y no arranca si faltan (ver el paso 5 y «Esquema»).
- Cada unidad con `ExecStartPre=/usr/local/sbin/jax-checkout-de-produccion-sano.sh /srv/jax-prod/jax` se **niega a arrancar** si el checkout no está en la rama `master` o tiene cualquier cambio sin comitear (`git status --porcelain` no vacío, también archivos sin seguimiento). Es el freno que impide desplegar código sin revisar por un reinicio automático (2026-09-20). **Toda la marcha atrás tiene que dejar el checkout en `master` y limpio.**

## 0. Antes de tocar nada

1. Fija el SHA destino (`<sha>`: el que se anotó en el paso 1 de `docs/runbooks/memoria-b9-reactivacion.md`; si no se anotó, no se improvisa: se escala).
   **Pre-chequeo del destino (obligatorio, antes de parar nada):** `jax-platform` lee de este checkout las migraciones B9 y la de autoridad de proyecto en cada arranque (ver el paso 5). El SHA destino tiene que contener las cuatro cosas (001, 002, 003 y 005):

   ```bash
   sudo git -c safe.directory=/srv/jax-prod/jax -C /srv/jax-prod/jax ls-tree --name-only <sha> \
     jax/memory/b9_migrations/001_b9_shared_memory.sql jax/memory/b9_migrations/002_b9_hardening.sql \
     jax/memory/project_authority_migrations.py     # tienen que salir las tres rutas (001, 002 y el archivo de 003+005)
   # La 005 NO tiene archivo propio (vive solo en Python, dentro del archivo anterior): se comprueba su contenido
   # y que el destino descienda del commit que la agregó.
   sudo git -c safe.directory=/srv/jax-prod/jax -C /srv/jax-prod/jax show <sha>:jax/memory/project_authority_migrations.py \
     | grep -c _apply_project_lifecycle_migration       # >= 1
   sudo git -c safe.directory=/srv/jax-prod/jax -C /srv/jax-prod/jax merge-base --is-ancestor caa49501242ecff50d74e8da69f7fc993914b334 <sha> && echo ok
   ```

   Si falta alguna, **no es un destino válido por sí solo**: volver ahí con `jax-platform` actual la deja sin arrancar. Opciones: elegir un SHA posterior que las tenga, o volver también `jax-platform` a una versión sin ese llamador (mismo procedimiento sobre `/srv/jax-prod/jax-platform`, con su propio freno y su propio SHA anotado, y con las dos unidades paradas durante los dos `reset --hard`). El orden: los dos checkouts quedan en su SHA con `jax-platform` parada, y recién después arrancan (paso 5). Esa segunda marcha atrás es otra decisión con su propio GO.
2. Registra el estado actual en un archivo fechado fuera de `/` y de `~` (por ejemplo bajo `/srv/almacen`):
   - `sudo git -c safe.directory=/srv/jax-prod/jax -C /srv/jax-prod/jax rev-parse HEAD` y `... status --porcelain`;
   - estado de **todos** los timers: `systemctl list-unit-files 'jax-*.timer'` y `systemctl list-timers 'jax-*' --all`;
   - copia de las unidades y drop-ins instalados: `sudo cp -a` de `/etc/systemd/system/jax-*.service`, `jax-*.timer` y `jax-*.service.d/` a una carpeta con fecha. Esta copia es la fuente del paso 3.

## 1. Detener las unidades

Orden: timers primero (para que nada nuevo arranque), después se **espera** a las unidades `oneshot` de memoria y, solo entonces, se paran las tres de larga vida.

```bash
# `disable --now` (no `stop`): un reinicio del servidor durante la ventana no rearma los habilitados.
# Solo los que el registro del paso 0 marque `enabled`; el paso 6 los vuelve a habilitar.
sudo systemctl disable --now jax-memory-worker.timer jax-memory-embedding.timer jax-memory-synthesis.timer \
  jax-memory-lifecycle.timer jax-memory-vector-health.timer jax-revisar-indice-vectorial.timer
# esperar a que ninguna corrida esté en curso (TimeoutStartSec=15min como tope):
systemctl is-active jax-memory-worker.service jax-memory-embedding.service jax-memory-synthesis.service \
  jax-memory-lifecycle.service jax-memory-vector-health.service jax-revisar-indice-vectorial.service   # todas `inactive` o `failed`, nunca `activating`
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

**El `.venv` no vuelve con `reset --hard`**: `/srv/jax-prod/jax/.venv` no está versionado y conserva las dependencias del despliegue que se retira. Antes de reiniciar se compara con lo que pide el SHA destino: `sudo -u jaxsvc /srv/jax-prod/jax/.venv/bin/python -m pip install --dry-run -r /srv/jax-prod/jax/requirements.txt` (ruta absoluta: con el `reset --hard` ya hecho es el `requirements.txt` del SHA destino; no instala nada; muestra lo que cambiaría) y `... -m pip check`. Si hay diferencias, resincronizar con `pip install -r /srv/jax-prod/jax/requirements.txt` es un cambio en producción con su propio GO; si el destino corre con las versiones actuales, se anota que se comprobó. Un paquete nuevo que el SHA retirado importaba y el destino no, no estorba; uno que el destino necesita y el `.venv` no tiene (o tiene en otra versión) hace fallar el arranque.

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

**El arranque de `jax-platform` depende del checkout de jax** (leído de `backend/db/migrations.py` de `/srv/jax-prod/jax-platform`, solo lectura, 2026-10-05; `JAX_REPO_PATH=/srv/jax-prod/jax` en `/etc/jax/.env`). `run_migrations()` llama, en cada arranque, a:

- `_apply_jax_b9_core_migrations`: ejecuta B9 **001 y 002** leyendo `jax/memory/b9_migrations/001_b9_shared_memory.sql` y `002_b9_hardening.sql` del checkout de jax. Si el directorio o alguno de los dos archivos no existe, levanta `RuntimeError("JAX-owned B9 migrations are missing")` y **`jax-platform` no arranca**.
- `_apply_jax_project_authority_migration`: carga `jax/memory/project_authority_migrations.py` del mismo checkout y llama a `apply_project_authority_migration` (003 y 005). Si falta el archivo o la función, también `RuntimeError` y no arranca.

O sea: con un SHA de jax **anterior a los commits que agregaron cada migración**, cada una con su origen exacto (hallado con `git log -S` sobre el historial completo del checkout de producción, 2026-10-05): **001 desde 291fb440; 002 desde 8bbdd2c5; 003 desde 97e2abe2; 005 desde caa49501** (`caa49501242ecff50d74e8da69f7fc993914b334`, 2026-09-25, «autoridad de proyecto sobre B9 — D1–D5, §3-bis y migración 005»). 97e2abe2 trae la 003 y `apply_project_authority_migration`, pero **no** la 005. Qué pasa según dónde caiga el destino: sin 001/002 (antes de 291fb440 o de 8bbdd2c5) o sin el archivo de 003 (antes de 97e2abe2), `jax-platform` **no arranca**; con 97e2abe2 pero antes de caa49501 arranca, pero la 005 **no se aplica** (el hook solo trae la 003), y si el esquema ya la tiene queda un esquema más nuevo que el código — el efecto sobre `jax-platform` no está verificado, y por eso el pre-chequeo exige las cuatro, `jax-platform` **falla al arrancar**, y no hay forma de «ignorarlo». Con un SHA que sí los contiene, 001, 002, 003 y 005 se reaplican en cada arranque (idempotentes).

El freno de `jax-platform` vigila `/srv/jax-prod/jax-platform`, **no** el checkout de jax: que la unidad arranque no confirma que el SHA de jax sea el esperado. Eso se confirma con el freno de las otras unidades (la línea del journal de `jax-ejecutor-proxy` o `jax-las-manos`) y con `rev-parse HEAD` del paso 2.

## 6. Devolver los timers a su estado

Solo los que estaban habilitados en el registro del paso 0, y solo los de esa lista: `sudo systemctl enable --now <timer>` para los que estaban `enabled`; los `disabled` (en 2026-10-05, los de worker y synthesis) se quedan como están. Verifica con `systemctl list-timers 'jax-*' --all` que los próximos disparos coinciden con los del registro.

## Esquema: cuándo se baja y cuándo no

**La marcha atrás de código no baja el esquema por sí sola**, y casi nunca debe. Las migraciones 001-013 son aditivas (tablas, columnas, índices) y se aplican aparte del despliegue y antes del código que las usa.

**No hace falta bajar el esquema** cuando se vuelve a un código que ignora lo nuevo: las tablas `memory_extraction_job_events` (007), `embedding_generation_attempts` (009), los índices de 010-013 y las columnas añadidas no estorban a un código anterior que no las nombra. Verificado en `jax/`, `jacobs/` y `las_manos/` (2026-10-05): ningún `FORCE INDEX`/`USE INDEX` nombra `idx_conversation` ni `idx_embedding_generation_revision`. Dejar el esquema donde está es lo reversible: bajarlo, no.

**Sí hay que mirarlo (y posiblemente bajarlo)** en estos casos, siempre con respaldo restaurable antes (Principio VI):

1. **La migración misma es el defecto.** 011, 012 y 013 bloquean escrituras 1-3 s (`ALGORITHM=COPY`); si eso es lo que causó el incidente, la bajada es la «Marcha atrás de 007–013» de `jax/memory/b9_migrations/README.md`, en orden inverso (013, 012, 011, 010, 009, 008, 007). Bajar 007 borra la auditoría del re-encolado y bajar 009 los contadores de intentos: exportarlos antes si importan.
2. **El código anterior no soporta lo nuevo.** La UNIQUE de 008 `(revision_id, embedding_space_id)` rechaza duplicados: si el código al que se vuelve puede insertar dos generaciones para el mismo par, fallará con `IntegrityError`. **No verificado** para cada SHA: se lee el `embedding_worker` de ese SHA antes de decidir. Para bajar 008 primero se restaura el índice de 010 (la FK lo necesita).
3. **Migración 005 (proyectos).** La reaplica `jax-platform` en **cada arranque** (llama a `apply_project_authority_migration` del checkout de jax, incondicionalmente). Bajarla (`scripts/b9_revertir_005.py`, dry-run por defecto, `--aplicar` no es transaccional) solo sirve si **ya no hay quien la llame**, y eso exige volver **también `jax-platform`** a una versión sin ese llamador, además del checkout de jax: quitar el archivo del checkout de jax no basta, porque entonces `jax-platform` no arranca (ver el paso 5). Hacerlo con `jax-platform` actual y reiniciarlo la vuelve a crear. Falla cerrado si quedan proyectos ARCHIVED/HIDDEN/DISABLED o `project_documents` con filas.
4. **001-004 y 006** (esquema base, autoridad de proyecto, `tenant_id` en bindings y revisiones con su trigger y FK, tablas de jobs) **no tienen bajada escrita**. No se inventa una: se restaura del respaldo con restauración probada. El código del worker actual usa 007 y 009 en cada corrida; el anterior no las conoce y las ignora.

Regla corta: **código atrás, esquema donde está**, salvo que la migración sea la causa o el código anterior choque con ella. Ante la duda se escala a Fernando; no se baja a ojo.

## Verificación final

`systemctl is-active` de las tres unidades, el journal con la línea del freno y el SHA correcto, `systemctl list-timers` contra el registro, y `memory_extraction_jobs`: `python -m jax.memory.worker --atascados` no debe mostrar nada nuevo atribuible a la marcha atrás.
