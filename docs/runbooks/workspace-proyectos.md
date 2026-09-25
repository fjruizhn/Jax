# Workspace proyectos/ — permisos compartidos jaxsvc/fruiz
## Purpose
`proyectos/` del workspace de JAX (`$JAX_WORKSPACE_DIR/proyectos`, hoy
`/home/fruiz/jax-workspace/proyectos` en hall9000) tiene que ser escribible
tanto por la cuenta de servicio `jaxsvc` (LAS MANOS y jax-platform corren como
ella, `UMask=0022` medido con `systemctl show`) como por `fruiz` (corre
`scripts/procesar_archivos.py` a mano), con herencia para todo lo que se cree
después — spec
`docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md` §5.
Medido el 2026-09-25 **antes** de aplicar nada: `proyectos/` es `fruiz:fruiz
775` sin ACL y `sudo -u jaxsvc test -w proyectos` da NO — jaxsvc no puede
escribir ahí hoy.

## Scope
`$JAX_WORKSPACE_DIR/proyectos` y todo lo que haya debajo. Nunca
`$JAX_WORKSPACE_DIR` completo (ahí conviven `calculadora.html`,
`run_tests.sh`, etc. que no son parte de este contrato) ni ninguna otra ruta.
`ops/permisos-proyectos.sh` lo hace cumplir: rechaza una RAIZ vacía, `/`, o
una ruta sin `proyectos/`.

## Preconditions
- `setfacl`/`getfacl` instalados (`/usr/bin/setfacl`, `/usr/bin/getfacl` en
  hall9000, verificado).
- `sudo -n` disponible para quien aplica: hace falta para `chgrp` (fruiz no
  es miembro del grupo `jaxsvc`, así que no puede cambiar un archivo a ese
  grupo sin privilegio — `CAP_CHOWN`) y para el `chmod g+s` final (ver más
  abajo, "por qué el orden importa").
- Los grupos/cuentas `fruiz` y `jaxsvc` ya existen en el host de producción.

## Authority impact
Cambia quién puede leer/escribir/borrar dentro de `proyectos/` — no toca
autenticación ni autorización de ningún servicio, sólo permisos POSIX de
archivo. **`--aplicar` contra el workspace real de producción
(`/home/fruiz/jax-workspace`) necesita el GO explícito de Fernando** antes de
correr — no es una operación de solo lectura, y toca el árbol donde viven los
proyectos de clientes reales (p.ej. `lacteos-victoria`). `--verificar` es
siempre de solo lectura y no necesita GO.

## Safe procedure
**Verificar primero, siempre** (no hace falta GO):
```bash
cd /ruta/al/checkout/de/jax
ops/permisos-proyectos.sh --verificar
```
Sin argumentos, usa `JAX_WORKSPACE_DIR` de `/etc/jax/.env` (leído con
`sudo -n grep`, nunca sourceado directo — `/etc/jax/.env` es `root:jaxsvc
640`) o, si esa variable no está, `/home/fruiz/jax-workspace`. Sale `0` si
todo `proyectos/` cumple; `1` e imprime cada ruta que no cumple si no.

**Aplicar, con el GO de Fernando ya obtenido:**
```bash
ops/permisos-proyectos.sh --aplicar
```
Antes de tocar nada, guarda un respaldo completo y restaurable en
`~/respaldos-permisos/proyectos-<fecha>.acl` (`getfacl -R -p`, con las rutas
absolutas y el flag `-s-`/setgid incluido — verificado en hall9000 el
2026-09-25 que ese archivo alcanza para reponer owner/group/ACL/setgid de
punta a punta con `setfacl --restore`, ver "Cómo revertir" más abajo). Luego
aplica, en este orden exacto — **no reordenar, ver la nota debajo**:
1. `sudo chgrp -R jaxsvc proyectos/`
2. `setfacl -R -m u:fruiz:rwX,g:jaxsvc:rwX,m::rwx proyectos/` (ACL de acceso)
3. `setfacl -R -d -m u:fruiz:rwX,g:jaxsvc:rwX,m::rwx proyectos/` (ACL por
   defecto — esto es lo que hace que un archivo nuevo, creado por cualquiera
   de las dos cuentas con cualquier umask, herede el grupo y quede
   escribible: **el umask se ignora cuando hay ACL por defecto puesta**,
   verificado empíricamente, no supuesto)
4. `sudo find proyectos/ -type d -exec chmod g+s {} +` (setgid en
   directorios, **al final**)
Es idempotente: correrlo dos veces deja el mismo estado y termina en `0`.

**Por qué el orden del paso 4 importa (verificado en hall9000, 2026-09-25).**
`fruiz` no pertenece al grupo `jaxsvc`. Cualquier llamada a `setfacl(1)`
corrida como `fruiz` sobre un directorio cuyo grupo es `jaxsvc` **limpia en
silencio el bit setgid** que ya estuviera puesto — es el mismo mecanismo del
kernel que limpia `S_ISGID` en `chmod(2)` cuando el llamador no pertenece al
grupo del archivo y no tiene `CAP_FSETID`; Linux lo aplica también a
directorios, aunque la letra de `chmod(2)` sólo hable de archivos regulares.
Por eso el `chmod g+s` va siempre **después** de los dos `setfacl` y siempre
con `sudo` (a root nunca se le limpia el bit). Poner el `chmod g+s` antes
"funciona" en apariencia (`chmod -v` confirma el cambio) pero el siguiente
`setfacl` se lo vuelve a quitar sin avisar.

## Verification
```bash
ops/permisos-proyectos.sh --verificar
```
`0` = cumple (grupo `jaxsvc`, setgid en cada directorio, ACL de acceso y por
defecto con `fruiz`/`jaxsvc` en cada directorio, grupo y ACL de acceso en
cada archivo). `1` = no cumple, con la lista completa de rutas. El PR que
agrega este guion incluye `tests/test_permisos_proyectos.py`, que lo ejercita
contra un árbol temporal en cada corrida de CI (job `permisos-proyectos` de
`.github/workflows/policy.yml`) y, sólo en el host de producción de jax
(donde exista `/home/fruiz/jax-workspace/proyectos` y `sudo -n -u jaxsvc
true` funcione), corre además una prueba real: `sudo -u jaxsvc` crea y borra
un archivo dentro de `proyectos/`.

## Fail-closed condition
Si `--verificar` da `1` contra producción, `proyectos/` sigue sin ser
escribible por una de las dos cuentas — no se asume que "probablemente ya
está bien". Si `--aplicar` termina pero su `--verificar` final también da
`1` (mensaje "`--aplicar` terminó pero `--verificar` final encontró fallos"),
el árbol quedó en un estado parcial: no declarar la tarea cumplida, revisar
la salida y, si hace falta, revertir (ver abajo) antes de reintentar.

## Recovery / escalation
**Revertir** al estado de antes de `--aplicar`, con el respaldo que el
propio `--aplicar` dejó en `~/respaldos-permisos/`:
```bash
sudo setfacl --restore=~/respaldos-permisos/proyectos-<fecha>.acl
```
Verificado en hall9000 (2026-09-25): esto repone owner, group, ACL de acceso
y por defecto, y el bit setgid, exactamente como estaban antes — de punta a
punta, con una sola orden, corrida como root (el grupo a restaurar es
`jaxsvc`, y restaurar `chgrp` necesita el mismo privilegio que aplicarlo).
Si el respaldo no alcanza o el estado quedó irreconocible, escalar a
Fernando antes de intentar nada más manual sobre `proyectos/` — es el árbol
de proyectos de clientes reales.

## Prohibited actions
- No correr `--aplicar` contra el workspace real sin el GO explícito de
  Fernando (`--verificar` sí, en cualquier momento, es de solo lectura).
- No reordenar los cuatro pasos de `--aplicar` (ver "por qué el orden
  importa" arriba) ni mover el `chmod g+s` fuera de `sudo`.
- No cambiar el dueño (`fruiz` se queda) ni tocar nada fuera de
  `$RAIZ/proyectos` — `ops/permisos-proyectos.sh` ya lo rechaza, pero
  tampoco se hace a mano por fuera del guion.
- No usar `chmod -R` a secas sobre `proyectos/` para "simplificar": pisa la
  ACL sin avisar y dos cuentas pueden volver a quedar sin escritura cruzada.

## Riesgos conocidos
**El `mkdir` por defecto de este host (hall9000) tiene un defecto verificado
que agrega bits espurios a directorios nuevos creados bajo un padre con ACL
por defecto.** `/usr/bin/mkdir` resuelve hoy a `coreutils-from-uutils`
(uutils-coreutils 0.8.0, paquete experimental de Ubuntu), no a GNU
coreutils. Verificado el 2026-09-25, con el mismo directorio padre (ACL por
defecto con entradas nombradas ya puesta) y sin ningún otro cambio:

| Herramienta que crea el directorio nuevo | Resultado |
|---|---|
| `/usr/bin/mkdir` (uutils, el que resuelve por `$PATH`) | `7775` — agrega **setuid + sticky** espurios |
| `/usr/bin/gnumkdir` (GNU coreutils real, mismo host) | `775` — correcto |
| `os.mkdir()` de Python | `775` — correcto |

El sticky espurio le **impide a `fruiz` borrar un archivo de `jaxsvc`** (y
viceversa) dentro de ese directorio — exactamente lo que este esquema de
permisos existe para permitir. Los servicios de `jaxsvc` (LAS MANOS,
jax-platform) crean directorios con Python, no con `mkdir`, así que no les
pega; **cualquier script de shell que use `mkdir` a secas bajo `proyectos/`
sí queda expuesto** el día que alguien lo escriba. `ops/permisos-proyectos.sh`
no crea directorios nuevos (sólo corrige los que ya existen), así que no lo
sufre. No se decidió una corrección de host (`update-alternatives` a GNU
mkdir, reportar el defecto río arriba a uutils, u otra) en esta ronda — queda
para quien tenga la autoridad de tocar el paquete de coreutils del host.
