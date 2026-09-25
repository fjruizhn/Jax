# Workspace proyectos/ — permisos compartidos jaxsvc/fruiz
## Purpose
`proyectos/` del workspace de JAX (`$JAX_WORKSPACE_DIR/proyectos`, hoy
`/home/fruiz/jax-workspace/proyectos` en hall9000) tiene que ser escribible
tanto por la cuenta de servicio `jaxsvc` (LAS MANOS y jax-platform corren como
ella, `UMask=0022` medido con `systemctl show`) como por `fruiz` (corre
`scripts/procesar_archivos.py` a mano), con herencia para todo lo que se cree
después — spec `docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md`
§5. **Corrección de brief, 2026-09-25**: el dueño nuevo es `jaxsvc`, no `fruiz`
(una decisión anterior lo tenía al revés; era un error). Medido el 2026-09-25
antes de aplicar nada: `proyectos/` es `fruiz:fruiz 775` sin ACL y
`sudo -u jaxsvc test -w proyectos` da NO — jaxsvc no puede escribir ahí hoy.

## Scope
`$JAX_WORKSPACE_DIR/proyectos` y todo lo que haya debajo, **salvo** cualquier
entrada cuyo nombre empiece con `.` (estado propio de alguna herramienta —
p.ej. algo como `.claude-flow`, típicamente `700` — que no se toca ni se
recorre: `--verificar` la ignora, `--aplicar` no le otorga nada). Nunca
`$JAX_WORKSPACE_DIR` completo. `ops/permisos_proyectos.py` lo hace cumplir:
rechaza una RAIZ vacía, `/`, una ruta sin `proyectos/`, o cuyo `proyectos/`
sea (o cuelgue de) un symlink.

## Preconditions
- `setfacl`/`getfacl` instalados.
- `sudo -n` disponible para quien aplica: **todo** el trabajo mutante corre
  como root (`chown`, `chmod`, `setfacl` — nunca como el usuario que invoca,
  ver "Por qué todo corre como root" más abajo).
- Las cuentas `jaxsvc` y `fruiz` ya existen en el host de producción, y el
  grupo primario de `fruiz` es un grupo también llamado `fruiz` (lo que
  `useradd` crea por defecto).

## Authority impact
Cambia quién puede leer/escribir/borrar dentro de `proyectos/` — no toca
autenticación ni autorización de ningún servicio, sólo permisos POSIX de
archivo y ACL. **`--aplicar` contra el workspace real de producción
(`/home/fruiz/jax-workspace`) necesita el GO explícito de Fernando** antes de
correr. `--verificar` es siempre de solo lectura (corre sin privilegio, sin
sudo) y no necesita GO.

## Safe procedure
**Verificar primero, siempre** (no hace falta GO, no hace falta sudo):
```bash
cd /ruta/al/checkout/de/jax
python3 ops/permisos_proyectos.py --verificar
```
Sin argumentos, usa `JAX_WORKSPACE_DIR` de `/etc/jax/.env` (leído con
`sudo -n grep`) o, si esa variable no está, `/home/fruiz/jax-workspace`. Sale
`0` si todo `proyectos/` cumple; `1` e imprime cada ruta que no cumple si no.

**Aplicar, con el GO de Fernando ya obtenido:**
```bash
python3 ops/permisos_proyectos.py --aplicar
```
Antes de tocar nada, corre `sudo -n getfacl -R -p` sobre TODO el árbol hacia
un archivo con nombre único (`tempfile.mkstemp`, nunca predecible) en
`/home/fruiz/respaldos-permisos/`, y **exige código 0 y tamaño > 0 antes de
mutar nada** — si el respaldo falla por cualquier motivo, `--aplicar` aborta
sin haber tocado el árbol (verificado con un `sudo` que falla a propósito).
El mensaje impreso trae la **ruta absoluta** del respaldo — nunca `~`, que
bajo `sudo` no se expande igual (`HOME` puede cambiar).

Luego, todo el trabajo mutante corre en un solo proceso hijo lanzado con
`sudo -n python3 ops/permisos_proyectos.py --nucleo-privilegiado <proyectos>`
(un detalle interno del propio guion, no algo que se invoque a mano), que
para cada directorio y archivo del árbol:
1. `fchown` a `jaxsvc:fruiz` (sobre el descriptor ya abierto, nunca sobre una
   ruta de texto — ver "Por qué todo corre como root").
2. `setfacl` de acceso (`u:jaxsvc:rwX,g:fruiz:rwX,m::rwx`) y, en
   directorios, también por defecto — vía `/proc/self/fd/N`, nunca la ruta.
3. En directorios: asegura el bit setgid y **quita** cualquier setuid/sticky
   que hubiera (ver "El `mkdir` de este host" más abajo). En archivos: quita
   cualquier bit especial (setuid/setgid/sticky) que hubiera.
4. Si el objeto es un **hardlink** (`nlink > 1`), **no lo muta** — lo reporta
   como fallo (ver "Hardlinks" más abajo).
5. Si un nombre resultó ser (o se volvió, a mitad de la corrida) un symlink,
   **nunca lo sigue** — lo salta y lo reporta.

Es idempotente: correrlo dos veces deja el mismo estado y termina en `0`
(salvo que haya un hardlink, que siempre falla — ver abajo).

## Por qué todo corre como root, y por qué recorre con descriptores, no rutas
**Amenaza real, no teórica.** `jaxsvc` tiene escritura de grupo sobre
`$JAX_WORKSPACE_DIR` (`fruiz:jaxsvc 775`) — un proceso de jaxsvc con un bug,
o comprometido, puede en cualquier momento borrar `proyectos` y ponerle en su
lugar un symlink a, por ejemplo, `/etc`. Si el recorrido de este guion
(que corre como root, porque el dueño/grupo/ACL nuevos ya no son los de quien
lo invoca) siguiera ese symlink, terminaría haciendo `chown`/`chmod`/`setfacl`
como root sobre archivos arbitrarios del sistema.

La defensa: cada componente del árbol se abre con `O_NOFOLLOW` relativo al
descriptor del directorio padre (`dir_fd`), nunca con una ruta de texto
vuelta a resolver desde la raíz. Si un nombre se convirtió en symlink —
incluso a mitad de la corrida, después de haber sido listado como
directorio — el `open(..., O_NOFOLLOW)` sobre ese nombre falla con `ELOOP`
en vez de seguirlo. Las mutaciones actúan sobre el descriptor ya abierto
(`fchown`, `fchmod`) o, para `setfacl` (que no tiene una forma nativa de
operar sobre un descriptor), sobre `/proc/self/fd/N` — que apunta al inodo
del descriptor, no a un nombre que se pueda haber vuelto a sustituir.

Probado empíricamente en hall9000 (2026-09-25, no supuesto): un directorio
reemplazado por un symlink a un directorio real DESPUÉS de haberlo abierto
con `O_NOFOLLOW` no contamina el objetivo del symlink cuando la ACL se
aplica vía `/proc/self/fd`; y un directorio raíz (`proyectos/`) que sea
symlink se rechaza antes de tocar nada. Los dos casos están automatizados en
`tests/test_permisos_proyectos.py`
(`test_symlink_en_el_punto_de_partida_se_rechaza`,
`test_symlink_sustituido_a_mitad_de_la_corrida_no_contamina_el_objetivo`).

## El permiso EFECTIVO, no el texto de la ACL
Un archivo creado con `tempfile.mkstemp()` (modo `0600` explícito) bajo un
directorio con ACL por defecto recibe entradas de ACL que **siguen
mostrando "rwx" en el texto** pero cuyo permiso **efectivo** (entrada ∧
máscara) es `---`: el algoritmo de creación de ACL interseca el modo pedido
con la ACL por defecto, y `0600` pide grupo=0, así que la máscara resultante
también es 0 — verificado empíricamente en hall9000 el 2026-09-25.
`--verificar` calcula ese efectivo bit a bit (`_permiso_efectivo` en el
propio guion), nunca confía en el texto pedido. El escritor real que
disparaba esto — `las_manos/motor_registry/tool_authority.py::_write_file`
— se corrigió en el mismo cambio que este guion: `os.fchmod(fd, 0o664)`
antes de `os.replace`, con test
(`las_manos/_tool_authority_test.py::test_5c_...`).

## El `mkdir` de este host agrega bits espurios — mitigado, no sólo documentado
`/usr/bin/mkdir` en hall9000 resuelve a `coreutils-from-uutils`
(uutils-coreutils 0.8.0, paquete experimental de Ubuntu), no a GNU
coreutils. Verificado el 2026-09-25, con el mismo directorio padre (ACL por
defecto con entradas nombradas ya puesta): `/usr/bin/mkdir` crea el
directorio nuevo en `7775` (agrega **setuid + sticky** espurios), mientras
que `/usr/bin/gnumkdir` (GNU real) y `os.mkdir()` de Python dan `775`
(correcto) sobre el mismo padre. El sticky espurio le impide a `fruiz`
borrar un archivo de `jaxsvc` (y viceversa) — exactamente lo que este
esquema existe para permitir.

**Ya no es sólo un riesgo documentado**: `--verificar` exige que ningún
directorio tenga setuid ni sticky (el setgid sí, es el bit legítimo) y que
ningún archivo tenga ningún bit especial; `--aplicar` los quita y reporta
cada ruta corregida (`test_verificar_detecta_y_aplicar_quita_bits_espurios_de_un_directorio`,
que reproduce el `7775` real). No se decidió una corrección de host
(`update-alternatives` a GNU mkdir, reportar el defecto río arriba a
uutils) — sigue pendiente para quien tenga autoridad sobre el paquete de
coreutils del host; mientras tanto, cada `--aplicar` limpia lo que haya.

## Hardlinks
Un archivo con `nlink > 1` comparte inodo con alguna otra ruta que puede
estar fuera de `proyectos/` por completo — cambiarle dueño o permiso desde
acá cambiaría también esa otra ruta. `--verificar` y `--aplicar` lo rechazan
sin tocarlo: `--aplicar` termina en `1` si encontró alguno (reportado en la
salida como `HARDLINK RECHAZADO`). Medido el 2026-09-25: la producción real
hoy no tiene ninguno (`find proyectos -type f -links +1` da 0) — es una
defensa hacia adelante, no una situación actual conocida.

## Verification
```bash
python3 ops/permisos_proyectos.py --verificar
```
`0` = cumple (dueño `jaxsvc`, grupo `fruiz`, setgid en cada directorio, sin
setuid/sticky espurios, ACL de acceso y por defecto con permiso EFECTIVO
suficiente para `jaxsvc`/`fruiz` en cada directorio, grupo y ACL de acceso
efectiva en cada archivo, sin bits especiales en archivos, sin hardlinks).
`1` = no cumple, con la lista completa de rutas y el motivo de cada una.

`tests/test_permisos_proyectos.py` lo ejercita en cada corrida de CI (job
`permisos-proyectos` de `.github/workflows/policy.yml`, piso medido "15
passed, 1 skipped") y, sólo en el host de producción de jax, corre además
una prueba de lectura/escritura cruzada real: un subdirectorio temporal
propio dentro de `proyectos/` (que el propio test borra al terminar) donde
`jaxsvc` crea algo y `fruiz` lo lee/escribe, y al revés.

## Fail-closed condition
Si `--verificar` da `1` contra producción, `proyectos/` sigue sin estar en
el estado que ambas cuentas necesitan. Si `--aplicar` encuentra un hardlink,
termina en `1` aunque haya corregido todo lo demás — un hardlink sin resolver
no es un estado parcial aceptable, es una decisión pendiente sobre qué
hacer con ese archivo. Si el respaldo previo falla, `--aplicar` no aplica
nada (ver "Safe procedure").

## Recovery / escalation
**Revertir** al estado de antes de `--aplicar`, con el respaldo (ruta
absoluta, impresa por el propio `--aplicar`):
```bash
sudo setfacl --restore=/home/fruiz/respaldos-permisos/proyectos-XXXXXXXX.acl
```
Verificado en hall9000 (2026-09-25): repone owner, group, ACL de acceso y
por defecto, y el bit setgid, de punta a punta, con una sola orden. Si el
respaldo no alcanza o el estado quedó irreconocible, escalar a Fernando
antes de intentar nada más manual sobre `proyectos/`.

## Prohibited actions
- No correr `--aplicar` contra el workspace real sin el GO explícito de
  Fernando (`--verificar` sí, en cualquier momento).
- No mutar nada por ruta de texto "a mano" (un `chown -R`/`chmod -R` directo
  se salta toda la defensa contra symlinks de este guion).
- No cambiar `USUARIO`/`GRUPO` en el guion sin actualizar el spec aprobado
  primero — es una decisión de Fernando, no un parámetro de conveniencia.
- No tocar nada fuera de `$RAIZ/proyectos`, ni las entradas cuyo nombre
  empieza con `.`.
- Nunca invocar `--nucleo-privilegiado` a mano: es el mecanismo interno con
  el que `--aplicar` re-ejecuta el núcleo mutante como root; asume que ya
  se validó todo lo que el modo público valida primero.
