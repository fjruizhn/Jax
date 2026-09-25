# Workspace proyectos/ — permisos compartidos jaxsvc/fruiz
## Purpose
`proyectos/` del workspace de JAX (`$JAX_WORKSPACE_DIR/proyectos`, hoy
`/home/fruiz/jax-workspace/proyectos` en hall9000) tiene que ser escribible
tanto por la cuenta de servicio `jaxsvc` (LAS MANOS y jax-platform corren como
ella, `UMask=0022` medido con `systemctl show`) como por `fruiz` (corre
`scripts/procesar_archivos.py` a mano), con herencia para todo lo que se cree
después — spec `docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md`
§5. El dueño es `jaxsvc`, el grupo es `fruiz`. Medido el 2026-09-25 antes de
aplicar nada: `proyectos/` es `fruiz:fruiz 775` sin ACL y `sudo -u jaxsvc test
-w proyectos` da NO — jaxsvc no puede escribir ahí hoy.

**Segunda ronda de auditoría (2026-09-25).** La primera reescritura en Python
(commit 4f117a7) recibió 3 BLOCK, 3 MAJOR y 6 MINOR. Cada uno se corrigió y
quedó probado; este runbook documenta el diseño YA CORREGIDO. Quien quiera el
historial completo de qué estaba mal y por qué, lo tiene en el docstring de
`ops/permisos_proyectos.py` y en el mensaje del commit correspondiente.

## Scope
`$JAX_WORKSPACE_DIR/proyectos` y todo lo que haya debajo, **salvo** los
nombres en la lista explícita `NOMBRES_EXCLUIDOS` del propio guion (hoy sólo
`.claude-flow`) — decisión del coordinador hasta que Fernando diga otra cosa;
antes era "cualquier nombre con punto", demasiado amplio. Esas rutas no
reciben nada y `--verificar`/`--aplicar` las imprimen como excluidas, nunca
las tratan como incumplimiento. Nunca `$JAX_WORKSPACE_DIR` completo.
`ops/permisos_proyectos.py` rechaza una RAIZ vacía, `/`, una ruta sin
`proyectos/`, o cuyo `proyectos/` sea (o cuelgue de) un symlink.

## Preconditions
- `setfacl`/`getfacl` instalados.
- `sudo -n` disponible: **todo** el trabajo mutante (aplicar, respaldar,
  revertir) corre como root, invocando el núcleo **instalado** (ver abajo) —
  nunca el checkout del repo directamente.
- **El núcleo privilegiado instalado** en `/usr/local/sbin/jax-permisos-proyectos`
  (root:root, 0755, con toda la cadena de directorios padre también de root y
  sin escritura de grupo/otros), con el mismo contenido byte a byte que
  `ops/permisos_proyectos.py` del checkout que se está usando. `--aplicar` se
  niega a correr si esto no se cumple (ver "Instalar/actualizar el núcleo"
  abajo) — no hay fallback silencioso a `__file__`.
- Las cuentas `jaxsvc` y `fruiz` ya existen en el host de producción, y el
  grupo primario de `fruiz` es un grupo también llamado `fruiz`.

## Authority impact
Cambia quién puede leer/escribir/borrar dentro de `proyectos/` — no toca
autenticación ni autorización de ningún servicio, sólo permisos POSIX de
archivo y ACL. **`--aplicar` y `--revertir` contra el workspace real de
producción (`/home/fruiz/jax-workspace`) necesitan el GO explícito de
Fernando** antes de correr. `--verificar` es siempre de solo lectura, corre
sin privilegio (no usa `sudo` para nada que mute) y no necesita GO.

## Instalar/actualizar el núcleo (BLOCK-2)
**Por qué existe este paso.** El núcleo mutante corre como root. Si se
re-ejecutara desde el propio checkout (`sudo -n python3 ops/permisos_proyectos.py
...`), y ese checkout viviera en un directorio escribible por `jaxsvc` (la
misma cuenta cuya escritura sin control es la amenaza de fondo de todo este
esquema), `jaxsvc` podría plantar un `json.py` en `ops/` que se importaría en
vez del `json` de la librería estándar — y correría como root. Reproducido
empíricamente en hall9000 el 2026-09-25: SIN el flag `-I` de Python, un
`json.py` de mentira en el directorio del script se importa y se ejecuta como
root; CON `-I`, `sys.path` no incluye el directorio del script y se importa el
`json` real. `ops/permisos_proyectos.py` usa `-I` siempre para esto, Y ADEMÁS
se instala en una ruta que sólo root puede escribir, por las dos razones:
defensa en profundidad, y porque `-I` sólo te salva de ESTE vector concreto,
no de que alguien reemplace el propio archivo.

```bash
sudo install -o root -g root -m 0755 \
  /ruta/al/checkout/de/jax/ops/permisos_proyectos.py \
  /usr/local/sbin/jax-permisos-proyectos
```

Idempotente — correrlo de nuevo tras cada cambio al guion. `--verificar`
compara el sha256 de lo instalado contra el del checkout que se está usando y
avisa si no coinciden (no lo cambia el exit code de la parte de ACL, pero
queda impreso, y `--aplicar` SÍ se niega a correr si no coinciden o si la
cadena de directorios no es segura).

**El núcleo privilegiado, además, sólo acepta la RAIZ configurada.** Antes de
esta ronda, se le podía pedir literalmente `/etc` y lo recorría igual. Ahora
`--nucleo-privilegiado` re-deriva de forma independiente cuál es la RAIZ
correcta (lee `/etc/jax/.env` directo, ya es root) y rechaza cualquier otra —
no confía en lo que el proceso sin privilegio le haya pasado.

## Safe procedure
**Verificar primero, siempre** (no hace falta GO, no hace falta sudo para
mutar nada):
```bash
cd /ruta/al/checkout/de/jax
python3 ops/permisos_proyectos.py --verificar
```
Sin argumentos, usa `JAX_WORKSPACE_DIR` de `/etc/jax/.env` (leído con
`sudo -n grep` desde el proceso sin privilegio — si `sudo -n` no está
disponible y no se dio una RAIZ explícita, esto FALLA CERRADO en vez de
adivinar un valor fijo). Sale `0` si todo `proyectos/` cumple; `1` e imprime
cada ruta que no cumple si no.

**Aplicar, con el GO de Fernando ya obtenido:**
```bash
python3 ops/permisos_proyectos.py --aplicar
```
Antes de tocar nada, el núcleo instalado respalda TODO el árbol con
`getfacl -R -p` a un archivo con nombre único (`tempfile.mkstemp`) en
`/var/backups/jax-permisos/` (root, 0700 — **no** el `$HOME` de `fruiz`: un
runner o una cuenta de servicio sin directorio home propio no tiene por qué
tenerlo, y esta ruta no depende de eso). Si el respaldo falla por cualquier
motivo, el archivo parcial se borra y no se aplica ningún cambio. El mensaje
impreso trae la **ruta absoluta** del respaldo.

Luego el núcleo instalado, corriendo como root, recorre el árbol con un
caminante que **nunca abre nada por nombre resuelto desde la raíz**: cada
componente se abre con `O_PATH|O_NOFOLLOW` relativo al descriptor del padre,
y se clasifica con `fstat` sobre ESE MISMO descriptor — no hay ventana entre
"mirar qué es" y "actuar sobre eso" (ver "Por qué el recorrido es seguro" más
abajo). Para cada directorio/archivo:
1. `fchown` a `jaxsvc:fruiz`.
2. `setfacl` de acceso (`u:jaxsvc:rwX,g:fruiz:rwX,m::rwx`) y, en
   directorios, también por defecto.
3. En directorios: asegura setgid y quita setuid/sticky si los hubiera. En
   archivos: quita cualquier bit especial que hubiera.
4. Un **hardlink** (`nlink > 1`) nunca se muta — se reporta como fallo (ver
   "Hardlinks" abajo).
5. Un **symlink** — incluso si apareció a mitad de la corrida, después de
   haber sido listado como directorio — nunca se sigue: se salta y se
   reporta.

Es idempotente (salvo que haya un hardlink, que siempre hace fallar la
corrida hasta que se resuelva a mano).

## Por qué el recorrido es seguro (B3, MAJOR-2)
`jaxsvc` tiene escritura de grupo sobre `$JAX_WORKSPACE_DIR` — un proceso de
jaxsvc con un bug, o comprometido, puede en cualquier momento reemplazar
`proyectos` (o cualquier subdirectorio) por un symlink a, por ejemplo,
`/etc`. `O_PATH|O_NOFOLLOW` es la defensa: verificado empíricamente en
hall9000 (2026-09-25) que **no requiere ningún permiso sobre el objetivo**
(ni de lectura, ni de escritura, ni de ejecución — sólo travesía sobre el
padre, que ya se tiene por haber llegado hasta ahí), que `fstat`/`getfacl`/
`setfacl` vía `/proc/self/fd/N` de un descriptor `O_PATH` funcionan siempre
(incluso contra un archivo 0600 de otro dueño — esto es lo que resuelve
MAJOR-1, ver abajo), que sobre una FIFO **no bloquea** (a diferencia de un
`open()` normal, que colgaría el núcleo root esperando un escritor que nunca
llega — MAJOR-2), y que sobre un symlink **no falla** — da un descriptor que
`fstat` identifica como symlink, sin haber tocado el objetivo. `fchown`/
`fchmod` sí necesitan un descriptor "real" (no `O_PATH`) — se reabren vía
`/proc/self/fd/N`, lo que corriendo como root nunca falla por permisos del
objetivo. Con todo esto, clasificar y actuar son la MISMA apertura: no hay
ventana entre "mirar qué es" y "usarlo" en la que algo se pueda haber
sustituido — ni para un symlink, ni para un hardlink, ni para una FIFO.

Un directorio al que el proceso sin privilegio (`--verificar`) no puede
**listar** (`EACCES` al reabrir para escanear hijos) se reporta como NO CUMPLE
y no se desciende ahí — nunca revienta con un traceback a medias (MAJOR-1).

## Reversión (BLOCK-1 — reescrita esta ronda)
**`sudo setfacl --restore=<respaldo>` YA NO ES el procedimiento de
reversión.** Esa orden resuelve las rutas del respaldo por NOMBRE, como root,
sin ninguna protección contra symlinks: si algo (`jaxsvc`, que escribe en el
padre) reemplazó un subdirectorio por un symlink a un directorio real de
root, `setfacl --restore` le aplicaría alegremente las entradas de ACL del
respaldo a lo que el symlink apunte HOY — contaminando un directorio
arbitrario del sistema como root. Reproducido y corregido el 2026-09-25.

```bash
python3 ops/permisos_proyectos.py --revertir /var/backups/jax-permisos/proyectos-XXXXXXXX.acl
```

Corre el mismo caminante seguro que `--aplicar` (nunca por nombre resuelto
desde la raíz). Para cada objeto que el respaldo recuerda, si el objeto
ACTUAL en esa ruta es del mismo tipo (directorio/archivo) que el respaldo
registró, restaura dueño, grupo, ACL de acceso (y por defecto, si es
directorio). Si el objeto es hoy un symlink, o cambió de tipo desde el
respaldo, **se salta y se reporta** — nunca se toca. Verificado con el ataque
exacto: un subdirectorio ya respaldado, reemplazado por `jaxsvc` con un
symlink a un directorio `root:700` real — `--revertir` lo saltea, lo reporta,
y el directorio de root queda exactamente como estaba (`root:700`, sin
ninguna entrada de ACL nueva).

Un objeto que existe hoy pero no estaba en el respaldo NO se toca (criterio
conservador: revertir repone lo que había, no "limpia" lo nuevo). Un objeto
del respaldo que ya no existe, simplemente no tiene nada que revertir.

## El permiso EFECTIVO, no el texto de la ACL (B1, MAJOR-3)
Un archivo creado con `tempfile.mkstemp()` (modo `0600` explícito) bajo un
directorio con ACL por defecto recibe entradas de ACL que **siguen
mostrando "rwx" en el texto** pero cuyo permiso **efectivo** (entrada ∧
máscara) es `---` — verificado empíricamente. `--verificar` calcula esa
intersección bit a bit, nunca confía en el texto pedido. El escritor real que
disparaba esto — `las_manos/motor_registry/tool_authority.py::_write_file` —
se corrige en el mismo cambio: `os.fchmod(fd, 0o660)` antes de `os.replace`
(0660, no 0664 — MINOR m-c: fuera de `proyectos/` no hay ACL por defecto que
recorte "other", así que 0664 dejaría el archivo legible por cualquiera).

## El `mkdir` de este host agrega bits espurios — detectado y corregido
`/usr/bin/mkdir` en hall9000 resuelve a `coreutils-from-uutils` (uutils-coreutils
0.8.0), no a GNU coreutils. Bajo un padre con ACL por defecto de entradas
nombradas, crea el directorio nuevo con setuid+sticky espurios (`7775`)
mientras que GNU mkdir y `os.mkdir()` de Python dan `775` sobre el mismo
padre. `--verificar` exige que ningún directorio tenga setuid ni sticky (el
setgid sí, es legítimo) y que ningún archivo tenga ningún bit especial;
`--aplicar` los quita y reporta cada ruta corregida.

## Hardlinks
Un archivo con `nlink > 1` comparte inodo con alguna otra ruta que puede
estar fuera de `proyectos/` por completo. `--verificar` y `--aplicar` lo
rechazan sin tocarlo — `--aplicar` termina en `1` si encontró alguno.

## Verification
```bash
python3 ops/permisos_proyectos.py --verificar
```
`0` = cumple (dueño `jaxsvc`, grupo `fruiz`, setgid en cada directorio, sin
setuid/sticky espurios, ACL de acceso y por defecto con permiso EFECTIVO
suficiente, sin bits especiales en archivos, sin hardlinks). `1` = no cumple.

`tests/test_permisos_proyectos.py` lo ejercita en cada corrida de CI (job
`permisos-proyectos`, piso medido "28 passed, 1 skipped" **en un contenedor
`ubuntu:24.04` limpio con un usuario `runner` con sudo sin contraseña — no en
hall9000**, a propósito: el propio hall9000, donde la sesión que desarrolla
esto corre literalmente como `fruiz`, no habría destapado que el respaldo
dependía del `$HOME` de fruiz, ni que varias operaciones de test asumían sin
decirlo que "quien corre pytest" y "fruiz" son la misma identidad — falso en
cualquier runner de GitHub Actions, donde el usuario se llama `runner`) y,
sólo en el host de producción de jax, una prueba de lectura/escritura
cruzada real en un subdirectorio temporal propio.

## Fail-closed condition
Si `--verificar` da `1`, `proyectos/` sigue sin estar en el estado correcto.
Si `--aplicar` encuentra un hardlink, termina en `1` aunque haya corregido
todo lo demás. Si el respaldo previo falla, `--aplicar` no aplica nada. Si el
núcleo instalado no existe, no coincide (sha256) con el checkout, o su cadena
de directorios no es segura, `--aplicar` se niega a correr.

## Recovery / escalation
```bash
python3 ops/permisos_proyectos.py --revertir /var/backups/jax-permisos/proyectos-XXXXXXXX.acl
```
(Ver "Reversión" arriba — **no** `sudo setfacl --restore`, que ya no es el
procedimiento recomendado.) Si `--revertir` reporta rutas saltadas
(symlinks o tipo cambiado), esas rutas quedan sin revertir a propósito —
resolverlas a mano, con Fernando, antes de asumir que el árbol volvió al
estado anterior. Si el estado quedó irreconocible, escalar a Fernando antes
de intentar nada más manual sobre `proyectos/`.

## Prohibited actions
- No correr `--aplicar` ni `--revertir` contra el workspace real sin el GO
  explícito de Fernando.
- No usar `sudo setfacl --restore=<respaldo>` como reversión — sigue
  symlinks por nombre como root, es exactamente la amenaza que `--revertir`
  existe para evitar.
- No mutar nada por ruta de texto "a mano".
- No cambiar `USUARIO`/`GRUPO` en el guion sin actualizar el spec aprobado
  primero.
- No tocar nada fuera de `$RAIZ/proyectos`, ni las entradas en
  `NOMBRES_EXCLUIDOS`.
- Nunca invocar `--nucleo-privilegiado`/`--nucleo-respaldo`/`--nucleo-revertir`
  a mano: son el mecanismo interno con el que `--aplicar`/`--revertir`
  re-ejecutan el núcleo instalado como root.
- No dejar que `ops/` (o cualquier ancestro del núcleo instalado) quede
  escribible por grupo u otros -- `--aplicar` ya se niega a correr si lo
  detecta, pero tampoco se relaja ese chequeo "para que ande".
