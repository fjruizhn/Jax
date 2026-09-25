# Workspace proyectos/ — permisos compartidos jaxsvc/fruiz
## Purpose
`proyectos/` del workspace de JAX (`$JAX_WORKSPACE_DIR/proyectos`, hoy
`/home/fruiz/jax-workspace/proyectos` en hall9000) tiene que ser escribible
tanto por la cuenta de servicio `jaxsvc` (LAS MANOS y jax-platform corren como
ella, `UMask=0022` medido con `systemctl show`) como por `fruiz` (corre
`scripts/procesar_archivos.py` a mano), con herencia para todo lo que se cree
después — spec `docs/superpowers/specs/2026-09-22-proyectos-y-selector-design.md`
§5. El dueño tras `--aplicar` es `jaxsvc`, el grupo es `fruiz`. Medido el
2026-09-25 antes de aplicar nada: `proyectos/` es `fruiz:fruiz 775` sin ACL y
`sudo -u jaxsvc test -w proyectos` da NO.

**Tercera ronda de auditoría (2026-09-25).** Las dos primeras reescrituras
(commits 4f117a7 y e1354b2) recibieron 3+3 BLOCK, 3+0 MAJOR y 6+0 MINOR. Esta
tercera ronda rechazó el DISEÑO de la reversión de la segunda ronda (2 BLOCK
nuevos) — ver "Reversión" más abajo, que ya NO usa el respaldo para nada.
Quien quiera el historial completo, está en el docstring de
`ops/permisos_proyectos.py` y en el mensaje de cada commit.

## Scope
`$JAX_WORKSPACE_DIR/proyectos` y todo lo que haya debajo, **salvo** los
nombres en `NOMBRES_EXCLUIDOS` (hoy sólo `.claude-flow`) -- y SÓLO cuando
aparecen en el PRIMER nivel de cada proyecto (`proyectos/<proyecto>/.claude-flow`):
ni en `proyectos/` mismo, ni más profundo. Esas rutas no reciben nada y se
reportan como excluidas, nunca como incumplimiento. Nunca
`$JAX_WORKSPACE_DIR` completo. `ops/permisos_proyectos.py` rechaza una RAIZ
vacía, `/`, una ruta sin `proyectos/`, o cuyo `proyectos/` sea (o cuelgue de)
un symlink.

## Preconditions
- `setfacl`/`getfacl` instalados.
- `sudo -n` disponible: **todo** el trabajo mutante (aplicar, respaldar,
  deshacer) corre como root, invocando el núcleo **instalado** -- nunca el
  checkout del repo directamente.
- **El núcleo privilegiado instalado** en `/usr/local/sbin/jax-permisos-proyectos`
  (root:root, 0755, con toda la cadena de directorios padre también de root,
  sin escritura de grupo/otros, y sin ningún symlink en la cadena -- se
  revisa con `lstat`, nunca `stat`), con el mismo contenido que
  `git show HEAD:ops/permisos_proyectos.py` del checkout que se está usando
  -- no el working tree, el COMMIT. `--aplicar`/`--deshacer` se niegan a
  correr si esto no se cumple.
- Las cuentas `jaxsvc` y `fruiz` ya existen en el host de producción, y el
  grupo primario de `fruiz` es un grupo también llamado `fruiz`.
- `/etc/jax/.env` legible (por `sudo -n` desde el proceso sin privilegio, o
  directo desde el núcleo, que ya es root): sin esto, TODO lo que necesita
  `PROYECTOS` falla cerrado -- no hay ningún valor de reserva.

## Authority impact
Cambia quién puede leer/escribir/borrar dentro de `proyectos/` -- no toca
autenticación ni autorización de ningún servicio. **`--aplicar` y
`--deshacer` contra el workspace real de producción necesitan el GO explícito
de Fernando.** `--verificar` es siempre de solo lectura, corre sin privilegio,
y no necesita GO.

## Instalar/actualizar el núcleo
```bash
sudo install -o root -g root -m 0755 \
  /ruta/al/checkout/de/jax/ops/permisos_proyectos.py \
  /usr/local/sbin/jax-permisos-proyectos
```
**Por qué existe este paso, y por qué siempre con `-I`.** El núcleo mutante
corre como root. Si se re-ejecutara desde el propio checkout, y ese checkout
viviera en un directorio escribible por `jaxsvc` (la misma cuenta cuya
escritura sin control es la amenaza de fondo), `jaxsvc` podría plantar un
`json.py` que se importaría en vez del de la librería estándar -- y correría
como root. Reproducido empíricamente: SIN `-I`, un `json.py` de mentira en el
directorio del script se importa y se ejecuta como root; CON `-I` (lo que
`--aplicar`/`--deshacer` usan siempre, invocando SIEMPRE la copia instalada,
nunca el checkout), no. La instalación es idempotente -- correrla de nuevo
tras cada cambio al guion (y sólo tras COMMITEARLO: la verificación compara
contra HEAD, no contra el working tree).

**Las tres entradas del núcleo (`--nucleo-privilegiado`, `--nucleo-respaldo`,
`--nucleo-deshacer`) no aceptan NINGÚN argumento.** Cada una resuelve
`PROYECTOS` de forma independiente, siempre desde `/etc/jax/.env` -- nunca
confían en una ruta que les pase el proceso que las invoca. Un argumento de
más (`--nucleo-respaldo /etc/ssh`, `--nucleo-deshacer /var/tmp/x`) se
rechaza sin tocar nada. Modos repetidos (`--aplicar --verificar`) también.

## Safe procedure
**Verificar primero, siempre:**
```bash
python3 ops/permisos_proyectos.py --verificar
```
Sale `0` si todo `proyectos/` cumple; `1` e imprime cada ruta que no cumple
si no.

**Aplicar, con el GO de Fernando ya obtenido:**
```bash
python3 ops/permisos_proyectos.py --aplicar
```
Antes de mutar nada: (1) se revisa que el núcleo instalado sea de confianza;
(2) se guarda un respaldo con `getfacl -R -p` en `/var/backups/jax-permisos/`
(root, 0700) -- **registro forense únicamente, ver más abajo, nunca se usa
para reconstruir nada**; el respaldo se valida solo (`fsync`, se relee, se
cuenta cuántas entradas tiene y se compara contra un recorrido independiente
del árbol real -- `getfacl -R -p ... > archivo` puede dar código de salida 0
aunque la escritura haya fallado de verdad, verificado empíricamente contra
`/dev/full`, así que el código de salida NUNCA es suficiente evidencia por
sí solo) y si no cuadra, se descarta y `--aplicar` aborta sin mutar nada.

Luego el núcleo instalado recorre el árbol con `O_PATH|O_NOFOLLOW` (nunca
sigue un symlink, ni siquiera uno plantado a mitad de la corrida) y para
cada directorio/archivo: `fchown` a `jaxsvc:fruiz`; `setfacl` de acceso
(`u:jaxsvc:rwX,g:fruiz:rwX,m::rwx`) y, en directorios, también por defecto;
asegura setgid y quita setuid/sticky espurios; un **hardlink** (`nlink > 1`)
nunca se muta, se reporta como fallo.

## Reversión — `--deshacer`, DETERMINISTA (reescrita en la ronda 3)
**El respaldo de `getfacl` ya NO se usa para revertir nada.** La segunda
ronda intentaba reconstruir el árbol leyendo ese respaldo, y tenía cuatro
fallas de diseño reales: un directorio SIN ACL todavía (el estado de HOY) se
leía como archivo porque el parser decidía "es directorio" mirando si había
líneas `default:`; no restauraba setgid ni el resto de los bits; un espacio
al final de un nombre se recortaba al desescapar; y `--nucleo-revertir
<respaldo> <ruta>` aceptaba CUALQUIER respaldo fabricado y cualquier ruta,
dándole a un proceso root instrucciones de chown arbitrario. Ninguna de esas
cuatro es un parche menor -- por eso se retiró por completo, no se corrigió.

```bash
python3 ops/permisos_proyectos.py --deshacer
```
Sin argumentos -- siempre actúa sobre la RAIZ configurada. DETERMINISTA: no
lee ningún archivo de estado. Lleva cada directorio y archivo al único
estado que hall9000 tiene medido HOY con `stat` real, fuera de
`.claude-flow` (que este modo tampoco toca):

| | dueño | grupo | modo | ACL |
|---|---|---|---|---|
| 108 directorios medidos | fruiz | fruiz | 0775 | ninguna |
| 249 archivos medidos | fruiz | fruiz | 0664 | ninguna |

En vez de fijar `0775`/`0664` a fuego, el modo se DERIVA del `rwx` que el
DUEÑO ya tiene en cada objeto en el momento de deshacer (lo que en el árbol
ya aplicado da exactamente esos dos números: grupo y "otros" se recalculan a
partir del propietario -- `grupo = propietario`, `otros = propietario sin
escritura`). Así, si algún objeto real dentro del alcance de este guion
resultara ser una excepción legítima algún día, `--deshacer` no lo fuerza a
`0775`/`0664` -- preserva lo que su dueño ya podía hacer. **Las únicas
excepciones medidas hoy** (2 directorios en `0700`, 1 archivo en `0600`)
viven DENTRO de `.claude-flow`, y por lo tanto están fuera del alcance de
`--verificar`/`--aplicar`/`--deshacer` los tres.

Igual que `--aplicar`: recorrido `O_PATH|O_NOFOLLOW`, symlinks y hardlinks
nunca se tocan, se reportan. Verificado con el ataque de la ronda 2 (un
enlace plantado a un directorio real) y con los casos de borde de la ronda 3
(un nombre con espacio al final, un archivo con uid sin entrada en
`/etc/passwd`): ninguno hace fallar ni contamina nada fuera del árbol.

**Orden interno importante, por si se toca este código de nuevo:** la ACL
se quita ANTES de calcular/aplicar el modo final, no después. Verificado
empíricamente (en un contenedor limpio, no en hall9000, donde el propio
umask del creador original enmascaró el defecto): cuando un objeto tiene ACL
extendida, el bit de "grupo" que `stat` muestra sin leer la ACL es la
MÁSCARA, no la entrada `group::` real -- `_mutar_directorio`/`_mutar_archivo`
sólo tocan la entrada NOMBRADA `g:fruiz:`, nunca `group::`, así que esa
entrada puede seguir arrastrando el valor que tenía al crearse el objeto
(`0644` si quien lo creó tenía umask `022`, como `jaxsvc` en producción). Si
el modo se calculara ANTES de quitar la ACL, `setfacl -b` (que corre
después) revelaría esa entrada vieja y pisaría el modo recién puesto.

## Verification
```bash
python3 ops/permisos_proyectos.py --verificar
```
Piso de CI medido en un contenedor `ubuntu:24.04` limpio (usuario `runner`
con sudo sin contraseña, sin `/etc/jax/.env` -- no en hall9000): **39
passed, 2 skipped**. Los dos skips son estructurales fuera de un host de jax
real (uno necesita `/etc/jax/.env` legible para probar el mensaje específico
de RAIZ-no-configurada; el otro es la prueba de lectura/escritura cruzada
contra `/home/fruiz/jax-workspace/proyectos` real).

## Fail-closed condition
Si `--verificar` da `1`, `proyectos/` sigue sin estar en el estado correcto.
Si `--aplicar` encuentra un hardlink, termina en `1` aunque haya corregido
todo lo demás. Si el respaldo forense no se puede validar (fsync, conteo de
entradas), `--aplicar` no aplica nada. Si el núcleo instalado no existe, no
coincide con HEAD, o su cadena de directorios no es segura, ni `--aplicar`
ni `--deshacer` corren.

## Recovery / escalation
```bash
python3 ops/permisos_proyectos.py --deshacer
```
(Ver "Reversión" arriba.) Si `--deshacer` reporta symlinks o hardlinks
saltados, esas rutas quedan sin tocar a propósito -- resolverlas a mano, con
Fernando, antes de asumir que el árbol volvió al estado anterior. El
respaldo en `/var/backups/jax-permisos/` queda como registro forense de qué
había antes de cada `--aplicar` -- útil para AUDITAR qué cambió, nunca para
reconstruirlo automáticamente. Si el estado quedó irreconocible, escalar a
Fernando antes de intentar nada más manual sobre `proyectos/`.

## Prohibited actions
- No correr `--aplicar` ni `--deshacer` contra el workspace real sin el GO
  explícito de Fernando.
- No usar el respaldo de `/var/backups/jax-permisos/` para reconstruir nada
  a mano (`setfacl --restore` o similar) -- es un registro forense, no un
  mecanismo de reversión; usar `--deshacer`.
- No mutar nada por ruta de texto "a mano".
- No cambiar `USUARIO`/`GRUPO`/`DUENO_ORIGINAL` en el guion sin actualizar
  el spec aprobado primero.
- No tocar nada fuera de `$RAIZ/proyectos`, ni las entradas en
  `NOMBRES_EXCLUIDOS` en su primer nivel.
- Nunca invocar `--nucleo-privilegiado`/`--nucleo-respaldo`/`--nucleo-deshacer`
  a mano, ni con ningún argumento: son el mecanismo interno con el que
  `--aplicar`/`--deshacer` re-ejecutan el núcleo instalado como root.
- No dejar que `ops/` (o cualquier ancestro del núcleo instalado) quede
  escribible por grupo u otros, ni que ningún ancestro sea un symlink.
- No instalar el núcleo desde el working tree sin commitear primero: la
  verificación compara contra HEAD, así que un working tree con cambios
  sueltos deja el núcleo instalado sin poder pasar esa comprobación (a
  propósito).
