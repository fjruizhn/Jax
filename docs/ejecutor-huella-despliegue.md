# Desplegar el arreglo de la huella (jax#260, rondas 2-3) — runbook

## Qué es esto

`jax#260` (mergeado) introdujo la huella de controles del Ejecutor tomándola con la
llave PERSONAL del administrador (`revocacion.argv_admin`, sin `-i`, resolución de
identidad por default de ssh). Desde el 2026-09-17 `jax-platform.service` corre como
`jaxsvc` (cuenta de servicio, drop-in `cuenta-de-servicio.conf`), y `jaxsvc` no puede
leer `~fruiz/.ssh/*` — medido en producción con
`scripts/ejecutor_contratos/mision_de_humo.py`: `vigia_no_latio=true rc=2`, el Ejecutor
bloqueado por completo.

**BLOCK-1 de la ronda 2 de auditoría:** las variables que este arreglo necesita
(`JAX_EJECUTOR_HUELLA_LLAVE`, `JAX_EJECUTOR_HUELLA_KNOWN_HOSTS`,
`JAX_EJECUTOR_HUELLA_ORIGEN_IP`) **no existen todavía en `/etc/jax/.env`**. Desplegar el
código SIN antes haber instalado el camino nuevo en las remotas deja el Ejecutor **igual
de bloqueado** — sólo cambia el mensaje de error (de `Permission denied` con la llave
vieja a `Permission denied`/`command not found` con la llave nueva, porque las remotas
todavía no la conocen). El orden de abajo evita esa ventana.

## Los tres valores exactos a sembrar en `/etc/jax/.env`

Fernando los agrega a mano (este documento NO edita `/etc/jax/.env` — es producción, y
esa edición es de él):

```bash
JAX_EJECUTOR_HUELLA_LLAVE=/etc/jax/controlador/id_ejecutor_huella
JAX_EJECUTOR_HUELLA_KNOWN_HOSTS=/etc/jax/controlador/known_hosts_huella
JAX_EJECUTOR_HUELLA_ORIGEN_IP=172.16.20.5
```

- `JAX_EJECUTOR_HUELLA_LLAVE`/`_KNOWN_HOSTS`: bajo `/etc/jax/controlador/`, que YA es
  `jaxsvc:jaxsvc 700` (el mismo directorio de `JAX_EJECUTOR_CONTROLADOR_LLAVE`, un PAR
  de llaves DISTINTO — ver `jax/ejecutor/contratos/huella.py`). Los crea, la primera vez
  que se corre, `ops/ejecutor/instalar_huella_en_maquina.sh` (paso 1).
- `JAX_EJECUTOR_HUELLA_ORIGEN_IP=172.16.20.5`: la IP LAN de hall9000 (`br0`, verificada
  con `ip -4 -o addr show` el 2026-09-22 — la misma que documenta `CLAUDE.md`). Es el
  `from=` que la línea de `authorized_keys` de cada remota exige: sólo conexiones desde
  esa IP pueden usar la llave del servicio.

**No hace falta agregar nada más a mano** (MAJOR-4, ronda 3): el instalador escribe
además, EN CADA REMOTA, `/etc/ejecutor-huella/admin_usuario` (root 0644, una sola línea
con `$JAX_EJECUTOR_ADMIN_USUARIO`) -- es lo que `ejecutor-huella` lee para saber de
quién es el `authorized_keys` a medir, en vez de tener la cuenta hardcodeada en el
guion. No es una variable de `/etc/jax/.env`: vive en cada remota, la escribe el Paso 1.

## El orden correcto (por qué importa)

**Primero TODAS las remotas, después el `.env`, recién entonces el Python.** Si se
invierte, hay una ventana donde el código nuevo ya está corriendo pero alguna remota
todavía no acepta la llave del servicio — mismo bloqueo que hoy, en otra máquina.

### Paso 1 — Instalar en TODAS las máquinas remotas del inventario

Hoy son **cuatro**: `atemai`, `bridge`, `ejecutor-prueba`, `prod` (verificado contra
`/etc/jax-ejecutor/politica.json` el 2026-09-22 — `hall9000` es la única `es_local` y
NUNCA es destino de este mecanismo; si el inventario cambió, confirmar con
`jq -r '.hosts[] | select(.es_local==false) | .nombre' /etc/jax-ejecutor/politica.json`).

**ADVERTENCIA (ronda 3, MINOR):** instalar CAMBIA la huella de cada máquina (agrega el
script, el sudoers.d y la línea de `authorized_keys` -- las tres, parte de lo que la
huella mide). Si hay una MISIÓN EN VUELO contra esa máquina en ese momento, el próximo
cierre de turno ve ese cambio como `huella_cambio_no_declarado` y PAUSA el Ejecutor --
hay que aceptar esa huella después (`docs/ejecutor-huella-aceptar.md`). Instalar con el
Ejecutor SIN misiones activas contra esa máquina evita el trámite.

```bash
set -euo pipefail  # MINOR (ronda 3): sin esto, un fallo en una máquina no corta el
                    # bucle -- las siguientes se instalarían igual, con la anterior a
                    # medias. `set -e` hace que el `for` se corte en la PRIMERA que falle.
set -a; . <(sudo -n cat /etc/jax/.env); set +a
export JAX_EJECUTOR_HUELLA_LLAVE=/etc/jax/controlador/id_ejecutor_huella
export JAX_EJECUTOR_HUELLA_KNOWN_HOSTS=/etc/jax/controlador/known_hosts_huella
export JAX_EJECUTOR_HUELLA_ORIGEN_IP=172.16.20.5

for maquina in atemai bridge ejecutor-prueba prod; do
  ./ops/ejecutor/instalar_huella_en_maquina.sh "$maquina"
done
```

Cada corrida imprime `maquina_huella_instalada="<nombre>" script_sha256="..."` (o
`maquina_local_sin_remoto=...` para un host local, que no debería salir acá). Con
`set -euo pipefail` puesto arriba, un fallo CORTA el bucle ahí mismo -- no hace falta
"no seguir" a mano; revisar el mensaje (`codigo=sshd_...`, `codigo=llave_huella_...`)
antes de reintentar. El guion es idempotente: correrlo de nuevo sobre una máquina ya
instalada no duplica nada (MAJOR-3, ronda 2/3: `huella.actualizar_authorized_keys_admin`
converge por MARCA, no por la clave pública actual -- sigue funcionando aunque la llave
del servicio se haya regenerado entre medio).

**Verificar cada una** antes de seguir (el propio mecanismo, de punta a punta, sin tocar
código Python todavía).

**BLOCK-1 de la ronda 3 de auditoría (corregido acá):** este paso, tal como estaba
escrito, mandaba a `fruiz` a leer directamente `$JAX_EJECUTOR_HUELLA_LLAVE` -- pero esa
llave es `jaxsvc:jaxsvc 700`/`600` (paso 1 la crea con ese dueño a propósito, para que
`jax-platform.service`, que corre como `jaxsvc`, pueda leerla). `fruiz` SOLO, sin
`sudo -u jaxsvc`, se encuentra con `Permission denied` al abrir el ARCHIVO DE LA LLAVE
-- antes de siquiera intentar conectar -- y el runbook decía "no seguir" ante ese mismo
mensaje que él mismo provocaba. Verificado el 2026-09-22 (sin tocar la llave real ni
ninguna máquina remota): un archivo de prueba `jaxsvc:jaxsvc 700/600` da
`Permission denied` leído como `fruiz` a secas, y `sudo -u jaxsvc cat` lo lee bien --
misma identidad que ya usa `docs/ejecutor-huella-aceptar.md` para la CLI de aceptación.
El comando correcto:

```bash
sudo -u jaxsvc ssh -F /dev/null -i "$JAX_EJECUTOR_HUELLA_LLAVE" -o BatchMode=yes \
  -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile="$JAX_EJECUTOR_HUELLA_KNOWN_HOSTS" \
  -p 58291 fruiz@<ip-de-la-maquina> ejecutor-huella
```

(`sudo -u jaxsvc` necesita que las variables `JAX_EJECUTOR_HUELLA_*` lleguen al entorno
de jaxsvc -- si se corre en la MISMA terminal donde ya se hizo `export` más arriba,
`sudo -u jaxsvc` con la sesión de `fruiz` activa las hereda; si no, exportarlas de nuevo
antes de este comando.)

Tiene que imprimir varias líneas `<sha256>  <ruta>` / `D <ruta>` / `L <ruta> -> <destino>`
ordenadas — la huella real de esa máquina. Una salida vacía, un `Permission denied` o un
`command not found` significa que el paso 1 no terminó bien en esa máquina: no seguir.

### Paso 2 — Sembrar `/etc/jax/.env` (lo hace Fernando)

Con **las cuatro** máquinas verificadas, agregar las tres líneas de arriba a
`/etc/jax/.env` y reiniciar `jax-platform.service` (`EnvironmentFile=/etc/jax/.env` sólo
se relee al arrancar el proceso):

```bash
sudo systemctl restart jax-platform.service
```

**ADVERTENCIA (ronda 3, MINOR):** este reinicio CORTA cualquier misión del Ejecutor que
esté en vuelo en ese momento (el vigía es un subproceso de `jax-platform`, muere con
él -- el turno en curso queda a medias, y `vigia_servicio.py` lo trata como `SIGKILL`:
la marca de huella queda `ABIERTA`, no `CERRADA`, y la próxima misión la revisa antes de
abrir). Programar este paso para un momento SIN misión activa, igual que el paso 1.

### Paso 3 — Desplegar/mergear el código Python (esta rama)

Recién ahora. Con las cuatro remotas ya aceptando la llave del servicio Y
`jax-platform.service` ya reiniciado con las tres variables nuevas en su entorno, el
código que llama a `huella.argv_huella_servicio` encuentra todo lo que necesita desde el
primer turno.

### Verificación final

```bash
scripts/ejecutor_contratos/mision_de_humo.py ejecutor-prueba
```

Tiene que dar `arranco=true` sin `vigia_no_latio`/`huella_no_medible`/
`huella_cambio_no_declarado` en el resultado. Si la máquina de humo (`ejecutor-prueba`)
no estaba en el paso 1 por algún motivo, el vigía bloquea la misión ANTES de arrancar
(fail-closed) — es la señal de que faltó un paso, no un defecto del arreglo.

## Reversión (si algo sale mal)

Por máquina, en orden inverso (después de haber quitado las tres variables de
`/etc/jax/.env` y reiniciado `jax-platform.service`, para que el código deje de intentar
usar el camino nuevo):

```bash
set -euo pipefail  # MINOR (ronda 3): corta en la primera máquina que falle -- ver Paso 1.
for maquina in atemai bridge ejecutor-prueba prod; do
  ./ops/ejecutor/revertir_huella_en_maquina.sh "$maquina"
done
```

Cada corrida sale con `maquina_huella_revertida="<nombre>" verificado=true` SÓLO si de
verdad no queda nada (MAJOR-4, ronda 2: el guion verifica que el script, el sudoers y la
línea de `authorized_keys` ya no estén, y sale con código ≠0 si algo sigue ahí — nunca
declara "revertida" sin comprobarlo). MAJOR-3 (ronda 3): la reversión filtra y verifica
por la MARCA (`ejecutor-huella-servicio`), no por la clave pública actual -- funciona
aunque la llave del servicio se haya regenerado entre la instalación y la reversión. La
llave del servicio (`JAX_EJECUTOR_HUELLA_LLAVE`), su `known_hosts` y
`/etc/ejecutor-huella/admin_usuario` NO se borran por este camino: son compartidos por
todo el inventario (los dos primeros) o siguen haciendo falta mientras el script exista
(el tercero); se borran a mano sólo si se da de baja el mecanismo entero.
