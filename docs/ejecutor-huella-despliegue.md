# Desplegar el arreglo de la huella (jax#260, ronda 2) — runbook

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

## El orden correcto (por qué importa)

**Primero TODAS las remotas, después el `.env`, recién entonces el Python.** Si se
invierte, hay una ventana donde el código nuevo ya está corriendo pero alguna remota
todavía no acepta la llave del servicio — mismo bloqueo que hoy, en otra máquina.

### Paso 1 — Instalar en TODAS las máquinas remotas del inventario

Hoy son **cuatro**: `atemai`, `bridge`, `ejecutor-prueba`, `prod` (verificado contra
`/etc/jax-ejecutor/politica.json` el 2026-09-22 — `hall9000` es la única `es_local` y
NUNCA es destino de este mecanismo; si el inventario cambió, confirmar con
`jq -r '.hosts[] | select(.es_local==false) | .nombre' /etc/jax-ejecutor/politica.json`).

```bash
set -a; . <(sudo -n cat /etc/jax/.env); set +a
export JAX_EJECUTOR_HUELLA_LLAVE=/etc/jax/controlador/id_ejecutor_huella
export JAX_EJECUTOR_HUELLA_KNOWN_HOSTS=/etc/jax/controlador/known_hosts_huella
export JAX_EJECUTOR_HUELLA_ORIGEN_IP=172.16.20.5

for maquina in atemai bridge ejecutor-prueba prod; do
  ./ops/ejecutor/instalar_huella_en_maquina.sh "$maquina"
done
```

Cada corrida imprime `maquina_huella_instalada="<nombre>" script_sha256="..."` (o
`maquina_local_sin_remoto=...` para un host local, que no debería salir acá). Si alguna
falla, **no seguir** — revisar el mensaje (`codigo=sshd_...`, `codigo=llave_huella_...`)
antes de tocar la siguiente máquina. El guion es idempotente: correrlo de nuevo sobre
una máquina ya instalada no duplica nada (MAJOR-3, ronda 2:
`huella.actualizar_authorized_keys_admin` converge).

**Verificar cada una** antes de seguir (el propio mecanismo, de punta a punta, sin tocar
código Python todavía):

```bash
ssh -F /dev/null -i "$JAX_EJECUTOR_HUELLA_LLAVE" -o BatchMode=yes -o IdentitiesOnly=yes \
  -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$JAX_EJECUTOR_HUELLA_KNOWN_HOSTS" \
  -p 58291 fruiz@<ip-de-la-maquina> ejecutor-huella
```

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
for maquina in atemai bridge ejecutor-prueba prod; do
  ./ops/ejecutor/revertir_huella_en_maquina.sh "$maquina"
done
```

Cada corrida sale con `maquina_huella_revertida="<nombre>" verificado=true` SÓLO si de
verdad no queda nada (MAJOR-4, ronda 2: el guion verifica que el script, el sudoers y la
línea de `authorized_keys` ya no estén, y sale con código ≠0 si algo sigue ahí — nunca
declara "revertida" sin comprobarlo). La llave del servicio
(`JAX_EJECUTOR_HUELLA_LLAVE`) y su `known_hosts` NO se borran por este camino: son
compartidos por todo el inventario; se borran a mano sólo si se da de baja el mecanismo
entero.
