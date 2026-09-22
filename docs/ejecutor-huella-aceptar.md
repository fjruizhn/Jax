# Aceptar una huella `reportada` del Ejecutor (M-1, ronda 7; corregido en ronda 8)

## Qué es esto

La huella del Ejecutor (`jax/ejecutor/contratos/huella.py`) mide, en cada máquina con
sudo, los controles propios del Ejecutor -- `/etc/sudoers`, `sudoers.d/*`,
`/etc/ssh/sshd_config`, `sshd_config.d/*`, `/etc/ssh/authorized_keys.d/*`,
`/root/.ssh/authorized_keys` y los binarios `/usr/local/sbin/ejecutor-*`. Nada de eso
debe cambiar NUNCA durante una misión. Si cambia, la huella pone la pausa del Ejecutor
y la marca de esa máquina/misión queda en estado `reportada`, con el diff de lo que
cambió guardado.

Una marca `reportada` **bloquea cualquier misión nueva en esa máquina** hasta que
alguien la acepte explícitamente -- a propósito: no se re-mide sola ni se destraba con
el tiempo. El único camino de salida es este.

## Por qué existe este camino

La ronda 6 quitó `/etc/passwd`/`group`/`shadow` de la huella (crear cuentas de sistema
es administración legítima), pero los seis controles que quedan (sudoers, sshd,
authorized_keys, binarios `ejecutor-*`) SÍ pueden cambiar por trabajo legítimo en casos
más raros -- y cuando eso pasa, alguien (Fernando) tiene que MIRAR el cambio y decidir,
no el sistema solo. Antes de esta ronda, una marca `reportada` no tenía salida
programada: había que editar archivos a mano en el controlador. Ahora hay un camino
único, auditable y con registro.

## Cómo aceptar

Desde hall9000, como `jaxsvc` -- **NO** como `fruiz` a secas, y **NO** como `root`
(M-1, ronda 8): el registro append-only de C3 (`/var/log/jax-ejecutor/registro.jsonl`)
y el árbol de misiones (`JAX_EJECUTOR_MISIONES`) son de `jaxsvc`; `fruiz` sólo tiene
`r--`/`r-x` sobre ellos por ACL (verificado en producción, `getfacl`, 2026-09-22) --
NO puede escribir ahí. `sudo python -m ...` tampoco sirve: escribiría como `root`, con
un dueño distinto al resto del árbol. El comando exacto:

```bash
set -a; . <(sudo -n cat /etc/jax/.env); set +a
sudo -u jaxsvc PYTHONPATH=.:las_manos JAX_EJECUTOR_MISIONES="$JAX_EJECUTOR_MISIONES" \
  JAX_EJECUTOR_ADMIN_USUARIO="$JAX_EJECUTOR_ADMIN_USUARIO" JAX_EJECUTOR_REGISTRO="$JAX_EJECUTOR_REGISTRO" \
  JAX_EJECUTOR_POLITICA="$JAX_EJECUTOR_POLITICA" JAX_EJECUTOR_PAUSA="$JAX_EJECUTOR_PAUSA" \
  JAX_EJECUTOR_CUENTA="$JAX_EJECUTOR_CUENTA" \
  python3 -m jax.ejecutor.contratos.huella aceptar \
  --host <nombre-de-la-maquina> --mision <mision_id>
```

`sudo -u jaxsvc` cambia la identidad del PROCESO, pero `SUDO_UID` sigue siendo el uid
de quien tecleó `sudo` (`fruiz`, o quien sea) -- por eso `aceptado_por` en la marca y en
el registro sigue siendo la persona real, no `jaxsvc` (ver M-2 más abajo). `axioma`
nunca puede correr esto: la CLI se niega si el uid invocante (resuelto por `SUDO_UID`)
es el de la cuenta configurada en `JAX_EJECUTOR_CUENTA`.

**Requisito nuevo en ronda 8:** la marca tiene que estar en estado `reportada` (no
`abierta` ni `cerrada`) para que `aceptar` haga algo -- si no, devuelve
`codigo=huella_no_reportada`. Es a propósito: no hay nada que aceptar si nunca se
comparó, o si ya está limpia.

Esto:

1. Muestra el diff completo de lo que cambió (léelo -- es la razón por la que este
   camino existe, no un trámite).
2. Toma una huella NUEVA de la máquina, AHORA MISMO, y la deja como la nueva línea
   base -- lo que hay en la máquina en este momento pasa a ser "lo normal" de ahí en
   adelante.
3. Registra la aceptación en el registro append-only de C3 (el mismo que audita cada
   paso del cerebro, cadena encadenada en hall9000): quién aceptó (resuelto de
   `SUDO_UID` vía `pwd`, nunca de `$SUDO_USER`/`$USER` como texto suelto), cuándo, el
   diff completo que se aceptó, y el motivo si vino con `--sin-medir` -- evento
   `huella_aceptada`.
4. Deja la marca de esa máquina/misión en estado `abierta` de nuevo: dejó de bloquear.
5. Borra la pausa del Ejecutor (`JAX_EJECUTOR_PAUSA`) -- **pero SOLO si es la pausa de
   ESTA huella** (mismo origen=`huella`, mismo host, misma misión; ronda 8, B-1). Si
   C4 o C5 pausaron por su cuenta, o pausó la huella de OTRO host/misión, esa pausa se
   deja intacta -- avisa con `codigo=pausa_de_otro_origen`, y el Ejecutor sigue frenado
   por lo que sea que la puso.

Con eso, el turno que sigue (de la misma misión) y cualquier misión nueva en esa
máquina vuelven a abrir (salvo que siga pausado por otro motivo, ver el punto 5).

## La máquina ya no existe, o no responde

Si la máquina de la marca `reportada` (o `abierta`, ronda 8: `--sin-medir` también
acepta una marca que nunca se llegó a comparar) se dio de baja, o no contesta por ssh
(y no vas a poder tomar una huella nueva de verdad), usa `--sin-medir` -- que desde
ronda 8 EXIGE `--motivo`, sin excepción:

```bash
sudo -u jaxsvc ... python3 -m jax.ejecutor.contratos.huella aceptar \
  --host <nombre> --mision <mision_id> --sin-medir --motivo "máquina dada de baja el 2026-09-22"
```

Hace lo mismo, pero SIN intentar tomar una huella nueva: la línea base que queda es la
misma que tenía antes de reportarse (no una medición fresca), y el registro de C3 deja
constancia de que se aceptó `sin_medir=true` junto con el `motivo` -- para que quien lea
el registro después sepa POR QUÉ esta aceptación no está respaldada por una medición
reciente de la máquina. Sin `--motivo`, la CLI se niega con `codigo=motivo_obligatorio`.

## Qué NO hace este camino

- No cambia nada en la máquina remota: sólo la MIDE (salvo `--sin-medir`, que ni eso).
- No acepta un cambio que todavía no pasó: si la marca está `abierta` (nunca se comparó)
  no hay nada que aceptar -- `aceptar` devuelve `codigo=huella_no_encontrada` si la
  marca no está en estado que tenga un diff guardado, o si simplemente no existe.
- No es axioma quien acepta: la CLI RECHAZA correr si el uid invocante (resuelto de
  `SUDO_UID`, validado con `pwd`) es el de la cuenta `JAX_EJECUTOR_CUENTA` -- no hay
  forma de que la propia jaula se autorice. LÍMITE declarado: un `root` arbitrario
  podría falsear `SUDO_UID` antes de invocar esto; eso no se puede cerrar desde acá.

## Verificar que quedó aceptada

```bash
cat "$JAX_EJECUTOR_MISIONES/<mision_id>/huella/<host>.json"   # estado=abierta, aceptada_por, aceptada_en
grep huella_aceptada "$JAX_EJECUTOR_REGISTRO" | tail -1        # el evento en el registro de C3
```
