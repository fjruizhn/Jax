# Aceptar una huella `reportada` del Ejecutor (M-1, ronda 7)

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

Desde hall9000, como `fruiz` (nunca como `axioma` -- la cuenta de la jaula no puede
aceptar sus propios cambios):

```bash
set -a; . <(sudo -n cat /etc/jax/.env); set +a
PYTHONPATH=.:las_manos python3 -m jax.ejecutor.contratos.huella aceptar \
  --host <nombre-de-la-maquina> --mision <mision_id>
```

Esto:

1. Muestra el diff completo de lo que cambió (léelo -- es la razón por la que este
   camino existe, no un trámite).
2. Toma una huella NUEVA de la máquina, AHORA MISMO, y la deja como la nueva línea
   base -- lo que hay en la máquina en este momento pasa a ser "lo normal" de ahí en
   adelante.
3. Registra la aceptación en el registro append-only de C3 (el mismo que audita cada
   paso del cerebro, cadena encadenada en hall9000): quién aceptó (`$SUDO_USER`/`$USER`),
   cuándo, y el diff completo que se aceptó -- evento `huella_aceptada`.
4. Deja la marca de esa máquina/misión en estado `abierta` de nuevo: dejó de bloquear.
5. Borra la pausa del Ejecutor (`JAX_EJECUTOR_PAUSA`) -- sin esto el paso 4 no alcanza:
   el arranque de la próxima misión (C5) sigue viendo la pausa puesta aunque la marca
   ya no bloquee.

Con eso, el turno que sigue (de la misma misión) y cualquier misión nueva en esa
máquina vuelven a abrir.

## La máquina ya no existe, o no responde

Si la máquina de la marca `reportada` se dio de baja, o no contesta por ssh (y no vas a
poder tomar una huella nueva de verdad), usa `--sin-medir`:

```bash
python3 -m jax.ejecutor.contratos.huella aceptar --host <nombre> --mision <mision_id> --sin-medir
```

Hace lo mismo, pero SIN intentar tomar una huella nueva: la línea base que queda es la
misma que tenía antes de reportarse (no una medición fresca), y el registro de C3 deja
constancia de que se aceptó `sin_medir=true` -- para que quien lea el registro después
sepa que esta aceptación NO está respaldada por una medición reciente de la máquina.

## Qué NO hace este camino

- No cambia nada en la máquina remota: sólo la MIDE (salvo `--sin-medir`, que ni eso).
- No acepta un cambio que todavía no pasó: si la marca está `abierta` (nunca se comparó)
  no hay nada que aceptar -- `aceptar` devuelve `codigo=huella_no_encontrada` si la
  marca no está en estado que tenga un diff guardado, o si simplemente no existe.
- No es axioma quien acepta: la CLI corre como `fruiz` (o quien sea `$SUDO_USER`), y eso
  es lo que queda en el registro -- no hay forma de que la propia jaula se autorice.

## Verificar que quedó aceptada

```bash
cat "$JAX_EJECUTOR_MISIONES/<mision_id>/huella/<host>.json"   # estado=abierta, aceptada_por, aceptada_en
grep huella_aceptada "$JAX_EJECUTOR_REGISTRO" | tail -1        # el evento en el registro de C3
```
