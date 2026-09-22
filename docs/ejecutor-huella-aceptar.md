# Aceptar una huella `reportada` del Ejecutor (M-1, ronda 7; corregido en ronda 8)

## Qué es esto

La huella del Ejecutor (`jax/ejecutor/contratos/huella.py`) mide, en cada máquina con
sudo, los controles propios del Ejecutor -- `/etc/sudoers`, `sudoers.d/*`,
`/etc/ssh/sshd_config`, `sshd_config.d/*`, `/etc/ssh/authorized_keys.d/*`,
`/root/.ssh/authorized_keys`, el `authorized_keys` del administrador (LÍMITE 9, ronda 2:
donde vive el acceso privilegiado real, y donde el arreglo de jax#260 puso la llave
propia del servicio -- no medirlo dejaría ese mismo cambio invisible a la huella) y los
binarios `/usr/local/sbin/ejecutor-*`. Nada de eso debe cambiar NUNCA durante una
misión. Si cambia, la huella pone la pausa del Ejecutor y la marca de esa
máquina/misión queda en estado `reportada`, con el diff de lo que cambió guardado.

**Ronda 4 (auditoría adversarial 2026-09-22):** cada ruta que se mide reporta
exactamente uno de cuatro estados: un hash (existe, es un archivo), `D <ruta>`
(existe, es un directorio, con su listado), `A <ruta>` (medida y CONFIRMADA ausente)
o `E <ruta> <motivo>` (no se pudo medir -- esto SÍ bloquea). Antes, "ausente" y "no
medible" daban lo mismo (cero líneas) y la huella trataba las dos como inválidas,
bloqueando el Ejecutor en una máquina SANA.

**MAJOR-G (ronda 5, auditoría adversarial 2026-09-22): hecho corregido.** El párrafo
de arriba, en la ronda 4, decía que `/root/.ssh/authorized_keys` "no existe en
hall9000, atemai NI prod" -- **ERA FALSO**. Medido el 2026-09-22: esa ruta **SÍ
existe** (archivo vacío, 0600, dueño root) en hall9000, atemai y prod; **sólo falta en
`bridge`**. El arreglo de la ronda 4 sigue haciendo falta igual -- `bridge` es la
máquina SANA donde de verdad da `A`, y la ronda 3 la bloqueaba ahí como si la
medición hubiera fallado -- pero el hecho que se citaba como ejemplo estaba mal.

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
un dueño distinto al resto del árbol.

**Desde el arreglo del bug de producción (jax#260, ronda 2, 2026-09-22):** esta CLI
también toma una huella nueva de la máquina (a menos que se use `--sin-medir`) -- y para
eso necesita, además, `JAX_EJECUTOR_HUELLA_LLAVE` y `JAX_EJECUTOR_HUELLA_KNOWN_HOSTS` (la
llave PROPIA del servicio, no la personal de `fruiz` -- ver
`jax/ejecutor/contratos/huella.py::argv_huella_servicio` y
`docs/ejecutor-huella-despliegue.md`). Las dos se exigen SIEMPRE, aunque la corrida use
`--sin-medir`: mismo criterio que ya regía para `JAX_EJECUTOR_POLITICA`/
`JAX_EJECUTOR_ADMIN_USUARIO`. El comando exacto:

```bash
set -a; . <(sudo -n cat /etc/jax/.env); set +a
sudo -u jaxsvc PYTHONPATH=.:las_manos JAX_EJECUTOR_MISIONES="$JAX_EJECUTOR_MISIONES" \
  JAX_EJECUTOR_ADMIN_USUARIO="$JAX_EJECUTOR_ADMIN_USUARIO" JAX_EJECUTOR_REGISTRO="$JAX_EJECUTOR_REGISTRO" \
  JAX_EJECUTOR_POLITICA="$JAX_EJECUTOR_POLITICA" JAX_EJECUTOR_PAUSA="$JAX_EJECUTOR_PAUSA" \
  JAX_EJECUTOR_CUENTA="$JAX_EJECUTOR_CUENTA" \
  JAX_EJECUTOR_HUELLA_LLAVE="$JAX_EJECUTOR_HUELLA_LLAVE" \
  JAX_EJECUTOR_HUELLA_KNOWN_HOSTS="$JAX_EJECUTOR_HUELLA_KNOWN_HOSTS" \
  python3 -m jax.ejecutor.contratos.huella aceptar \
  --host <nombre-de-la-maquina> --mision <mision_id>
```

`sudo -u jaxsvc` cambia la identidad del PROCESO, pero `SUDO_UID` sigue siendo el uid
que `sudo` DECLARA para quien lo invocó (`fruiz`, o quien sea) -- por eso `aceptado_por`
en la marca y en el registro sigue mostrando a esa persona, no `jaxsvc`. Ojo: es un dato
DECLARADO, no verificado por este proceso -- el registro lo etiqueta
`identidad_declarada_por` (ver M-2 más abajo). Hoy, en hall9000, `axioma` directamente
no puede correr esto de punta a punta: no tiene sudo hacia `jaxsvc` (se le quitó la
noche del 2026-09-22). La CLI ADEMÁS se niega si el uid invocante declarado es el de
`JAX_EJECUTOR_CUENTA` -- freno del error accidental, no del que de verdad tiene acceso.

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
- No es un chequeo de identidad quien evita que axioma acepte su propia huella --
  ES el acceso: `axioma` no tiene sudo hacia `jaxsvc` en hall9000 (se le quitó la
  noche del 2026-09-22, `sudo -l -U axioma` → no permitido), y sin eso no puede
  siquiera escribir el registro o la marca. La CLI ADEMÁS rechaza correr si el uid
  invocante (declarado por `SUDO_UID`, resuelto con `pwd`) es el de
  `JAX_EJECUTOR_CUENTA` (ronda 9, MAJOR-2) -- pero eso es protección contra el ERROR
  ACCIDENTAL, no una barrera anti-suplantación: `SUDO_UID` es un dato que el entorno
  DECLARA, no una identidad que este proceso verifica. LÍMITE, sin cerrar: cualquiera
  con una regla `ALL` (puede correr como CUALQUIER usuario, no hace falta ser root)
  puede encadenar `sudo -u <alguien> sudo -u jaxsvc ...` y hacer que `SUDO_UID`
  declare el uid de `<alguien>` en vez del propio.

## Verificar que quedó aceptada

```bash
cat "$JAX_EJECUTOR_MISIONES/<mision_id>/huella/<host>.json"   # estado=abierta, aceptada_por, aceptada_en
grep huella_aceptada "$JAX_EJECUTOR_REGISTRO" | tail -1        # el evento en el registro de C3
```
