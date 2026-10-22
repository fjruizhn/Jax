# Anti-fabricación: activación del detector de estructura (capas 2 y 4)

Código: `jax/anti_fabricacion/` (symlink `las_manos/anti_fabricacion`), cableado en
`las_manos/motor_registry/worker.py`. Pre-registro y umbrales:
`docs/superpowers/specs/2026-10-22-anti-fabricacion-preregistro.md`.

## Orden (no negociable)

1. **Antes de reiniciar LAS MANOS con este código**, insertar en `axioma_config` la clave
   `anti_fabricacion.modo='sombra'`. Sin la clave el modo es `duro` por omisión (falla
   cerrado) y cada falso positivo falla un job de producción.
   - Simular: `python3 scripts/anti_fabricacion/sembrar_modo.py`
   - Aplicar (una persona con GO; `INSERT IGNORE`, nunca pisa un valor existente):
     `python3 scripts/anti_fabricacion/sembrar_modo.py --aplicar`
   - Comprobar: `python3 scripts/anti_fabricacion/sembrar_modo.py --verificar` (0 = bien).
2. Reiniciar LAS MANOS. Si la clave faltara igual, el primer job de cada proceso registra un
   ERROR y manda un aviso por Telegram: «el modo salió 'duro' por omisión». No es silencioso.
3. Fase **sombra**: el estado de los jobs no cambia; cada job guarda `_estructura` y
   `_unbacked_claims` en su JSONL, y cada marca deja un evento `ESTRUCTURA_SIN_RESPALDO` en
   `audit.jsonl` y un aviso inmediato. Hacen falta 30 salidas reales nuevas con veredicto
   (las de `cuenta_para_sombra=false`, detector caído, no cuentan), cada marca clasificada por
   un escalón 3 distinto del que implementó. Si a los 21 días no hay 30, avisar a Fernando.
4. **Pasar a `duro` lo decide Fernando tras U5** (y tras disparar la alerta una vez a propósito,
   U7). Cambiar el valor de la clave es la única acción; no hace falta reiniciar (se lee por job).

## Qué mirar en la sombra

- `usos_nuevo` y `usos_hipotesis` (en `_estructura` y en el evento): cuántas veces una salida se
  escapó por `nuevo:` o por una oración con «?». Un modelo que aprende a abusarlos se ve ahí.
- Categorías `no_verificable` frecuentes (`archivos`, `jax_checkout`, `systemd`, `heredado`,
  `tamano`): una fuente opcional caída o un límite declarado.
- Avisos de «detector no disponible»: un aviso por motivo cada 15 minutos
  (`MOTOR_ANTI_FABRICACION_VENTANA_AVISO_S`).

## Variables de entorno

- `JAX_ANTI_FAB_JAX_CHECKOUT`: checkout desplegado de jax (solo `git ls-files`); por defecto
  `/srv/jax-prod/jax`.
- `JAX_AUDIT_LOG_PATH`: la misma de LAS MANOS (se lee al emitir el evento).

## Reversa

Para apagar el efecto sin quitar código: dejar la clave en `sombra`. El detector caído en
`sombra` no falla jobs. No existe un modo «apagado»: sacar el detector es revertir el PR.
