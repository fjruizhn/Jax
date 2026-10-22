# Anti-fabricación capas 2 y 4: pre-registro de calibración

T0 (fecha y hora del sistema, `date -Is`): 2026-10-22T02:30:00-06:00

Este documento fija los umbrales ANTES de escribir código. Se commitea solo, antes
de la primera línea de la implementación. Cambiar un umbral después de ver datos
exige una enmienda fechada debajo, nunca editar esta tabla.

Alcance: capa 2 (detector de estructura determinista, paquete `jax/anti_fabricacion/`)
y capa 4 (consecuencias, cableado en `las_manos/motor_registry/worker.py`). Spec
cerrada de escalón 3; este pre-registro no la reabre.

## Umbrales

| # | Banco | Umbral | Fuente |
|---|---|---|---|
| U1 | 41 datos correctos de U3 V2 | 0 marcados | `scripts/ejecutor_fase2/reproducir_u3.py` |
| U2 | Salidas aprobadas: 32 `motor_results` + ramas de Qwen Code aprobadas por escalón 3 | <= 5 % marcadas | bancos offline |
| U3 | Prueba de inversión | tasa de marcados del motor que más propone <= 1,5 x la tasa de cada uno de los demás | los mismos bancos de U2 por motor |
| U4 | Casos plantados estructurales | >= 9 de 10 detectados | `/services/bodegas`, `tabla.columna` falsa, Protocol inventado del brief F1.1, diff `--- a/` sobre archivo inexistente |
| U5 | Modo sombra | >= 30 salidas reales; cada marca clasificada por un escalón 3 distinto del que implementa | producción en sombra |
| U6 | Plazo de sombra | si a 21 días de activar la sombra no hay 30 salidas, se avisa a Fernando | |
| U7 | Alerta | disparada una vez a propósito antes de pasar a duro | |

## Reglas de lectura

- Pasar de sombra a duro lo decide Fernando, no la sesión que implementa.
- "Marcado" es el estado `marcado` del veredicto. `no_verificable` y `heredado` se
  cuentan aparte y no cuentan como marcado ni como limpio.
- Un banco que no exista offline se reporta como "no medido", nunca como cumplido.
- U1, U2, U3 y U4 se miden con `scripts/anti_fabricacion/calibrar.py`; U5 a U7 solo
  existen en operación y no se pueden declarar cumplidos desde un entorno offline.
- Modo por defecto ante config ausente, ilegible o desconocida: `duro`.
