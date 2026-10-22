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

## Enmiendas (fechadas; los umbrales de arriba NO se tocan)

- 2026-10-22 (primera corrida de `scripts/anti_fabricacion/calibrar.py`): U1 dio 1 respuesta
  final marcada de 5 (tarea 8, de clientes) por un dotfile suelto entre backticks (`.xxx.yyyy`)
  tratado como ruta inexistente. Se ajustó el DETECTOR (un nombre oculto sin directorio no es
  candidato; prueba `test_nombre_oculto_sin_directorio_no_es_candidato`, roja antes del ajuste),
  no el umbral. Segunda corrida: U1 = 0 marcados en los 41 datos y en las 5 respuestas finales.
- 2026-10-22 (aclaración de lectura, antes de medir nada con `heredado`): en las "Reglas de
  lectura" de arriba se dijo que `heredado` no cuenta como marcado. La implementación lo trata
  como `marcado` con `clase=heredado` (la salida de otro modelo nunca respalda, así que un
  candidato sin respaldo cuyo único testigo es un paso previo sigue sin respaldo; "no limpio").
  Vale la implementación; para U1 a U4 un `heredado` cuenta como marcado. Los bancos offline no
  traen pasos previos, así que ninguna cifra de arriba cambia por esto.

## Enmienda 1, 2026-10-22 (decisiones del escalón 3 tras ver los datos)

Causa. La primera calibración dio U2 = 5/32 marcadas (15,6 %); descontando 23 JSON de
procesamiento de archivos (315 a 373 bytes, no son salida de modelo), son 5/9 salidas reales
(56 %). Dos causas: (a) rutas que existen en disco pero `git ls-files` no las ve (8 archivos
versionados contra 370 en disco); (b) identificadores snake_case sueltos en backticks de
prosa de diseño (propuestas). Los UMBRALES de la tabla NO cambian. U2 pasa a ser
CALIBRACIÓN (la regla cambió tras ver los datos): ya no valida nada por sí sola; la validación
es la fase sombra (30 salidas reales NUEVAS). U4 lo replantea alguien distinto del
implementador, a ciegas para éste; el conjunto actual se reporta solo como humo.

Reglas.
1. Raíces. Workspace = `git ls-files` + árbol en disco visible con el mismo encierro de
   `tool_authority` (nada fuera de WORKSPACE_ROOT, sin seguir enlaces); exclusiones en UNA
   constante compartida (.git, .claude-flow, node_modules, .venv, __pycache__); tope de 20 000
   entradas: si se pasa, la categoría archivos es `no_verificable` (nunca limpio). Los no
   versionados aportan SOLO la ruta; los símbolos (ast) solo de versionados. El checkout de jax
   es fuente OPCIONAL: solo `git ls-files` del SHA desplegado en /srv/jax-prod/jax (sin
   recorrer disco).
2. Identificadores. Los campos que PROPONEN nunca se revisan. En campos que afirman y en texto
   libre solo hay 4 formas candidatas: rutas con «/», `tabla.columna` con izquierda tabla real,
   `*.service|timer`, y las rutas de `files_modified` y «--- a/». Un identificador suelto entre
   backticks (sin «/» ni «.») deja de ser candidato. Límite declarado: «`bodegas` es una
   tabla» no se atrapa; esos casos de U4 quedan como «fuera de alcance».
3. `heredado` = `no_verificable` (visible, nunca limpio, no falla en duro) SOLO si el job de
   origen tiene veredicto propio guardado; si no hay origen o no tiene veredicto, es `marcado`.
   (Reemplaza la aclaración anterior de esta sección.)
4. Detector caído en sombra: el job continúa con `no_corrio` y hay aviso inmediato por
   Telegram; esa salida no cuenta entre las 30. En duro: FAILED antes de llamar al modelo.
   Esto corrige el texto de la spec.
5. `objective` no llega a /motor/dispatch: límite declarado, fuera de este PR; va a DEUDA.md.
6. La huella de `git status` excluye la misma constante de exclusiones.
