# Emisor de sub-pipelines de Ada — diseño (frente G)

> Fecha: 2026-09-16 · Autor: Mr. Hyde · Diseño aprobado por Fernando por secciones (chat,
> 2026-09-16). Depende del frente F (`2026-09-16-hallazgos-auditoria-jax-design.md` §F y su
> plan con la ENMIENDA: profundidad por defecto 3; padre activo = pipeline padre `running`,
> token atado al step de Ada que delegó en cualquier estado). Reglas comunes: spec de
> jax-platform `2026-09-16-hallazgos-auditoria-design.md` §0.

## 1. Propósito

Ada (faceta arquitecta) está en el sistema por su capacidad multiagente. Dos usos, ambos
requeridos por Fernando: **descomponer y repartir** un objetivo grande entre facetas, y
**enjambres de código** (Ada diseña, Kimi implementa en paralelo, Ada revisa).
Autonomía: **Ada lanza sola dentro de techos holgados; lo que los excede pide aprobación
de Fernando y le notifica.** Nunca rechazo silencioso.

## 2. Dos modos permanentes (no uno provisional)

| Modo | Cuándo sirve | Requisito |
|---|---|---|
| **1. Plan de delegación** | Descomponer primero: el reparto se decide al inicio | Ninguno nuevo sobre Ada (sigue por HTTP directo) |
| **2. Tools** | Reaccionar en marcha: Ada descubre a mitad de trabajo que algo necesita un enjambre | Ada migrada al camino gobernado (Motor Registry + bucle GAP2) **y** medición real de tool-calling de GLM que pase |

Orden de construcción: modo 1 → migración de Ada + medición → modo 2. Si GLM no pasa la
medición, el modo 1 queda operativo y el 2 espera a un modelo que pase (se registra con fecha
de revisión). Los dos comparten contrato (F), techos, cola, auditoría, kill switch y aprobación.

## 3. Flujo

### 3.1 Modo 1
1. Step de Ada con capability **`delegate`** (fila nueva en `capability`,
   `allowed_callers` incluye `jacobs`, `output_schema = plan_delegacion.v1`).
2. `output_validator` valida **estrictamente** `plan_delegacion.v1` (sale de
   `_KNOWN_UNIMPLEMENTED_SCHEMAS`):
   `{"subobjetivos": [{"id", "objetivo", "faceta", "capability", "depende_de": [ids], "skip_on_fail": bool}], "integracion": {"objetivo"}}`.
   Faceta validada contra la tabla `facet`; capability contra `capability`; `depende_de` sin
   ciclos y solo a ids del mismo plan.
3. **`jacobs/delegacion.py`** (único punto de decisión):
   - carga el árbol del padre (`root_pipeline_id`, profundidad, pipelines del árbol, consumo);
   - proyecta: profundidad del hijo, hijos de este step, pipelines del árbol tras lanzar,
     tokens = consumido + `JAX_ADA_RESERVA_TOKENS_HIJO` × hijos nuevos, USD = consumido con precio;
   - **dentro de techos** → por cada sub-objetivo, `emitir_token_subpipeline(padre, step)` (F)
     y creación del hijo con `invoked_by="ada"` + token por el `create_pipeline` endurecido;
   - **excede algún techo** → padre a `awaiting_approval`, evento `DELEGACION_EXCEDE_TECHO`
     (qué techo, valor proyectado, límite), notificación Telegram; no se emite ningún token.
4. **Cola:** un hijo que supera `MAX_PARALLEL_PIPELINES` (global) queda **`queued`**; un
   despachador (mismo loop del reaper, cadencia configurable) admite en orden FIFO por
   `queued_at` cuando hay lugar. Espera máxima `JAX_ADA_COLA_MAX_ESPERA_SEGUNDOS` → `expired`
   con evento.
5. **Integración:** cuando todos los hijos terminaron (completados, fallados declarados o
   saltados), corre un step de Ada con capability **`integrate`** que recibe, por hijo:
   id, sub-objetivo, estado, resumen acotado y referencia a su `output_ref`.

### 3.2 Modo 2
- Tools `lanzar_subpipeline(objetivo, faceta, capability, depende_de, skip_on_fail)` y
  `leer_resultado_subpipeline(pipeline_id)` en el catálogo de tools de GAP2.
- `tool_authority` valida cada llamada invocando **la misma** `delegacion.py` (mismos techos,
  mismo token, misma cola). `leer_resultado_subpipeline` solo lee hijos de su propio árbol.
- Prerrequisitos: (a) extender la traducción de `_CAPABILITY_MAP` al camino de Ada
  (diagnóstico ronda 8: `analysis`→`pipeline_analysis`, `review`→`refactor`); (b) mover `ada`
  a `MOTOR_FACETS` auditando cada capability que usa hoy contra los 8 checks de
  `MotorPolicy`; (c) `motor.has_tool_access=TRUE` para ada; (d) **medición** (§6).

### 3.3 Invariantes
- jax-platform nunca emite tokens ni fija profundidad (F lo prueba).
- El kill switch corta el árbol entero (frente B): Jacobs no admite hijos ni despacha la cola
  con el freno puesto.

## 4. Datos

- `jacobs_pipelines`: `parent_pipeline_id` y `depth` (F) + **`root_pipeline_id`** (índice) +
  estado **`queued`** + `queued_at`.
- `jacobs_subpipeline_tokens`: tabla del frente F.
- **`jacobs_arbol_consumo`** (PK `root_pipeline_id`): `pipelines_total`, `tokens_total`,
  `costo_usd_total`, `tokens_sin_precio`, `techo_tokens_aprobado` NULL,
  `techo_usd_aprobado` NULL, `actualizado_at`. Se actualiza **en la misma transacción** que
  registra el uso de cada step (writers de uso de jacobs y motor_registry), así el techo se
  compara contra un número vivo.
- Techos en `/etc/jax/.env`, validados al arrancar (fail-closed, rango declarado):
  `JAX_ADA_MAX_DEPTH=3` (= `JAX_MAX_SUBPIPELINE_DEPTH` de F; una sola fuente: el emisor lee
  la de F), `JAX_ADA_MAX_HIJOS_POR_STEP=10`, `JAX_ADA_MAX_PIPELINES_ARBOL=40`,
  `JAX_ADA_MAX_TOKENS_ARBOL=2000000`, `JAX_ADA_MAX_USD_ARBOL=10`,
  `JAX_ADA_RESERVA_TOKENS_HIJO=50000`, `JAX_ADA_COLA_MAX_ESPERA_SEGUNDOS=3600`.
- Eventos (`jacobs_events`): `DELEGACION_PROPUESTA`, `DELEGACION_APROBADA`
  (`por: auto|fernando`), `DELEGACION_EXCEDE_TECHO`, `DELEGACION_TECHO_CRUZADO`,
  `SUBPIPELINE_ENCOLADO`, `SUBPIPELINE_ADMITIDO`, `SUBPIPELINE_EXPIRADO_EN_COLA`,
  más los del contrato F.

## 5. Techos en acción, errores y aprobación

- **Cruce a mitad de camino:** si `jacobs_arbol_consumo` cruza tokens o USD, los hijos aún no
  arrancados pasan a `awaiting_approval` (evento `DELEGACION_TECHO_CRUZADO` + Telegram); lo que
  ya corre termina su step.
- **Aprobación de Fernando:** vía el flujo de aprobación de Jacobs expuesto en el admin;
  aprobar fija `techo_*_aprobado` **solo para ese árbol**, auditado con usuario y hora.
- **Plan inválido:** un reintento con el error explícito; si falla otra vez, step `failed`
  sin hijos. Nunca se lanza un plan parcial.
- **Hijo fallado:** respeta `skip_on_fail`; sin él, sus dependientes no arrancan y la
  integración recibe el fallo declarado.
- **Padre abortado** (kill switch, cancelación, reaper): hijos `pending`/`queued` →
  `cancelled`; los que corren abortan por el freno; ningún huérfano sigue gastando.
- **DB no disponible al medir techos:** fail-closed, no se delega (evento si se puede escribir).

## 6. Pruebas

- Contrato F: su matriz de ataques.
- Emisor: plan válido → N hijos con token; cada techo excedido → aprobación y cero tokens;
  aprobación → lanza; cruce a mitad de camino → frena lo pendiente; padre abortado → sin
  huérfanos; cola respeta el límite global y admite FIFO; espera máxima → `expired`; plan
  inválido → reintento y fallo declarado; ciclo en `depende_de` → rechazo.
- Carga (DB de test): creación de hijos y despachador de cola con p95 registrado.
- **En vivo con GO de Fernando (gasta):** árbol real chico (Ada delega 3 sub-objetivos
  baratos) con techos bajados a propósito para forzar una aprobación.
- **Modo 2:** medición de tool-calling de GLM con GO (5 llamadas reales al formato de GAP2;
  pasa si ≥ 4/5 llamadas producen tool_calls válidos con argumentos que validan el schema, y
  0 invenciones de nombres de tool); luego las mismas pruebas del emisor por la tool.

---

## Aprobación

- **2026-09-16 (chat):** Fernando revisó y aprobó este spec ("todo aprobado tienes mi GO").
  GO incluido para: medición de tool-calling de GLM (llamadas reales a Z.ai), árbol real chico
  en vivo con techos bajados, y deploys de los frentes A-G.
