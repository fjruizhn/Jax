# Faro F1.1 — RULE AUTHORITY KERNEL

**Fecha:** 2026-10-05  
**Base:** JAX `f69f045a35fc5d6143b1d6a17934ee01b42d55d1`  
**Estado:** contrato de diseño para revisión humana  
**Autoridad conceptual:** Fernando Ruiz, aprobación en persona del 2026-10-05

## 1. Objetivo

F1.1 implementa el núcleo que convierte una regla individual de `policy/faro/*.yaml`
en una decisión de autoridad verificable. Una regla solo puede producir `PERMIT` si:

1. el snapshot confiable cargó la regla desde un commit Git exacto;
2. sus bytes exactos coinciden con una ratificación individual firmada por
   `human:fernando` en Block 4;
3. la regla sigue presente, sin cambios, en el snapshot cargado;
4. la ratificación no está revocada;
5. la regla y la ratificación están dentro de vigencia;
6. el alcance y los límites cubren exactamente la solicitud;
7. STOP es conocido e inactivo;
8. la decisión y, si corresponde, el `RulePermit` quedan registrados de forma durable.

F1.1 no ejecuta acciones. Su salida positiva es un permiso corto, ligado a una solicitud
y consumible una sola vez. Ese permiso no es una autorización de Block 6 ni una
capacidad para hacer dispatch.

## 2. Alcance cerrado

### Incluye

- schema cerrado para `policy/faro/*.yaml`;
- loader desde objetos Git de un snapshot fijado;
- hash SHA-256 de los bytes exactos de cada regla;
- eventos Block 4 `RULE_RATIFICATION_GRANTED` y
  `RULE_RATIFICATION_REVOKED`;
- firma Ed25519 por la identidad constitucional `human:fernando`;
- vigencia;
- comprobación de presencia en el snapshot de `policy/` cargado;
- STOP con fallo cerrado;
- decisiones `PERMIT`, `DENY` y `MISSING_RULE`;
- `RulePermit` corto, ligado a una solicitud y de un solo uso;
- auditoría durable y encadenada;
- corrección completa del adapter de persistencia de Block 4, writer y decoder,
  necesaria para persistir y reconstruir todos los payloads existentes y nuevos;
- fronteras confiables para el pin activo, el checkpoint externo, STOP, reloj y
  clasificación de capacidades.

### Excluye

- motor o dispatcher real;
- jaula nueva;
- Qwen;
- SSH;
- operaciones en GitHub;
- despliegue o escrituras en producción;
- activación de RL01;
- creación o ratificación de una regla real;
- control acumulado de frecuencia y monto entre múltiples efectos externos;
- atomicidad entre consumo del permiso y un efecto externo futuro.

## 3. Contratos que se preservan

- `RATIFICATION_GRANTED` y `RATIFICATION_REVOKED` conservan su semántica actual de
  corpus completo.
- Una ratificación de corpus nunca cuenta como ratificación individual.
- C14N/3 y `policy_corpus_hash` siguen identificando el corpus semántico actual; nunca
  sustituyen el hash exacto de una regla.
- `Memory != Authority`, `Memory != Evidence`, `Memory != Current Truth` y
  `MODEL_AUDITED != VERIFIED`.
- `policy/**` sigue reservado para integración por Fernando.
- Block 3, Block 5 y Block 6 conservan sus contratos actuales.
- STOP falla cerrado cuando está activo, ilegible o sin configurar.

## 4. Alternativas consideradas

### A. Incorporar las reglas al corpus C14N/3

Reutilizaría el loader y el hash del corpus, pero no distingue cambios de bytes que no
cambian la semántica canónica. También convertiría una ratificación individual en una
ratificación del conjunto completo.

**Decisión:** rechazada.

### B. Snapshot raw-byte y eventos individuales dentro de Block 4

El loader lee los blobs exactos de Git, calcula el hash antes de parsear y agrega eventos
individuales a la cadena de autoridad que ya tiene raíz Ed25519, replay y checkpoint.

**Decisión:** elegida.

### C. Bundle Faro firmado y ledger separados

Crearía una segunda raíz de autoridad, otra rotación de claves, otra cadena y otra
resolución de conflictos.

**Decisión:** rechazada.

## 5. Identidades y procedencia

Las siguientes identidades son distintas y no se sustituyen entre sí:

| Campo | Significado |
|---|---|
| `rule_content_hash` | `sha256:<hex>` de todos los bytes del blob de la regla |
| `rule_blob_oid` | OID Git del blob exacto |
| `ratified_policy_revision` | commit Git desde cuyo snapshot sellado se creó el grant |
| `ratified_policy_tree_oid` | OID del árbol `policy/` en ese commit |
| `loaded_policy_revision` | commit exacto del snapshot que evalúa la solicitud |
| `loaded_policy_tree_oid` | OID del árbol `policy/` cargado |
| `policy_snapshot_hash` | hash de la lista ordenada de entradas Faro del snapshot |
| `policy_corpus_hash` | identidad semántica C14N/3 del contrato existente |

`rule_content_hash` se calcula sobre los bytes crudos antes de YAML. No hay conversión
de newline, Unicode, espacios, comentarios ni orden de claves.

### Ratificación una vez por contenido

El grant registra el commit y árbol donde Fernando ratificó la regla. La evaluación no
exige que todo el árbol cargado sea idéntico al árbol ratificado, porque eso haría que un
cambio ajeno a la regla obligara a ratificarla de nuevo.

Para usar el grant, el snapshot cargado debe cumplir todo lo siguiente:

- `ratified_policy_revision` es ancestro de `loaded_policy_revision`;
- la misma ruta contiene un blob regular;
- `rule_id`, `rule_blob_oid` y `rule_content_hash` coinciden exactamente;
- el loader verificó el commit, el árbol y la procedencia del pin cargado.

Un cambio de un byte cambia el hash y deja el grant sin coincidencia. Una eliminación
produce `MISSING_RULE`. Una regla idéntica que siga en la historia descendiente conserva
su ratificación. La revocación explícita prevalece siempre.

La comparación es entre los extremos ratificado y cargado. Si la historia contiene
`A(bytes h) → B(modificada o eliminada) → C(bytes h)`, el grant original vuelve a
coincidir en C si continúa siendo el grant más reciente y no fue revocado. Esto implementa
“ratificación una vez por contenido”. Retirar autoridad de forma permanente exige un
evento de revocación; borrar temporalmente el archivo no sustituye ese evento.

## 6. Snapshot confiable

### Interfaz

```python
load_trusted_policy_snapshot(
    repo: Path,
    pin: TrustedPolicyPin,
) -> TrustedPolicySnapshot
```

`TrustedPolicyPin` contiene el repositorio esperado, el commit exacto, el OID esperado
del árbol `policy/` y la procedencia del pin. El pin no acepta nombres móviles como raíz
de autoridad.

### Proceso

El loader:

1. comprueba que el commit y el árbol esperados existen y coinciden, y recalcula
   el OID del commit desde sus bytes crudos;
2. usa objetos Git, nunca el working tree;
3. hereda o generaliza las defensas de `jax/faro/git_objetos.py`: entorno limpio,
   hooks y fsmonitor apagados, replace objects y lazy fetch desactivados;
   recalcula los OID de todos los árboles recorridos desde esos mismos bytes;
   no interpreta una segunda vista de `ls-tree`;
4. enumera exclusivamente blobs `100644` con ruta directa
   `policy/faro/<nombre>.yaml`;
5. rechaza symlinks, submodules, modos desconocidos, rutas anidadas, nombres no
   canónicos y duplicados;
6. obtiene todos los blobs antes de validar uno;
7. calcula OID y hash raw antes de parsear;
8. parsea YAML con loader seguro que rechaza claves duplicadas, aliases y merge keys, y
   valida shape cerrado;
9. rechaza el snapshot completo si una regla falla o si hay `rule_id` duplicado;
10. devuelve objetos profundamente inmutables con un sello privado que los callers no
    pueden fabricar por la API pública.

El presupuesto es cerrado: formato Git SHA-1 o SHA-256 del repositorio, máximo
1.024 objetos contando commit, árboles y blobs; profundidad máxima de 16; 8 MiB
agregados entre objetos recorridos y blobs; y 1 MiB por blob. El cargador mide
cantidad y tamaño acumulados antes de materializar el lote de blobs. Un límite
excedido niega el snapshot completo.

`policy_snapshot_hash` usa un dominio propio y la lista ordenada por ruta de
`(path, mode, blob_oid, rule_content_hash)`. Sirve para auditar qué se cargó; no reemplaza
el hash individual.

El sello Python impide fabricación accidental por la API pública; no es aislamiento frente
a código arbitrario dentro del proceso. La confianza se apoya además en tipos cerrados,
inmutabilidad profunda, providers de arranque y validación al cruzar cada frontera.

## 7. Schema de regla F1.1

```yaml
schema_version: "1.0"
kind: JAX_FARO_RULE
rule_id: ejemplo-regla
effect: PERMIT
action_class: REVERSIBLE

scope:
  subjects: [actor:ejemplo]
  capabilities: [CAPABILITY_ID]
  objectives: [OBJECTIVE_ID]

obligation_limits:
  quantity: null
  amount: null
  frequency: null

validity:
  not_before_utc: "2026-10-05T00:00:00Z"
  not_after_utc: null

permit:
  ttl_seconds: 60
```

Reglas del schema:

- top level y objetos internos cerrados; campos extra se rechazan;
- `schema_version`, `kind` y `effect` tienen los únicos valores mostrados;
- identificadores en NFC, con vocabularios cerrados donde corresponda;
- colecciones no vacías donde el contrato las exige, sin duplicados;
- `not_before_utc` y `not_after_utc` son UTC; el intervalo es
  `[not_before_utc, not_after_utc)`;
- `ttl_seconds` es positivo y no supera el máximo de configuración confiable;
- para `OBLIGATING`, `objectives`, vigencia finita, frecuencia y exactamente uno de
  `quantity` o `amount` son obligatorios;
- cantidad declara unidad y máximo positivo;
- monto declara moneda ISO 4217 y máximo en unidades menores;
- frecuencia declara máximo de ocurrencias y ventana en segundos;
- una regla `REVERSIBLE` puede tener fin abierto;
- las reglas heredadas de `policy/rules/*.yaml` no entran en este snapshot.

`action_class` es una afirmación verificable de la regla, no la fuente de verdad. El
kernel consulta un `CapabilityClassificationProvider` confiable para cada capability. Una
capability desconocida, una clasificación no disponible o una discrepancia producen
`DENY`. La regla nunca puede rebajar una capability obligante a `REVERSIBLE`. En F1.1 el
provider es una interfaz obligatoria y solo hay implementación de prueba; sin un registro
operativo confiable no se emiten permisos consumibles para esa capability.

Unidades y monedas se comparan de forma exacta con el contrato de la capability. Cantidad,
monto, máximos y valores solicitados deben ser enteros positivos dentro del rango del
schema; se rechazan NaN, infinitos, coerciones, unidades o monedas ausentes y combinaciones
incompatibles.

F1.1 agrega el schema y documentación, pero ninguna regla activa. Los ejemplos
ratificables viven únicamente como fixtures de prueba.

## 8. Block 4: eventos individuales

Se agregan dos variantes sin cambiar las seis existentes:

```python
RULE_RATIFICATION_GRANTED
RULE_RATIFICATION_REVOKED
```

### Grant

El payload incluye:

- `rule_id`;
- `rule_path`;
- `rule_blob_oid`;
- `rule_content_hash`;
- `ratified_policy_revision`;
- `ratified_policy_tree_oid`;
- `ratified_policy_snapshot_hash`;
- `valid_from_utc`;
- `valid_until_utc`.

La única fábrica pública recibe un `TrustedPolicySnapshot` sellado y un `rule_id`. Copia
todos los campos desde ese objeto atómico y fija `actor_id=human:fernando`. No acepta
hashes, rutas ni revisiones sueltas.

La firma Ed25519 vigente de Block 4 cubre el tipo, payload, actor, posición, timestamp,
predecessor y evidencia.

### Revocación

El payload contiene `rule_ratification_event_id`, que apunta al `event_id` exacto del
grant. Un grant desconocido o una segunda revocación del mismo grant invalidan el replay.

Para cada `rule_id`, el grant válido más reciente desplaza definitivamente a los
anteriores. Revocar el más reciente deja la regla sin grant y no reactiva uno antiguo.
Solo un grant posterior puede establecer un nuevo estado.

### Adapter de persistencia como condición previa

`event_from_storage_row()` hoy descarta payloads y solo reconstruye tipo, actor y
evidencia. El writer MariaDB actual también serializa el dataclass completo, que puede
incluir sellos internos, y omite la columna `evidence_refs NOT NULL` de la migración 001.

Antes de añadir eventos se implementan:

- proyecciones públicas cerradas de intent y evento, sin sellos ni estado interno;
- encoder y decoder cerrados por variante para los seis tipos existentes y los dos nuevos;
- INSERT completo de `canonical_intent`, `canonical_event` y `evidence_refs`;
- preservación byte por byte de la proyección firmada de los seis eventos históricos: no
  se agregan campos nulos ni defaults a sus bytes firmados;
- integración MariaDB `firmar → insertar → leer → reconstruir → verificar firma/replay`
  para los ocho tipos.

Cualquier campo faltante, extra o de tipo incorrecto falla. El replay histórico debe
producir el mismo estado previo.

## 9. Solicitud y decisión

```python
RuleEvaluationRequest(
    request_id,
    rule_id,
    subject,
    capability,
    objective,
    resource_id,
    arguments,
    quantity=None,
    amount=None,
    request_hash=...,
)
```

`request_hash` usa una proyección canónica cerrada y un dominio propio. No acepta una
proyección aportada por el caller que difiera de los campos. La proyección incluye el
identificador estable del recurso u objetivo concreto y todos los argumentos relevantes
para el efecto; no se permiten campos opacos fuera del hash.

`request_id` es idempotente: el store tiene una restricción única y una solicitud exacta
solo puede producir un permiso. Repetir el mismo ID y hash devuelve la decisión existente;
repetir el ID con otro hash falla cerrado.

El camino que puede emitir permisos no acepta snapshot, estado, STOP ni hora aportados por
el caller:

```python
RuleAuthorityKernel(
    active_policy_pin_provider,
    authority_ledger_store,
    trusted_root,
    trusted_checkpoint_store,
    stop_provider,
    trusted_clock,
    capability_classification_provider,
    rule_authority_store,
).evaluate(request) -> RuleDecision
```

Los providers son dependencias de arranque confiables y selladas. Su configuración no
sale del request ni de la regla:

- `ActivePolicyPinProvider` entrega el pin activo y su procedencia externa;
- `TrustedCheckpointStore` es obligatorio; `checkpoint_store=None` nunca habilita emisión;
- `StopProvider` lee la fuente única actual y entrega estado más versión o huella;
- `TrustedClock` entrega UTC consciente y detecta retroceso respecto de emisión;
- `CapabilityClassificationProvider` fija clase, unidad o moneda y forma de límites.

Los providers de pin, STOP y clasificación implementan además un protocolo común de
leases. Evaluación y consumo adquieren leases compartidos; cualquier escritor que active
STOP, cambie el pin o reemplace un contrato de capability necesita el lease exclusivo de
su provider. El lease mantiene valor y versión estables hasta después del commit de DB.
Una implementación que solo permita releer un valor, aunque tenga contador monotónico,
no satisface el contrato y no puede emitir ni consumir permisos.

Orden global de adquisición: pin activo → clasificación de capability → STOP → permit DB
→ `authority_ledger_head` → audit head. Todos los dobles y adapters siguen ese orden. Los
writers de pin, clasificación y STOP toman su lease exclusivo correspondiente; Block 4
serializa sobre su head. Esto evita invertir locks entre evaluación, consumo y mutaciones.

Una función separada de evaluación histórica puede aceptar valores explícitos, pero su
tipo de resultado no contiene `RulePermit` y no es aceptado por el store de consumo. Un
`ReconstructedAuthorityState` con sello interno pero sin checkpoint externo coincidente
no es autoridad actual.

Orden de evaluación:

1. pin activo obtenido del provider y snapshot exacto cargado;
2. snapshot confiable y sellado;
3. ledger verificado contra trusted root y checkpoint externo obligatorio;
4. STOP actual conocido e inactivo;
5. regla presente;
6. grant individual actual presente y no revocado;
7. actor y firma constitucional válidos;
8. procedencia histórica y bytes de la regla coincidentes;
9. vigencia de regla y grant;
10. clasificación confiable de la capability coincidente;
11. subject, capability y objective dentro del scope;
12. cantidad o monto dentro del máximo y del contrato de unidad o moneda;
13. límites estructurales obligatorios presentes;
14. persistencia durable de la decisión;
15. creación y persistencia atómica del permiso, solo para `PERMIT`.

El grant actual se selecciona primero por la mayor secuencia del ledger para ese
`rule_id`, antes de filtrar vigencia, hash o snapshot. Si ese grant es futuro, vencido o
incompatible, la decisión es `DENY`; nunca se cae hacia un grant anterior.

Resultado:

- regla ausente: `MISSING_RULE`;
- regla presente pero sin autoridad, vencida, revocada, cambiada, frenada o fuera de
  alcance: `DENY`;
- todo el contrato satisfecho y persistido: `PERMIT`.

Toda negativa contiene `required_rule_id` y un reason code estable. Un error interno o
de storage nunca se convierte en `PERMIT`.

F1.1 valida los límites de una solicitud individual. El conteo acumulado por frecuencia
pertenece al futuro borde transaccional del motor; hasta que exista, ninguna regla que
dependa de ese conteo puede autorizar un efecto real.

## 10. RulePermit

El permiso incluye:

- `permit_id`, `request_id` y `request_hash`;
- `rule_id`, `rule_path`, `rule_blob_oid` y `rule_content_hash`;
- revisión, árbol y hash del snapshot cargado;
- `ratification_event_id`;
- checkpoint del authority ledger;
- checkpoint de STOP;
- identificador y versión inmutable del contrato de capability;
- `issued_at_utc`, `expires_at_utc` y `permit_hash`.

Propiedades:

- TTL corto tomado de la regla y limitado por configuración confiable;
- `expires_at_utc` es el mínimo entre `issued_at + ttl`, `rule.not_after_utc` y
  `grant.valid_until_utc`, ignorando solamente límites abiertos;
- ligado a una única solicitud exacta;
- persistido antes de devolverse;
- una instancia construida o deserializada por un caller carece de procedencia;
- solo el store autoritativo devuelve un permiso confiable;
- no implementa ni importa motor, executor, SSH, GitHub o subprocess de ejecución.

### Consumo

```python
RuleAuthorityKernel.consume(
    permit_id,
    expected_request_hash,
) -> RulePermitConsumption
```

El kernel vuelve a consultar sus providers; el caller no aporta estado, snapshot, STOP ni
hora. Antes de abrir la transacción adquiere leases compartidos del pin activo, contrato
de capability y STOP, en el orden global. Conserva esos leases hasta después del commit.
En una sola transacción MariaDB y con orden fijo de bloqueos:

1. bloquea el permit;
2. verifica hash, request, procedencia y expiración;
3. bloquea `authority_ledger_head`, la misma fila y protocolo que usa el writer Block 4;
4. exige que el head bloqueado coincida con el checkpoint externo actual;
5. reconstruye autoridad a ese head y confirma que el grant sigue siendo el más reciente
   y no está revocado;
6. usa el pin protegido por el lease y vuelve a cargar la regla, exigiendo misma ruta,
   blob y bytes;
7. revalida que el contrato de capability protegido por lease conserva identidad,
   versión, clase, unidad o moneda y forma de límites guardadas en el permiso;
8. comprueba nuevamente los intervalos de regla, grant y permiso con hora UTC confiable,
   sin retroceso respecto de `issued_at_utc`;
9. exige que STOP, protegido por su lease, siga inactivo;
10. bloquea el head de auditoría de Rule Authority;
11. inserta el consumo con `permit_id` como clave única;
12. inserta el registro durable encadenado del consumo y actualiza su head;
13. confirma la transacción;
14. publica y hace `fsync` del nuevo checkpoint externo de auditoría;
15. relee el log de checkpoints y comprueba que contiene ese head exacto antes de devolver
    éxito.

El writer de Block 4 y el consumidor serializan sobre `authority_ledger_head`. Una
revocación que obtiene el lock primero hace que el consumo niegue; si el consumo confirma
primero, ese es su punto de linearización y la revocación es posterior. Un head DB todavía
no publicado en el checkpoint externo bloquea consumos hasta que ambos coincidan.

Los providers deben entregar versiones monotónicas sin ABA y coordinar readers/writers con
leases. Si la implementación disponible no ofrece ambas propiedades, el kernel puede
evaluar y negar, pero no emitir ni consumir permisos. F1.1 implementa las interfaces y
dobles deterministas; no instala un provider operativo.

Dos consumidores concurrentes producen exactamente un consumo exitoso. Las pruebas de
carrera usan barreras después de la última validación para cubrir doble consumo,
revocación, cambio de pin, cambio de clasificación y activación de STOP en ambos órdenes.

El consumo F1.1 solo registra que el permiso se gastó. No produce efecto externo. El
futuro dispatcher tendrá que consumir y comprobar STOP en su propia frontera inmediata
antes del efecto. F1.1 garantiza single-use y orden respecto de Block 4 en el commit de
consumo; no garantiza atomicidad entre ese commit y un efecto futuro.

## 11. Persistencia y auditoría durable

La migración nueva crea, sin modificar migraciones publicadas:

- `rule_authority_decisions`;
- `rule_permits`;
- `rule_permit_consumptions`;
- `rule_authority_audit_head`.

Los registros usan proyecciones canónicas, hash propio y predecessor. Triggers rechazan
`UPDATE` y `DELETE` en decisiones, permits y consumos. `rule_authority_audit_head` es un
puntero mutable bloqueado y actualizado por la misma transacción; no se presenta como
registro inmutable. Los índices cubren las búsquedas por request, regla, grant, estado,
expiración y consumo. Las consultas reales se verifican con `EXPLAIN` en MariaDB efímera.

Un `RuleAuditCheckpointStore` externo, monotónico y obligatorio ancla el head de
auditoría. Restaurar una copia antigua de MariaDB produce mismatch y bloquea emisión y
consumo. Los permits pendientes no vuelven a ser utilizables hasta una reconciliación
explícita y auditada; F1.1 no contiene un procedimiento automático que los bendiga.

Antes de mutar, la transacción exige igualdad entre el audit head de DB y el checkpoint
externo. Después del commit, el head DB queda temporalmente por delante y cualquier otra
emisión o consumo falla cerrado. El proceso que confirmó publica de forma monotónica el
checkpoint exacto, hace `fsync`, lo relee y solo entonces devuelve `PERMIT` o consumo
exitoso. Otra operación posterior puede haber agregado un head descendiente; la prueba de
éxito es que el log durable contiene el head exacto confirmado por esta operación. El
checkpoint rechaza retrocesos y heads conflictivos.

Si el proceso cae o falla después del commit y antes de completar el anclaje, el cambio
puede existir en DB pero no se reconoce como éxito y el mismatch bloquea nuevas
operaciones. La recuperación es explícita: verifica cadena y DB, publica el head pendiente
o invalida por un procedimiento auditado futuro; F1.1 no reconcilia automáticamente.

Si el crash ocurre después del anclaje durable y antes de responder, DB y checkpoint ya
coinciden. El retry idempotente recupera la decisión o consumo anclado y no abre una
reconciliación. Nunca devuelve un permiso cuyo head exacto no figure en el log durable.
Las pruebas inyectan crashes después del commit, durante publicación y después del
anclaje antes de responder.

`PERMIT` y su `RulePermit` se insertan atómicamente. Si la auditoría falla, no se entrega
permiso. `DENY` y `MISSING_RULE` se intentan persistir antes de responder; si el store
falla, se devuelve un error cerrado que el caller trata como denegación.

Los avisos inmediatos son fail-soft y se emiten después de la decisión durable. Incluyen
identificador de regla, reason code y hashes necesarios; excluyen contenido completo y
argumentos sensibles.

## 12. Estructura de código prevista

### Nuevos

- `policy/faro/schemas/rule-v1.schema.json`
- `policy/faro/README.md`
- `policy/rule_authority/{models,schema,snapshot,evaluator,permit,storage,serialization,canonical,errors}.py`
- `policy/rule_authority/migrations/001_rule_authority_kernel.sql`
- pruebas unitarias y de MariaDB bajo `tests/policy/`
- fixtures exclusivamente bajo `tests/policy/fixtures/faro_rules/`

### Modificados

- `policy/authority_ledger/models.py`
- `policy/authority_ledger/replay.py`
- `policy/authority_ledger/service.py`
- `policy/authority_ledger/serialization.py`
- `policy/authority_ledger/__init__.py`
- `jax/faro/git_objetos.py`, solo para generalizar lectura endurecida reutilizable;
- `jax/faro/aviso.py`, para avisos de negativas;
- documentación y pisos de CI aplicables.

La implementación puede ajustar esta distribución si conserva los límites de autoridad y
evita ciclos entre Block 4, el loader y el store.

## 13. Pruebas obligatorias

### Bytes y snapshot

- espacio, comentario, newline, CRLF/LF, NFC/NFD y reordenamiento YAML cambian el hash;
- el hash YAML canónico nunca se acepta como `rule_content_hash`;
- una mutación del checkout durante la carga no cambia el snapshot;
- mover una branch no cambia un pin ya cargado;
- hooks, fsmonitor, replace objects y variables `GIT_*` no desvían la lectura;
- commit, árbol raíz, árbol `policy/`, árboles `faro/` anidados y blobs adulterados
  bajo su OID nominal niegan antes de producir un snapshot;
- SHA-1 y SHA-256 usan el formato real del repositorio; árboles truncados,
  inválidos o que excedan el presupuesto niegan;
- symlink, submodule, modo o ruta inválidos fallan cerrado;
- claves YAML duplicadas, aliases y merge keys fallan cerrado;
- un clone con contenido anidado modificado no conserva procedencia;
- `rule_id` duplicado o un YAML inválido rechazan el snapshot completo.

### Autoridad

- los seis eventos históricos y los dos nuevos hacen roundtrip exacto;
- replay histórico conserva su estado;
- ratificación de corpus sola nunca produce `PERMIT`;
- actor, clave o firma incorrectos fallan;
- grant de otra ruta, bytes o historia falla;
- un cambio ajeno en un snapshot descendiente conserva el grant;
- una secuencia `h → distinto/ausente → h` vuelve a coincidir si no hubo revocación;
- un cambio de un byte invalida el grant;
- una revisión cargada que no desciende de la ratificada falla;
- revocación desconocida o duplicada falla replay;
- revocar el grant nuevo no reactiva el anterior;
- un grant posterior futuro, vencido o incompatible niega sin fallback al anterior;
- writer y reader MariaDB preservan bytes firmados y evidencia de los ocho eventos;
- ledger truncado, no anclado o con checkpoint discordante falla.

### Vigencia y STOP

- antes de `not_before`: `DENY`;
- exactamente en `not_before`: elegible;
- exactamente en `not_after`: `DENY`;
- `expires_at` se recorta por fin de regla y grant y se prueba exactamente en cada borde;
- reloj sin timezone: `DENY`;
- retroceso del reloj respecto de emisión: `DENY`;
- STOP activo, ilegible o sin configurar: `DENY`;
- provider sin versión monotónica: no emite ni consume;
- STOP activado entre emisión y consumo rechaza el consumo;
- carreras con barrera prueban ambos órdenes de STOP, cambio de pin, cambio de contrato de
  capability y revocación frente al consumo.

### Permiso y store

- modificar un campo o usar otro request hash falla;
- repetir `request_id` con el mismo hash es idempotente y con otro hash falla;
- capability desconocida, sin clasificación o rebajada por YAML niega;
- unidad, moneda, cantidad o monto incompatibles niegan;
- permiso vencido falla;
- doble consumo secuencial falla;
- dos consumidores concurrentes dejan un solo consumo;
- storage o auditoría caídos no entregan permiso;
- rollback de DB frente al checkpoint externo bloquea emisión y consumo;
- crash después de commit y antes del anclaje nunca declara éxito y deja el kernel cerrado
  hasta reconciliación;
- crash después del anclaje y antes de responder recupera el resultado anclado de forma
  idempotente, sin emitir otro permiso;
- triggers impiden mutación y borrado;
- `EXPLAIN` confirma índices de caminos de evaluación y consumo;
- ningún módulo F1.1 despacha, ejecuta procesos o realiza efectos externos.

## 14. Secuencia de implementación

1. Corregir y probar writer y codec cerrado de Block 4.
2. Agregar modelos y replay de los eventos individuales.
3. Agregar schema y snapshot Git sin reglas reales.
4. Agregar modelos de solicitud, evaluación y decisiones con store en memoria para TDD.
5. Agregar migración y adapter MariaDB transaccional.
6. Agregar `RulePermit` y prueba de carrera de consumo.
7. Agregar interfaces confiables de pin, checkpoint, STOP, reloj y clasificación; sin
   wiring operativo.
8. Conectar avisos sin hacerlos parte de la autoridad.
9. Ejecutar suites Block 3/4/5/6, Faro y MariaDB efímera.
10. Auditar el SHA exacto antes de presentar el PR.

No hay backfill: ningún evento existente ratifica una regla individual. RL01 sigue siendo
propuesta no ratificada.

## 15. Criterios de aceptación

F1.1 queda listo para revisión cuando:

- todas las pruebas negativas anteriores están verdes;
- un grant solo nace de un snapshot sellado;
- cualquier cambio de byte niega la regla;
- una regla sin grant individual nunca produce `PERMIT`;
- STOP y fallos de auditoría cierran el paso;
- el permiso se consume como máximo una vez bajo carrera real;
- el codec conserva todos los payloads históricos;
- no existe integración con motor ni efecto externo;
- el SHA final recibe auditoría adversarial Tier 3 sin bloqueadores;
- Fernando integra los cambios bajo `policy/**`.

## 16. Decisiones de implementación (2026-10-07)

Lo que la implementación dejó distinto o más preciso que este texto, registrado con quién
decidió y dónde vive el código. Procedencia de los veredictos y SHAs:
`~/encargos-codex/LEDGER-2026-10-05-noche.md` (jornada 2026-10-07).

1. **La tabla de decisiones se llama `rule_decisions`** (§11 la nombra
   `rule_authority_decisions`). Decisión de Hyde (ledger 05:09: «se queda
   `rule_decisions`, se alinea el diseño»). Código: migración
   `policy/rule_authority/migrations/001_rule_authority_kernel.sql` (crea
   `rule_authority_audit_head`, `rule_decisions`, `rule_permits`,
   `rule_permit_consumptions`) — Jax#375 (`6ae6986a`, APROBADO), presente también en la
   cadena de #371.

2. **Capability desconocida, sin clasificar o rebajada NIEGA** (§13 ya lo decía; queda
   confirmado como cierre obligatorio de auditoría). Hyde rechazó #373 r2 por permitirlo y
   fijó: «capability desconocida NIEGA per §13» (ledger 04:20). Implementado en
   `policy/rule_authority/` — Jax#373 (`72b02be7`, APROBADO).

3. **Procedencia del pin con forma cerrada y anclada**: `refs/heads/<rama>` o
   `refs/tags/<tag>` exacta, sensible a mayúsculas, sin espacios, e IGUAL a la procedencia
   real del pin cargado. Código: `policy/rule_authority/providers.py`
   (`_RE_PROCEDENCIA`) — Jax#373 (`72b02be7`); el rechazo que la introdujo fue el MAJOR-2
   de la ronda 2 (`b16216d7`).

4. **El catálogo de topes sale del snapshot sellado del pin**: SOLO el snapshot emite
   `CatalogoTopes` (registro único de emisor en el proceso) y exige el OID del catálogo en
   el pin. Decisión de Hyde en la auditoría de #370 r6 (ledger 03:50: «solo el snapshot
   emite CatalogoTopes»). Código: `jax/faro/catalogo_topes.py` — Jax#370
   (`28f1eac7`, APROBADO).

5. **D-4 como piso de código con lista negra ampliada**: subid y sus segmentos/raíces se
   niegan por lista negra ampliada con sinónimos en inglés (cerrados como MINOR de #370,
   `a7757685`); la lista BLANCA en código queda recomendada para cuando se cablee el
   runtime (recomendación de Hyde, ledger 05:04; decisión de Fernando pendiente). En la
   misma cadena, `Tope.period` solo acepta subids de frecuencia del catálogo
   (`28f1eac7`, #370) y `Cantidad.unit`/`Monto.currency` se validan contra el catálogo —
   Jax#378 (`0c2822b5`, APROBADO; cerró el hueco del schema de #370 anotado en el ledger
   05:09).

6. **Los límites derivados llevan la regla atada**: `LimitesObligatorios` (los límites que
   la evaluación deriva de la regla) incluye `rule_id` y `rule_hash` ligados a la
   solicitud; nacieron del rechazo de #371 r3 («límites sin rule_id», ledger 05:12) y se
   cerraron en `002fc9ba`; `limites_de` además valida `Tope.period` contra el catálogo
   (`b967dff0`). Final: Jax#371 (`d4ffe62f`, APROBADO) —
   `policy/rule_authority/models.py`.

7. **`request_hash` incluye el catálogo**: `RuleEvaluationRequest` valida y proyecta el
   catálogo sellado (con su OID) dentro del `domain_hash` del `request_hash`; un
   catálogo distinto es una solicitud distinta. Código:
   `policy/rule_authority/models.py` (`canonical_projection` → `request_hash`) —
   Jax#371 (`d4ffe62f`).

8. **Avisos (paso 8) con entrega al-menos-una-vez** (§11 los describía fail-soft sin este
   mecanismo): `resumen_diario` trabaja en DOS FASES — reclama los rotados con un lease
   (`lease_s`) y NO borra nada hasta que el llamador envía y llama
   `confirmar_resumen(token)`; sin confirmar, pasado el lease se re-entregan. Archivos
   ilegibles o inseguros a cuarentena (`.cuarentena`, `O_NOFOLLOW`), y los envíos
   fallidos se marcan `envio_fallido` y vuelven a la cola (no se descartan). Código:
   `jax/faro/aviso.py` — Jax#376 (`ea7fa5e3`, APROBADO; hitos `153560d0`, `82801e36`,
   `92b98d36`).

9. **Ledger de Block 4 endurecido**:
   - sello `init=False` — `dataclasses.replace` no transporta el sello del snapshot de
     ratificación (`policy/authority_ledger/models.py`) — Jax#377 (`d082cc89`,
     APROBADO);
   - replay previo al append — `append_authority_event` re-verifica TODO el stream
     existente y el evento nuevo contra el estado reconstruido ANTES de escribir
     (`policy/authority_ledger/service.py`) — Jax#377 (`d082cc89`);
   - checkpoint externo tras el append con fallo cerrado — el evento no se acepta hasta
     que su checkpoint queda escrito; si el checkpoint falla, el evento ya append-only
     queda en el ledger y la operación falla cerrada exigiendo reconciliación explícita
     (`service.py`) — Jax#381 (`ae4e9338`, en auditoría al escribir esto);
   - `OVERLAY_ISSUED` exige ratificación vigente — un overlay solo es válido contra un
     corpus con ratificación no revocada en ese punto del stream (`replay.py`) —
     Jax#381 (`ae4e9338`).

Nota de procedencia: las decisiones 4 y 5 se registran con la atribución que el ledger
respalda (Hyde, en sus auditorías de #370); una co-atribución a Fernando del 2026-10-06
para el catálogo no consta ahí. La lista blanca de D-4 sigue pendiente de decisión de
Fernando.
