# Historial de pipelines y arreglos del encadenamiento — diseño

**Fecha:** 2026-09-18
**Autor:** Mr. Hyde, con Fernando Ruiz
**Origen:** el informe de `b8f80733` (`~/Documents/HTML/informe-pipeline-b8f80733-2026-09-18.html`)
y el pedido de Fernando: *"debería haber una forma de ver el resultado del pipeline,
no supe qué hacer cuando terminó… un lugar donde se listen los pipelines que se hicieron
y se pueda leer el prompt que generaste"*.

---

## 1 · El problema, en dos partes

Son dos problemas distintos que se veían como uno solo.

**El primero es que el trabajo no se puede ver.** Un pipeline termina y no hay a dónde ir.
Lo que produjo queda en la base y en `~/jax/repo/documents/`, pero la Mesa no lo muestra.
Fernando no supo qué hacer cuando terminó porque, literalmente, no había nada que hacer.

**El segundo es que el trabajo salió mal, y por una razón concreta y verificada.** Los seis
pasos de `b8f80733` arrancaron todos en el mismo instante — `2026-09-12 09:55:25.67` — con
`depends_on: []`. Corrieron **en paralelo, no encadenados**. Eso explica cada síntoma que el
informe describió por separado:

- Thot dijo que no había recibido nada: era cierto, no había nada todavía.
- Cuatro pasos declararon "sin fuentes": corrieron antes de que existiera una fuente.
- Cinco hojas de ruta que no convergen: nadie leyó a nadie, y nadie decidió al final.

No fue un fallo de los modelos. Fue que se les pidió construir sobre algo que aún no existía.

---

## 2 · Lo que ya está guardado (verificado, no supuesto)

Verificado el 2026-09-18 leyendo `jacobs_pipelines.plan` de `b8f80733` en producción. El
plan es un JSON que **ya guarda, por paso**:

| Campo | Qué es |
|---|---|
| `input.prompt` | el prompt exacto que se despachó |
| `output_ref` | la salida completa (`result_full`) |
| `facet` | qué faceta lo hizo |
| `capability` | qué se le pidió |
| `status` | cómo terminó |
| `started_at` / `finished_at` | tiempos reales |
| `trace_id` | para cruzar con el registro de uso |
| `depends_on` | de quién dependía |

**No hay que construir el dato: hay que mostrarlo.** Eso cambia el tamaño del trabajo y es la
razón de que el historial no sea una ronda grande.

La única excepción, y es el "Modelo desconocido" del informe: **`motor` viene en `null`**. La
faceta se resuelve a modelo en el momento del despacho (`resolve_facet()`), y ese resultado no
se escribe en el paso. Saber que actuó `jax_local` no dice qué modelo corrió: el binding cambia,
y un historial que no lo fija miente con el tiempo.

---

## 3 · Alcance de esta ronda

### 3.1 · Historial de pipelines en la Mesa

Una sección propia que lista los pipelines del dueño — los que corren y los terminados —
con nombre, estado, cuándo, cuánto duró y cuánto costó.

Al abrir uno, el detalle muestra **paso por paso**:

- qué faceta y **qué modelo real** lo ejecutó;
- **el prompt exacto que se le envió** (el pedido explícito de Fernando);
- **la salida completa**, no el resumen de 200 caracteres;
- duración y costo del paso;
- de qué pasos dependía.

Reglas de la casa que aplican sin excepción: todo texto por i18n, colores por tokens de tema
en claro y oscuro, y **ninguna confirmación por diálogo del navegador**.

Rendimiento, declarado antes de escribir el código (LAS CUATRO):
- **Indexing:** el listado filtra por dueño y ordena por fecha. Va contra índice, y se verifica
  con `EXPLAIN` sobre la consulta real — no sobre el diseño en la cabeza. Nada de `filesort`
  en el camino caliente.
- **Async:** el detalle de un pipeline largo no se arma dentro del request del listado.
- **Load testing:** se mide con el peor caso — el pipeline con más pasos y más salida, con
  concurrencia. Sin número medido, no hay GO.

### 3.2 · El modelo real queda escrito en el paso

En el momento del despacho, el modelo que resolvió la faceta se persiste en el paso. Es un
**HECHO con procedencia**: qué corrió de verdad, no qué corre hoy.

### 3.3 · Aviso al terminar — correo y Telegram

Cuando un pipeline termina (bien o mal), se avisa por **correo y por Telegram**, con el enlace
a su detalle.

- El correo usa el SMTP ya configurado en `axioma_config` (`smtp.*`).
- Telegram reusa `send_telegram_alert()` de `jacobs/reaper.py`, que **ya existe y ya se usa**
  para las alertas del reaper. No se construye un canal nuevo.
- El aviso **sale del camino del usuario**: se encola, no se manda dentro del request.
- Un aviso que falla **no cambia el estado del pipeline** y no se traga en silencio: queda
  registrado que no se pudo avisar.

### 3.4 · Encadenar por defecto

Hoy un plan sin `depends_on` se lanza todo junto. El cambio:

> **Si el plan no declara dependencias, Jacobs encadena en orden:** el paso *N* depende del
> *N-1* y recibe su salida.

El paralelo deja de ser lo que pasa por omisión y pasa a ser algo que el plan **pide**. Se
distinguen en el JSON: **ausencia de `depends_on` ≠ lista vacía declarada**. Un plan que
quiere paralelo lo dice; uno que se olvidó, se encadena.

Verificación: un plan de tres pasos sin `depends_on` tiene que producir `started_at` que **no
se solapan**, y el contexto del paso 3 tiene que contener la salida del 2. **Ese test se corre
primero contra el código de hoy y tiene que salir rojo.** Si con el código viejo también pasa,
no está midiendo el cambio.

### 3.5 · Un paso truncado falla, no entrega

Si el proveedor corta por tope de salida, hoy el texto a medias sigue viaje como resultado
válido y el paso siguiente construye sobre una frase cortada.

El cambio: corte por longitud → el paso **falla con su propio código**, el pipeline se detiene
y **queda continuable**. Con el pre-vuelo y Continuar ya desplegados (jax#209 /
jax-platform#100), se retoma sin repetir los pasos buenos.

Es fallo cerrado: ante la duda, se detiene, no se entrega.

### 3.6 · Un paso árbitro al final

El último paso recibe todas las salidas anteriores y produce **una decisión y un plan**, donde
**cada punto cita el paso que lo sostiene**. Lo que no tiene paso que lo respalde, no entra
al plan — es el Principio VIII en la salida del pipeline.

**Lo hace siempre Thot**, con regla de sala limpia: **si el plan pone a Thot también como
productor, el plan se rechaza al crearse.** El que produce no arbitra. Es una validación en la
creación, no un aviso que alguien pueda ignorar.

### 3.7 · Permiso de lectura para los pasos

Los modelos pedían permiso para leer el repo, no lo tenían, y respondían de memoria — de ahí
salen las invenciones que contó el calificador. Los pasos reciben **lectura** del árbol de
trabajo.

**Escritura no.** Ver §4.

### 3.8 · Volver a medir la carga de U5

El p95 de 2,99 ms que quedó registrado medía a FastAPI devolviendo un literal, no al servicio.
Es una VERDAD OPERACIONAL falsa y hay que reemplazarla por un número medido de verdad, con el
camino real y concurrencia.

---

## 4 · Fuera de alcance, con su razón

**Escritura en el repo por los pasos.** Ronda propia. Necesita clon dentro del jail (nunca el
árbol vivo), rama propia por pipeline y compuerta humana antes de cualquier merge. Habilitar
la capacidad antes que esos tres contratos es Principio IX — deuda ética, no técnica.

**Que Qwen haga todo.** Se queda en "recolector con verificación" hasta que haya número: las
mismas misiones, Qwen contra el reparto actual, contadas las invenciones con el calificador de
tres capas. Moverlo sin esa medición es cambiar un sistema que funciona por una corazonada.

**Memoria a largo plazo que lo haga automejorar.** La memoria **ya existe** — `messages` con
806 filas, embeddings bge-m3, `embedding_worker` y `synthesis_worker` corriendo. La ronda
futura no es construirla: es **enchufarle el historial de pipelines**, que es el dato crudo del
que aprendería. Va después de esta ronda porque necesita que el historial exista primero, y
necesita decidir antes qué se guarda con qué firma y tipo (Protocolo de la Memoria Viva) y
cómo se prueba que mejoró en vez de que se acostumbró.

**La auditoría de `PUT /api/admin/config`.** Estaba en el alcance y **ya está hecha y viva**
(jax-platform#113, `c00403c`): toda escritura de configuración queda con actor, valor anterior,
valor nuevo e IP, en la misma transacción. Sale del alcance por estar cerrada, no por diferirse.

**La compuerta `ejecutor.c5_auditor_admite_datos_de_clientes`.** Cerrada (`false`) desde el
2026-09-18 06:02:34. Del cierre no hay registro de auditoría porque la auditoría entró 40
minutos después, y de quién la abrió el 2026-09-17 no hay forma de saberlo. Queda dicho como
HISTORIA, no se inventa un responsable.

**CI contra MariaDB 12.3.3.** Ya en curso, fuera de esta ronda: jax#219 y jax-platform#115.

---

## 5 · Cómo se verifica que esto funcionó

No alcanza con que los tests pasen. Esta ronda se da por buena cuando:

1. Un pipeline nuevo de tres pasos **encadena solo**, y se ve en sus `started_at` que no se
   solapan.
2. Ese pipeline aparece en el historial, y su detalle muestra el prompt exacto, la salida
   completa y **el modelo real** de cada paso.
3. Al terminar llega **el correo y el mensaje de Telegram**, con el enlace que abre ese detalle.
4. El paso árbitro produce una decisión donde **cada punto cita su paso**.
5. Los tests del encadenamiento y del truncado **se vieron rojos contra el código viejo**
   antes de verse verdes contra el nuevo.
6. Hay un número de carga medido, con fecha, en la Biblioteca del proyecto.

El punto 5 no es ceremonia: un control que no falla cuando debería fallar no valida nada.

---

## 6 · Repos que toca

- **jax** (Jacobs): encadenar por defecto, truncado falla, paso árbitro, permiso de lectura,
  modelo real en el paso, disparo del aviso.
- **jax-platform** (Mesa): sección de historial, detalle del pipeline, i18n, tema, carga.

En memoria de Jairo Urbina.
