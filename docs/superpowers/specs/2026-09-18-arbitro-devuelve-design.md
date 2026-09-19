# El árbitro devuelve el trabajo mal hecho — diseño

**Fecha:** 2026-09-18
**Autor:** Mr. Hyde, con Fernando Ruiz
**Origen:** Fernando, hoy, tras leer el primer pipeline con árbitro:
*"estoy de acuerdo, pero debió habérselo regresado a ADA para que lo arreglara"*.

---

## 1. El problema, con el caso real

El pipeline `e570ac1c` corrió hoy con árbitro por primera vez. El paso 4 (`ada`, `glm-5.3`)
produjo 15.238 caracteres de arquitectura que se ven impecables y **están mal**: propuso el
esquema en sintaxis PostgreSQL (`BIGSERIAL`, `TIMESTAMPTZ`, `JSONB`) sobre un destino MariaDB, y
un modelo de movimientos con extremos nulos donde el brief exige ubicaciones virtuales.

El árbitro lo detectó y lo dijo bien:

> *"no aprobar todavía el DDL ni la arquitectura de los pasos 4–5 para implementación"* —
> citando `[paso 0] [paso 1] [paso 4] [paso 5]`.

Y ahí se acabó. **El error quedó escrito, el pipeline terminó "completado", y nadie lo corrigió.**
Detectar sin poder devolver convierte al árbitro en un crítico: tiene razón y no cambia nada.

## 2. La idea, en una línea

Una devolución es un **Continuar apuntado a propósito**: invalidar el paso señalado y lo que
dependía de él, subir la época, y rehacerlo **con la crítica del árbitro adentro del contexto**.

La maquinaria ya existe entera (`jacobs/continuar.py`, épocas, reuso de pasos, pre-vuelo de lo
que falta). Hoy la usamos cuando un pipeline **muere**; esto la usa cuando un pipeline
**produce algo malo**. No se construye un mecanismo nuevo: se le da una segunda razón de uso al
que ya está probado — hoy mismo rescató este pipeline dos veces.

## 3. Lo que hay que construir

### 3.1 · Un veredicto que el sistema pueda accionar

Hoy el árbitro escribe prosa: un humano la entiende y el código no puede hacer nada con ella.
El árbitro pasa a emitir, **además de su texto**, un veredicto con estructura:

- a qué paso devuelve,
- por qué, en palabras que sirvan a quien lo rehace (no "está mal": *"usaste `BIGSERIAL` y
  `TIMESTAMPTZ`; el destino es MariaDB"*),
- y la cita del paso que lo sostiene, igual que el resto de sus afirmaciones.

**Fallo cerrado:** si el veredicto no viene con estructura válida, **no hay devolución**. El
pipeline termina como hoy, con la crítica en prosa, y queda registrado que el árbitro quiso
devolver y no se pudo parsear. Nunca se adivina a qué paso se refería.

### 3.2 · La crítica viaja con el trabajo devuelto

El paso rehecho recibe la crítica en su contexto. Sin esto, `ada` reintenta a ciegas y lo más
probable es que repita el mismo error — y paguemos dos veces por la misma equivocación.

Es el mismo principio que hace útil una revisión de código: el hallazgo viaja con el archivo,
no en la cabeza del revisor.

### 3.3 · Un tope de vueltas, y quién lo decide

**Dos devoluciones por pipeline.** A la tercera, el pipeline **para y avisa**, entregando las
dos versiones para que Fernando decida.

Sin tope, dos modelos pueden discutir toda la noche gastando dinero real. Y el que decide
cuándo se acabó la discusión **no puede ser ninguno de los dos**: el árbitro no se concede
vueltas a sí mismo. El tope vive en configuración, no en el código.

### 3.4 · El dinero: la devolución gasta dentro de lo que ya aceptaste

Hoy `costo_max_aceptado_usd` es un parámetro por pedido y **no se persiste** (verificado:
`jacobs/continuar.py:213-228` lo recibe, `store.py` no lo guarda). Una devolución automática no
tendría contra qué presupuesto medirse.

Se persiste el tope aceptado en el pipeline. La devolución **cabe dentro de lo que el humano ya
aceptó, o no ocurre**: si rehacer los pasos supera lo que queda de ese tope, el pipeline para y
lo pide. No se inventa un permiso nuevo para gastar — se respeta el que ya se dio.

### 3.5 · El árbitro devuelve, no reescribe

**La línea que no se cruza.** Si el árbitro corrige el trabajo de `ada`, se volvió productor, y
su próxima auditoría es sobre sí mismo. Puede decir qué está mal y por qué; **no puede poner la
versión buena**.

Es la misma regla de sala limpia que ya hace cumplir `_con_arbitro` al crearse el plan, aplicada
al momento de devolver.

## 4. Qué NO se construye

- **Devolución a criterio del humano paso por paso.** El árbitro decide devolver dentro del tope
  y del presupuesto; si hiciera falta aprobación en cada vuelta, volvemos a un humano en el
  medio de cada paso, que es exactamente lo que esta ronda quita.
- **Que el árbitro escriba la corrección.** Ver §3.5.
- **Devolver a un paso distinto del que produjo el error.** Se devuelve a quien lo hizo. Si el
  error viene de más atrás, esa es una segunda devolución, no un salto.

## 5. Cómo se sabe que funcionó

1. Se corre el mismo caso real: un plan donde el paso de arquitectura propone sintaxis de otro
   motor. El árbitro lo devuelve, `ada` lo rehace **sin ese error**, y el pipeline termina con la
   arquitectura corregida.
2. Un veredicto sin estructura válida **no** dispara devolución y queda registrado.
3. Al tercer intento el pipeline para y avisa, con las dos versiones.
4. Una devolución que no cabe en el tope aceptado **no ocurre**: el pipeline para y lo pide.
5. Los tests de 1 a 4 se ven **rojos contra el código de hoy** antes de verse verdes.

En memoria de Jairo Urbina.
