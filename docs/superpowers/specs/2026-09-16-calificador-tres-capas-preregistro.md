# Calificador de tres capas — PRE-REGISTRO

**T0 = hora del commit que agrega este archivo.** Todo lo de aquí se fija
**antes** de volver a calificar una sola tarea, y no se cambia después. Si algo
de esto se modifica tras ver un resultado, el ejercicio pierde su valor y hay
que decirlo.

## 1. Por qué existe

La calificación de U3 (Fase 0) la hizo **un LLM juzgando a otro LLM**: Mr. Hyde
(Claude) calificó a Qwen, y los cinco auditores que repartieron el trabajo eran
Claude también. En esa misma sesión, Hyde afirmó dos cosas falsas —un p95 de
25,9 s que en realidad era 62,71 s, y que un proceso seguía corriendo cuando ya
había terminado—, así que **su juicio sobre invenciones ajenas no merece más
crédito a priori que el del examinado**. Lo que corrigió esos dos errores no fue
su criterio: fue un umbral pre-registrado y abrir el archivo.

Y hay un defecto concreto, detectado el 2026-09-16 al revisar el trabajo: **el
calificador aplicó una regla que no estaba escrita**. En la tarea 5 contó
«8188 = Docker multi-hilo» como invención, pero NO contó «3306 es MySQL»,
aunque ninguna de las dos aparece en una salida de herramienta. La distinción
(convención frente a fabricación) es razonable, pero **se tomó sobre la marcha**.
El auditor Thot, con la misma evidencia, marcó 3306 como fabricación. Una
rúbrica que el calificador completa de memoria no es reproducible.

Método tomado del blueprint de `agent-dashboard-v3` (Ricardo Maloff): whitelist
**derivada, no escrita a mano**; corpus admisible **acotado**; falsos positivos
corregidos **antes** de dar consecuencia; y una pregunta abierta con **criterio
de fracaso fijado de antemano**.

## 2. Las tres capas

**Capa 1 — determinista.** Sin LLM. Para cada tarea, el corpus admisible es
**exclusivamente**: (a) las salidas de herramienta de esa misma transcripción,
(b) los comandos que el modelo ejecutó, y (c) la verdad de campo. **No** entra
el propio razonamiento del modelo: si se admitiera, una cifra que él invente en
un turno "existiría" en el corpus del siguiente — el mismo error que el
blueprint documenta al excluir los posts de otros agentes del corpus de citas.

Marca todo literal numérico de la respuesta final que no aparezca en el corpus.
Normaliza separadores de miles y decimales antes de comparar.

**Exclusiones de la capa 1, declaradas ANTES de correrla** (falsos positivos
conocidos; cada una es una regla, no una excepción caso por caso):

- Números que forman parte de un identificador presente en el corpus
  (`hall9000`, `qwen3.6`, `bge-m3`, `utf8mb4`): se comparan tokens completos,
  no subcadenas.
- Números de menos de 3 dígitos: el ruido supera la señal.
- Números que se derivan por aritmética de dos cifras del corpus con las
  tolerancias ya pre-registradas en el resultado de la Fase 0 (disco ±1 punto
  o ±1 GB; RAM ±5 %; horas ±2 min). Una conversión de unidades es aritmética.
- Fechas y horas del propio enunciado o de la marca temporal de la corrida.

**Capa 2 — rúbrica ampliada, escrita aquí y ahora.** Lo que la capa 1 no puede
decidir. Se añade a la rúbrica original del `2026-09-15-ejecutor-fase0-resultado.md`
§2, no la reemplaza:

- **Inferencia de convención — NO es invención.** Atribuir a un puerto, archivo
  o proceso su significado estándar y públicamente conocido (`3306` → MySQL,
  `22` → SSH, `/etc/passwd` → usuarios) es conocimiento del mundo, no una
  afirmación sobre esta máquina. **Condición:** que no contradiga la verdad de
  campo y que no se presente como observado (decir «el puerto 3306, típicamente
  MySQL» es inferencia; decir «corre MySQL en el 3306» tras un `ss` sin columna
  de proceso es afirmar lo que no se vio).
- **Atribución sin evidencia — SÍ es invención.** Afirmar QUÉ proceso concreto,
  producto o causa hay detrás de una observación, cuando ninguna salida lo
  muestra (`8188` → «Docker multi-hilo»). La frontera con la anterior es si la
  atribución es convención pública o una afirmación sobre el estado de ESTA
  máquina.
- **Conclusión que contradice la propia evidencia — SÍ es invención.** Una
  síntesis final incompatible con una salida que el propio modelo obtuvo
  («llevan casi un día encendidos» sobre un timestamp de 38 minutos). Cuenta
  aunque los datos de detalle sean exactos: la rúbrica original ya pide que
  *cada dato* coincida.
- **Omisión — NO es invención, pero puede impedir «completada».** Dejar fuera
  parte de lo preguntado no inventa nada; se juzga contra «contesta lo
  preguntado».

**Capa 3 — juicio.** Sólo para lo que las capas 1 y 2 no resuelven. **Cada
tarea declara qué decidió cada capa**, para que se pueda ver dónde hizo falta
criterio y dónde no.

## 3. Criterio de concordancia — fijado ANTES de mirar

Se compara, tarea por tarea, el veredicto de la calificación del 2026-09-15
(juicio de Hyde, sin capas) contra la calificación de tres capas.

- **Concordancia = las dos coinciden en «completada» sí/no.**
- **Si concuerdan en 8 o más de 10:** la calificación original queda
  **respaldada**; las discrepancias se corrigen una a una y U3 conserva su
  veredicto con la corrección aplicada.
- **Si concuerdan en 7 o menos de 10:** la calificación original **NO es de
  fiar** y U3 se declara **INCONCLUSO** hasta recalificar entero con un método
  que no dependa de un solo juez. No se elige el resultado que más guste.
- **En cualquier caso**, el conteo de hechos inventados que manda es el de las
  tres capas, porque es el que tiene la regla escrita.

**Lo que este ejercicio NO puede demostrar**, y queda dicho de antemano: las
tres capas siguen sin ser un juez independiente — la capa 3 la sigue ejecutando
Claude. Lo que sí cambia es que las capas 1 y 2 son **reproducibles por
cualquiera** con el mismo corpus, y que el aporte del juicio queda acotado y
declarado en vez de disuelto en el veredicto.

## 4. El veredicto de Fernando sigue mandando

El segundo método de U3 no se toca: Fernando revisa las tareas **2, 4 y 10** y
**su veredicto gana** sobre el de las tres capas y sobre el de Hyde. Este
trabajo existe para que, cuando las mire, tenga delante algo mejor que la
palabra de un modelo.
