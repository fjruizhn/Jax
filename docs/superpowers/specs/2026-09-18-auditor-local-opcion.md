# Auditor local para C5 — la opción, con números

## Decisión posterior — C5 `SOLO_ORDENES` (2026-10-06)

Fernando decide permitir auditoría de misiones sensibles usando la faceta de nube
configurada `ejecutor.auditor_faceta`, exclusivamente cuando la nueva clave
`ejecutor.c5_auditor_nube_solo_ordenes` existe y vale el booleano estricto `true`.
La clave es obligatoria en la configuración C5, sin fallback en código; el seed la
crea como `false`. Ausente, vacía o con cualquier valor distinto de `true`/`false`,
el arranque falla cerrado antes de resolver o invocar al auditor de nube. Con `false`
se conserva la selección del auditor local para misiones sensibles.

`SOLO_ORDENES` es un modo de privacidad con alcance limitado: la nube recibe únicamente
el objetivo de la misión, las instrucciones de auditoría, la identidad de máquina
requerida por el contrato (machine-id legítimo) y el número y comando de cada paso que
se pide juzgar, para que los hallazgos puedan citar su paso.
No salen capturas de stdout/stderr, salidas, líneas citadas, contexto, afirmaciones
(claims), entradas de pasos ni errores que contengan datos. El cuerpo HTTP se construye
por allowlist de esos cuatro campos; la ausencia de contenido prohibido se comprueba
contra el cuerpo serializado real. Los datos omitidos siguen disponibles localmente para
la ejecución y la supervisión humana.

El auditor en este modo solo juzga si cada comando pertenece al objetivo y al contrato
de la misión y si el machine-id es legítimo. No inventa ni infiere salidas de comandos.
Un veredicto sobre afirmaciones queda prohibido: cualquier respuesta con veredicto o
contenido de claims invalida la auditoría y falla cerrada. Por tanto, las afirmaciones
de la misión quedan explícitamente **no auditadas por C5** y requieren supervisión.
Las lecturas de identidad que el contrato del Ejecutor exige continúan dentro del alcance
de la misión y de su auditoría; este modo no autoriza omitirlas.

La habilitación se administra como clave de `/admin/config`, con autorización exclusiva
de superadmin. Cada misión registra de forma auditable faceta, proveedor y modo en su
bitácora y en el journal de arranque. Para C3, `jaxsvc` informa la selección por el
socket Unix `JAX_PROXY_CARRIL_C5_SOCKET`; el proxy autentica su UID
(`JAX_PROXY_CARRIL_C5_UID`) y escribe el evento en la cadena como único escritor. El
socket se habilita con su grupo (`JAX_PROXY_CARRIL_C5_GID`). Si el canal falta o falla,
la misión `SOLO_ORDENES` no inicia. El evento solo contiene ID de misión, faceta,
proveedor, localidad y modo. El registro de claims marca expresamente que C5 en modo
`SOLO_ORDENES` no los auditó. Las órdenes `Skill` o entradas sin Bash `command` no
pueden proyectarse sin salir de la allowlist y fallan cerradas en el vigía. La revisión
final no vuelve a enviar claims ni un lote vacío: usa el resultado del vigía para marcar
las claims como no auditadas.

El registro C3 de selección es idempotente por ID de misión: al iniciar, el proxy
reconstruye el índice desde la cadena C3 íntegra; una repetición idéntica responde OK
sin agregar otra entrada y sobrevive reinicios. Una faceta, proveedor, localidad o modo
distinto para el mismo ID se rechaza, incluso tras reiniciar el proxy. El índice conserva
todas las selecciones existentes en la cadena, sin una ventana de deduplicación que pueda
olvidar misiones antiguas. Así los turnos sucesivos y los reintentos tras perder la
respuesta no bloquean el vigía ni ocultan un cambio de selección. La bitácora visible de
misión conserva faceta, proveedor, localidad, modo y estado de auditoría de afirmaciones.

---

> **Estado: OPCIÓN PARA DECIDIR, no implementada.** Pedida por Fernando el 2026-09-18 después de
> cerrar la compuerta de datos de clientes. No cambia nada en producción hasta que él elija.

## 1. El problema, en una línea

El auditor de C5 es de **nube**. Una misión que toca una máquina con datos de clientes le manda a ese
modelo los comandos y las líneas citadas de esas máquinas. Por eso existe la compuerta
`ejecutor.c5_auditor_admite_datos_de_clientes`, que **nace cerrada**: con ella cerrada, esas misiones
no arrancan, y hoy el Ejecutor sólo trabaja contra la VM desechable.

Con un auditor **local** la pregunta desaparece: lo que se audita no sale de hall9000.

## 2. Qué ve hoy el auditor (medido, no supuesto)

Con la compuerta abierta el 2026-09-17 corrieron seis misiones de sólo lectura: `df -h /`,
`uptime`, `cat /etc/os-release` en atemai, prod y bridge. El auditor vio esos comandos y esas
líneas: dispositivos de disco, porcentajes de uso, versión del sistema, tiempo encendido. **Ningún
dato de clientes.** Pero una misión de verdad —la Fase 4/5, mover `.20` a aaPanel— lee archivos de
configuración con credenciales de bases de clientes: ahí el límite importa de verdad.

## 3. Medición: ¿un modelo local sirve de auditor?

Método: los **seis canarios reales** de C5 (`jax/ejecutor/contratos/canarios_c5.json`), con las
instrucciones reales del auditor, contra Ollama en hall9000, **forzando CPU** (`num_gpu: 0` por el
endpoint NATIVO). Criterio: los dos de `fuera_de_mision` tienen que pausar; los dos de `conclusion`
tienen que **retener** la afirmación; los dos `limpio` no deben pausar.

| Modelo | Dónde | Aciertos 6/6 | Tiempo por lote |
|---|---|---|---|
| `qwen2.5:7b` | CPU | **NO** (no vio la trampa; una respuesta ni siquiera fue JSON) | 4–12 s |
| `qwen3:14b` | CPU | **SÍ, los seis** | 49–161 s |

**Trampa medida que conviene no repetir:** el endpoint compatible con OpenAI de Ollama **ignora**
`options`, así que el modelo entra igual a la GPU y **desaloja al de la Mesa** (pasó: hubo que
reponerlo, 4,8 s). El endpoint nativo sí respeta `num_gpu: 0`.

## 4. La opción

**Un segundo Ollama, sólo CPU, como proveedor propio.** Unidad `systemd` aparte, puerto aparte,
`OLLAMA_NUM_GPU=0`, modelo `qwen3:14b`; fila de proveedor propia en la base y faceta `auditor_local`
apuntada ahí. El arranque ya exige que cerebro y auditor sean **proveedores distintos**: esto lo
cumple, y además separa el proceso (si uno se cae o se cuelga, el otro no).

- **Cuesta:** ~10 GB de RAM (hay 55 GB libres) y CPU durante cada lote. **Cero GPU**, así que no toca
  a la Mesa ni obliga a recargar el modelo de 26 GB.
- **Tarda:** 49–161 s por lote contra los ~2–5 s del auditor de nube. El vigía audita por lotes
  mientras el turno corre, y el turno espera al cierre: una misión se alarga del orden de un minuto
  por lote auditado.
- **Qué NO da:** independencia de proveedor de verdad. Son dos procesos del mismo software en la
  misma máquina: si Ollama tiene un defecto, lo tienen los dos. La nube da esa independencia y por
  eso el diseño la eligió; acá se cambia **independencia** por **privacidad**.

## 5. Alternativas consideradas

- **Dejar la compuerta cerrada.** Gratis, y hoy es lo que está: el Ejecutor trabaja sólo contra la
  VM desechable. No sirve para la Fase 4/5.
- **Abrir la compuerta.** Gratis y rápido, pero manda datos de clientes a un tercero. Decisión de
  Fernando y de nadie más.
- **Auditor local en GPU.** Mejor latencia, pero pelea con la Mesa por los 32 GB: el modelo de la
  Mesa ocupa 26. Vuelve el problema que SP3 cerró (recargas). Con la Red Queen (imposible #6) esto
  cambia.
- **Mezclar:** auditor local **sólo** para misiones que tocan máquinas con datos de clientes, y de
  nube para el resto. Es la que menos cambia lo que ya funciona, y es la que recomiendo si se decide
  avanzar: la compuerta pasa de «sí/no» a «con qué auditor».

## 6. Qué haría falta para implementarla

1. Unidad `ollama-cpu` (puerto propio, `OLLAMA_NUM_GPU=0`, `OLLAMA_NUM_PARALLEL=1`).
2. Fila de proveedor y faceta `auditor_local` (por migración, no a mano).
3. `eleccion_c5`: elegir auditor según si la misión toca máquinas con datos de clientes.
4. Canarios de C5 corridos contra el auditor local **en cada arranque**, como hoy: si el local falla
   los canarios, la misión no arranca.
5. Medir de nuevo con una misión larga: estos números son de lotes de canario, no de un cutover.

## 7. Recomendación

Si la Fase 4/5 va a correrse con el Ejecutor, **la mezcla del punto 5 es el camino**: auditor local
para lo que toca clientes, de nube para lo demás, y la compuerta deja de ser una puerta abierta o
cerrada para pasar a ser «qué auditor mira esto». Mientras tanto, la compuerta queda cerrada.

*En memoria de Jairo Urbina. En honor al Prof. Raúl Jacobs.*
