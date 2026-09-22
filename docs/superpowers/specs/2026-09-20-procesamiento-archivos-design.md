# Procesamiento de archivos — Fase 1

**Fecha:** 2026-09-20 · **Autor:** Mr. Hyde, con Fernando Ruiz
**Estado:** diseño aprobado en conversación; pendiente revisión del spec escrito
**Rama:** `feat/procesamiento-archivos` (desde `origin/master` = `efb0159`)

---

## 1 · El problema, con número

Hoy cada dependencia de un pipeline de Jacobs reenvía **hasta 60.000 caracteres por paso
y por corrida**. El mismo brief, el mismo esquema, el mismo expediente, una y otra vez.

El ahorro **no** está en elegir mejor motor. Meter un PDF en un LLM es meterlo en el
contexto, que es exactamente el gasto que se quiere evitar. La conversión archivo → texto
es trabajo **determinista**.

> El ahorro está en **convertir una vez y cachear**.

## 2 · Categoría nueva: herramientas, no motores

Esto NO es una faceta ni un motor. Es una categoría nueva de **herramientas deterministas**.

| | Motor (faceta) | Herramienta (esto) |
|---|---|---|
| Cola de GPU | compite | **no la toca** |
| Costo | por token | **$0** |
| Resultado | probabilístico | **determinista, reproducible** |
| Verificación | calificando salidas | **test de igualdad exacta** |

### Simplificación que se gana por extraer ANTES del pipeline

Como la extracción ocurre antes de que arranque la cadena, **la fase 1 no toca Jacobs**:
ni el planner, ni el validador de planes, ni el esquema de steps. El extracto queda como
un archivo más y las facetas lo leen con el `file_read` que ya existe.

Esto saca del alcance la pieza más riesgosa que se había considerado: un tipo de step sin
faceta. Queda documentada como mejora futura (§10), no se implementa.

## 3 · Fuera de alcance (explícito)

Embeddings · búsqueda semántica · audio · video · explorador de archivos del lado servidor ·
compartir extractos entre trabajos distintos · escáner por hardware (SANE) · subida desde
el teléfono como camino propio · PaddleOCR (condicionado, ver §6).

## 4 · Componentes

| # | Componente | Qué hace | Dónde |
|---|---|---|---|
| 1 | **Embudo** | trae lo que llega solo: Nextcloud (sólo lectura, un sentido) + carpetas vigiladas | timer systemd, hall9000 |
| 2 | **Selector** | diálogo nativo de abrir archivo + resumen + confirmación | jax-platform (frontend) |
| 3 | **Ingesta** | copia a `fuente/`, calcula `sha256`, escribe la ficha | herramienta nueva |
| 4 | **Extractores** | convierten según tipo, con compuerta determinista | herramienta nueva |
| 5 | **Caché** | decide si extraer o si ya está hecho | dentro de 3 y 4 |

### Almacenamiento

```
jax-workspace/<trabajo>/
    fuente/      ← originales. INMUTABLES. con su sha256
    procesado/   ← extractos + fichas. BORRABLE y regenerable
```

**Invariante central:** `procesado/` se puede borrar entero en cualquier momento y el
sistema lo reconstruye. Nada que no sea reconstruible vive ahí. Un caché del que no te
podés fiar para borrarlo no es un caché: es una segunda base de datos.

Vive **dentro de `JAX_WORKSPACE_DIR`** para reusar el jail de rutas ya probado
(`tool_authority.resolve_jailed_path`: forma canónica, symlinks, `..`, rutas absolutas,
con tests adversariales). **No se escribe un jail nuevo** — el código de seguridad es el
peor lugar para estrenar.

### Entradas

Dos, que terminan en el mismo sitio:

1. **El Selector** — el diálogo nativo del sistema operativo. Navega la máquina donde corre
   el NAVEGADOR. Como Chrome corre en hall9000, ya ve `/home/fruiz/atem-ai` (sshfs a
   `172.16.20.11:/`, verificado 2026-09-20) sin que escribamos navegación ni guardemos
   credenciales de red. Límite conocido: desde la Mac o el iPad muestra los archivos de
   ESE equipo, no los de hall9000.
2. **El Embudo** — lo que llega solo. Nextcloud vive en macmini-bridge
   (`172.16.20.20`, verificado: `nextcloud.rich-hn.com`, 184 `.xlsx` en su datadir).

**Nextcloud se lee, nunca se escribe.** Copia de un solo sentido vía WebDAV (rclone), con
usuario propio de sólo lectura y credencial en `/etc/jax/.env`. Motivo: son los documentos
originales de los clientes. Con lectura de un sentido, un bug nuestro **no puede** borrarle
el expediente a un cliente. Nextcloud sigue siendo la fuente de verdad.

### Antes de copiar, se dice qué se va a copiar

El Selector muestra cuántos archivos, cuánto pesan, de qué tipos y cuáles se ignoran, y
pide confirmación. "Seleccioné una carpeta y se colgó todo" es un modo de fallo evitable.

La confirmación va en **ventana propia** con i18n y tema, nunca `confirm()`/`alert()` del
navegador (política de Fernando, 2026-09-15). El control automático debe buscar también la
forma DESNUDA (`confirm(`, `alert(`, `prompt(` sin `window.` delante).

## 5 · La compuerta — nadie gasta CPU sin motivo

```
archivo
   │
   ├─ .docx ─────────► python-docx        │ .doc viejo → LibreOffice
   │
   ├─ .xlsx / .xls ──► openpyxl, TODAS las hojas
   │                     └─ celda sin valor cacheado → LibreOffice recalcula
   │
   ├─ .pdf ─┬─ ¿capa de texto? ──SÍ──► pdfplumber (conserva tablas)
   │        │                            └─ tabla fallida → Docling
   │        └──────────────────────NO──► OCR (tesseract)
   │
   ├─ imagen ────────► OCR (tesseract)
   │
   └─ otro ──────────► se copia a fuente/, estado `sin_extractor`
                       (NO se inventa un extracto vacío)
```

Cada capa corre **sólo si la anterior no alcanzó**. La detección de "PDF con capa de texto"
es determinista: se le pide el texto y si vuelve casi vacío, es escaneado.

## 6 · Herramientas: lo medido el 2026-09-20

Todo lo de abajo se verificó en hall9000 ese día, con archivos reales de Fernando.

| Herramienta | Estado | Evidencia |
|---|---|---|
| LibreOffice 26.2.5.2 | instalado | convierte `.xlsx` en **0,4 s** |
| tesseract 5.5.0 + `spa` | **instalado hoy** | OCR correcto con tildes y guion largo |
| pypdf 6.19.0 | instalado | — |
| Pillow 12.1.1 | instalado | — |
| `openpyxl`, `pdfplumber`, `python-docx` | **faltan** | se instalan, en venv propio |
| `pandas` | **NO se instala** | peso sin beneficio: sirve para analizar, no para extraer |

### El defecto que justifica openpyxl

Sobre `EEFF Cierre 2022 A 06-2025 NETEADOS.xlsx` (6 hojas, con fórmulas):

> **LibreOffice `--convert-to csv` convirtió UNA sola hoja de seis. Sin error, sin aviso.**

Verde con el trabajo sin hacer — el modo de fallo dominante de esta casa. `openpyxl` lee
las seis, distingue fórmula de dato y ve celdas combinadas.

Medición adicional: de 16 celdas con fórmula, **15 traían valor cacheado y 1 no**. Por eso
LibreOffice no se retira: entra a recalcular justo esa.

### PaddleOCR — condicionado, no por las dudas

Los clientes mandan documentos escaneados **con el app del teléfono**, que ya endereza,
recorta y corrige contraste. Eso no es una foto cruda y tesseract se defiende.

**PaddleOCR se instala SOLO si tesseract falla** el criterio §7.A.2 contra PDFs de escáner
de teléfono reales. Sin esa evidencia, no entra.

### Docling — condicionado, misma regla

`pdfplumber` es el mejor extractor de tablas de PDF nativo y pesa una fracción de Docling.
Se prueban los dos sobre un estado financiero real y **gana el que conserve mejor la tabla**.
Docling sólo entra si pdfplumber pierde.

## 7 · Criterio de éxito

### A · Correctitud — sin esto no se despliega

1. `EEFF Cierre 2022 A 06-2025 NETEADOS.xlsx` produce **6 extractos, no 1**.
   *(Hoy produce 1: el test DEBE fallar contra el código actual, o no mide nada.)*
2. De un estado financiero escaneado, **10 cifras elegidas a mano por Fernando** salen correctas.
3. Un PDF nativo con tabla conserva filas y columnas, verificado contra el original.
4. Nada entra a `fuente/` sin `sha256` y ficha.
5. Se cambia un original → el sistema **regenera**. Probado.
6. Se borra `procesado/` entero → se reconstruye idéntico. Probado.
7. **Cero escrituras hacia Nextcloud**, verificado DESDE Nextcloud, no desde nuestro lado.

### B · Utilidad — CORREGIDO el 2026-09-21, y puede decir que NO

> **El criterio anterior estaba MAL PLANTEADO, y el error fue de Hyde.** Decía: *"los
> tokens de entrada bajan al menos 30 %, medido contra el pipeline del ERP"*. Los tres
> archivos de ese pipeline (`brief-multistore.md`, `ateneaerp-estructura.md`,
> `ateneaerp-esquema.sql`) son **texto plano**: la compuerta responde `sin_extractor`, y
> hace bien, porque no hay nada que extraer. Medir ahí habría dado cero ahorro por la razón
> equivocada.
>
> **Este sistema no reduce tokens sobre texto que ya era texto.** Hace usables documentos
> binarios que antes no se podían leer, y evita reextraerlos. El criterio corregido mide
> eso. *(Corregido con GO de Fernando, 2026-09-21.)*

**Medido el 2026-09-21 con documentos reales de Nextcloud** (borrados tras medir):

| Documento | Original | Extracto | Reducción | Tiempo | Estado |
|---|---:|---:|---:|---:|---|
| EEFF NETEADOS `.xlsx` | 68.464 B | 22.261 B | 3,1× | 0,09 s | ok |
| Prospección `.xlsx` | 410.001 B | 1.522 B | 269× | 0,01 s | ok |
| EEFF 2022 `.pdf` escaneado, 17 pág. | 3.131.619 B | 38.292 B | 81,8× | 46,2 s | parcial |

#### B.1 · Lo que MATA el proyecto si falla

**Las 10 cifras que elige Fernando** de un estado financiero escaneado real salen
correctas. Si el OCR no lee sus documentos de forma confiable, todo lo demás da igual:
un extracto que no se puede creer no sirve para nada.

#### B.2 · Utilidad — binario y atado a un límite real del sistema

**El extracto de cada documento tiene que caber en el tope de 200 KB de `file_read`**,
y el original no.

No es un porcentaje inventado: es el límite que ya existe en `tool_authority.MAX_READ_BYTES`.
El PDF de 3,1 MB **no cabe**; su extracto de 38 KB **sí**. Esa es, literalmente, la
diferencia entre "no se puede usar" y "se puede usar".

Se mide sobre una muestra de **al menos 20 documentos reales** y se registra cuántos caben.

#### B.3 · El caché

Segundo pase sobre el mismo archivo: **cero trabajo**. Binario. Ya probado en el Task 8.

#### B.4 · Que `parcial` signifique algo

Medido el 2026-09-21: el PDF escaneado real dio **16 de 17 páginas con dudas**, en su
mayoría por los **puntos suspensivos del índice**, que el OCR lee como basura con confianza
0,0.

El sistema hace lo que se le pidió, pero **si `parcial` es el estado de casi todos los
documentos reales, deja de ser una señal y pasa a ser ruido de fondo**.

Criterio: sobre la muestra de 20 documentos, **`parcial` no puede ser el estado de más del
60 %**. Si lo es, el umbral está mal calibrado y hay que corregirlo antes del GO.

### C · Rendimiento — las cuatro reglas

- PDF escaneado de 50 páginas: tiempo **medido y registrado**, no estimado.
- 100 archivos de golpe no tumban nada **ni le quitan GPU a las facetas**. La GPU ya está
  al 80 % (27,5 de 34,2 GB, medido 2026-09-20): que esto no la toque se **verifica**, no
  se supone.

### D · Sin regresión

Tests adversariales del jail siguen verdes · 4 servicios siguen `active` · ningún test
existente se rompe.

## 8 · Cómo falla — cerrado, siempre

| Situación | Comportamiento |
|---|---|
| El extractor falla | `estado: "error"` + razón. **No escribe extracto.** |
| Extrae de menos (5 de 6 hojas) | `estado: "parcial"` y lo dice. Nunca pasa por completo. |
| El original cambió (sha256 distinto) | detecta, regenera, no sirve el viejo |
| Tipo sin extractor | `estado: "sin_extractor"`. El archivo se guarda igual. |
| Una faceta pide un extracto en error | **recibe el error, no un texto vacío** |

La última fila es la línea entre esto y un desastre silencioso: un extracto vacío entregado
como bueno hace que el modelo **invente** lo que no pudo leer (Principio VIII).

## 9 · Contrato del extracto

El formato **depende del origen**. Forzar todo a `.txt` plano destruye lo que importa.

| Origen | Extracto | Por qué |
|---|---|---|
| PDF, Word | Markdown | conserva secciones y tablas |
| Excel | **un CSV por hoja** | conserva filas y columnas |
| Imagen | texto plano + OCR crudo | no hay estructura que conservar |

Ficha por extracto:

```json
{
  "sha256": "…",
  "origen": "fuente/EEFF-2025.xlsx",
  "extractor": "openpyxl",
  "extractor_version": "3.1.5",
  "fecha": "2026-09-20T23:14:00-06:00",
  "hojas": 6,
  "hojas_extraidas": 6,
  "estado": "ok"
}
```

`extractor` + `extractor_version` permiten reprocesar todo cuando cambie una herramienta,
**sin releer los originales a ciegas**. Es lo que faltó en la migración a bge-m3.

El `sha256` se guarda desde el día uno aunque hoy no se comparta entre trabajos: el día que
se quiera deduplicar, la huella ya está y no hay que reprocesar nada.

## 9-bis · Limitaciones CONOCIDAS de la fase 1

No son descuidos: son decisiones, y se escriben acá para que nadie las descubra en
producción creyendo que son defectos.

### PDF híbrido: los anexos escaneados no llegan a OCR

`tiene_capa_de_texto()` es todo-o-nada: basta **una** página con texto para que el PDF
entero se clasifique como nativo. Un expediente con portada impresa y anexos escaneados
extrae la portada y **no manda los anexos a OCR**.

**Mitigación, y es la razón por la que se acepta:** la pérdida **no es silenciosa**.
`pdf.py` devuelve `parcial` y nombra en `detalle["paginas_sin_texto"]` exactamente qué
páginas no dieron nada.

> La regla de esta casa no es *"no perder nunca"* —eso es imposible—: es **"no perder en
> silencio"**. Una pérdida declarada es un pendiente; una pérdida callada es un dato
> inventado esperando a ocurrir.

Arreglarlo bien significa OCR por página dentro de un PDF mixto y decidir cómo se mezclan
los dos extractos. Es una decisión de diseño nueva, va a fase 2, y **Fernando está avisado**.

### Restricción operativa nueva: el trabajo vive DENTRO del workspace

Desde el arreglo del jail (2026-09-21), `ingerir()` **rechaza con `ValueError`** cualquier
`trabajo` cuya forma canónica quede fuera de `tool_authority.WORKSPACE_ROOT`
(`/home/fruiz/jax-workspace` en producción). Una carpeta de cliente fuera de ahí ya no
funciona.

Es exactamente lo que se pidió —reusar el jail probado en vez de estrenar código de
seguridad— pero **cambia el contrato de uso** y por eso se escribe acá: quien llame a la
ingesta tiene que armar el trabajo dentro del workspace.

### El umbral de confianza del OCR: CALIBRADO el 2026-09-21, sin cambio justificado

Se midió sobre **10 documentos escaneados reales** (51 páginas, 1.295 cifras) usando la
**aritmética como verdad de referencia**: una cifra que participa en una suma que cuadra
está verificadamente bien leída, sin que nadie tenga que transcribir nada.

| | Cifras **verificadas** | Cifras **no verificadas** |
|---|---|---|
| Confianza mediana | 92,59 | **96,22** |

**Los dos grupos no se separan** — el no verificado tiene confianza *más alta*. Mover el
umbral de 60 a 80 apenas mejora el rechazo (7 % → 13 %) y empieza a perder cifras buenas
(96,8 % → 92 % de aciertos).

**Los valores se dejan como están** (`UMBRAL_CONFIANZA_PROMEDIO = 70`,
`CONFIANZA_MINIMA_PALABRA = 60`), ahora **sabiendo que son arbitrarios** en vez de creer
que estaban calibrados. Eso vale más que el número anterior.

> **Salvedad del método, que no se borra:** *"no verificada"* no es lo mismo que
> *"incorrecta"* — el método sólo prueba positivos. Esto no demuestra que la confianza de
> tesseract sea inútil en general; demuestra que **en este rango, con este método, no
> discrimina**. Si alguna vez hace falta una señal que sí separe, hay que buscar otra cosa.

### El DPI de rasterizado: 300, decidido el 2026-09-21 con datos reales

Un experimento **sintético** prometía −71 % de tiempo sin perder exactitud. Medido sobre
los 10 documentos reales, el ahorro es **59,2 %** — y la exactitud **sí se pierde**, justo
en el caso que más importa:

En los **3 documentos que son fotos de celular** (tipo CamScanner, que es lo que mandan los
clientes), 150 DPI destruye las cifras. Uno pierde el **100 %** de sus sumas verificables
(12 → 0); otro el 80 % (5 → 1). El caso del inventario que este spec cita como ejemplo
—las cuatro cifras que suman 25.847.583,99— **desaparece entero**: esa página del balance
cae de 49 cifras reconocidas a 17.

El sintético daba 20/20 porque era texto renderizado limpio. Una foto de celular tiene
desenfoque y ruido que a 150 DPI ya no se recuperan.

> **Lección de método: un experimento sintético puede mentir en la dirección cómoda.** Daba
> exactamente el resultado que uno quería oír, y con documentos reales se cayó.

Queda anotada, sin implementar, una tercera vía: **rasterizar a 150 y reintentar a 300 si
las cifras no cuadran** — el mismo principio de la compuerta, el caro sólo si el barato no
alcanzó. Es diseño nuevo con su propio riesgo, para ahorrar lo que menos falta hace.

### El costo de leer dos veces no está medido

Excel abre el libro dos veces (valores y fórmulas) y el OCR invoca tesseract dos veces por
página (texto y confianza), más el rasterizado. Se aceptó a cambio de no perder datos en
silencio. **Sin número medido no hay GO**: lo mide el Task 10 (regla 4 del rendimiento).

## 10 · Mejoras futuras (NO ahora)

Step de herramienta sin faceta dentro de Jacobs · dedup de extractos entre trabajos ·
explorador de archivos del lado servidor con contrato de autoridad propio · escáner por
hardware · embeddings y búsqueda semántica · audio y video · montaje de atem-ai en
`/etc/fstab` (hoy es manual y **no sobrevive a un reinicio**, verificado 2026-09-20).

## 11 · Pruebas

TDD. **Cada test se corre primero contra el código de hoy y tiene que fallar** — un test
que pasa antes del cambio no mide el cambio.

Archivos de prueba: **reales, de Fernando**, de los 184 `.xlsx` y los PDFs de Nextcloud,
copiados a un juego fijo. Nada de PDFs de juguete: el caso difícil es el estado financiero
con celdas combinadas y dos reportes en la misma hoja — ya verificado que existe en sus
archivos.

Los archivos de prueba contienen datos de clientes: **no entran al repositorio**. Van a un
directorio de pruebas fuera de git, y el test se salta con un mensaje claro si no están.
