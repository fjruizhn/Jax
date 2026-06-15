# MISIÓN: ESTUDIO DE MERCADO ATENEAERP — v2 (research verificado)

> Orquestador: JAX · Búsqueda: Hipatia · Estructura: Jekyll · Redacción y verificación: Hyde
> Esta misión se ejecuta por bloques. No intentar todo de una vez.

---

## REGLA DE HIERRO (no negociable)

**CERO datos sin fuente verificable.**

1. Toda cifra, porcentaje, monto o afirmación de hecho DEBE traer su URL real de origen.
2. Si no se encuentra fuente para un dato: se escribe literalmente **"NO VERIFICADO"** — NUNCA se inventa, NUNCA se rellena de memoria, NUNCA se pone un dato "ilustrativo".
3. Las citas de usuarios (quotes de foros) solo se incluyen si son textuales de una fuente con link. Si no hay link, se OMITE la cita. Prohibido reconstruir quotes de memoria.
4. Las gráficas se construyen SOLO con datos verificados y con su fuente al pie. Si no hay dato verificado, no hay gráfica — se pone una nota "dato pendiente de investigación primaria".
5. Si dos fuentes se contradicen, se marca **"CONFLICTO DE FUENTES"** y se citan AMBAS con sus links. Nunca inventar un conflicto para ilustrar.

Este informe es para uso interno del solicitante, que exige que su información sea MÁS rigurosa que la que daría a un tercero. Un dato sin fuente es peor que ningún dato.

---

## CONTEXTO DEL PRODUCTO

AteneaERP es un ERP SaaS multi-tenant para PyMEs. Propuesta de valor: el usuario solo registra compras y ventas; el sistema genera la contabilidad automáticamente. Multi-país (fiscalidad e identificadores por país). Precio en evaluación: ~$150/mes (a validar contra el mercado).

TESIS DE NEGOCIO A PROBAR: ERP ordena el negocio → data financiera limpia → la PyME se vuelve sujeto de crédito → el banco presta con menos riesgo. Canal hipotético: BANCOS que recomiendan/distribuyen el ERP a su cartera PyME.

---

## ALCANCE GEOGRÁFICO

GLOBAL. No asumir dónde está la oportunidad antes de medirla.
- Mirar: LATAM completo, Norteamérica, Europa, Asia-Pacífico, África.
- Medir dónde está la mayor necesidad insatisfecha (PyMEs sin ERP, dolor documentado) cruzada con dónde es alcanzable.
- Centroamérica se evalúa como un dato más (no como foco impuesto).
- La DATA decide el foco recomendado, no la suposición. Si la mayor oportunidad está en Asia o en otro lado, decirlo.

---

## LAS 3 PREGUNTAS QUE EL INFORME DEBE RESPONDER CON DATOS DUROS

Si estas tres no quedan contestadas con evidencia y fuente, el informe falló:

**P1 — PRECIO:** ¿El precio (~$150/mes u otro) valida contra lo que cobra la competencia REAL en cada mercado grande? Traer precios reales y verificables de los competidores principales por región, con URL.

**P2 — TESIS ERP→CRÉDITO:** ¿Hay evidencia dura (estudios, reportes, casos) de que usar software contable mejora el acceso a crédito de una PyME? ¿O sigue siendo hipótesis a probar en campo? Distinguir claramente: evidencia documentada vs. hipótesis. Citar fuentes.

**P3 — EL HUECO:** ¿Dónde está la mayor necesidad insatisfecha (qué mercado/segmento), y algún banco o player ya lo está llenando, o está abierto? Documentar con datos. Si la respuesta es "no hay evidencia de que un banco lo haga", eso ES un hallazgo (oportunidad), documentarlo como ausencia verificada.

---

## EJECUCIÓN POR BLOQUES (uno a la vez, para no saturar ni cortar por timeout)

### BLOQUE 1 — Tamaño y forma del mercado (Hipatia busca)
- Tamaño del mercado ERP para PyMEs/SMB: global y por región (USD, CAGR), con fuente y año.
- Penetración de ERP en PyMEs por región (% que usa ERP vs Excel/manual), con fuente.
- Dónde crece más rápido y dónde hay más PyMEs sin digitalizar.
- ENTREGABLE: lista de cifras, cada una con su URL y fecha. Marcar "NO VERIFICADO" lo que no se halle.

### BLOQUE 2 — Competencia y precios reales (Hipatia busca → responde P1)
- Competidores principales por región grande (LATAM: Siigo, Alegra, Bind, Aspel, CONTPAQi, Defontana, Nubox, ContaAzul; global: QuickBooks, Xero, Zoho, SAP B1, Odoo, etc.).
- PRECIO REAL de cada uno (plan de entrada, moneda local y USD), con URL a su página de precios.
- Localización fiscal: qué países cubre cada uno.
- ENTREGABLE: tabla competidor → precio real → países → fuente URL.

### BLOQUE 3 — Tesis ERP→crédito (Hipatia busca → responde P2)
- Estudios/reportes (BID, CAF, IFC, Banco Mundial, académicos) sobre si software contable mejora acceso a crédito PyME.
- Tasas de rechazo de crédito PyME por falta de estados financieros.
- Casos reales de bancos/fintechs que usen datos contables para scoring crediticio.
- ENTREGABLE: evidencia con fuente, separando claramente "documentado" de "no hay evidencia / hipótesis".

### BLOQUE 4 — El hueco y el canal bancario (Hipatia busca → responde P3)
- ¿Algún banco en el mundo ya ofrece/subsidia/distribuye un ERP a su cartera PyME? Buscar casos reales con fuente.
- Dónde está la mayor necesidad insatisfecha (cruce de PyMEs sin ERP + dolor documentado + obligación fiscal como facturación electrónica).
- ENTREGABLE: hallazgos con fuente. La ausencia de casos, si se confirma, es un hallazgo válido (documentar como "ausencia verificada").

### BLOQUE 5 — Dolor del usuario (Hipatia busca, con links)
- Quejas reales y verificables sobre ERPs existentes (precio, complejidad, soporte, localización).
- SOLO citas con link a la fuente (Reddit, G2, Capterra, Trustpilot con URL). Sin link = se omite.
- ENTREGABLE: quejas categorizadas, cada cita con su URL.

### ESTRUCTURA (Jekyll arma)
Con los hallazgos verificados de los 5 bloques, Jekyll organiza:
- Tabla comparativa de competidores (precio/cobertura/fuente).
- Las barreras de adopción ordenadas por frecuencia documentada.
- Cruce de oportunidad: necesidad insatisfecha vs. alcanzabilidad por mercado.
- Marcar cada dato con su origen.

### REDACCIÓN Y VERIFICACIÓN (Hyde redacta y comprueba)
Hyde produce el informe final en HTML, y ANTES de escribir cada dato, lo verifica con su propia búsqueda web (ya tiene WebSearch habilitado). Hyde:
- Pega la URL de fuente en cada cifra.
- Responde explícitamente P1, P2, P3 en secciones propias.
- Marca "NO VERIFICADO" lo que no pudo confirmar.
- NO inventa quotes, NO rellena gráficas con datos ilustrativos.
- Incluye una sección final "LIMITACIONES Y DATOS NO VERIFICADOS" listando honestamente qué quedó sin confirmar.

---

## FORMATO DEL INFORME HTML FINAL

- Archivo único, estilos embebidos en `<style>`, sin dependencias externas EXCEPTO Chart.js vía CDN (https://cdn.jsdelivr.net/npm/chart.js).
- Paleta: fondo #f0f0f0, header #0f0f0f, texto oscuro legible, acento violeta #7c3aed. PROHIBIDO verde como color principal.
- Tipografía: system-ui o Inter.
- Cada gráfica y tabla lleva su fuente al pie.
- Nombre del archivo: `ateneaerp_research_verificado.html`
- Guardar en: `/home/fruiz/jax/workspace/`

---

## RESPONSABLE FINAL DE AUDITORÍA

El informe terminado se entrega al solicitante, quien lo lleva a su revisor estratégico (Claude.ai) para auditar dato por dato que cada cifra tenga su fuente real antes de darlo por bueno. Hyde debe dejar el trabajo en condiciones de pasar esa auditoría: si un dato no aguanta la pregunta "¿cuál es la fuente?", no debe estar en el informe.
