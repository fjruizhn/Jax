# Extracción de La Gaceta a dos columnas

**Fecha:** 2026-10-03. **Trabajo:** biblioteca-legal-hn, Decreto 198-2026. **Responsable técnico:** Codex en hall9000, por encargo de Fernando vía jax-14.

## Hallazgo

La Gaceta 37,258 del 29-sep-2026 tiene 112 páginas con capa de texto y dos columnas. `procesamiento.extractores.pdf.extraer` sin indicación de diseño mezcló en la página 1 considerandos de columnas distintas dentro de una tabla falsa y devolvió `ok`. En la página 2 intercaló renglones de izquierda y derecha y devolvió `parcial` por una tabla fallida. La etiqueta de estado no detectaba el orden semántico roto. Fuente local: `~/Downloads/GACETA 29 DE SEPTIEMBRE DE 2026 37,258; SECCIÓN A MA; COMPLETA.pdf`, SHA-256 `157c62a1b68d6d85b59fb62a399d780c0cfd7387adc080d5ea6d21d65edcc72d`.

## Cambio

Se añadió `modo_lectura="dos_columnas"` al extractor PDF existente. Es opt-in: la compuerta y todas las ingestas anteriores mantienen el modo normal, orientado a documentos financieros con tablas. El nuevo modo lee la mitad izquierda completa y después la derecha, sin detección de tablas. Siempre devuelve `parcial` con la lista de páginas que exigen contraste visual. Un modo no reconocido devuelve `error` sin salida.

## Límite y comprobación

El corte geométrico puede partir encabezados que cruzan la página (se vio en la portada) y no resuelve artículos por sí solo. El consumidor debe comparar el texto con el PDF página por página; no debe promover el resultado a transcripción completa por su estado o por tener capa de texto. El test sintético alterna operadores de ambas columnas y fija orden, ausencia de tabla falsa y estado `parcial`. El smoke local de la Gaceta comprobó 112 etiquetas de página y cero tablas en este modo. No usó GPU.

**Alternativa descartada:** activar por heurística global en todo PDF. Habría cambiado expedientes financieros y arriesgado tablas, que son contenido crítico para esa ruta. **Pendiente:** el proyecto de la biblioteca legal debe estructurar artículos y contrastarlos contra el PDF antes de publicar `texto.md` como completo.
