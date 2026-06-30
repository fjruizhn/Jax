# Misión Hyde — Router de intención de 7 facetas (consola + web)

> Ejecutar con: `jax --task ~/jax/missions/router-7facetas.md`
> Hipatia inicia con `/using-superpowers` y `/ruflo`.

---

## 1. Objetivo

Dejar el enrutamiento automático coherente con las **7 facetas reales**, en las
**dos** superficies (consola `router.py` y web `chat.py`). Hoy el clasificador de
consola solo conoce 4 facetas y manda lo técnico a Hyde (bug latente); la web no
conoce a Ada. Se corrige todo con un solo criterio.

Diseño validado por revisión externa cruzada (GPT + DeepSeek). Tres principios:
1. **Hyde FUERA del auto-routing de usuario** (es ejecutor, no conversador).
2. **Scoring con umbral**, no primer-match (≥2 señales, o 1 señal distintiva).
3. **Keywords distintivas**: el verbo manda sobre el tema; desempate
   crítica > formalización > interpretación.

## 2. Principios (HYDE activo)
- No suponer; leer antes de editar. Backup antes de modificar.
- Ningún cambio sin prueba: el gate (§7) valida cada faceta.
- Declarar incertidumbres.

## 3. Reconocimiento (read-only)
```bash
# Confirmar la función de normalización existente (quita acentos)
grep -n -B2 -A4 'unicodedata.combining' ~/jax/jax/core/router.py
# Rangos exactos a tocar en consola
grep -n 'TECH_KEYWORDS\|RESEARCH_KEYWORDS\|ARTS_KEYWORDS\|CLASSIFIER_PROMPT\|VALID_FACETAS\|def _keyword_route\|def _classify' ~/jax/jax/core/router.py
# Rangos exactos en web
grep -n '_AUTO_ROUTER\|def _auto_route' ~/jax-platform/backend/api/chat.py
```
Si no existe función de normalización en `router.py`, declararlo (la usamos en §5).

## 4. Diseño — dominios y keywords (normalizadas: minúsculas, sin acentos)

Matching: normalizar **texto y keyword** con la función existente; frases (con
espacio) se chequean como substring; palabras sueltas con límite de palabra.

**kimi** (implementación técnica)
`codigo, programar, programa, script, funcion, clase, metodo, modulo, libreria, api, endpoint, backend, frontend, implementar, implementa, construir, refactor, refactorizar, debug, depurar, bug, traceback, excepcion, compilar, test, tests, pytest, variable, bucle, array, regex, fastapi, react, typescript, javascript, python, sql, docker, nginx, commit, branch, merge`
STRONG: `refactor, implementar, debug, depurar, pytest, fastapi, docker, nginx, endpoint`

**hipatia** (investigación / actualidad)
`busca, buscar, investiga, investigar, verifica, verificar, fuentes, fuente, citas, referencias, noticias, noticia, actualidad, reciente, ultima, ultimo, vigente, precio, precios, cotizacion, mercado, ley, regulacion, normativa, paper, papers, estudio, informe, estadistica, lanzamiento, "version actual", "quien es"`
STRONG: `busca, buscar, investiga, investigar, noticias, fuentes`

**jekyll** (humanidades / reflexión)
`poesia, poema, cuento, novela, literatura, ensayo, arte, pintura, musica, filosofia, etica, estetica, humanidades, barroco, renacimiento, romanticismo, mito, mitologia, simbolo, simbolismo, metafora, narrativa, personaje, estilo, interpretacion, sentido, significado, reflexion, reflexiona, contempla, humanista, cultura, "historia del arte", "historia cultural"`
STRONG: `poema, poesia, filosofia, literatura, mitologia, "historia del arte"`

**thot** (crítica / auditoría / riesgo)
`audita, auditar, auditoria, critica, criticar, criticamente, cuestiona, cuestionar, adversarial, "abogado del diablo", riesgo, riesgos, falla, fallas, debilidad, debilidades, vulnerabilidad, vulnerabilidades, amenaza, amenazas, "threat model", "modelo de amenazas", ataque, "donde se rompe", "punto ciego", supuesto, supuestos, contraargumento, refuta, refutar, "no-go", "revisa criticamente"`
STRONG: `audita, auditar, auditoria, vulnerabilidad, "threat model", adversarial, refuta`

**ada** (formalización / lógica / matemática)
`formaliza, formalizar, formalizacion, "modelo formal", pseudocodigo, logica, demuestra, demostrar, demostracion, "prueba formal", teorema, lema, corolario, axioma, proposicion, invariante, invariantes, precondicion, postcondicion, "maquina de estados", automata, complejidad, "big o", "o(n)", "estructura de datos", grafo, arbol, matriz, vector, ecuacion, optimizacion, "funcion objetivo", matematica, calculo, algebra, probabilidad, determinista, induccion, algoritmo`
STRONG: `formaliza, formalizar, demuestra, demostrar, teorema, invariante, precondicion, postcondicion, complejidad, "maquina de estados"`

**jax_local** (default / fallback) — sin set propio; gana cuando nada llega al umbral.

**hyde** — NO participa del auto-routing. Solo por nombre explícito (vía `ALIASES`)
o modo Comando. Su encadenamiento inter-agente vive en el diseño de Las Manos, aparte.

### Regla de decisión (idéntica en ambas superficies)
1. Calcular `score[faceta]` = nº de keywords que matchean (frase = 1 hit).
2. `top` = faceta con mayor score. Empate → desempate por prioridad:
   **hipatia > thot > ada > kimi > jekyll**.
3. Si `score[top] >= 2` → enrutar a `top`.
4. Si `score[top] == 1` **y** la keyword matcheada es STRONG de `top` → enrutar a `top`.
5. Si no → **consola:** caer al clasificador LLM. **web:** caer a `jax_local`.

## 5. Cambios — CONSOLA (`~/jax/jax/core/router.py`)
- Backup `router.py.backup-router7-$TS`.
- **Reemplazar** `TECH_KEYWORDS / RESEARCH_KEYWORDS / ARTS_KEYWORDS` por los 5
  sets de §4 (`KIMI_KW, HIPATIA_KW, JEKYLL_KW, THOT_KW, ADA_KW`) + sus `*_STRONG`.
- **Reescribir** `_keyword_route()` con la regla de decisión de §4 (scoring +
  umbral + STRONG + desempate). Normalizar texto con la función existente.
  Hyde NO es un destino posible aquí.
- **Extender** `CLASSIFIER_PROMPT` a 6 facetas (sin hyde):
  ```
  Sos un clasificador de intencion. Responde con UNA SOLA PALABRA eligiendo la faceta:
  - jax_local = charla casual, saludos, conversacion cotidiana, nada de lo de abajo.
  - kimi = codigo, programacion, implementacion, debugging, infraestructura tecnica.
  - hipatia = investigacion, buscar info actual, noticias, fuentes, hechos verificables.
  - jekyll = humanidades, arte, literatura, filosofia, musica, interpretacion, reflexion.
  - thot = auditoria critica, riesgos, fallas, vulnerabilidades, revision adversarial.
  - ada = formalizacion, logica, algoritmos, demostraciones, matematica, invariantes, complejidad.
  Si dudas: interpretacion->jekyll; critica/riesgo->thot; formalizacion/demostracion->ada;
  codigo/implementacion->kimi; actualidad/fuentes->hipatia.
  NUNCA elijas hyde (es ejecutor, no conversador).
  Mensaje:
  {texto}
  Responde SOLO con: jax_local, kimi, hipatia, jekyll, thot o ada
  ```
- En `_classify()`, parsear la salida contra una lista **sin hyde**
  (`AUTO_FACETAS = ("jax_local","kimi","hipatia","jekyll","thot","ada")`),
  no contra `VALID_FACETAS` (que conserva hyde para invocación explícita).

## 6. Cambios — WEB (`~/jax-platform/backend/api/chat.py`)
- Backup `chat.py.backup-router7-$TS`.
- **Reemplazar** `_AUTO_ROUTER` por los 5 sets de §4 (sin hyde) + sus STRONG.
- **Reescribir** `_auto_route()` con la misma regla de decisión (scoring + umbral
  + STRONG + desempate). Sin clasificador LLM (fase 1).
- **Agregar logging:** por cada ruteo, registrar `(mensaje[:80], faceta_elegida,
  score, via)` a un log (ej. `logger.info`) para alimentar la decisión de Fase 2
  (si las keywords no alcanzan, ahí se evalúa un clasificador LLM web).

## 7. Gate de prueba (obligatorio, ambas superficies)
Probar este set y verificar el destino. **Consola:** `jax --task <archivo>` y leer
la línea `[tarea] Faceta:`. **Web:** importar y llamar `_auto_route(frase)` directo.

| Frase | Esperado |
|---|---|
| "implementa la funcion de hash" | kimi |
| "refactoriza este modulo" | kimi |
| "demuestra que el router es determinista" | ada |
| "cual es la complejidad de quicksort" | ada |
| "define invariantes para esta funcion" | ada |
| "audita la seguridad de este diseño" | thot |
| "encontra vulnerabilidades en el deploy" | thot |
| "analiza la metafora en este poema" | jekyll |
| "que significa el barroco" | jekyll |
| "busca noticias sobre el barroco" | hipatia |
| "cual es la version actual de fastapi" | hipatia |
| "hola maje" | jax_local |
| "gracias, listo" | jax_local |
| "configura nginx en el servidor" | kimi (NO hyde) |

**Casos ambiguos a monitorear** (no rompen el gate, se observan 2 semanas vía log):
`"optimiza esta funcion"` (ada↔kimi), `"implementa un algoritmo recursivo"`
(kimi↔ada), `"es etico este algoritmo"` (thot↔jekyll). El log de §6 los caza.

**Criterio de aceptación del gate:** las 14 filas caen donde dice, en consola y web.

## 8. Reporte final
Archivos tocados + diffs, tabla de resultados del gate (las 14, ambas superficies),
incertidumbres declaradas, rollback (`*.backup-router7-$TS`).
