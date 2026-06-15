# CONTEXT.md — JAX 2.0 (Biblioteca de Alejandría)

> Documento canónico del proyecto JAX. Última actualización: **15 de junio de 2026** (FIX GROUNDING de Hipatia — fallo silencioso muerto, etiquetas de verificación + origen de autoridad en todas las facetas). Previas: 14-jun (LAS MANOS, kill switch en vuelo); 4-jun (hilo compartido + qwen2.5:7b + VOZ Kokoro + clasificador local).
> Regla de la casa: **NO SUPONER. Medir dos veces, cortar después.** Si algo de este documento contradice la realidad del servidor, gana la realidad — y se corrige este documento.

## 1. Qué es JAX

JAX (se pronuncia **"Yax"**, J española) es el asistente personal y futura plataforma de orquestación de Fernando Ruiz. Un solo ser con **4 facetas** (no 4 asistentes). Núcleo orquestador propio en **Python puro + asyncio**, sin frameworks. Honra la memoria de **Jairo Urbina**, amigo, socio y pionero del software libre en Honduras — su espíritu va en el system prompt de cada faceta. Easter egg: `IDE1990`.

Visión: JAX es base y laboratorio de un producto comercial globalmente escalable (multi-AI, multi-usuario, pricing por niveles) y fundación futura para AteneaERP y HAMMURABI.

## 2. Hardware y entorno (hall9000)

- **hall9000** (172.16.20.5): Ryzen 5 8500G, 30GB RAM, GPU AMD RX 9060 XT 16GB vía **Vulkan** (RADV GFX1200; ROCm inmaduro para gfx1200, diferido). Ubuntu 24.04.4. SSH puerto **58292** (verificado jun-2026; las notas viejas decían 58291).
- Proyecto: `~/jax/`, venv `~/jax/.venv` (Python 3.12; httpx, pydantic; **sin torch**).
- Credenciales: `/etc/jax/.env` (chmod 600): DEEPSEEK_API_KEY, GEMINI_API_KEY, JAX_DB_*.
- Kill Switch: `/etc/jax/PAUSE` (si existe, JAX no invoca músculos).
- Audio: parlantes en `plughw:2,0` (card 2, ALC897 Analog, conector verde).
- Arranque: `cd ~/jax && set -a; source /etc/jax/.env; set +a && PYTHONPATH=. .venv/bin/python -m jax.core.main`

## 3. Las 4 facetas (estado real)

| Faceta | Icono | Músculo | Modelo | Rol | Voz |
|---|---|---|---|---|---|
| JAX local | 🏠 | OllamaMuscle (GPU local) | **qwen2.5:7b** (default desde jun-4; 58-84 t/s, 100% GPU, 8.2GB VRAM, ctx 32K) | Conversación cotidiana, privada. Tono hondureño **sobrio**: "maje" con medida, sin groserías, sin modismos ajenos | em_alex @1.0 |
| Hyde | 🔧 | SubprocessMuscle (`claude -p`) | sonnet | Técnico: código, infra | em_santa @1.0 |
| Jekyll | 🧠 | HttpMuscle (DeepSeek) | deepseek-chat | Humanista erudito, español neutro formal | em_santa @0.85 |
| Hipatia | 🔍 | HttpMuscle (Gemini + grounding) | gemini | Investigación con fuentes web | ef_dora @1.0 |

Notas: llama3.2:3b sigue en `models_allowed` (tareas mecánicas rápidas). qwen2.5:14b permitido, no medido. ⚠️ Gemini en plan gratuito: **rate limits frecuentes** (Hipatia se quedó sin tokens 2 veces el 3-4 jun) — pendiente subir tier o espaciar uso. ⚠️ Jekyll **no tiene búsqueda web**: jamás usarlo como investigador (alucina con confianza; lección aprendida — recomendó config TTS inexistente).

## 4. Arquitectura del código (`~/jax/jax/`)

- `core/main.py` — REPL. Input vía **run_in_executor** (no congela el event loop; la voz suena de fondo). Comandos: `/voz on|off`, `/callate`, `/fact ...`, `salir`. Hilo de conversación en RAM + guardado MariaDB + voz como task de fondo. Kill switch antes de cada invoke. Errores humanizados.
- `core/router.py` — Híbrido: easter egg → adiós → invocar/fijar → manual → saludos puros → keywords de dominio → **clasificador LLM local** → default. **Blindado a tildes/voseo** (`_sin_tildes`: "traé"="trae", "adiós"="adios"; jun-4). Clasificador: jax_local (qwen2.5:7b), medido **~200 ms**, 4/4 aciertos; fallback a default si falla; migrable vía `set_classifier()`.
- `muscles/base.py` — Contrato `Muscle.invoke(prompt, model=None, history=None)`. HttpMuscle (DeepSeek/Gemini; grounding adjunta fuentes). Excepciones: ModelNotAllowed (fallo duro), Timeout, Invocation.
- `muscles/ollama_muscle.py` — localhost:11434/api/chat, **GPU_SEMAPHORE=1** (una inferencia GPU a la vez).
- `muscles/subprocess_muscle.py` — Hyde: `claude -p --append-system-prompt`, historial serializado a texto (la CLI no acepta array), kill+wait al timeout.
- `voice/kokoro_worker.py` — Proceso dedicado de TTS. Corre con el venv de Kokoro (`~/kokoro-test/.venv/bin/python`), **NUNCA** con el de JAX. `lang_code='e'` (⚠️ 'a' es inglés — error que propuso Deep, corregido). Voces directas (sin mapas). Protocolo: JSON por línea en stdin → 4 bytes LE (tamaño) + WAV por stdout; tamaño 0 = error.
- `voice/tts.py` — VoiceEngine: lazy (worker arranca al primer uso), lock (una locución a la vez), `/callate` mata aplay **sin tomar el lock** (anti-deadlock), limpieza para voz (markdown/código/URLs/emojis fuera), recorte a `MAX_PALABRAS_VOZ` con oraciones completas + "El resto te lo dejo en pantalla". La voz jamás tumba el latido (errores silenciosos). Texto completo SIEMPRE en pantalla.
- `memory/db.py` + worker.py — MariaDB (abajo).

## 5. Memoria (dos sistemas independientes)

1. **Hilo de conversación (corto plazo, RAM):** lista compartida entre TODAS las facetas (un solo JAX), `MAX_TURNS=10` pares, snapshot previo al turno (sin duplicación), historial estructurado role/content insertado en el formato nativo de cada API. Decisión: **compartido** (jun-4).
2. **Persistente (largo plazo, MariaDB 11.8.8):** base `jax_memory`, 9 tablas (conversations, messages, facts, decisions, projects, errors, people, action_items, jax_metadata), VECTOR(768) preparado para embeddings (nomic-embed-text, Fase 2). Usuario `jax_user`@localhost confinado. `messages.conversation_id` es **int**, rol = enum(user/jax_local/jekyll/hyde/hipatia). Escritura fire-and-forget; tolerante a fallos (sin DB, JAX conversa igual). Comandos `/fact list|verify|delete`. Jairo Urbina = primer registro en people (honor_memory=1).

## 6. Voz (Kokoro — Fase 1 COMPLETA, jun-4)

- **Motor:** Kokoro-82M (Apache 2.0, uso comercial libre), español latino, **CPU puro** (torch CPU en `~/kokoro-test/.venv`) — esquiva por completo el muro CUDA/ROCm de la GPU AMD. Calidad validada a oído: **85/100** ("fluido, no perfecto pero funciona").
- **Voces español disponibles:** em_alex, em_santa (masculinas), ef_dora (femenina). Jekyll comparte timbre con Hyde pero a 0.85 (profesor pausado).
- **Medición clave (hall9000):** generación ~4× tiempo real → 650 palabras = 43.3 s de generación para 179 s de audio. El diseño Fase 1 genera TODO antes de sonar → respuestas largas tienen 20-40 s de silencio inicial. `MAX_PALABRAS_VOZ` (en tts.py) controla el techo; decidido subirlo de 90 a ~300 (verificar valor vigente en el archivo).
- **Fase 2 (PRIORIDAD SIGUIENTE): streaming por oraciones** — reproducir la oración 1 mientras se genera la 2; latencia de arranque ~2 s sin importar el largo, techo eliminado. Prompt de diseño para Deep ya redactado; pendiente cruzar su respuesta (⚠️ validar contra evidencia local: Deep ya alucinó lang_code y voice_map).
- Escalera si algún día se quiere premium (pitch del producto financiero): Google TTS → ElevenLabs. Para JAX personal, Kokoro basta.

## 7. Método de trabajo (innegociable)

- **NO SUPONER.** "El que supone se equivoca; es real hasta que él lo sabe." Si no estás 100% seguro, es un posible error y debe saberse. Método científico: medir antes de decidir (ej.: clasificador ~200 ms medido; voz 43.3 s/650 palabras medido).
- Pasos chicos verificables; mostrar el comando antes de correr; validar en cada fase (wc -l exacto + py_compile).
- Archivos largos: **NUNCA heredoc pegado** (se corrompe en silencio). Claude crea → Fernando descarga → mv/scp. Heredoc solo para scripts cortos throwaway, verificando.
- Backup antes de tocar (`*.backup-pre-<cambio>`).
- Decisiones de arquitectura: cruzar Claude + Deep (+ Hipatia con fuentes). El que tenga evidencia local gana. Los registros/comentarios que mienten se corrigen al detectarse (print "DeepSeek" cuando era local; header del router).
- Fernando decide cuándo termina la sesión; no opinar de horarios.
- Productos sensibles → eventualmente modelos locales (soberanía); desconfianza sana de nubes de terceros.
- **Transparencia del origen (grabado 15-jun):** un dato inventado disfrazado de verificado es peor que un 'no sé' honesto. Cada faceta declara su **origen de autoridad** y NINGUNA respuesta sale sin etiqueta de verificación (🔍 web / 🧠 interno / 📜 solo-input / ✗ abortado / 🏠 local / 🔧 técnico). Hipatia en `required_web` busca de verdad o falla cerrado — jamás inventa fuentes. La voz de la biblioteca, no la del oráculo.

## 8. Pendientes (en orden de intención)

1. **Voz Fase 2: streaming por oraciones** (prompt a Deep listo; cruzar, corregir, construir).
2b. **EL OIDO**: COMPLETADO jun-4. Pendiente menor: push-to-talk/wake word — va con la TUI propia (LA CARA terminal) o webapp.
2c. **LA CARA**: webapp ojo de HAL 9000 (disenada 31-may: ojo pulsa al procesar, ondas al hablar, color por faceta; reutiliza Black Diamond Chat + Reverb + React de AteneaERP, ~60% ya existe). Incluye mostrar imagenes y paneles.
2d. **LAS MANOS**: ✅ COMPLETADO (14-jun, ver §9). Sistema de capacidades/ejecución con permisos por faceta + kill switch en vuelo probado. Pendiente menor: extender a más operaciones (http_get, validate_*) y exponerlo a las facetas vía tool calling.
2e. Lanzador `jax` instalado (~/.local/bin/jax) — hecho jun-4.
2. **Chat multiagente / multi-usuario** — diseñar con Deep ANTES de codear. Atado a la decisión de hardware: el cerebro local no escala en una GPU (GPU_SEMAPHORE=1); definir si el producto multi-user usa nube o fierro propio.
3. **Decisión de hardware** (¿devolver RX 9060 XT por NVIDIA 24-32GB?) — sesión propia, con cabeza fresca, NO de madrugada. La GPU solo importa para Ollama local + voz local; 3 de 4 facetas son nube.
4. Hipatia / Gemini: resolver rate limits (tier o espaciado).
5. Easter egg IDE1990 con voz (ahora que Kokoro existe).
6. get_datetime como primer superpoder de JAX local (tool calling Ollama).
3. **LA MEMORIA VIVA** (subida de prioridad jun-4): embeddings + worker de hechos — recuperar de jax_memory lo relevante e inyectarlo al contexto de la faceta. Cura estructural de la confabulacion del 7b (no recordar: leer). Disenar con Deep.
8. Rotación confirmada de la DeepSeek key filtrada en chat (verificar si ya se hizo).
9. Producto financiero (SaaS análisis de crédito): visión central, necesita su propio CONTEXT.md y nombre. La voz premium se justifica ahí, no en JAX personal.

## 9. Historial de hitos

- **Jun 2-3:** Semana 1 — 4 facetas end-to-end, router con clasificador, primer latido, easter egg, base de memoria MariaDB (9 tablas + VECTOR), comandos /fact.
- **Jun 3-4 (esta sesión):** Hilo de conversación compartido (contrato history). Modelo jax_local → qwen2.5:7b (medido). Tono hondureño sobrio. Clasificador migrado a LOCAL (medido ~200 ms, 4/4). **VOZ Fase 1 completa** (Kokoro, 4 voces, /voz, /callate, worker dedicado). Router blindado a tildes/voseo. Registros mentirosos corregidos.
- 2026-06-04 (madrugada): FASE 2 COMPLETA — JAX habla y escucha.
  VOZ streaming v2.1: PCM por oracion + un aplay raw continuo + prefetch real
  (fix underruns: pedir N+1 antes de escribir N). Primera oracion en ~2-3s.
  OIDO: whisper_worker (small int8, 1.5s por 6s de audio) + ears.py + /escucha.
  Mic jack ALC897 calibrado (Capture 70%, boost 0). Defensas: gate RMS 0.01 +
  filtro alucinaciones + initial_prompt con nombres de la casa (fix "John") +
  normalizar_nombres. Boca pronuncia "Yax". Probado con DOS voces: Fernando y
  Claudia (primera invitada). Easter egg con voz. config: "Tu nombre es JAX"
  (fix "Soy Sos JAX") + REGLA DE ORO de la mama en jax_local (derivar
  cuentos/datos; limite documentado: en modo fijo el 7b igual confabula).
  Clasificador local rutea cuentos a Jekyll solo (verificado: 3 cerditos).
  Backups: *.backup-pre-fase2, -pre-nombres, -pre-yax, -pre-egg-voz,
  -pre-regla-oro, -pre-regla-arriba.

- **2026-06-14: LAS MANOS — sistema de capacidades de JAX, COMPLETADO.** Diseño validado por Thot (contención) + Jekyll (arquitectura). API REST local **127.0.0.1:7777** (FastAPI, venv propio `~/jax/las_manos/.venv`). Flujo: **intención → planner → policy engine → human gate → dry-run → worker → audit** (JSONL append-only en `logs/audit.jsonl`). Permisos por faceta: hipatia (solo lectura), jekyll (staging), thot (auditoría), hyde (ejecución completa con human gate en prod). Workers: ssh_worker, file_worker (write atómico `.tmp`+`mv` con snapshot `.bak`), rsync_worker (pull desde destino).
  **DECISIÓN ARQUITECTÓNICA CLAVE: pre-flight gate ≠ in-flight abort.** El kill switch en el portón no frena lo que ya corre. Solución: **watcher async que sondea `/etc/jax/PAUSE` cada 250ms + `ssh -tt`** para propagar SIGHUP al proceso remoto (mata el `sleep` antes del `&& touch`).
  **Cinco garantías probadas en fuego real contra staging (.11):** SSH real (leyó /etc/hostname → atemai-net), policy deny, human gate (sin token deniega / con token permite / token un-solo-uso), dry-run antes de mutar, **kill switch en vuelo** (la prueba brutal: PAUSE a mitad de un `sleep 10 && touch` → el touch NO se creó, verificado con list_dir; audit registró `KILL_SWITCH CRITICAL`).
  **Caveat honesto:** aborta lo que corre, no revierte lo ya escrito a disco. Para eso el snapshot `.bak`. Defensa en profundidad, no magia. ⚠️ Nota de puerto: LAS MANOS usa SSH **58291** para .11/.10 (probado y funcional); hall9000 (.5) usa 58292 (§2). Reliquia del bautismo: `/tmp/las_manos_bautismo.txt` en .11.

- **2026-06-15: FIX GROUNDING de Hipatia — el bug del fallo silencioso, muerto.** Síntesis de tres voces (Thot/Jekyll/Claude). El bug: `google_search` de Gemini es capacidad, no obligación; cuando "cree saber" inventa con confianza sin avisar, y el código entregaba ese invento como válido. **8 decisiones en un solo cambio íntegro:** (1) `grounding_policy` por tarea (off/auto/required_web/local_context_only) reemplaza `grounding: bool`; (2) Hipatia default **required_web**; (3-4) el SISTEMA pone la etiqueta, SIEMPRE, nunca el modelo; (5) valida `chunks` Y `supports`; (6) retry estricto único antes de fallar cerrado; (7) **origen de autoridad para TODAS las facetas** (jekyll 🧠 / thot 📜 / hyde 🔧 / hipatia dinámico / jax_local 🏠); (8) juramento de Hipatia en su system_prompt. Archivos: `muscles/base.py` (grounding_policy + retry + `_append_authority` + `verificacion_label`), `muscles/subprocess_muscle.py` + `muscles/ollama_muscle.py` (param `authority_origin`), `core/main.py` (lee `grounding:` del .md), `config.toml`. **`invoke(decorate=False)`** para usos internos (clasificador del router + extractor de memoria) — la etiqueta no contamina parseos. **Probado en fuego:** prueba determinista (Hipatia intenta inventar noticias → MuscleInvocationError ✗) + 3 casos reales contra Gemini en vivo (requiere-web → 🔍 5 fuentes; noticias de la semana → 🔍 2 fuentes; local_context_only → 📜). Principio grabado (§7 y Six Impossible Things VIII): "un dato inventado disfrazado de verificado es peor que un 'no sé' honesto".

*En memoria de Jairo Urbina. La máquina al servicio de quien construye, no al revés.*
