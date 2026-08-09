# CONTEXT.md — JAX 2.0 (Biblioteca de Alejandría)

> Documento canónico del proyecto JAX. Última actualización: **30 de junio de 2026** (Jacobs Director de Orquesta (wave scheduler) + contrato de capabilities cerrado de raíz en producción/master; AUTONOMIA_ANTIERROR.md creado (Fase A hecha)). Previas: 18-jun (Thot cerrado — GPT-5.5 operativo; Ada bautizada — GLM-5.2, sexta faceta, arquitecta de código, key Z.ai pendiente semana 22-jun; Hefesto/Kimi y Cassandra/Grok definidos como motores futuros por la Mesa); 14-jun (LAS MANOS, kill switch en vuelo); 4-jun (hilo compartido + qwen2.5:7b + VOZ Kokoro + clasificador local).
> Regla de la casa: **NO SUPONER. Medir dos veces, cortar después.** Si algo de este documento contradice la realidad del servidor, gana la realidad — y se corrige este documento.

## 1. Qué es JAX

JAX (se pronuncia **"Yax"**, J española) es el asistente personal y futura plataforma de orquestación de Fernando Ruiz. Un solo ser con **7 facetas operativas** (jax_local, jekyll, hyde, hipatia, thot, kimi, ada — no 7 asistentes). Núcleo orquestador propio en **Python puro + asyncio**, sin frameworks. Honra la memoria de **Jairo Urbina**, amigo, socio y pionero del software libre en Honduras — su espíritu va en el system prompt de cada faceta. Easter egg: `IDE1990`.

Visión: JAX es base y laboratorio de un producto comercial globalmente escalable (multi-AI, multi-usuario, pricing por niveles) y fundación futura para AteneaERP y HAMMURABI.

## 2. Hardware y entorno (hall9000)

- **hall9000** (172.16.20.5): Ryzen 5 8500G, 30GB RAM, GPU AMD RX 9060 XT 16GB vía **Vulkan** (RADV GFX1200; ROCm inmaduro para gfx1200, diferido). Ubuntu 24.04.4. SSH puerto **58292** (verificado jun-2026; las notas viejas decían 58291).
- Proyecto: `~/jax/`, venv `~/jax/.venv` (Python 3.12; httpx, pydantic; **sin torch**).
- Credenciales: `/etc/jax/.env` (chmod 600): DEEPSEEK_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, KIMI_API_KEY, JAX_DB_*. ZHIPU_API_KEY pendiente (Z.ai abre API semana del 22-jun).
- Kill Switch: `/etc/jax/PAUSE` (si existe, JAX no invoca músculos).
- Audio: parlantes en `plughw:2,0` (card 2, ALC897 Analog, conector verde).
- Arranque: `cd ~/jax && set -a; source /etc/jax/.env; set +a && PYTHONPATH=. .venv/bin/python -m jax.core.main`

## 3. Las 7 facetas (estado real)

| Faceta | Icono | Músculo | Modelo | Rol | Voz |
|---|---|---|---|---|---|
| JAX local | 🏠 | OllamaMuscle (GPU local) | **qwen3:14b** (actualizado jun-2026; GPU local vía Vulkan) | Conversación cotidiana, privada. Tono hondureño **sobrio**: "maje" con medida, sin groserías, sin modismos ajenos | em_alex @1.0 |
| Hyde | 🔧 | SubprocessMuscle (`claude -p`) | sonnet | Técnico: código, infra | em_santa @1.0 |
| Jekyll | 🧠 | HttpMuscle (DeepSeek) | deepseek-chat | Humanista erudito, español neutro formal | em_santa @0.85 |
| Hipatia | 🔍 | HttpMuscle (Gemini + grounding) | gemini | Investigación con fuentes web | ef_dora @1.0 |
| Thot | 📜 | HttpMuscle (OpenAI) | gpt-5.5 | Crítico: abogado del diablo, guardián del largo plazo | em_alex @0.9 |
| Kimi | ⚙️ | HttpMuscle (Moonshot AI) | kimi-k2.7-code | Motor de enjambre: coding agéntico, refactors amplios, exploración paralela. Subordinado a Ada/Hyde. | em_alex @1.0 |
| Ada | 🏛️ | HttpMuscle (Z.ai) | glm-5.2 | Arquitecta de código: 1M contexto, largo horizonte. Nombrada en honor a Ada Lovelace. **Pendiente key Z.ai (semana 22-jun)** | ef_dora @1.0 |

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

**POLÍTICAS ADN — instaladas 18-jun-2026 en 3 capas:**
- `~/.claude/CLAUDE.md` — global para Hyde en cualquier proyecto
- `~/jax-platform/CLAUDE.md` — específico de Axioma Platform
- `~/jax/CLAUDE.md` — específico de JAX
- System prompts de Jekyll, Thot y Kimi en `config/config.toml`

Políticas no negociables: **i18n SIEMPRE** (cero strings hardcodeados), **Dark/Light mode SIEMPRE** (CSS variables, nunca colores hardcodeados), **sin hardcoding de ningún tipo**, backup antes de modificar, verificar con evidencia antes de declarar éxito.



- **NO SUPONER.** "El que supone se equivoca; es real hasta que él lo sabe." Si no estás 100% seguro, es un posible error y debe saberse. Método científico: medir antes de decidir (ej.: clasificador ~200 ms medido; voz 43.3 s/650 palabras medido).
- Pasos chicos verificables; mostrar el comando antes de correr; validar en cada fase (wc -l exacto + py_compile).
- Archivos largos: **NUNCA heredoc pegado** (se corrompe en silencio). Claude crea → Fernando descarga → mv/scp. Heredoc solo para scripts cortos throwaway, verificando.
- Backup antes de tocar (`*.backup-pre-<cambio>`).
- Decisiones de arquitectura: cruzar Claude + Deep (+ Hipatia con fuentes). El que tenga evidencia local gana. Los registros/comentarios que mienten se corrigen al detectarse (print "DeepSeek" cuando era local; header del router).
- Fernando decide cuándo termina la sesión; no opinar de horarios.
- Productos sensibles → eventualmente modelos locales (soberanía); desconfianza sana de nubes de terceros.
- **Transparencia del origen (grabado 15-jun):** un dato inventado disfrazado de verificado es peor que un 'no sé' honesto. Cada faceta declara su **origen de autoridad** y NINGUNA respuesta sale sin etiqueta de verificación (🔍 web / 🧠 interno / 📜 solo-input / ✗ abortado / 🏠 local / 🔧 técnico). Hipatia en `required_web` busca de verdad o falla cerrado — jamás inventa fuentes. La voz de la biblioteca, no la del oráculo.

## 8. Pendientes (en orden de intención)

0. **▶ PRÓXIMO HITO — secuenciado por la Mesa, NO diferido (NO-GO operacional de Thot).** Construir el **INTENT ENVELOPE en LAS MANOS**: 16 campos, **rechaza intenciones incompletas**. Nada se conecta a LAS MANOS hasta que el Envelope exista. Después conectar facetas EN ORDEN: **Thot (auditoría) → Hipatia (lectura) → Jekyll (staging) → Hyde (ejecución)**. Principio: *primero quien mira, después quien sabe, después quien construye, al final quien ejecuta.* ⚠️ Procedencia honesta: es decisión VERBAL de la Mesa — aún NO escrita en `six-impossible-things.html` ni en `missions/` (verificado por grep 15-jun); registrada aquí como pendiente vinculante.

1. **Voz Fase 2: streaming por oraciones** (prompt a Deep listo; cruzar, corregir, construir).
2b. **EL OIDO**: COMPLETADO jun-4. Pendiente menor: push-to-talk/wake word — va con la TUI propia (LA CARA terminal) o webapp.
2c. **LA CARA**: webapp ojo de HAL 9000 (disenada 31-may: ojo pulsa al procesar, ondas al hablar, color por faceta; reutiliza Black Diamond Chat + Reverb + React de AteneaERP, ~60% ya existe). Incluye mostrar imagenes y paneles.
2d. **LAS MANOS**: ✅ COMPLETADO (14-jun, ver §9). Sistema de capacidades/ejecución con permisos por faceta + kill switch en vuelo probado. Pendiente menor: extender a más operaciones (http_get, validate_*) y exponerlo a las facetas vía tool calling.
2e. Lanzador `jax` instalado (~/.local/bin/jax) — hecho jun-4.
2. **Chat multiagente / multi-usuario** — diseñar con Deep ANTES de codear. Atado a la decisión de hardware: el cerebro local no escala en una GPU (GPU_SEMAPHORE=1); definir si el producto multi-user usa nube o fierro propio.
3. **Decisión de hardware** (¿devolver RX 9060 XT por NVIDIA 24-32GB?) — sesión propia, con cabeza fresca, NO de madrugada. La GPU solo importa para Ollama local + voz local; 6 de 7 facetas son nube/externas (solo jax_local es local).
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

- **2026-06-15 (CIERRE — La Mesa Redonda ratifica la constitución viva).** `~/jax/six-impossible-things.html` pasó de ~98 KB (Apéndices A–B) a **127 KiB / 130 099 B** (md5 `0538c3d9…`; backup inmediato anterior: `six-impossible-things.html.backup-pre-jaxlocal-20260615-030330`, 98 KB / solo A–B). Hitos del día, ratificados por la Mesa (Thot + Jekyll):
  - **Thot es ahora la QUINTA faceta operativa** (ya no consultor externo): motor OpenAI GPT-5.5, provider `openai` en base.py, personalidad `thot` en config.toml. Las facetas operativas son **CINCO**: JAX Local (Qwen3:14b), Jekyll (DeepSeek), Hyde (Claude Code), Hipatia (Gemini), Thot (GPT-5.5). ⚠️ **DERIVA DOCUMENTO-REALIDAD detectada:** §1 y §3 de este mismo documento todavía dicen "4 facetas" — es la amenaza que el Threat Model (Apéndice E) lista como activa; pendiente reescribir §1/§3 con cabeza fresca (el cambio toca la tabla de facetas y el conteo). **[RESUELTO 15-jun: §1 y §3 corregidos a cinco facetas; Thot agregado a la tabla.]**
  - **Tabla canónica de roles** (a pedido de Thot): fija persona/faceta/voz/motor/función para que nadie infiera. Corrige la confusión histórica **Hyde ≠ Hipatia** (Hyde = ejecutor, Claude Code, "tiene las manos"; Hipatia = buscadora, Gemini, "voz de la biblioteca"). Claude (chat estratégico) es **voz** arquitecto, no faceta.
  - **Capítulo XI — Protocolo de la Mesa Redonda:** nadie a la cabecera; la autoridad es del que tiene razón, no del que tiene rango; el corregido convierte el error en guardrail.
  - **Tres meta-contratos nuevos (Apéndices C/D/E), refinados por Thot, NO DIFERIBLES (Principio IX):**
    - **C — Memoria Viva v0.1:** 6 tipos de recuerdo (HECHO/DECISIÓN/PENDIENTE/VERDAD OPERACIONAL/HISTORIA/DESEO·VISIÓN); firma+tipo obligatorios; HECHO y VERDAD OPERACIONAL exigen procedencia; corrección visible, nunca borrado silencioso; **"la memoria informa, no autoriza"**.
    - **D — Enmienda v0.1:** EDITORIAL (corrige hecho/rol/estado; cualquier voz verificando) vs CONSTITUCIONAL (principio/capítulo/contrato; consenso de la Mesa); "el escriba preserva, no gobierna"; **regla de emergencia** (parche que evita daño activo se aplica ya, ratifica después).
    - **E — Threat Model mínimo v0.1:** 10 amenazas activas con fallo cerrado. "El peor caso no es el agente rebelde; es el agente obediente con permisos excesivos y contexto equivocado."
  - **LAS MANOS probado en fuego** (las 5 garantías, ver entrada 14-jun) con permisos por faceta ya incluyendo a thot (auditoría). **Fix de grounding de Hipatia** (ver entrada anterior). **Sésamo operativo** (TrueNAS .6: Restic 2AM + Telegram + ~~Time Machine~~ respaldando Hall9000) ⚠️ aún NO respalda el ecosistema completo (faltan atemai .11 y rich-hn .10). **[CADUCÓ — Time Machine removido el 15-jun por la mañana; ver entrada siguiente.]**

- **2026-06-15 (mañana — EDITORIAL / Memoria Viva: Time Machine removido de Sésamo).** *VERDAD OPERACIONAL caducada → HISTORIA:* se intentó Time Machine en Sésamo; el dataset `timemachine` **no tenía cuota** y devoró **378 GiB del pool de 430**; además un Mac de 500+ GB no cabe en ese pool. Removido el 15-jun por la mañana. *NUEVA VERDAD OPERACIONAL* (procedencia: verificado en la web de TrueNAS, 15-jun): **Sésamo = backup de SERVIDORES.** Restic (hall9000) activo con sus **10 GiB intactos**; pool con **420 GiB libres**. Pendientes: conectar atemai (.11) y rich-hn (.10). Time Machine se mueve a un **disco USB dedicado** (pendiente comprar). El backup de los servidores (lo crítico) nunca dependió de Time Machine — eso era respaldo del Mac de Fernando, que ahora va por separado.

- **2026-06-18 (continuación): Motor Registry v0.2 + systemd + Jacobs bautizado.**
  - **Motor Registry v0.2 ✅:** Kimi real conectada al worker. reasoning_content logueado, invisible al caller. Output validator integrado. Kill switch en vuelo probado. Job lifecycle completo probado en fuego con código real (~94s análisis de worker.py).
  - **LAS MANOS como servicio systemd ✅:** `/etc/systemd/system/jax-las-manos.service`. Arranca con hall9000, reinicio automático en fallo. `curl /health → alive`.
  - **Jacobs bautizado ✅:** Director ejecutivo de pipelines multi-faceta. Nombre en honor al **Prof. Raúl Jacobs** — maestro, mentor, director del colegio donde todo comenzó. Módulo de orquestación puro, sin LLM propio. Frase constitucional (Thot): *"Jacobs dirige el proceso, no decide el propósito. Secuencia, delega y reporta; no inventa autoridad."* Diseño aprobado por Mesa unánime. Hyde construyendo v0.1.

- **2026-06-18 (sesión completa — cierre):**
  - **Jacobs v0.1+v0.2 ✅** — Director ejecutivo de pipelines multi-faceta. En honor al Prof. Raúl Jacobs. Plan builder con JAX Local real. Executor con context propagation (Hipatia→Jekyll→Thot encadenados). Modo autonomous habilitado. Modo supervised con botón APROBAR desde LA CARA. Endpoints: /jacobs/plan, /jacobs/pipeline, /jacobs/pipeline/{id}/resume, /jacobs/pipeline/{id}/approve-step. MariaDB: jacobs_pipelines + jacobs_steps.
  - **JAX Engine ✅** — Estado vivo del ecosistema, EventBus pub/sub, ResourceManager (admission control sin interrumpir workers en vuelo), WebSocket hub por usuario.
  - **LA CARA / Axioma Platform v0.2 ✅** — React 19 + Tailwind + Zustand + react-i18next. Tres paneles: facetas (izq), ojo HAL + chat (centro), Director Jacobs + Audit (der). Modos: Chat (todas las facetas), Comando (Hyde autónomo), Pipeline (Jacobs con modal). Kill switch visible siempre. LAS MANOS vivo en barra inferior. WebSocket con reconexión automática. Tres servicios systemd: jax-las-manos (:7777), jax-platform (:8080), jax-platform-frontend (:5173).
  - **Primer pipeline real de HAMMURABI ✅** — Hipatia (regulaciones CNBS, 10 componentes) → Jekyll (análisis humanista) → Thot (crítica regulatoria). 71KB guardados en ~/jax/workspace/hammurabi-credito-pipeline-001.json.
  - **Políticas ADN instaladas ✅** — CLAUDE.md global + por proyecto + system prompts de Jekyll/Thot/Kimi. i18n obligatorio, dark/light obligatorio, sin hardcoding.
  - **Six Impossible Things actualizado ✅** — Flujograma de dos capas (Infraestructura + Facetas), Ada corregida (GLM-5.2 Z.ai), Kimi en firma final, Jacobs en firma final, portada "Axioma · Infraestructura Cognitiva Personal" en serif dorado.
  - **Rename pendiente:** JAX Platform → **Axioma** (nombre definitivo del producto).

*En memoria de Jairo Urbina. La máquina al servicio de quien construye, no al revés.*

- **2026-06-18: Thot cerrado + Ada bautizada + Mesa Redonda sobre nuevas facetas.**
  - **Thot ✅ CERRADO:** GPT-5.5 operativo vía API OpenAI. Router actualizado (LABELS/ICONS/ALIASES/VALID_FACETAS). Prueba en fuego: "trae a thot" → juicio semántico real activo. Primera auditoría: identifica supuestos ocultos, pide evidencia antes de validar — comportamiento correcto.
  - **Ada bautizada — sexta faceta:** GLM-5.2 (Z.ai/Zhipu), nombrada en honor a Ada Lovelace. Arquitecta de código, 1M tokens de contexto, largo horizonte. Especialidad: leer repos completos, sostener coherencia arquitectónica, coding agéntico multi-hora. Supera a Claude Opus 4.7/4.8 en coding blind test (LMArena #2 global). Key Z.ai pendiente — API general abre semana del 22-jun. Motor ya implementado en base.py (_call_openai compatible); solo falta ZHIPU_API_KEY + entrada en config.toml.
  - **Mesa Redonda — evaluación de Kimi y Grok:** Prompt enviado a Jekyll y Thot. Consenso: Kimi K2.7 Code → **Hefesto** (motor/agente de enjambre subordinado a Ada/Hyde, no faceta constitucional aún). Grok 4.3 → **Cassandra** (radar de señales externas para Hipatia/HAMMURABI, después de Hefesto). Ninguno entra como faceta plena todavía — período probatorio primero. Regla: "No toda inteligencia merece una silla en la Mesa."
  - **Hardware evaluado:** RTX 5090 32GB para hall9000 — desbloquea qwen3:32b completo (~40+ tok/s), CUDA nativo, fin del dolor ROCm. RAM recomendada: 64GB DDR5 (de 32GB actuales). GLM-5.2 local requiere 256GB+ — no entra ni en RTX 5090 ni en RED QUEEN (96GB); siempre vía API.
  - **Router:** Thot agregado a LABELS, ICONS, ALIASES, VALID_FACETAS. Backup: router.py.backup-pre-thot-*.

---
- **2026-06-30 (madrugada — Jacobs "Director de Orquesta" + contrato de capabilities cerrado de raíz). EN PRODUCCIÓN, mergeado a master, pusheado a origin.**
  Sesión de arquitectura+verificación (Fernando + Claude estratégico + Hipatia/Claude Code en hall9000). Método Hyde estricto: cada paso verificado con evidencia, productor≠auditor, producción bajo control de Fernando. El método atrapó CINCO falsos-positivos/estados-rotos: (1) un "test 10/10" que era script con sys.exit(0) y rompía pytest sin correr; (2) servicio sin --reload con código viejo en memoria; (3) catálogo viejo en memoria al intentar e2e prematuro; (4) un commit de Fase A que se saltó por comandos encadenados; (5) una corrección a CONTEXT.md que era no-op (el error vivía en los prompts, no en el doc).
  - **Wave scheduler "Director de Orquesta" (commit 01eacc4) ✅** — run_pipeline reescrito: ejecuta por OLAS topológicas derivadas del DAG (depends_on), steps de una ola en PARALELO vía asyncio.gather. _compute_waves particiona; _run_one_step extrae el cuerpo del step. Estado derivado de refs en context (no cursor lineal) → sobrevive a /resume. Kill switch pre-ola, hyde gate pre-ola, manejo de fallo de ola (respeta skip_on_fail, aborta si falla sin skip), supervised pausa POR OLA (granularidad = ola, no step). _CLEANROOM_RULE en plan.py: auditor debe usar facet distinto al productor.
  - **Contrato de capabilities cerrado de raíz (commit 5fdae32) ✅** — Causa raíz: el planner emitía capability libre, solo kimi la validaba (asimetría), y había nombres sembrados (generate/reconcile/validate_consistency/design/critique/reason) ausentes del catálogo → PIPELINE_ABORTED latente. Fix en 3 capas: (1) config.toml ampliado con 6 capabilities nuevas (generate, reason, design, validate_consistency, reconcile, critique), todas verdes/sin-gate; jacobs añadido a allowed_callers de code_swarm/bug_hunt/architecture_review (gates de code_swarm/bug_hunt INTACTOS, requires_human_gate=true). (2) _CAPABILITY_MAP total (17 entradas, cero pass-through silencioso). (3) VALID_CAPABILITIES en plan.py (planner cerrado, degrada a 'reason' lo desconocido) + validate_capability PRE-dispatch en DOS NIVELES: Nivel A existencia (todos los facets, cierra la asimetría), Nivel B contrato de motor (solo facets-motor/kimi). Fail-open si el catálogo no carga (net secundario, no SPOF).
  - **Suite pytest real (commit edb8099) ✅** — tests/test_jacobs_director.py convertido de script-con-print a 10 funciones test_* con assert. 10/10 vía pytest, integrable a CI. Deuda técnica saldada.
  - **Verificado e2e en producción:** fan-out ada/generate ∥ kimi/generate → thot/validate_consistency. Olas reales [[0,1],[2]], parallel=2 confirmado por timestamps idénticos, orden DAG respetado (ola[0,1] completa antes de ola[2]), kimi/generate ACEPTADO por Motor Registry (202, bug original muerto), clean-room limpio. Pipeline completed.
  - **Servicio/infra:** jax-las-manos.service reiniciado por Fernando, PID vivo, /health 200. Endpoint correcto: POST /jacobs/pipeline (router prefix="/jacobs"). DB jacobs: jacobs_pipelines, jacobs_steps (output_ref LONGTEXT), jacobs_events (columna real event_type, NO type). Timeout reconcile=900s en config (no global). Repo: git@github.com:fjruizhn/Jax.git, master en commit 2c84dd8.
  - **NUEVO DOCUMENTO — AUTONOMIA_ANTIERROR.md:** diseño aprobado de autonomía anti-error en 4 capas. Visión de Fernando: "que el sistema trabaje por mí, no conmigo; auto-auditoría en cada etapa; que no dependa de mí y un yes; lo verdaderamente importante me notifica, lo demás para eso están backups/sandbox." Capa 1 (contrato cerrado) = Fase A, HECHA hoy. Pendientes: Fase B (auto-auditoría bloqueante + schemas), C (snapshot+rollback automático), D (gate de severidad verde/amarillo/rojo + Telegram). Caso de aceptación: ver el Mundial sin mirar el monitor.
  - **⚠️ DERIVA DOCUMENTO-REALIDAD pendiente (Apéndice E, cabeza fresca):** §2 describe hardware viejo (Ryzen 5 8500G / RX 9060 XT) — Fernando arma hall9000 v2.0 (Ryzen 9 9950X / R9700 32GB / 96GB DDR5) hoy. §3 lista Ada como "key Z.ai pendiente" — Ada/GLM-5.2 YA opera en producción vía Z.ai. §8 no menciona Jacobs v0.3 / wave scheduler. Reescribir §2/§3/§8 con cabeza fresca, NO de madrugada.
---

- **2026-08-09: MariaDB 11.8→12.3 LTS (Docker blue/green) + Fase 1 credenciales a la DB. Rama `infra/mariadb-12.3-migration` (jax + jax-platform), aún no mergeada a master.**
  - **Motor: MariaDB 12.3.2 LTS en Docker ✅** — contenedor `mariadb-12-3-jax`, imagen `mariadb:12.3.2` pinneada (nunca `:latest`), puerto `127.0.0.1:3308` (bind explícito a loopback, verificado con `ss`), volumen persistente `/var/lib/mariadb-12.3-docker/data` (bind mount real, no anónimo), red Docker con subred fija `172.30.5.0/24` (grants `@172.30.5.%`, no `@localhost` — el gateway del bridge, no `localhost`, es quien conecta). TZ `America/Tegucigalpa` explícita (el contenedor corría en UTC por default, detectado por un timestamp 6h adelantado). Migración vía dump lógico en 3 fases (schema sin índice VECTOR → carga de datos → `ALTER TABLE ADD VECTOR KEY` en bloque) — evita el riesgo declarado incierto de re-serialización de columnas VECTOR/768d. Paridad verificada: checksums idénticos en 20/20 tablas, `VEC_DISTANCE_COSINE` devuelve resultados idénticos a 11.8. **11.8 nativa apagada pero INSTALADA e intacta — conservar hasta 2026-08-23 (fecha del cutover + 2 semanas) antes de considerar desinstalarla.**
  - **Docker CE instalado desde el repo oficial de Docker Inc** (`download.docker.com`, suite `resolute` = Ubuntu 26.04) — nueva fuente de actualizaciones a vigilar en el ecosistema. `fruiz` NO está en el grupo `docker` (equivale a root) — todo `sudo docker` explícito, vía NOPASSWD acotado en `/etc/sudoers.d/jax-migration` (solo docker + start/stop/restart/status de los 4 servicios JAX — **eliminar esta regla tras el cutover verificado en verde**, `sudo rm /etc/sudoers.d/jax-migration`).
  - **UFW verificado activo el 2026-08-09** (default deny incoming/routed, `:8080`/`:5173` restringidos a LAN) — **R1 de la auditoría (bind 0.0.0.0) queda CERRADO** como riesgo alto; el bind a loopback del contenedor 12.3 es defensa en profundidad adicional, no un parche de firewall roto.
  - **DB como fuente única de verdad para credenciales de proveedor ✅** (resuelve R3: rotar una key no se propagaba sin restart). 3 tablas nuevas (`provider`, `credential`, `credential_audit`), resolver compartido `credential_resolver.py` (espejado en jax-platform/jax/las_manos, mismo patrón que `crypto_secrets.py`) con `resolve_credential()`/`resolve_credential_instrumented()`. `KeyProvider.get_master_key()` como única interfaz para `FERNET_KEY` (`EnvKeyProvider` como única implementación de esta fase). 4+1 consumidores migrados: jax-platform (chat.py/image.py), jax-las-manos/Jacobs (executor.py/plan.py), REPL+jax-memory-worker (vía `jax/muscles/base.py`, resolución por-request, ya no en `__init__`), y Motor Registry/Kimi (`motor_registry/worker.py` — consumidor que se había escapado del inventario original, encontrado durante la implementación). UI admin: rotar y revocar como acciones separadas (antes un solo botón hacía las dos cosas mal), salud persistida en `credential.last_health_status`/`last_verified_at` (antes vivía en estado de React y se perdía al recargar).
  - **TTL = SLA de revocación, explícito:** `CREDENTIAL_CACHE_TTL_SECONDS=30` (una key revocada sigue viva hasta 30s en el caso normal, DB sana) + `CREDENTIAL_STALE_MAX_SECONDS=300` (techo de tolerancia si la DB cae; pasado eso, fail-closed explícito, nunca una llamada silenciosa con credencial vieja). Ambos configurables por env. Verificado en vivo con evidencia real: propagación de rotación en ~15s, fail-closed inmediato tras revocar, degradación explícita tras superar el techo de stale.
  - **Corte con doble lectura instrumentada (B1.4), en curso** — `user_api_keys` (tabla legacy) INTACTA, no se toca hasta que el corte esté verificado. Criterio de salida: 7 días consecutivos sin ninguna lectura `source=env_fallback` en logs, incluyendo al menos una rotación real en la ventana. Si a los 7 días siguen apareciendo, hay un consumidor no mapeado — investigar antes de forzar el corte.
  - **R2 (FERNET_KEY co-ubicada con lo que cifra) — DEUDA ABIERTA, NO resuelta por esta fase.** El TDE de MariaDB protege archivos en disco pero `mariadb-dump` produce SQL en texto plano; el backup a R2 sigue conteniendo la llave y la tabla cifrada en el mismo snapshot restic (confirmado con evidencia: `backup-hall9000.sh` respalda `$STAGING` completo, que incluye tanto `mariadb-local/jax_memory.sql` como `jax-config/env`). `KeyProvider` deja el punto de cambio listo (mover a KMS/Vault el día que se decida es cambiar una implementación, no cazar referencias), pero el tratamiento real queda pendiente como iniciativa propia.
  - **Hallazgo adicional durante la migración:** `/etc/restic/mysql-backup-local.cnf` necesitó `protocol=TCP` explícito además de `host=`/`port=` — el cliente CLI de MariaDB prioriza el socket Unix local sobre host/port si ambos están disponibles; sin ese detalle, el backup hubiera seguido respaldando una base fantasma en silencio.
---
