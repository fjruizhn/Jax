"""
JAX 2.0 — El primer latido (memoria + hilo + VOZ streaming + OIDO).

Bucle de terminal (REPL) que cose todo: lee lo que escribis (o lo que
DECIS por el microfono), lo pasa por el Router, e invoca la faceta correcta.

Seguridad: antes de CADA invocacion de musculo se chequea el Kill Switch
(/etc/jax/PAUSE).

MEMORIA: hilo de sesion compartido en RAM (MAX_TURNS) + MariaDB persistente
(tolerante a fallos).

VOZ (salida, Fase 2 streaming): cada faceta habla con su voz Kokoro; la
primera oracion suena en ~2-3s y el resto se genera mientras suena.
/voz on|off, /callate.

OIDO (entrada): /escucha corta la locucion en curso, graba 8 segundos del
microfono (jack ALC897, calibrado), transcribe con Whisper local, y el
texto entra al flujo NORMAL del router como si lo hubieras tecleado.
Defensas: gate de silencio + filtro de alucinaciones (en el worker).

NOTA TECNICA input(): via run_in_executor para NO congelar el event loop —
asi la voz suena de fondo y /callate entra mientras JAX habla.

Uso:
    jax            (lanzador en ~/.local/bin/jax)
  o:
    cd ~/jax && set -a; source /etc/jax/.env; set +a
    PYTHONPATH=. .venv/bin/python -m jax.core.main

En memoria de Jairo Urbina.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from jax.core.router import Router
from jax.muscles.base import HttpMuscle, MuscleError, GROUNDING_POLICIES
from jax.muscles.subprocess_muscle import SubprocessMuscle
from jax.muscles.ollama_muscle import OllamaMuscle
from jax.memory.db import MemoryDB
from jax.voice.tts import VoiceEngine
from jax.voice.ears import EarEngine

CONFIG_PATH = "config/config.toml"

# Cuantos turnos del hilo de la sesion se conservan en RAM y se pasan a la
# faceta. Un "turno" = un par (user + respuesta). 10 turnos = 20 mensajes.
MAX_TURNS = 10

# Segundos que graba /escucha (v1: duracion fija, simple y predecible).
ESCUCHA_SEGUNDOS = 8


def build_muscles(cfg: dict, timeout_override: float | None = None) -> dict:
    """Arma las facetas desde el config. El 'type' decide la clase.
    timeout_override: si se pasa, reemplaza el timeout del config (modo tarea)."""
    timeout = timeout_override if timeout_override is not None else cfg["jax"]["timeout_seconds"]
    muscles: dict = {}

    for name, p in cfg["personalities"].items():
        ptype = p["type"]
        if ptype == "http":
            muscles[name] = HttpMuscle(
                name, p["provider"], p["model_default"], p["models_allowed"],
                p["system_prompt"], timeout,
                grounding_policy=p.get("grounding_policy", "off"),
                authority_origin=p.get("authority_origin", ""),
                api_url=p.get("api_url", ""),
            )
        elif ptype == "subprocess":
            muscles[name] = SubprocessMuscle(
                name, p["model_default"], p["models_allowed"],
                p["system_prompt"], timeout,
                authority_origin=p.get("authority_origin", ""),
            )
        elif ptype == "ollama":
            muscles[name] = OllamaMuscle(
                name, p["model_default"], p["models_allowed"],
                p["system_prompt"], timeout, api_url=p["api_url"],
                authority_origin=p.get("authority_origin", ""),
            )
        else:
            raise ValueError(f"Tipo de faceta desconocido: {ptype} ({name})")

    return muscles


def kill_switch_active(path: str) -> bool:
    """True si el Kill Switch esta puesto."""
    return Path(path).exists()


def _lanzar_workers_background() -> None:
    """Lanza worker de extraccion y embedding como subprocesos desacoplados.
    start_new_session=True: sobreviven al cierre de JAX. Fallos al Popen se ignoran."""
    env = os.environ.copy()
    for modulo in ("jax.memory.worker", "jax.memory.embedding_worker"):
        try:
            subprocess.Popen(
                [sys.executable, "-m", modulo],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception:
            pass


def humanizar_error(label: str, err: Exception) -> str:
    """Traduce errores tecnicos comunes a un mensaje humano y breve."""
    msg = str(err)
    if "503" in msg or "UNAVAILABLE" in msg or "high demand" in msg:
        return f"{label}: El servicio esta saturado en este momento. Proba de nuevo en un rato."
    if "timeout" in msg.lower() or "sin respuesta" in msg:
        return f"{label}: Tarde demasiado en responder. Proba otra vez."
    if "429" in msg or "rate" in msg.lower() or "quota" in msg.lower():
        return f"{label}: Se alcanzo el limite de uso por ahora. Espera un momento."
    if "401" in msg or "403" in msg or "API_KEY" in msg or "api key" in msg.lower():
        return f"{label}: Problema de credenciales. Revisa la llave en /etc/jax/.env."
    if "connect" in msg.lower() or "connection" in msg.lower():
        return f"{label}: No me pude conectar. Revisa la red o que el servicio este arriba."
    # Si no reconocemos el error, mostramos algo corto (no el JSON crudo).
    corto = msg.split("\n")[0][:160]
    return f"[{label} fallo] {corto}"


async def handle_fact_command(db, line: str, pending_delete: dict) -> str:
    """Procesa comandos /fact. Devuelve el texto a mostrar.
    pending_delete: dict mutable {id: texto} para confirmar borrados."""
    parts = line.strip().split()
    sub = parts[1].lower() if len(parts) > 1 else "help"
    arg = parts[2] if len(parts) > 2 else ""

    # /fact list [--all] [--type X]   (atajo: ls)
    if sub in ("list", "ls"):
        only_unverified = "--all" not in line
        ftype = None
        m = re.search(r"--type (\w+)", line)
        if m:
            ftype = m.group(1)
        facts = await db.get_facts(only_unverified=only_unverified, fact_type=ftype)
        if facts is None:
            return "No pude leer la memoria (base no disponible)."
        if not facts:
            return "No hay hechos pendientes de revision."
        out = ["Hechos " + ("pendientes de revision:" if only_unverified else "(todos):"), ""]
        for f in facts:
            mark = "[OK]" if f["is_verified"] else "[..]"
            conf = f["confidence"]
            out.append(f"  {mark} #{f['id']} ({f['fact_type']}, conf {conf:.1f})")
            out.append(f"       {f['fact_text']}")
        out.append("")
        out.append("Usa: /fact verify <id>  |  /fact delete <id>")
        return "\n".join(out)

    # /fact verify <id>   (atajo: v)
    if sub in ("verify", "v"):
        if not arg.isdigit():
            return "Uso: /fact verify <id>"
        ok = await db.verify_fact(int(arg))
        return f"Hecho #{arg} verificado." if ok else f"No encontre el hecho #{arg}."

    # /fact delete <id>   (atajo: d) — SIEMPRE confirma
    if sub in ("delete", "del", "d"):
        if not arg.isdigit():
            return "Uso: /fact delete <id>"
        fid = int(arg)
        texto = await db.get_fact_text(fid)
        if texto is None:
            return f"No encontre el hecho #{fid}."
        pending_delete["id"] = fid
        pending_delete["text"] = texto
        return (f"Vas a borrar el hecho #{fid}:\n   \"{texto}\"\n"
                f"Escribi '/fact confirm' para borrarlo, o cualquier otra cosa para cancelar.")

    # /fact confirm — ejecuta el borrado pendiente
    if sub == "confirm":
        if "id" not in pending_delete:
            return "No hay ningun borrado pendiente de confirmar."
        fid = pending_delete.pop("id")
        pending_delete.pop("text", None)
        ok = await db.delete_fact(fid)
        return f"Hecho #{fid} borrado." if ok else f"No pude borrar el hecho #{fid}."

    # ayuda
    return (
        "Comandos de memoria:\n"
        "  /fact list            hechos pendientes de revision\n"
        "  /fact list --all      todos los hechos\n"
        "  /fact list --type X   filtra por tipo (user/technical/project/...)\n"
        "  /fact verify <id>     marca un hecho como correcto\n"
        "  /fact delete <id>     borra un hecho (pide confirmar)\n"
        "  atajos: ls, v, d"
    )


async def run_task(task_file: Path, facet_cli: str | None = None) -> None:
    """Ejecuta una tarea autónoma desde un archivo .md sin REPL interactivo.
    Escribe el resultado en <nombre>_result.md junto al archivo de entrada."""
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    kill_path = cfg["jax"]["kill_switch_path"]
    task_timeout = float(cfg["jax"].get("task_timeout_seconds", 600))
    default_faceta = cfg["jax"].get("default_personality", "jax_local")

    if not task_file.exists():
        print(f"[tarea] Error: no encontré el archivo '{task_file}'")
        sys.exit(1)

    contenido = task_file.read_text(encoding="utf-8").strip()

    # Detectar cabeceras opcionales 'faceta:' y 'grounding:' en las primeras
    # líneas (en cualquier orden). 'grounding:' permite override por TAREA de
    # la política de grounding (Decisión 1 y 3 del fix consolidado).
    lineas = contenido.splitlines()
    faceta_forzada: str | None = None
    grounding_override: str | None = None
    while lineas:
        m_f = re.match(r"^faceta\s*:\s*(.+)$", lineas[0], re.IGNORECASE)
        m_g = re.match(r"^grounding\s*:\s*(.+)$", lineas[0], re.IGNORECASE)
        if m_f:
            faceta_forzada = m_f.group(1).strip().lower()
            lineas = lineas[1:]
        elif m_g:
            grounding_override = m_g.group(1).strip().lower()
            lineas = lineas[1:]
        else:
            break
    contenido = "\n".join(lineas).strip()

    # Resultado: mismo directorio que el input, sufijo _result.md
    result_file = task_file.parent / (task_file.stem + "_result.md")

    # Músculos con timeout extendido
    muscles = build_muscles(cfg, timeout_override=task_timeout)

    router = Router(
        default_personality=default_faceta,
        classifier=muscles.get("jax_local"),
        debug=False,
    )

    voice = VoiceEngine()

    if kill_switch_active(kill_path):
        print(f"[tarea] PAUSE activo — JAX no puede ejecutar. "
              f"Borrá {kill_path} para reactivar.")
        sys.exit(1)

    # Elegir faceta (precedencia: CLI --facet > cabecera del archivo > router)
    if facet_cli:
        faceta = facet_cli
        print(f"[tarea] Faceta: {faceta} (FORZADA por --facet, router omitido)")
    elif faceta_forzada:
        if faceta_forzada not in muscles:
            print(f"[tarea] Error: la faceta '{faceta_forzada}' no existe en la config.")
            sys.exit(1)
        faceta = faceta_forzada
        print(f"[tarea] Faceta: {faceta} (forzada en el archivo)")
    else:
        # Pasamos un extracto al router para no saturarlo con todo el contenido
        decision = await router.route(contenido[:500])
        faceta = decision.personality if decision.kind == "route" else default_faceta
        print(f"[tarea] Faceta: {faceta} (elegida por router)")

    muscle = muscles[faceta]
    label = router.label(faceta)
    pconf = cfg["personalities"].get(faceta, cfg["personalities"][default_faceta])

    # Política de grounding por tarea (solo facetas que la soportan: HttpMuscle).
    # Precedencia: 'grounding:' del .md > default de Hipatia (required_web) >
    # default del config de la faceta. (Decisiones 1 y 2 del fix consolidado.)
    if hasattr(muscle, "grounding_policy"):
        if grounding_override:
            if grounding_override not in GROUNDING_POLICIES:
                print(f"[tarea] Error: grounding '{grounding_override}' inválido. "
                      f"Válidos: {GROUNDING_POLICIES}")
                sys.exit(1)
            muscle.grounding_policy = grounding_override
        elif faceta == "hipatia":
            # Hipatia ES investigación: sin especificar, exige web (Decisión 2).
            muscle.grounding_policy = "required_web"
        print(f"[tarea] grounding_policy: {muscle.grounding_policy}")

    print(f"[tarea] {label} procesando '{task_file.name}'...", flush=True)

    try:
        respuesta = await muscle.invoke(contenido, history=None)

        result_file.write_text(
            f"# Resultado de: {task_file.name}\n\n{respuesta}\n",
            encoding="utf-8",
        )
        print(f"\n[tarea] Resultado guardado en: {result_file}")

        voz_msg = f"Tarea completada. El resultado está en {result_file.name}."
        print(voz_msg)
        voice.enabled = True
        await voice.speak(
            voz_msg,
            voz=pconf.get("voice_id", "em_alex"),
            velocidad=float(pconf.get("voice_speed", 1.0)),
        )

    except (MuscleError, Exception) as e:
        error_msg = str(e) or repr(e) or "error sin detalle"

        result_file.write_text(
            f"# Error en tarea: {task_file.name}\n\n{error_msg}\n",
            encoding="utf-8",
        )
        print(f"\n[tarea] {error_msg}")
        print(f"[tarea] Error guardado en: {result_file}")

        voz_fallo = "La tarea falló. Revisá el archivo de resultado."
        print(voz_fallo)
        voice.enabled = True
        await voice.speak(
            voz_fallo,
            voz=pconf.get("voice_id", "em_alex"),
            velocidad=float(pconf.get("voice_speed", 1.0)),
        )

    finally:
        await voice.shutdown()


async def main() -> None:
    with open(CONFIG_PATH, "rb") as f:
        cfg = tomllib.load(f)

    kill_path = cfg["jax"]["kill_switch_path"]
    default_faceta = cfg["jax"].get("default_personality", "jax_local")

    # Bloque C: modelo real desde facet_binding, no config.toml. Se
    # sobrescribe model_default ANTES de build_muscles() — mismo mecanismo
    # existente (Muscle.invoke usa model_default como fallback), solo
    # cambia la fuente. Query unica al arrancar (REPL = sesion unica, no
    # servicio 24/7 — no justifica resolucion por-request como Jacobs/Mesa
    # web). Si la DB no responde al boot, cfg["personalities"] ya trae el
    # model_default de config.toml como fallback — no se rompe el arranque.
    from jax.core.facet_resolver import load_facet_registry
    try:
        registry = await load_facet_registry()
    except Exception as exc:
        logging.warning(f"No se pudo cargar facet_registry desde DB, usando config.toml: {exc}")
        registry = {}
    for key, info in registry.items():
        if key in cfg["personalities"]:
            cfg["personalities"][key]["model_default"] = info["model"]

    if registry:
        import jax.core.router as router_module
        router_module.VALID_FACETAS = tuple(registry.keys())
        router_module.AUTO_FACETAS = tuple(k for k, v in registry.items() if v["auto_selectable"])
        router_module.LABELS = {k: v["display_name"] for k, v in registry.items() if v["display_name"]}
        router_module.ICONS = {k: v["icon"] for k, v in registry.items() if v["icon"]}
    # Si registry esta vacio (DB no respondio), router.py conserva sus
    # constantes hardcodeadas como fallback de arranque — declarado en el
    # propio router.py, no es una fuente paralela silenciosa.

    muscles = build_muscles(cfg)

    # Router hibrido: clasificador LOCAL (jax_local / qwen2.5:7b) para lo
    # ambiguo. Medido en hall9000: ~200 ms, 4/4 aciertos. Soberania total.
    router = Router(
        default_personality=default_faceta,
        classifier=muscles.get("jax_local"),
        debug=True,
    )

    # --- Memoria persistente (tolerante a fallos) -----------------------
    # Identidad del REPL (sin hardcode): scope INDIVIDUAL de Fernando, para que
    # su memoria sea la MISMA entre consola y web (jax-platform /api/chat).
    def _repl_int(name):
        v = os.getenv(name, "")
        return int(v) if v.strip().isdigit() else None
    repl_uid = _repl_int("JAX_REPL_USER_ID")
    repl_tid = _repl_int("JAX_REPL_TENANT_ID")

    db = MemoryDB()
    conv_uuid = None
    db_ok = await db.connect(
        host=os.getenv("JAX_DB_HOST", "localhost"),
        user=os.getenv("JAX_DB_USER", ""),
        password=os.getenv("JAX_DB_PASSWORD", ""),
        database=os.getenv("JAX_DB_NAME", "jax_memory"),
    )
    if db_ok:
        conv_uuid = await db.start_conversation(
            source="terminal", user_id=repl_uid, tenant_id=repl_tid, project_id=None)

        # Inyectar facts en los system_prompts de todas las facetas (scope individual).
        facts = await db.get_facts(only_unverified=False, limit=20, user_id=repl_uid)
        if facts:
            lineas = [f"- {f['fact_text']}" for f in facts]
            memoria_str = "Lo que sé de Fernando:\n" + "\n".join(lineas)
            for nombre in ("jax_local", "jekyll", "hyde", "hipatia"):
                if nombre in muscles:
                    muscles[nombre].system_prompt = (
                        memoria_str + "\n\n" + muscles[nombre].system_prompt
                    )
    # --------------------------------------------------------------------

    # --- Hilo de conversacion en RAM (COMPARTIDO por todas las facetas) -
    # El contexto de sesiones anteriores se inyecta por busqueda semantica
    # en cada turno (ver mas abajo), no al inicio.
    historial: list[dict] = []
    # --------------------------------------------------------------------

    # --- Voz (salida) y Oido (entrada): apagados/lazy por defecto -------
    voice = VoiceEngine()
    ears = EarEngine()
    # --------------------------------------------------------------------

    # Estado para confirmar borrados de facts (comando /fact delete).
    pending_delete: dict = {}

    # Modo pesado: /pesado activa deepseek-v4-pro en Jekyll; /normal lo apaga.
    modo_pesado: bool = False
    MODELO_PESADO = "deepseek-v4-pro"
    FACETA_PESADO = "jekyll"

    loop = asyncio.get_running_loop()

    print("=" * 56)
    print("  JAX 2.0 — En memoria de Jairo Urbina.")
    print("  Escribi para hablar. 'salir' para terminar.")
    print("  Voz: /voz on | /voz off | /callate    Oido: /escucha")
    print("  Modo: /pesado (Jekyll -> v4-pro) | /normal")
    if not db_ok:
        print("  [memoria offline — converso, pero no guardo esta sesion]")
    print("=" * 56)

    try:
        while True:
            try:
                # input() en un thread del executor: el event loop queda
                # libre y la voz suena de fondo mientras esperamos teclas.
                user_text = (
                    await loop.run_in_executor(None, lambda: input("\n> "))
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nHasta luego.")
                break

            if not user_text:
                continue
            if user_text.lower() in ("salir", "exit", "quit"):
                print("Hasta luego.")
                break

            # --- Comandos de voz y oido: antes del router ----------------
            lt = user_text.lower()
            if lt == "/voz on":
                voice.enabled = True
                print("\n[voz activada — la primera locucion tarda unos "
                      "segundos extra mientras despierta el motor]")
                continue
            if lt == "/voz off":
                voice.enabled = False
                await voice.stop_playing()
                print("\n[voz desactivada]")
                continue
            if lt in ("/callate", "/silencio"):
                await voice.stop_playing()
                print("\n[locucion cortada]")
                continue

            if lt == "/escucha":
                # 1) Que JAX no se oiga a si mismo: cortar locucion y dejar
                #    que el buffer del hardware se vacie.
                await voice.stop_playing()
                await asyncio.sleep(0.3)
                # 2) Grabar (el REPL espera aqui: no hay input simultaneo).
                print(f"\n[🎤 grabando {ESCUCHA_SEGUNDOS} segundos — habla ya]",
                      flush=True)
                texto_voz = await ears.listen(ESCUCHA_SEGUNDOS)
                if not texto_voz:
                    motivo = ears.last_reason or "no entendi"
                    print(f"[oido: {motivo} — intenta de nuevo]")
                    continue
                # 3) El texto entra al flujo NORMAL, como si lo tecleara.
                print(f"\n> [voz] {texto_voz}")
                user_text = texto_voz
                lt = user_text.lower()
                # (sin continue: sigue al router como cualquier mensaje)
            # --------------------------------------------------------------

            if lt == "/pesado":
                modo_pesado = True
                modelo_default = cfg["personalities"][FACETA_PESADO].get("model_default", "")
                print(f"\n[modo pesado: Jekyll usara {MODELO_PESADO} "
                      f"(por defecto: {modelo_default}). /normal para volver.]")
                continue
            if lt == "/normal":
                modo_pesado = False
                print("\n[modo normal: Jekyll vuelve a su modelo por defecto.]")
                continue

            # Comandos de gestion de memoria: tampoco son dialogo.
            if lt.startswith("/fact"):
                salida = await handle_fact_command(db, user_text, pending_delete)
                print(f"\n{salida}")
                continue

            # Guardar lo que dijo Fernando en la base (fire-and-forget).
            db.save_message(conv_uuid, "user", user_text)

            decision = await router.route(user_text)

            # Easter egg o mensaje del propio router (no es dialogo con faceta).
            if decision.kind in ("easter_egg", "say"):
                print(f"\n{decision.text}")
                # El easter egg tambien merece voz (la voz de la casa).
                if voice.enabled:
                    pconf = cfg["personalities"][default_faceta]
                    asyncio.create_task(voice.speak(
                        decision.text,
                        voz=pconf.get("voice_id", "em_alex"),
                        velocidad=float(pconf.get("voice_speed", 1.0)),
                    ))
                continue

            # kind == "route": invocar la faceta elegida.
            faceta = decision.personality
            muscle = muscles[faceta]
            label = router.label(faceta)

            # Debug del router hibrido: muestra cuando decidio el clasificador.
            if router.debug and decision.via == "clasificador":
                print(f"\n[router -> {faceta} via clasificador local]")

            # Kill Switch: chequeo atomico JUSTO antes de invocar.
            if kill_switch_active(kill_path):
                print(f"\n[PAUSE activo — JAX no invoca musculos. "
                      f"Borra {kill_path} para reactivar.]")
                continue

            print(f"\n{label} esta pensando...", flush=True)
            try:
                # Busqueda semantica: contexto relevante de sesiones anteriores.
                # Se agrega SOLO a este turno — no entra al historial permanente.
                history_for_invocation = list(historial)
                if db_ok:
                    similares = await db.search_similar_messages(
                        user_text, limit=5, user_id=repl_uid, project_id=None)
                    relevantes = [r for r in similares if r["distancia"] < 0.8]
                    if relevantes:
                        lineas = []
                        for r in relevantes:
                            fecha = r["started_at"].strftime("%Y-%m-%d") if r["started_at"] else "?"
                            rol = "user" if r["role"] == "user" else "jax"
                            lineas.append(f"[{fecha}] {rol}: {r['content']}")
                        contexto_semantico = (
                            "Conversaciones relevantes de sesiones anteriores:\n"
                            + "\n".join(lineas)
                        )
                        history_for_invocation = [
                            {"role": "user", "content": "[memoria de sesiones anteriores]"},
                            {"role": "assistant", "content": contexto_semantico},
                        ] + history_for_invocation

                model_override = MODELO_PESADO if (modo_pesado and faceta == FACETA_PESADO) else None
                respuesta = await muscle.invoke(user_text, history=history_for_invocation, model=model_override)
                print(f"\n{label}: {respuesta}")

                # Voz en streaming, de fondo, con la voz/velocidad de la
                # faceta. La primera oracion suena en ~2-3s.
                if voice.enabled:
                    pconf = cfg["personalities"][faceta]
                    asyncio.create_task(voice.speak(
                        respuesta,
                        voz=pconf.get("voice_id", "em_alex"),
                        velocidad=float(pconf.get("voice_speed", 1.0)),
                    ))

                # Turno exitoso -> entra al hilo compartido en RAM.
                historial.append({"role": "user", "content": user_text})
                historial.append({"role": "assistant", "content": respuesta})
                if len(historial) > MAX_TURNS * 2:
                    del historial[: len(historial) - MAX_TURNS * 2]

                # Guardar la respuesta de la faceta en la base (con su modelo).
                db.save_message(
                    conv_uuid, faceta, respuesta,
                    facet=faceta, model=getattr(muscle, "model_default", None),
                )
            except MuscleError as e:
                print(f"\n{humanizar_error(label, e)}")
            except Exception as e:  # red de seguridad: nunca tumbar el latido
                print(f"\n{humanizar_error(label, e)}")
    finally:
        # Cierre limpio en CUALQUIER salida: oido, voz, luego memoria.
        await ears.shutdown()
        await voice.shutdown()
        if conv_uuid:
            await db.end_conversation(conv_uuid)
            if db_ok:
                print("\n[Procesando memoria de esta sesion en background...]")
                _lanzar_workers_background()
        await db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="JAX 2.0 — El primer latido. En memoria de Jairo Urbina.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", metavar="ARCHIVO.md", type=Path,
        help="Ejecutar una tarea autónoma desde un archivo .md (sin REPL).",
    )
    parser.add_argument(
        "--facet",
        metavar="NOMBRE",
        default=None,
        help="Fuerza la faceta en modo --task, saltando el router "
             "(ej: --facet ada). Sin esta bandera, el router decide.",
    )
    cli_args = parser.parse_args()

    if cli_args.facet and not cli_args.task:
        parser.error("--facet solo puede usarse junto con --task")

    if cli_args.facet:
        from jax.core.router import VALID_FACETAS
        if cli_args.facet not in VALID_FACETAS:
            parser.error(
                f"faceta '{cli_args.facet}' no válida. "
                f"Válidas: {', '.join(sorted(VALID_FACETAS))}"
            )

    if cli_args.task:
        asyncio.run(run_task(cli_args.task, facet_cli=cli_args.facet))
    else:
        asyncio.run(main())
