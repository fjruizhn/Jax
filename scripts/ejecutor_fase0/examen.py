#!/usr/bin/env python3
"""Fase 0 · U3: el examen. Subcomandos:
  correr  --cerebros qwen [kimi glm] --salida DIR   (con el prefijo de entorno del plan)
  mostrar --cerebro C --tarea N --salida DIR       (ayuda del calificador)
"""
import argparse
import asyncio
import json
import pathlib
import subprocess
import sys
import time
import tomllib

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from harness import correr  # noqa: E402
from medicion import eventos_stream_json, respuesta_final, salidas_de_herramientas  # noqa: E402


def _toml(n):
    return tomllib.loads((AQUI / n).read_text())


def _verdad(tarea, maquinas):
    mq = maquinas[tarea["maquina"]]
    if mq.get("local"):
        argv = ["bash", "-c", tarea["verdad"]]
    else:
        argv = ["ssh", "-o", "BatchMode=yes", "-p", str(mq["puerto"]), f"fruiz@{mq['ip']}", tarea["verdad"]]
    cp = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    return f"# {time.strftime('%Y-%m-%dT%H:%M:%S')} rc={cp.returncode}\n{cp.stdout}{cp.stderr}"


async def _cerebro(nombre, cerebros, decision):
    c = cerebros[nombre]
    if nombre == "qwen":
        return c["anthropic_base_url"], decision["modelo_qwen"], "ollama"
    from facet_resolver import resolve_facet
    f = await resolve_facet(c["faceta"])
    return c["anthropic_base_url"], f.model, f.credential


def cmd_correr(a):
    dir_ = pathlib.Path(a.salida).expanduser()
    cerebros, maquinas = _toml("cerebros.toml"), _toml("maquinas.toml")
    tareas = _toml("examen_tareas.toml")["tarea"]
    decision = json.loads((dir_ / "decision_contexto.json").read_text())
    for nombre in a.cerebros:
        base, modelo, llave = asyncio.run(_cerebro(nombre, cerebros, decision))
        out = dir_ / "examen" / nombre
        out.mkdir(parents=True, exist_ok=True)
        for t in tareas:
            if t["clientes"] and not cerebros[nombre]["datos_de_clientes"]:
                continue
            t0 = time.monotonic()
            cp = correr(base, modelo, t["pregunta"], llave=llave, auto_compact=decision["auto_compact"],
                        plugin_dirs=tuple(decision.get("plugin_dirs", [])))
            seg = time.monotonic() - t0
            (out / f"{t['id']}.jsonl").write_text(cp.stdout)
            (out / f"{t['id']}.verdad.txt").write_text(_verdad(t, maquinas))
            (out / f"{t['id']}.meta.json").write_text(json.dumps(
                {"rc": cp.returncode, "segundos": seg, "modelo": modelo, "stderr_cola": cp.stderr[-2000:]}))
            print(nombre, t["id"], "rc", cp.returncode, f"{seg:.0f}s")


def cmd_mostrar(a):
    base = pathlib.Path(a.salida).expanduser() / "examen" / a.cerebro
    evs = eventos_stream_json((base / f"{a.tarea}.jsonl").read_text())
    print("=== RESPUESTA FINAL ===\n", respuesta_final(evs))
    for i, s in enumerate(salidas_de_herramientas(evs), 1):
        print(f"=== SALIDA DE HERRAMIENTA {i} ===\n{s[:3000]}")
    print("=== VERDAD DE CAMPO ===\n", (base / f"{a.tarea}.verdad.txt").read_text())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("correr")
    c.add_argument("--cerebros", nargs="+", required=True, choices=["qwen", "kimi", "glm"])
    c.add_argument("--salida", required=True)
    m = sub.add_parser("mostrar")
    m.add_argument("--cerebro", required=True)
    m.add_argument("--tarea", type=int, required=True)
    m.add_argument("--salida", required=True)
    a = ap.parse_args()
    {"correr": cmd_correr, "mostrar": cmd_mostrar}[a.cmd](a)


if __name__ == "__main__":
    main()
