#!/usr/bin/env python3
"""Fase 0 · U2: tokens de arranque del arnés, por configuración de plugins.

Uso: python3 scripts/ejecutor_fase0/arranque.py --salida ~/ejecutor-fase0/resultados [--modelo M]
Corre como axioma (harness.py). Configs: A sin plugins; B +superpowers;
C +superpowers, frontend-design, ui-ux-pro-max, impeccable.
Dos métodos para el número (lección 15): usage.input_tokens del turno
principal y modelUsage[modelo].inputTokens (incluye llamadas laterales).
"""
import argparse
import json
import pathlib
import sys
import tomllib

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from harness import correr  # noqa: E402

P = "/opt/ejecutor/plugins"
CONFIGS = {
    "A": (),
    "B": (f"{P}/superpowers",),
    "C": (f"{P}/superpowers", f"{P}/frontend-design", f"{P}/ui-ux-pro-max", f"{P}/impeccable"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", required=True)
    ap.add_argument("--modelo", default=None)
    a = ap.parse_args()
    q = tomllib.loads((AQUI / "cerebros.toml").read_text())["qwen"]
    modelo = a.modelo or q["modelo"]
    dir_ = pathlib.Path(a.salida).expanduser()
    for nombre, dirs in CONFIGS.items():
        cp = correr(q["anthropic_base_url"], modelo, "responde solo con la palabra: ok",
                    plugin_dirs=dirs, formato="json")
        if cp.returncode != 0:
            print(nombre, "FALLÓ rc", cp.returncode, cp.stderr[-500:])
            sys.exit(1)
        d = json.loads(cp.stdout)
        fila = {"config": nombre, "modelo": modelo, "plugins": list(dirs),
                "input_tokens": d["usage"]["input_tokens"],
                "model_input_tokens": d["modelUsage"][modelo]["inputTokens"],
                "result": d.get("result")}
        with (dir_ / "arranque.jsonl").open("a") as f:
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
        print(json.dumps(fila, ensure_ascii=False))


if __name__ == "__main__":
    main()
