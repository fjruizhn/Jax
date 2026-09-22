#!/usr/bin/env python3
"""Genera el CLAUDE.md del usuario axioma. Nunca se escribe a mano (spec §6.1)."""
import pathlib
import sys
import tomllib

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from medicion import extraer_secciones  # noqa: E402


def generar(*, fuente_constitucion: pathlib.Path | None = None) -> str:
    """`fuente_constitucion=None` (producción, y `python3 generar_claude_md.py`) lee la
    ruta real de `cerebros.toml` -- esta máquina, /home/fruiz/claude-skills/... (host-
    bound, Fase 0). Pasarlo explícito es el seam para CI (B3/M1, auditoría adversarial
    2026-09-22): un runner sin esa ruta puede darle un DOBLE con las mismas secciones
    y correr las mismas afirmaciones, en vez de saltarse la prueba por completo."""
    c = tomllib.loads((AQUI / "cerebros.toml").read_text())["constitucion"]
    maquinas = tomllib.loads((AQUI / "maquinas.toml").read_text())
    lineas = ["# El Ejecutor de Axioma", "", c["identidad"], "", "## Máquinas", ""]
    for nombre, mq in maquinas.items():
        acceso = "local" if mq.get("local") else f"`ssh {mq['ip']}` (usuario axioma, puerto {mq['puerto']})"
        sudo = "sí" if mq.get("sudo") else "no"
        machine_id = mq.get("machine_id") or "SIN VERIFICAR — no operar acá hasta confirmarlo"
        lineas.append(f"- **{nombre}** — {mq['descripcion']} Acceso: {acceso}. sudo: {sudo}. "
                      f"machine-id: `{machine_id}`.")
    fuente = fuente_constitucion if fuente_constitucion is not None else pathlib.Path(c["fuente"])
    core = fuente.read_text()
    return "\n".join(lineas) + "\n\n" + extraer_secciones(core, c["secciones"])


if __name__ == "__main__":
    sys.stdout.write(generar())
