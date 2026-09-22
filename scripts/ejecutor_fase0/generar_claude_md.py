#!/usr/bin/env python3
"""Genera el CLAUDE.md del usuario axioma. Nunca se escribe a mano (spec §6.1)."""
import pathlib
import sys
import tomllib

AQUI = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
from medicion import extraer_secciones  # noqa: E402


def generar() -> str:
    c = tomllib.loads((AQUI / "cerebros.toml").read_text())["constitucion"]
    maquinas = tomllib.loads((AQUI / "maquinas.toml").read_text())
    lineas = ["# El Ejecutor de Axioma", "", c["identidad"], "", "## Máquinas", ""]
    for nombre, mq in maquinas.items():
        acceso = "local" if mq.get("local") else f"`ssh {mq['ip']}` (usuario axioma, puerto {mq['puerto']})"
        sudo = "sí" if mq.get("sudo") else "no"
        machine_id = mq.get("machine_id") or "SIN VERIFICAR — no operar acá hasta confirmarlo"
        lineas.append(f"- **{nombre}** — {mq['descripcion']} Acceso: {acceso}. sudo: {sudo}. "
                      f"machine-id: `{machine_id}`.")
    core = pathlib.Path(c["fuente"]).read_text()
    return "\n".join(lineas) + "\n\n" + extraer_secciones(core, c["secciones"])


if __name__ == "__main__":
    sys.stdout.write(generar())
