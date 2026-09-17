# jax/ejecutor/contratos/canario_c1.py
"""Canario permanente de C1 (spec 2026-09-15 §4): si no se dispara, el Ejecutor no arranca.

Tres observaciones, todas como `axioma`:
1. Autoprueba: cada ejemplo de cada regla da lo que dice, con el gancho INSTALADO.
2. Canario directo: el gancho instalado bloquea el evento canario (exit 2, regla
   canario_c1) y deja pasar un control inocuo (exit 0).
3. Canario por el arnés real dentro de la jaula, contra un upstream falso: el
   comando canario vuelve `is_error` con la regla canario_c1 y NO se ejecutó (su
   archivo no existe); el control SÍ se ejecutó (su nonce vuelve en la salida).

Devuelve los Fallo; vacío = C1 vivo. No lanza por un contrato roto: lo reporta.
"""
from __future__ import annotations

import json
import secrets
import shlex

from jax.ejecutor.contratos import canario_upstream, cuenta_axioma
from jax.ejecutor.contratos.fallo import Fallo

_TOPE_DIRECTO_S = 30
_TOPE_ARNES_S = 180
_MARCA = 'regla="canario_c1"'


def _evento(comando: str) -> bytes:
    return json.dumps({"tool_name": "Bash", "tool_input": {"command": comando}}).encode()


async def verificar_c1(c, *, puerto_canario: int, correr=cuenta_axioma.correr_en_la_cuenta,
                       upstream=canario_upstream.UpstreamCanario, nonce: str | None = None) -> tuple:
    nonce = nonce or secrets.token_hex(8)
    q = shlex.quote
    gancho = f"{q(str(c.lib / 'gancho.sh'))} {q(str(c.politica))}"
    fallos: list[Fallo] = []

    rc, salida, errores = await correr(c, f"{gancho} autoprueba", tope_s=_TOPE_DIRECTO_S)
    if rc == 255:
        return (Fallo("c1", "cuenta_inalcanzable", (("rc", rc),)),)
    if rc != 0:
        if b'codigo="politica_ilegible"' in errores:
            fallos.append(Fallo("c1", "politica_ilegible", (("stderr", errores.decode(errors="replace").strip()),)))
        else:
            fallos.append(Fallo("c1", "autoprueba_fallida", (("fallos", salida.decode(errors="replace")),)))

    rc, _, errores = await correr(c, gancho, entrada=_evento("echo ejecutor-canario-c1"), tope_s=_TOPE_DIRECTO_S)
    if rc != 2:
        fallos.append(Fallo("c1", "canario_directo_no_bloqueado", (("rc", rc),)))
    elif _MARCA.encode() not in errores:
        # Bloqueado, pero no por su regla: la política o el gancho están rotos.
        fallos.append(Fallo("c1", "canario_directo_sin_su_regla", (("stderr", errores.decode(errors="replace").strip()),)))
    rc, _, errores = await correr(c, gancho, entrada=_evento("uptime"), tope_s=_TOPE_DIRECTO_S)
    if rc != 0:
        fallos.append(Fallo("c1", "control_directo_bloqueado", (("stderr", errores.decode(errors="replace").strip()),)))

    marca = f"$HOME/.canario-c1-{nonce}"
    id_canario, id_control = f"toolu_canario_{nonce}", f"toolu_control_{nonce}"
    guion = [canario_upstream.guion_bash(id_canario, f'touch "{marca}" # ejecutor-canario-c1'),
             canario_upstream.guion_bash(id_control, f"echo control-c1-{nonce}")]
    async with upstream(guion, "127.0.0.1", puerto_canario) as up:
        remoto = cuenta_axioma.remoto_claude(c, base_url=f"http://127.0.0.1:{up.puerto}", modelo="canario",
                                             prompt="canario")
        await correr(c, remoto, entrada=b"canario\n", tope_s=_TOPE_ARNES_S)
        canario, control = up.resultados.get(id_canario), up.resultados.get(id_control)
    if canario is None:
        fallos.append(Fallo("c1", "claude_no_llego_al_canario", (("peticiones", len(up.peticiones)),)))
    elif not canario.es_error:
        fallos.append(Fallo("c1", "canario_no_bloqueado", (("contenido", canario.contenido[:500]),)))
    elif _MARCA not in canario.contenido:
        fallos.append(Fallo("c1", "canario_sin_su_regla", (("contenido", canario.contenido[:500]),)))
    if control is None or control.es_error or f"control-c1-{nonce}" not in control.contenido:
        fallos.append(Fallo("c1", "control_no_ejecutado",
                            (("contenido", "" if control is None else control.contenido[:500]),)))
    rc, _, _ = await correr(c, f'test -e "{marca}"', tope_s=_TOPE_DIRECTO_S)
    if rc == 0:
        fallos.append(Fallo("c1", "canario_ejecutado"))
    return tuple(fallos)
