# jax/ejecutor/mision_servicio.py
"""El proceso de UN turno de misión del Ejecutor, lanzado por la plataforma (SP2).

    python -m jax.ejecutor.mision_servicio  < pedido.json

- stdin: `{"mision_id", "n", "sesion", "objetivo", "instruccion", "hosts"}` (ver mision.py).
- stdout: una línea JSON por evento (`{"evento", "turno", "datos"}`) y SIEMPRE, al final, una
  línea `resultado` con el estado del turno. La plataforma es la dueña de sus tablas: guarda
  los eventos en la bitácora y el resultado en el turno. Este proceso no escribe en la base.
- Salida 0 = turno completado; 1 = rechazado/fallido; 2 = pedido ilegible o sin configurar.

Camino gobernado: las dependencias reales son las MISMAS piezas que la misión de humo
(scripts/ejecutor_contratos/mision_de_humo.py): `arranque.exigir_contratos`, el vigía de C5
como proceso (`jax.ejecutor.contratos.vigia_servicio`, el módulo de la unidad
ejecutor-vigia@), la jaula de la cuenta contra el proxy de C3, el registro encadenado, el
auditor de C5 y la pausa del Ejecutor.

Topes sin defaults (Principio IV): JAX_EJECUTOR_TURNO_TOPE_S (lo que puede durar el cerebro
en un turno) y JAX_EJECUTOR_VIGIA_ESPERA_S (lo que se espera a que el vigía verifique los
contratos y lata). Sin ellos, no se lanza nada.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import sys
from pathlib import Path

from jax.ejecutor import mision as M
from jax.ejecutor.contratos import arranque, cuenta_axioma, pausa, politica
from jax.ejecutor.contratos import auditor as A
from jax.ejecutor.contratos.registro import verificar_cadena

VARIABLE_TOPE = "JAX_EJECUTOR_TURNO_TOPE_S"
VARIABLE_ESPERA = "JAX_EJECUTOR_VIGIA_ESPERA_S"
_CIERRE_VIGIA_S = 200  # TimeoutStopSec de la unidad (150) + margen: el vigía audita lo pendiente al parar


class SinConfigurar(RuntimeError):
    """`args[0]` es la variable que falta o no vale."""


def _segundos(env, variable: str) -> float:
    try:
        valor = float(env[variable])
    except (KeyError, ValueError):
        raise SinConfigurar(variable) from None
    if not math.isfinite(valor) or valor <= 0:
        raise SinConfigurar(variable)
    return valor


def leer_pausa(ruta) -> dict:
    """La pausa del Ejecutor como datos. Fail-closed como `pausa.pausa_puesta`: si no se puede
    leer, está PUESTA (`legible=False`). Un campo con un tipo inesperado se omite, no se inventa."""
    vacia = {"origen": None, "motivo": None, "paso": None, "momento": None}
    if not pausa.pausa_puesta(ruta):
        return {"puesta": False, "legible": True, **vacia}
    try:
        doc = json.loads(Path(ruta).read_bytes())
    except (OSError, ValueError):  # fail-soft sobre la LECTURA del motivo; la pausa sigue PUESTA (fail-closed)
        return {"puesta": True, "legible": False, **vacia}
    if not isinstance(doc, dict):
        return {"puesta": True, "legible": False, **vacia}
    texto = {k: doc[k] if isinstance(doc.get(k), str) else None for k in ("origen", "motivo", "momento")}
    paso = doc.get("paso") if isinstance(doc.get("paso"), int) and not isinstance(doc.get("paso"), bool) else None
    return {"puesta": True, "legible": True, **texto, "paso": paso}


class Vigia:
    def __init__(self, proc, ruta: Path):
        self._proc, self._ruta = proc, ruta

    def vive(self) -> bool:
        return self._proc.returncode is None

    async def cerrar(self) -> tuple:
        try:
            if self._proc.returncode is None:
                self._proc.send_signal(signal.SIGTERM)
            try:
                salida, _ = await asyncio.wait_for(self._proc.communicate(), _CIERRE_VIGIA_S)
            except asyncio.TimeoutError:
                self._proc.kill()
                salida, _ = await self._proc.communicate()
            return self._proc.returncode, salida.decode(errors="replace")
        finally:
            self._ruta.unlink(missing_ok=True)


async def abrir_vigia(directorio, id_vigia: str, texto: str, hosts, *, argv=None) -> Vigia:
    ruta = Path(directorio) / f"{id_vigia}.json"
    await asyncio.to_thread(ruta.write_text, json.dumps({"mision": texto, "hosts": sorted(hosts)}), "utf-8")
    argv = argv or [sys.executable, "-m", "jax.ejecutor.contratos.vigia_servicio"]
    proc = await asyncio.create_subprocess_exec(*argv, str(ruta), stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.DEVNULL, start_new_session=True)
    return Vigia(proc, ruta)


def _eventos_desde(registro: Path, desde: int) -> list:
    with open(registro, "rb") as f:
        f.seek(desde)
        return [json.loads(l) for l in f.read().splitlines() if l.strip()]


def dependencias_reales(env, turno: M.Turno, *, tope_s: float, espera_s: float) -> M.Dependencias:

    async def contexto():
        return arranque.contexto_desde_entorno(env, turno.hosts)

    async def hosts(ctx):
        doc = json.loads(await asyncio.to_thread(ctx.cuenta.politica.read_bytes))
        return politica.validar(doc).hosts

    async def tamano_registro(ctx):
        return (await asyncio.to_thread(os.stat, ctx.registro)).st_size

    async def vigia(ctx, id_vigia, texto, maquinas):
        return await abrir_vigia(env["JAX_EJECUTOR_MISIONES"], id_vigia, texto, maquinas)

    async def latido(ctx):
        return await asyncio.to_thread(pausa.latido_fresco, ctx.latido, ctx.latido_max_s)

    async def cerebro(ctx, prompt, sesion, reanudar):
        remoto = cuenta_axioma.remoto_claude(
            ctx.cuenta, base_url=f"http://127.0.0.1:{ctx.puerto_proxy}", modelo=env["JAX_PROXY_CARRIL_MODELO"],
            prompt=prompt, herramientas="Bash", max_salida_tokens=int(env["JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS"]),
            sesion=sesion, reanudar=reanudar)
        rc, crudo, _ = await cuenta_axioma.correr_en_la_cuenta(ctx.cuenta, remoto, entrada=b"sin-clave\n",
                                                               tope_s=tope_s)
        return rc, crudo

    async def eventos(ctx, desde):
        return await asyncio.to_thread(_eventos_desde, ctx.registro, desde)

    async def auditar(texto, entrega, maquinas):
        from facet_resolver import resolve_facet
        from jacobs.store import conexion
        from jax.ejecutor.contratos import auditor_cliente, eleccion_c5
        # Spec 2026-09-18-auditor-local-opcion.md §4: `turno.hosts`, no `maquinas` (que ya
        # perdió el nombre plano por A.maquinas_de) -- son las mismas máquinas de la misión,
        # y elegir_y_resolver_auditor necesita nombres para consultar ejecutor_host.
        async with conexion(desechable=True) as conn:
            cfg = await eleccion_c5.leer_config(conn)
            faceta, _, _ = await eleccion_c5.elegir_y_resolver_auditor(
                conn, cfg=cfg, hosts_mision=turno.hosts, resolve_facet=resolve_facet)
        return await auditor_cliente.auditar(A.Lote(texto, (), A.afirmaciones_auditables(entrega), maquinas),
                                             faceta=faceta, max_tokens=cfg.max_tokens)

    async def cadena(ctx):
        return (await asyncio.to_thread(verificar_cadena, ctx.registro)).ok

    async def pausa_leida(ctx):
        return await asyncio.to_thread(leer_pausa, ctx.pausa)

    return M.Dependencias(contexto=contexto, hosts=hosts, exigir=arranque.exigir_contratos,
                          tamano_registro=tamano_registro, abrir_vigia=vigia, latido_fresco=latido,
                          correr_cerebro=cerebro, eventos_desde=eventos, auditar=auditar, cadena_ok=cadena,
                          leer_pausa=pausa_leida, espera_latido_s=espera_s)


def principal(entrada, salida, env) -> int:
    def emitir(linea: str) -> None:
        salida.write(linea + "\n")
        salida.flush()

    def resultado(turno, datos) -> None:
        emitir(json.dumps({"evento": "resultado", "turno": turno, "datos": datos}, ensure_ascii=False))

    try:
        turno = M.turno_desde_json(entrada.read())
    except M.TurnoIlegible as exc:
        resultado(None, {"estado": "fallido", "codigo": "turno_ilegible", "detalle": exc.args[0]})
        return 2
    try:
        deps = dependencias_reales(env, turno, tope_s=_segundos(env, VARIABLE_TOPE),
                                   espera_s=_segundos(env, VARIABLE_ESPERA))
        datos = asyncio.run(M.correr_turno(turno, deps, emitir))
    except (SinConfigurar, cuenta_axioma.CuentaSinConfigurar, pausa.PausaSinConfigurar) as exc:
        resultado(turno.n, {"estado": "fallido", "codigo": "sin_configurar", "detalle": exc.args[0]})
        return 2
    except KeyError as exc:  # una variable de entorno obligatoria que falta (contexto_desde_entorno lee env[...])
        resultado(turno.n, {"estado": "fallido", "codigo": "sin_configurar", "detalle": str(exc.args[0])})
        return 2
    except Exception as exc:  # fail-soft: el turno termina FALLIDO y lo dice; sólo el tipo, el mensaje puede traer cualquier cosa
        resultado(turno.n, {"estado": "fallido", "codigo": "runner_error", "tipo": type(exc).__name__})
        return 1
    resultado(turno.n, datos)
    return 0 if datos.get("estado") == "completado" else 1


if __name__ == "__main__":
    sys.exit(principal(sys.stdin.buffer, sys.stdout, os.environ))
