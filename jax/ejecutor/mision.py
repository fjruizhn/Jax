# jax/ejecutor/mision.py
"""Un TURNO de una misión del Ejecutor, por el camino gobernado (SP2, vía de producto).

Generaliza la misión de humo (scripts/ejecutor_contratos/mision_de_humo.py) a un objetivo
libre, con la sesión del arnés creada en el primer turno y RETOMADA por id en los siguientes
(spec 2026-09-15 §5). La plataforma lo lanza con `python -m jax.ejecutor.mision_servicio`
y el pedido por stdin; este módulo NO toca la base de la plataforma: emite eventos (una línea
JSON cada uno) y la plataforma, dueña de sus tablas, los guarda en la bitácora.

Orden de un turno (el mismo que el humo, sin atajos):
1. `exigir_contratos` con las máquinas de la misión (compuerta de datos de clientes incluida).
   Si se niega, el turno termina `rechazado` con los fallos (contrato, código, datos) y no se
   abre nada.
2. El vigía de C5 (`vigia_servicio`, el módulo que corre la unidad ejecutor-vigia@) arranca,
   vuelve a exigir los contratos y late: recién ahí el proxy de C3 deja de dar 423.
3. El cerebro corre en la jaula de la cuenta, SÓLO contra el proxy.
4. Capturas desde lo que la jaula devolvió; cada paso se ata al registro de C3 (tool_use_id y
   sha256). Afirmaciones → `transporte.entregar` (cita literal) → auditor de C5.
5. SIGTERM al vigía (fin normal), cadena del registro y pausa del Ejecutor.

DECISIÓN §2.0 de la Fase 2: el Ejecutor no escribe prosa. Lo que sale son pares (dato, cita)
y las salidas crudas (siempre, piso §2.3). Sin textos para personas: sólo códigos.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Awaitable, Callable

from jax.ejecutor import cita, transporte
from jax.ejecutor.contratos import auditor as A
from jax.ejecutor.contratos import destinos
from jax.ejecutor.contratos.arranque import ContratosNoVerificados

_TRUNCADO = re.compile(r"truncat", re.I)
CAMPOS = ("maquina", "comando", "linea", "dato", "proposito")
AUDITOR_ILEGIBLE = "auditor_ilegible"


class TurnoIlegible(ValueError):
    """`args[0]` es un código estable."""


@dataclass(frozen=True)
class Turno:
    mision_id: str
    n: int
    sesion: str
    objetivo: str
    instruccion: str
    hosts: frozenset

    @property
    def reanudar(self) -> bool:
        return self.n > 1

    @property
    def id_vigia(self) -> str:
        return f"{self.mision_id}-t{self.n}"

    @property
    def texto_de_mision(self) -> str:
        """Lo que juzgan el vigía y el auditor de C5. En un turno siguiente, el objetivo SOLO no
        alcanza: lo pedido en el turno quedaría `fuera_de_mision` (visto en real, 2026-09-17)."""
        if self.n == 1:
            return self.objetivo
        return f"{self.objetivo}\n\nTurno {self.n} de la misma misión: {self.instruccion}"


def _uuid_canonico(valor) -> bool:
    try:
        return isinstance(valor, str) and str(uuid.UUID(valor)) == valor
    except ValueError:
        return False


def turno_desde_json(datos: bytes) -> Turno:
    try:
        doc = json.loads(datos)
    except ValueError:
        raise TurnoIlegible("turno_no_es_json") from None
    if not isinstance(doc, dict):
        raise TurnoIlegible("turno_no_es_objeto")
    if not _uuid_canonico(doc.get("mision_id")):
        raise TurnoIlegible("turno_mision_invalida")
    n = doc.get("n")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise TurnoIlegible("turno_numero_invalido")
    if not _uuid_canonico(doc.get("sesion")):
        raise TurnoIlegible("turno_sesion_invalida")
    objetivo, instruccion, hosts = doc.get("objetivo"), doc.get("instruccion"), doc.get("hosts")
    if not isinstance(objetivo, str) or not objetivo.strip():
        raise TurnoIlegible("turno_sin_objetivo")
    if not isinstance(instruccion, str) or not instruccion.strip():
        raise TurnoIlegible("turno_sin_instruccion")
    if not isinstance(hosts, list) or not hosts or not all(isinstance(h, str) and h.strip() for h in hosts):
        raise TurnoIlegible("turno_sin_maquinas")
    return Turno(doc["mision_id"], n, doc["sesion"], objetivo.strip(), instruccion.strip(),
                 frozenset(h.strip() for h in hosts))


def prompt_del_turno(instruccion: str, hosts, de_la_mision: frozenset, reanudar: bool = False) -> str:
    maquinas = "\n".join(f"- {h.nombre}: ssh -tt -p {int(h.puerto)} axioma@{h.ip} <comando>"
                         for h in hosts if h.nombre in de_la_mision)
    vigencia = ("Los datos de turnos anteriores ya no valen: todo comando que respalde una afirmación se corre "
                "EN ESTE TURNO, aunque ya lo hayas corrido antes.\n" if reanudar else "")
    return (
        f"Misión: {instruccion}\n{vigencia}"
        f"Máquinas de esta misión (ninguna otra), y cómo se corre un comando en cada una:\n{maquinas}\n"
        "Corre los comandos con la herramienta Bash, uno por llamada, siempre con `ssh -tt` como arriba. "
        "Después responde SOLO un arreglo JSON, sin texto alrededor y sin bloque de código, con una "
        'afirmación por dato que responda la misión: {"maquina": <nombre de la lista>, '
        '"comando": <el comando COMPLETO tal como lo pasaste a Bash, con el `ssh -tt -p … axioma@…` delante, '
        'carácter por carácter>, "linea": <una línea COPIADA LITERAL de su salida>, '
        '"dato": <UN valor copiado tal cual de esa línea>, "proposito": <la pregunta que responde>}. '
        "El `dato` se busca entero dentro de la `linea`: no juntes dos valores, no calcules nada "
        "(ni porcentajes ni totales) y no agregues palabras tuyas. Si la misión pregunta dos cosas, "
        "manda una afirmación por cada una. Sin línea literal que lo respalde, un dato no se escribe."
    )


def evento(nombre: str, turno: int, /, **datos) -> str:
    return json.dumps({"evento": nombre, "turno": turno, "datos": datos}, ensure_ascii=False)


# --- lo que la jaula devolvió (antes vivía en el script de humo) ----------------------------

def _texto_de_resultado(contenido) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "".join(b.get("text", "") for b in contenido if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _eventos_del_stream(salida: bytes):
    for linea in salida.decode(errors="replace").splitlines():
        try:
            ev = json.loads(linea)
        except ValueError:
            continue
        if isinstance(ev, dict):
            yield ev


def pasos_del_stream(salida: bytes) -> tuple:
    """(pedidas {id: comando}, resultados {id: (contenido crudo, es_error)}, texto final)."""
    pedidas, resultados, final = {}, {}, None
    for ev in _eventos_del_stream(salida):
        mensaje = ev.get("message") if isinstance(ev.get("message"), dict) else {}
        for b in mensaje.get("content") or []:
            if not isinstance(b, dict):
                continue
            if ev.get("type") == "assistant" and b.get("type") == "tool_use" and b.get("name") == "Bash":
                pedidas[b.get("id")] = (b.get("input") or {}).get("command")
            elif ev.get("type") == "user" and b.get("type") == "tool_result":
                resultados[b.get("tool_use_id")] = (b.get("content"), bool(b.get("is_error", False)))
        if ev.get("type") == "result":
            final = ev.get("result")
    return pedidas, resultados, final


def sesion_anunciada(salida: bytes, sesion: str) -> bool:
    return any(ev.get("session_id") == sesion for ev in _eventos_del_stream(salida))


def capturas(pedidas: dict, resultados: dict, hosts) -> tuple:
    """Una captura por comando con resultado. La máquina la decide `destinos` (lo que el gancho
    ve), no el modelo; un comando que toca más de una máquina, o ninguna legible, no respalda."""
    salida = []
    for tid, comando in pedidas.items():
        if tid not in resultados or not isinstance(comando, str):
            continue
        try:
            tocadas = destinos.destinos(comando, hosts)
        except (destinos.HostDesconocido, destinos.ComandoIlegible):
            continue
        if len(tocadas) != 1:
            continue
        contenido, es_error = resultados[tid]
        texto = _texto_de_resultado(contenido)
        salida.append(cita.Captura(maquina=next(iter(tocadas)), comando=comando, salida=texto, stderr="",
                                   truncada=es_error or bool(_TRUNCADO.search(texto))))
    return tuple(salida)


def afirmaciones_del_texto(texto) -> tuple:
    """Fail-closed: un arreglo JSON de objetos, o JSON Lines donde TODA línea no vacía es un
    objeto (el cerebro local respondió así, 2026-09-17). Lo demás no afirma nada, y cada objeto
    sin los cinco campos de texto se cae. Ninguna forma afloja la cita: `transporte.entregar`
    y el auditor deciden qué sale."""
    if not isinstance(texto, str):
        return ()
    m = re.search(r"\[.*\]", texto, re.S)
    try:
        doc = json.loads(m.group(0) if m else texto)
    except ValueError:
        try:
            doc = [json.loads(l) for l in texto.splitlines() if l.strip()]
        except ValueError:
            return ()
        if not doc or not all(isinstance(d, dict) for d in doc):
            return ()
    if isinstance(doc, dict):  # JSON Lines de una sola línea
        doc = [doc]
    if not isinstance(doc, list):
        return ()
    return tuple(cita.Afirmacion(**{c: d[c] for c in CAMPOS}) for d in doc
                 if isinstance(d, dict) and all(isinstance(d.get(c), str) for c in CAMPOS))


def sha_de_resultado(contenido) -> str:
    return hashlib.sha256(json.dumps(contenido, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


# --- serialización hacia la plataforma --------------------------------------------------------

def _afirmacion(a) -> dict:
    return {c: getattr(a, c) for c in CAMPOS}


def _descartada(d) -> dict:
    return {"estado": d.estado, "codigo": d.motivo.codigo, "datos": dict(d.motivo.datos), **_afirmacion(d.afirmacion)}


def _cruda(c) -> dict:
    return {"maquina": c.maquina, "comando": c.comando, "salida": c.salida, "truncada": c.truncada}


def _retener_todo(entrega):
    motivo = cita.Motivo(AUDITOR_ILEGIBLE)
    return transporte.Entrega(entrega.crudas, (), entrega.descartadas + tuple(
        transporte.Descartada(a, A.RETENIDA_POR_AUDITOR, motivo) for a in entrega.respaldadas))


# --- el orquestador ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Dependencias:
    contexto: Callable[[], Awaitable]
    hosts: Callable
    exigir: Callable
    tamano_registro: Callable
    abrir_vigia: Callable
    latido_fresco: Callable
    correr_cerebro: Callable
    eventos_desde: Callable
    auditar: Callable
    cadena_ok: Callable
    leer_pausa: Callable
    espera_latido_s: float
    paso_espera_s: float = 0.5


def _resultado(estado, codigo, *, rechazo=(), entrega=None, verificacion=None, sesion_iniciada=False) -> dict:
    return {
        "estado": estado, "codigo": codigo, "rechazo": list(rechazo), "sesion_iniciada": sesion_iniciada,
        "afirmaciones": [_afirmacion(a) for a in entrega.respaldadas] if entrega else [],
        "descartadas": [_descartada(d) for d in entrega.descartadas] if entrega else [],
        "crudas": [_cruda(c) for c in entrega.crudas] if entrega else [],
        "verificacion": verificacion or {},
    }


async def correr_turno(turno: Turno, deps: Dependencias, emitir: Callable[[str], None]) -> dict:
    n = turno.n

    def dice(nombre, /, **datos):
        emitir(evento(nombre, n, **datos))

    dice("turno_lanzado", reanudar=turno.reanudar, maquinas=sorted(turno.hosts))
    ctx = await deps.contexto()
    try:
        await deps.exigir(ctx)
    except ContratosNoVerificados as exc:
        rechazo = [{"contrato": f.contrato, "codigo": f.codigo, "datos": dict(f.datos)} for f in exc.fallos]
        dice("arranque_rechazado", fallos=rechazo)
        dice("turno_rechazado", codigo="arranque_rechazado")
        return _resultado("rechazado", "arranque_rechazado", rechazo=rechazo)
    dice("arranque_verificado")
    hosts = await deps.hosts(ctx)
    desde = await deps.tamano_registro(ctx)
    vigia = await deps.abrir_vigia(ctx, turno.id_vigia, turno.texto_de_mision, turno.hosts)
    entrega, codigo, sesion_iniciada = None, None, False
    registro_cuadra, auditor_pauso, auditor_legible = False, False, True
    try:
        limite = time.monotonic() + deps.espera_latido_s
        while not await deps.latido_fresco(ctx):
            if not vigia.vive() or time.monotonic() > limite:
                codigo = "vigia_no_latio"
                dice("vigia_no_latio", vivo=vigia.vive())
                break
            await asyncio.sleep(deps.paso_espera_s)
        if codigo is None:
            dice("vigia_late")
            try:
                rc, crudo = await deps.correr_cerebro(ctx, prompt_del_turno(turno.instruccion, hosts, turno.hosts, turno.reanudar),
                                                      turno.sesion, turno.reanudar)
            except asyncio.TimeoutError:
                rc, crudo, codigo = None, b"", "cerebro_tope_vencido"
            sesion_iniciada = sesion_anunciada(crudo, turno.sesion)
            pedidas, resultados, final = pasos_del_stream(crudo)
            dice("cerebro_termino", rc=rc, pasos=len(pedidas), sesion_iniciada=sesion_iniciada)
            eventos = await deps.eventos_desde(ctx, desde)
            anotadas = {e.get("tool_use_id") for e in eventos if e.get("evento") == "herramienta_pedida"}
            devueltos = {e.get("tool_use_id"): e.get("sha256") for e in eventos
                         if e.get("evento") == "resultado_devuelto"}
            registro_cuadra = True
            for tid, comando in pedidas.items():
                en_registro = tid in anotadas
                cuadra = en_registro and (tid not in resultados or devueltos.get(tid) == sha_de_resultado(resultados[tid][0]))
                registro_cuadra = registro_cuadra and cuadra
                dice("paso", comando=comando, en_registro=en_registro, cuadra=cuadra)
            entrega = transporte.entregar(afirmaciones_del_texto(final), capturas(pedidas, resultados, hosts))
            try:
                revision = await deps.auditar(turno.texto_de_mision, entrega, A.maquinas_de(hosts, turno.hosts))
                auditor_pauso = revision.pausar
                entrega = A.aplicar_revision(entrega, revision)
            except Exception as exc:  # fail-soft: el turno entrega las crudas; fail-CLOSED para las afirmaciones: con el auditor ilegible o caído no sale ninguna
                auditor_legible = False
                dice(AUDITOR_ILEGIBLE, tipo=type(exc).__name__)
                entrega = _retener_todo(entrega)
            if auditor_pauso:
                dice("auditor_pauso", motivo=revision.motivo, paso=revision.paso)
            for a in entrega.respaldadas:
                dice("afirmacion_entregada", **_afirmacion(a))
            for d in entrega.descartadas:
                dice("afirmacion_descartada", estado=d.estado, codigo=d.motivo.codigo, dato=d.afirmacion.dato)
            if codigo is None and rc != 0:
                codigo = "cerebro_fallo"
    finally:
        rc_vigia, salida_vigia = await vigia.cerrar()
        cerro = rc_vigia == 0 and "cerrada=true" in salida_vigia
        dice("vigia_cerrado", rc=rc_vigia, cerrada=cerro)
    cadena = await deps.cadena_ok(ctx)
    pausa = await deps.leer_pausa(ctx)
    puesta = bool(pausa and pausa.get("puesta"))
    if puesta:
        dice("pausa_detectada", origen=pausa.get("origen"), motivo=pausa.get("motivo"), paso=pausa.get("paso"),
             legible=pausa.get("legible"))
    for condicion, cod in ((auditor_pauso, "auditor_pauso"),
                           (not registro_cuadra, "registro_no_cuadra"), (not cadena, "cadena_rota"),
                           (not cerro, "vigia_no_cerro"), (not auditor_legible, AUDITOR_ILEGIBLE)):
        if codigo is None and condicion:
            codigo = cod
    # Un pausa o un auditor que pausó mandan sobre un fallo de latido o de cerebro: es lo que hay que leer.
    if puesta:
        codigo = "pausa_puesta"
    verificacion = {"registro_cuadra": registro_cuadra, "cadena_ok": cadena, "pausa_puesta": puesta,
                    "auditor_pauso": auditor_pauso, "auditor_legible": auditor_legible}
    estado = "completado" if codigo is None else "fallido"
    dice("turno_completado" if codigo is None else "turno_fallido", codigo=codigo)
    return _resultado(estado, codigo, entrega=entrega, verificacion=verificacion, sesion_iniciada=sesion_iniciada)
