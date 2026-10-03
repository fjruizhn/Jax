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
2. El vigía de C5 (`vigia_servicio`, lanzado por `abrir_vigia` -- mision_servicio.py -- como
   SUBPROCESO DIRECTO, heredando la identidad de `jax-platform`, `fruiz`; no hay unidad
   systemd, ver DEUDA.md) arranca, vuelve a exigir los contratos y late: recién ahí el proxy
   de C3 deja de dar 423.
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
from jax.ejecutor.codigo.entrega import validar_owner_repo
from jax.ejecutor.contratos import auditor as A
from jax.ejecutor.contratos import destinos
from jax.ejecutor.contratos.arranque import ContratosNoVerificados

_TRUNCADO = re.compile(r"truncat", re.I)
CAMPOS = ("maquina", "comando", "linea", "dato", "proposito")
AUDITOR_ILEGIBLE = "auditor_ilegible"
TIPOS = ("servidor", "codigo")
#: Estados de `mision_codigo.entregar` que NO son un fallo del turno.
ENTREGA_SIN_FALLO = ("abierto", "sin_cambios")


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
    # Misión de código (spec 2026-09-28 v1.3): `repo` = {"owner_repo", "comandos_prueba" (tupla)},
    # ya validado por `turno_desde_json`. `None` en una misión de servidor.
    tipo: str = "servidor"
    repo: dict | None = None

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
    tipo = doc.get("tipo", "servidor")
    if tipo not in TIPOS:
        raise TurnoIlegible("turno_tipo_invalido")
    if tipo == "servidor" and doc.get("repo") is not None:
        raise TurnoIlegible("turno_repo_invalido")
    repo = _repo_del_turno(doc.get("repo")) if tipo == "codigo" else None
    return Turno(doc["mision_id"], n, doc["sesion"], objetivo.strip(), instruccion.strip(),
                 frozenset(h.strip() for h in hosts), tipo, repo)


def _repo_del_turno(repo) -> dict:
    """Fail-closed: exactamente `owner_repo` (dueño/repo validado; la URL la deriva la entrega,
    nunca llega como campo) y `comandos_prueba` (lista de textos no vacíos). Una clave de más
    -- p. ej. un `remoto_url` de la versión anterior del plan -- se rechaza, no se ignora."""
    if not isinstance(repo, dict) or set(repo) != {"owner_repo", "comandos_prueba"}:
        raise TurnoIlegible("turno_repo_invalido")
    try:
        owner_repo = validar_owner_repo(repo["owner_repo"])
    except ValueError:
        raise TurnoIlegible("turno_repo_invalido") from None
    comandos = repo["comandos_prueba"]
    if not isinstance(comandos, list) or not all(isinstance(c, str) and c.strip() for c in comandos):
        raise TurnoIlegible("turno_repo_invalido")
    return {"owner_repo": owner_repo, "comandos_prueba": tuple(c.strip() for c in comandos)}


def prompt_del_turno(instruccion: str, hosts, de_la_mision: frozenset, reanudar: bool = False) -> str:
    maquinas = "\n".join(f"- {h.nombre}: ssh -tt -p {int(h.puerto)} axioma@{h.ip} <comando>"
                         for h in hosts if h.nombre in de_la_mision)
    vigencia = ("Los datos de turnos anteriores ya no valen: todo comando que respalde una afirmación se corre "
                "EN ESTE TURNO, aunque ya lo hayas corrido antes.\n" if reanudar else "")
    return (
        f"Misión: {instruccion}\n{vigencia}"
        f"Máquinas de esta misión (ninguna otra), y cómo se corre un comando en cada una:\n{maquinas}\n"
        "Corre los comandos con la herramienta Bash, uno por llamada, siempre con `ssh -tt` como arriba. "
        "Todo lo que filtre o recorte (tubería, `grep`, `tail`, redirección) va DENTRO de las comillas del "
        "comando remoto: una tubería afuera corre en otra máquina, el comando pasa a tocar dos y entonces "
        "no respalda nada. "
        "Después responde SOLO un arreglo JSON, sin texto alrededor y sin bloque de código, con una "
        'afirmación por dato que responda la misión: {"maquina": <nombre de la lista>, '
        '"comando": <el comando COMPLETO tal como lo pasaste a Bash, con el `ssh -tt -p … axioma@…` delante, '
        'carácter por carácter>, "linea": <una línea COPIADA LITERAL de su salida>, '
        '"dato": <UN valor copiado tal cual de esa línea>, "proposito": <la pregunta que responde>}. '
        "El `dato` se busca entero dentro de la `linea`: no juntes dos valores, no calcules nada "
        "(ni porcentajes ni totales) y no agregues palabras tuyas. Si la misión pregunta dos cosas, "
        "manda una afirmación por cada una. Sin línea literal que lo respalde, un dato no se escribe."
    )


def instrucciones_de_codigo(repo: dict) -> str:
    """Lo que se agrega al pedido en una misión de código (es prompt del modelo, no UI: no va
    por i18n). La entrega al remoto y el PR los hace el sistema, fuera de la jaula."""
    comandos = "\n".join(f"- `{c}`" for c in repo["comandos_prueba"]) or "- (el repo no declara comandos)"
    return ("\n\nMISIÓN DE CÓDIGO. Trabajas en el directorio actual, un clon de "
            f"{repo['owner_repo']} en la rama de la misión. Haz commits locales (git add/commit). "
            "NO empujes: la entrega la hace el sistema. Pruebas del repo:\n" + comandos +
            "\nCita la salida real de cada prueba que corras; lo que no corriste, dilo como no corrido.")


def prompt_de_codigo(instruccion: str, repo: dict, maquina_local: str | None, reanudar: bool = False) -> str:
    """El pedido de una misión de CÓDIGO (MAJOR-3 de la auditoría de escalón 3). Sin lista de ssh: medido
    por el controlador con los usuarios reales, desde la jaula no hay salida (la llave de axioma no
    entra a axioma@hall9000 y el cerco rechaza 127.0.0.1:58291). Todo corre en el directorio actual,
    y la máquina de cada afirmación es la local -- la misma a la que `destinos` atribuye esas capturas."""
    maquina = maquina_local or "local"
    vigencia = ("Los datos de turnos anteriores ya no valen: todo comando que respalde una afirmación se corre "
                "EN ESTE TURNO, aunque ya lo hayas corrido antes.\n" if reanudar else "")
    return (
        f"Misión: {instruccion}\n{vigencia}"
        "Todos los comandos corren en ESTA máquina, en el directorio actual, con la herramienta Bash, uno por "
        "llamada. No hay red ni otras máquinas: no intentes conectarte a ninguna."
        + instrucciones_de_codigo(repo) + "\n"
        "Al terminar responde SOLO un arreglo JSON, sin texto alrededor y sin bloque de código, con una "
        f'afirmación por dato: {{"maquina": "{maquina}", "comando": <el comando COMPLETO tal como lo pasaste a '
        'Bash, carácter por carácter>, "linea": <una línea COPIADA LITERAL de su salida>, "dato": <UN valor '
        'copiado tal cual de esa línea>, "proposito": <la pregunta que responde>}. Sin línea literal que lo '
        "respalde, un dato no se escribe."
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


#: Herramientas que cuentan como un PASO de la misión: Bash (comandos) y Skill (las
#: tres skills declaradas en cerebros.toml, spec 2026-09-22 -- el arnés las tiene en
#: --allowedTools junto con Bash). Cualquier otra (p. ej. Read) NO cuenta: el Ejecutor
#: lee con `cat` (Bash), nunca con la herramienta Read.
_HERRAMIENTAS_QUE_CUENTAN_COMO_PASO = frozenset({"Bash", "Skill"})


def pasos_del_stream(salida: bytes) -> tuple:
    """(pedidas {id: comando}, resultados {id: (contenido crudo, es_error)}, texto final).

    `comando` es el texto del `command` para Bash; para Skill (sin ese campo) queda en
    `None` -- `capturas()` ya descarta cualquier `comando` que no sea `str`, así que un
    paso de Skill nunca se lee como si tocara una máquina por ssh. Lo que SÍ gana al
    contar como paso: entra en `pasos` de `cerebro_termino` y se verifica contra el
    registro de C3 (`registro_cuadra`) igual que un Bash -- antes se perdía en silencio."""
    pedidas, resultados, final = {}, {}, None
    for ev in _eventos_del_stream(salida):
        mensaje = ev.get("message") if isinstance(ev.get("message"), dict) else {}
        for b in mensaje.get("content") or []:
            if not isinstance(b, dict):
                continue
            if (ev.get("type") == "assistant" and b.get("type") == "tool_use"
                    and b.get("name") in _HERRAMIENTAS_QUE_CUENTAN_COMO_PASO):
                pedidas[b.get("id")] = (b.get("input") or {}).get("command")
            elif ev.get("type") == "user" and b.get("type") == "tool_result":
                resultados[b.get("tool_use_id")] = (b.get("content"), bool(b.get("is_error", False)))
        if ev.get("type") == "result":
            final = ev.get("result")
    return pedidas, resultados, final


def sesion_anunciada(salida: bytes, sesion: str) -> bool:
    return any(ev.get("session_id") == sesion for ev in _eventos_del_stream(salida))


def capturas(pedidas: dict, resultados: dict, hosts, permitidas: frozenset | None = None) -> tuple:
    """Una captura por comando con resultado. La máquina la decide `destinos` (lo que el gancho
    ve), no el modelo; un comando que toca más de una máquina, o ninguna legible, no respalda.
    `permitidas` (MINOR-2 de la auditoría): una captura de una máquina fuera de ese conjunto -- las
    máquinas de C5 del turno -- se descarta, de forma determinista, antes de la cita."""
    salida = []
    for tid, comando in pedidas.items():
        if tid not in resultados or not isinstance(comando, str) or destinos.ssh_no_literal(comando):
            continue
        try:
            tocadas = destinos.destinos(comando, hosts)
        except (destinos.HostDesconocido, destinos.ComandoIlegible):
            continue
        if len(tocadas) != 1 or (permitidas is not None and not tocadas <= permitidas):
            continue
        contenido, es_error = resultados[tid]
        texto = _texto_de_resultado(contenido)
        salida.append(cita.Captura(maquina=next(iter(tocadas)), comando=comando, salida=texto, stderr="",
                                   truncada=es_error or bool(_TRUNCADO.search(texto))))
    return tuple(salida)


_VALLA = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```", re.S)


def _sin_valla_de_codigo(texto: str) -> str:
    """Devuelve el contenido del primer bloque ``` ``` ```, o el texto tal cual si no hay uno.

    INCIDENTE 2026-09-20 (misiones 445ac19c y 10707ccc): el cerebro hizo TODO bien
    --corrio el ssh, copio la linea literal, armo el objeto con los cinco campos-- y lo
    entrego dentro de un bloque ```json. Un arreglo dentro de la valla ya se leia (el
    `re.search` de `[...]` lo encuentra igual); un OBJETO UNICO no, porque el respaldo de
    JSON Lines se atraganta con las lineas de la valla. Las dos misiones salieron
    "completada" con CERO afirmaciones: un cero silencioso que se lee como exito.

    El prompt pide "sin bloque de codigo" y el modelo lo pone igual: una instruccion no es
    un contrato. Esto NO afloja la cita -- la valla es envoltorio del transporte, no
    contenido, y el objeto que sale es identico. `transporte.entregar` y el auditor siguen
    decidiendo que se publica; lo unico que cambia es que deja de tirarse a la basura."""
    m = _VALLA.search(texto)
    return m.group(1) if m else texto


def afirmaciones_del_texto(texto) -> tuple:
    """Fail-closed: un arreglo JSON de objetos, o JSON Lines donde TODA línea no vacía es un
    objeto (el cerebro local respondió así, 2026-09-17). Lo demás no afirma nada, y cada objeto
    sin los cinco campos de texto se cae. Ninguna forma afloja la cita: `transporte.entregar`
    y el auditor deciden qué sale."""
    if not isinstance(texto, str):
        return ()
    texto = _sin_valla_de_codigo(texto)
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


def _maquinas_para_c5(turno: Turno, hosts) -> frozenset:
    """Las máquinas que C5 recibe como «de la misión». En una misión de CÓDIGO (ruling 4a, Tarea 9)
    el Bash del cerebro corre en la jaula, en la máquina local: `destinos` ya le atribuye esas
    capturas a la ÚNICA máquina `es_local` del inventario, y C5 tiene que verla como de la misión o
    una afirmación sobre las pruebas corridas le llega como ajena. Si no hay exactamente una local,
    no se suma nada (`destinos` tampoco produce capturas locales en ese caso)."""
    if turno.tipo != "codigo":
        return turno.hosts
    locales = [h.nombre for h in hosts if h.es_local]
    return turno.hosts | frozenset(locales) if len(locales) == 1 else turno.hosts


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
    #: `poner_pausa(ctx, motivo)`: si el vigia no cierra, la mision frena (el ultimo lote pudo quedar
    #: sin auditar). Las dependencias reales siempre lo traen; None solo en pruebas de otras cosas.
    poner_pausa: Callable | None = None
    # Misión de código: `preparar_codigo(ctx) -> Clon` y
    # `entregar_codigo(ctx, clon, entrega, auditor_legible) -> dict` (mision_codigo.entregar).
    preparar_codigo: Callable | None = None
    entregar_codigo: Callable | None = None


def _resultado(estado, codigo, *, rechazo=(), entrega=None, verificacion=None, sesion_iniciada=False,
               entrega_codigo=None) -> dict:
    extra = {} if entrega_codigo is None else {"entrega_codigo": entrega_codigo}
    return {**extra,
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
    es_codigo = turno.tipo == "codigo"
    if es_codigo and (deps.preparar_codigo is None or deps.entregar_codigo is None):
        # Fail-closed: un turno de código sin con qué preparar Y entregar no arranca nada.
        dice("turno_fallido", codigo="codigo_sin_dependencias")
        return _resultado("fallido", "codigo_sin_dependencias")
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
    resultado_entrega, clon = None, None
    try:
        limite = time.monotonic() + deps.espera_latido_s
        while not await deps.latido_fresco(ctx):
            if not vigia.vive() or time.monotonic() > limite:
                codigo = "vigia_no_latio"
                # `espera_s` no es adorno: el 2026-09-20 una mision fallo asi y para
                # saber si el vigia estaba MUERTO o solo lento hubo que medir a mano,
                # contra la base, la distancia entre `turno_lanzado` y `vigia_late` de
                # las misiones que si latieron. Con `el_juez` se tarda 125-199 s contra
                # un presupuesto de 180: el tope estaba calibrado para el auditor de
                # nube (16 s). `vivo` distingue los dos casos -- muerto es un fallo del
                # vigia, vivo y sin latir es un presupuesto corto.
                dice("vigia_no_latio", vivo=vigia.vive(), espera_s=deps.espera_latido_s)
                break
            await asyncio.sleep(deps.paso_espera_s)
        maquinas_c5 = _maquinas_para_c5(turno, hosts)
        if codigo is None:
            dice("vigia_late")
            if es_codigo:
                # Antes del cerebro: espejo + clon + dependencias (turno 1) o fetch del espejo (≥ 2).
                try:
                    clon = await deps.preparar_codigo(ctx)
                except Exception as exc:  # fail-closed: sin clon no hay cerebro ni entrega; a la bitácora solo el tipo
                    codigo = "preparar_fallo"
                    dice("preparar_fallo", tipo=type(exc).__name__)
                else:
                    dice("codigo_preparado", rama=clon.rama, base=clon.rama_por_omision,
                         dependencias=list(clon.dependencias))
        if codigo is None:
            if es_codigo:
                locales = [h.nombre for h in hosts if h.es_local]
                prompt = prompt_de_codigo(turno.instruccion, turno.repo, locales[0] if len(locales) == 1 else None,
                                          turno.reanudar)
            else:
                prompt = prompt_del_turno(turno.instruccion, hosts, turno.hosts, turno.reanudar)
            try:
                rc, crudo = await deps.correr_cerebro(ctx, prompt, turno.sesion, turno.reanudar)
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
            entrega = transporte.entregar(afirmaciones_del_texto(final), capturas(pedidas, resultados, hosts,
                                                                                     maquinas_c5))
            try:
                revision = await deps.auditar(turno.texto_de_mision, entrega,
                                              A.maquinas_de(hosts, maquinas_c5))
                auditor_pauso = revision.pausar
                entrega = A.aplicar_revision(entrega, revision)
            except Exception as exc:  # fail-soft: el turno entrega las crudas; fail-CLOSED para las afirmaciones: con el auditor ilegible o caído no sale ninguna
                auditor_legible = False
                # `motivo` SOLO para AuditorIlegible: su codigo es una constante (proveedor_fallo,
                # json_invalido...). El texto de cualquier otra excepcion puede traer una llave
                # o un cuerpo HTTP y no viaja.
                dice(AUDITOR_ILEGIBLE, tipo=type(exc).__name__,
                     **({"motivo": exc.codigo} if isinstance(exc, A.AuditorIlegible) else {}))
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
        rc_vigia, salida_vigia, err_vigia = await vigia.cerrar()
        cerro = rc_vigia == 0 and "cerrada=true" in salida_vigia
        # El stderr SOLO cuando algo salio mal, y solo la cola: en el camino feliz son
        # lineas de INFO de httpx que no dicen nada y ensucian la bitacora. Va redactado
        # desde `Vigia.cerrar`. El 2026-09-20 una mision fallo con `vigia_no_latio` y no
        # habia una sola linea para investigar.
        dice("vigia_cerrado", rc=rc_vigia, cerrada=cerro,
             **({} if cerro else {"stderr": err_vigia[-2000:]}))
        if not cerro and deps.poner_pausa is not None:
            # Fail-closed: un vigia que no cerro (muerto a SIGKILL por el plazo de cierre, rc != 0, sin la
            # linea cerrada=true) pudo dejar el ultimo lote sin auditar y SIN pausa propia. Se frena.
            try:
                await deps.poner_pausa(ctx, "vigia_no_cerro")
                dice("pausa_puesta_por_vigia_no_cerro")
            except Exception as exc:  # fail-soft sobre la traza; la mision ya termina como fallo vigia_no_cerro
                dice("pausa_no_puesta", tipo=type(exc).__name__)
    cadena = await deps.cadena_ok(ctx)
    pausa = await deps.leer_pausa(ctx)
    puesta = bool(pausa and pausa.get("puesta"))
    if puesta:
        dice("pausa_detectada", origen=pausa.get("origen"), motivo=pausa.get("motivo"), paso=pausa.get("paso"),
             legible=pausa.get("legible"), **({"detalle": A.detalle_conocido(pausa["detalle"])} if "detalle" in pausa else {}))
    for condicion, cod in ((auditor_pauso, "auditor_pauso"),
                           (not registro_cuadra, "registro_no_cuadra"), (not cadena, "cadena_rota"),
                           (not cerro, "vigia_no_cerro"), (not auditor_legible, AUDITOR_ILEGIBLE),
                           # AL FINAL a proposito (2026-09-20). Si el auditor pauso, si el
                           # registro no cuadra, si la cadena se rompio o si el vigia no cerro,
                           # ESE es el motivo del cero y es el que hay que leer; esto es el caso
                           # RESIDUAL: todo lo demas salio bien y aun asi no salio nada.
                           #
                           # Por que es fallo: el Ejecutor existe para producir afirmaciones
                           # RESPALDADAS; cero entregadas es cero trabajo entregado. Las misiones
                           # 445ac19c y 10707ccc salieron "completada" con cero y nadie las miro
                           # -- eso convirtio un defecto de parseo (jax#229) en un FALSO EXITO.
                           #
                           # NO distingue "el cerebro no afirmo" de "el auditor las descarto
                           # todas": las dos entregan cero. Cual fue se lee en `descartadas`,
                           # que viaja en el mismo resultado.
                           # `entrega is None` = el cerebro ni corrio (el vigia no latio, por
                           # ejemplo): ese codigo ya explica el cero y se puso mas arriba.
                           # `descartadas` vacio ademas de `respaldadas`: si el auditor RETUVO
                           # algo, el turno sigue "completado" -- el sistema hizo su trabajo y el
                           # cero SE VE en `descartadas`. Lo que esto caza es el cero INVISIBLE:
                           # nada propuesto y nada descartado.
                           (entrega is not None and not entrega.respaldadas
                            and not entrega.descartadas, "sin_afirmaciones")):
        if codigo is None and condicion:
            codigo = cod
    # Un pausa o un auditor que pausó mandan sobre un fallo de latido o de cerebro: es lo que hay que leer.
    if puesta:
        codigo = "pausa_puesta"
    if es_codigo:
        # Ruling 3 (Tarea 9): la entrega va AL FINAL y SOLO si todo lo demás pasó -- vigía que latió
        # y cerró, cerebro sin fallo, C3 (registro) que cuadra, cadena entera, C5 legible y sin
        # pausa, C4 (pausa) libre y afirmaciones entregadas. Si algo falló, no se toca GitHub: queda
        # `sin_entregar` con el código del turno como motivo.
        if codigo is None and clon is not None:
            resultado_entrega = await deps.entregar_codigo(ctx, clon, entrega, auditor_legible)
            if resultado_entrega["estado_entrega"] not in ENTREGA_SIN_FALLO:
                codigo = resultado_entrega["estado_entrega"]
        else:
            resultado_entrega = {"estado_entrega": "sin_entregar", "motivo": codigo or "codigo_sin_clon",
                                 "pr_url": None, "violaciones": [], "notas": [], "rama_empujada": False,
                                 "sha": None}
            codigo = codigo or "codigo_sin_clon"
        dice("entrega_codigo", **resultado_entrega)
    verificacion = {"registro_cuadra": registro_cuadra, "cadena_ok": cadena, "pausa_puesta": puesta,
                    "auditor_pauso": auditor_pauso, "auditor_legible": auditor_legible}
    estado = "completado" if codigo is None else "fallido"
    dice("turno_completado" if codigo is None else "turno_fallido", codigo=codigo)
    return _resultado(estado, codigo, entrega=entrega, verificacion=verificacion, sesion_iniciada=sesion_iniciada,
                      entrega_codigo=resultado_entrega)
