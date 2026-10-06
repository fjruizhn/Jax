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
como proceso (`abrir_vigia`, más abajo, lanza `jax.ejecutor.contratos.vigia_servicio` como
SUBPROCESO DIRECTO -- no hay unidad systemd: `ejecutor-vigia@.service` se retiró el
2026-09-22, código muerto que nunca arrancó en producción, ver DEUDA.md), la jaula de la
cuenta contra el proxy de C3, el registro encadenado, el auditor de C5 y la pausa del
Ejecutor.

El vigía HEREDA la identidad de quien lanza ESTE proceso: en producción, `jax-platform`
-- y desde el 2026-09-17 (decisión de Fernando, cuenta de servicio) ESO es `jaxsvc`
(`/etc/systemd/system/jax-platform.service.d/cuenta-de-servicio.conf: User=jaxsvc`), no
`fruiz`. CORREGIDO (bug de producción jax#260, 2026-09-22): este párrafo decía lo
contrario, con un "verificado con `systemctl cat jax-platform.service`" que en realidad
sólo había mirado la unidad BASE, sin sus drop-ins -- `systemctl cat` los lista a los
tres si se lee la salida completa; medido de nuevo, `sudo cat` de cada
`.service.d/*.conf`, 2026-09-22. El vigía corre como `jaxsvc`, y `jaxsvc` no puede leer
la llave personal de `fruiz` (`~fruiz/.ssh/*`, 600) -- por eso la huella
(`vigia_servicio.py::_principal`) YA NO arma el ssh con `revocacion.argv_admin` (que
resuelve la identidad por default, la de quien invoca): usa una llave PROPIA del
servicio (`huella.argv_huella_servicio`, `JAX_EJECUTOR_HUELLA_LLAVE`, jaxsvc:jaxsvc),
autorizada en cada remota por comando forzado -- mismo patrón que ya usa C4
(`ejecutor-freno-remoto`). El CONTROLADOR nominal sigue siendo el mismo administrador
(`fruiz`, vía `JAX_EJECUTOR_ADMIN_USUARIO`); lo que cambió es la credencial, no la
cuenta. Para C5 (elección y llamada al auditor, `eleccion_c5.py`/`canario_c5.py`) NO hay
acoplamiento con esta identidad: la elección sale de la DB y la llamada al auditor es
HTTP saliente, ninguna de las dos depende de qué cuenta del sistema operativo lanzó el
proceso -- si mañana el vigía corriera bajo otra cuenta con el mismo acceso a la DB y a
la red, C5 seguiría igual.

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
from dataclasses import dataclass, replace
from pathlib import Path

import redaccion

from jax.core.cliente_http_compartido import obtener_cliente_http
from jax.ejecutor import cita
from jax.ejecutor import mision as M
from jax.ejecutor.codigo import entrega as E
from jax.ejecutor.codigo import mision_codigo as MC
from jax.ejecutor.codigo import preparar as P
from jax.ejecutor.contratos import arranque, cuenta_axioma, pausa, politica
from jax.ejecutor.contratos import auditor as A
from jax.ejecutor.contratos.registro import verificar_cadena

VARIABLE_TOPE = "JAX_EJECUTOR_TURNO_TOPE_S"
VARIABLE_ESPERA = "JAX_EJECUTOR_VIGIA_ESPERA_S"
#: Cierre del vigia (MINOR ronda 6: ya no hay unidad systemd de la que citar un TimeoutStopSec). El
#: presupuesto se DERIVA del plazo del auditor (`ejecutor.c5_tope_s`, `ConfigC5.tope_s`), no es una
#: constante: al recibir SIGTERM el vigia termina el lote en vuelo y audita el ultimo pendiente -- dos
#: llamadas que pueden durar casi `tope_s` cada una -- y luego toma la huella de cierre de cada maquina. Una constante de 200 s,
#: calculada para el viejo plazo de 120 s, mataba al vigia DENTRO de esa llamada (BLOCK-1 de la
#: auditoria del 2026-10-03): el ultimo lote quedaba sin auditar y sin pausa.
HUELLA_CIERRE_S = 30   # = vigia_servicio._TOPE_HUELLA_S, por maquina de la mision
MARGEN_CIERRE_S = 60   # borrar el latido, escribir la pausa si toca, salir del proceso


def presupuesto_cierre_s(tope_s: float, n_maquinas: int) -> float:
    """Cuanto se espera al vigia tras el SIGTERM antes de matarlo. Peor caso: al llegar el SIGTERM hay un lote
    EN VUELO (hasta `tope_s`) y, al terminar, el vigia audita ademas el ULTIMO lote pendiente (otro `tope_s`):
    dos llamadas en serie, 2 * tope_s; despues la huella de cierre de cada maquina (30 s * n) y un margen.
    Sin maquinas o con un plazo no positivo no hay presupuesto que derivar: ValueError (falla cerrado, la
    mision no abre el vigia)."""
    if not (isinstance(tope_s, (int, float)) and math.isfinite(tope_s) and tope_s > 0) or n_maquinas < 1:
        raise ValueError("presupuesto_cierre_invalido")
    return 2 * tope_s + HUELLA_CIERRE_S * n_maquinas + MARGEN_CIERRE_S


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
    # `detalle` (codigo constante, p. ej. el motivo de un auditor_ilegible) solo si el vigia lo puso:
    # sin la clave, el dict sigue siendo el de siempre.
    # Solo el detalle de una pausa de C5 con un codigo conocido de AuditorIlegible se copia (si no es conocido:
    # `detalle_invalido`); el de cualquier otro origen es `detalle_no_copiado`.
    detalle = {"detalle": A.detalle_de_pausa(texto["origen"], doc["detalle"])} if "detalle" in doc else {}
    return {"puesta": True, "legible": True, **texto, "paso": paso, **detalle}


#: Cuánto stderr del vigía se conserva. Se guarda la COLA, no la cabeza: una traza
#: aparece DESPUÉS de las líneas de INFO, así que la cabeza es el ruido y la cola el
#: diagnóstico.
TOPE_STDERR_VIGIA = 8192
#: El stdout del vigía es una línea (`cerrada=...`), pero un pipe sin drenar
#: bloquea igual: se drena con un tope holgado y también por la COLA, que es
#: donde está la línea de cierre.
TOPE_STDOUT_VIGIA = 65536


async def _drenar(flujo, tope: int) -> bytes:
    """Lee `flujo` hasta el EOF conservando los últimos `tope` bytes.

    POR QUÉ NO ALCANZA CON `stderr=PIPE` (2026-09-20). El vigía escribe INFO de httpx
    en stderr --tres líneas por lote auditado, medido-- y el pipe del sistema son
    ~64 KB. Con nadie drenando, un turno largo lo llena y el vigía se cuelga en su
    propio `write`. El síntoma sería `vigia_no_latio`: el MISMO fallo que esto viene a
    poder diagnosticar. Por eso se drena en continuo desde que el proceso arranca."""
    cola = bytearray()
    while True:
        trozo = await flujo.read(4096)
        if not trozo:
            return bytes(cola)
        cola.extend(trozo)
        if len(cola) > tope:
            del cola[:-tope]


class Vigia:
    def __init__(self, proc, ruta: Path, salida=None, error=None, *, cierre_s: float):
        self._proc, self._ruta, self._cierre_s = proc, ruta, cierre_s
        self._salida, self._error = salida, error

    def vive(self) -> bool:
        return self._proc.returncode is None

    async def cerrar(self) -> tuple:
        """(rc, stdout, stderr). El stderr ya no se tira: el 2026-09-20 una misión falló
        con `vigia_no_latio` y NO había una sola línea para investigar -- el diagnóstico
        salió corriendo el vigía a mano, que es lo que un log existe para evitar. Va
        redactado: una traza puede traer la llave.

        NO se usa `communicate()`: los dos flujos ya los drenan tareas propias desde que
        el proceso arranca, y `communicate()` intentaría leerlos otra vez ("read() called
        while another coroutine is already waiting")."""
        try:
            if self._proc.returncode is None:
                self._proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(self._proc.wait(), self._cierre_s)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
            salida, error = b"", b""
            for tarea, destino in ((self._salida, "salida"), (self._error, "error")):
                if tarea is None:
                    continue
                try:
                    trozo = await asyncio.wait_for(tarea, self._cierre_s)
                except (asyncio.TimeoutError, asyncio.CancelledError):  # fail-soft: sin un flujo se sigue; el rc manda
                    tarea.cancel()
                    trozo = b""
                if destino == "salida":
                    salida = trozo
                else:
                    error = trozo
            return (self._proc.returncode, salida.decode(errors="replace"),
                    redaccion.redactar_secretos(error.decode(errors="replace")) or "")
        finally:
            self._ruta.unlink(missing_ok=True)


async def abrir_vigia(directorio, id_vigia: str, texto: str, hosts, *, cierre_s: float, argv=None, tipo=None) -> Vigia:
    # `tipo` (Tarea 12/13, ruling del coordinador): el vigía es un PROCESO APARTE
    # (`vigia_servicio._principal`) que exige los contratos antes de que el proxy sirva -- sin
    # el tipo del turno en este archivo, "codigo" queda dormido ahí aunque `arranque.py` ya lo
    # sepa correr. `None` (misión de humo, sin turno): igual que antes.
    ruta = Path(directorio) / f"{id_vigia}.json"
    await asyncio.to_thread(ruta.write_text, json.dumps({"mision": texto, "hosts": sorted(hosts), "tipo": tipo}), "utf-8")
    argv = argv or [sys.executable, "-m", "jax.ejecutor.contratos.vigia_servicio"]
    proc = await asyncio.create_subprocess_exec(*argv, str(ruta), stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.PIPE, start_new_session=True)
    # Los drenajes arrancan YA, no al cerrar: si se esperara, el pipe se llena y el vigía
    # se cuelga. Ver el comentario largo de `_drenar`. Los DOS flujos, porque un pipe sin
    # drenar bloquea igual sea stdout o stderr.
    return Vigia(proc, ruta,
                 asyncio.create_task(_drenar(proc.stdout, TOPE_STDOUT_VIGIA)),
                 asyncio.create_task(_drenar(proc.stderr, TOPE_STDERR_VIGIA)), cierre_s=cierre_s)


def _eventos_desde(registro: Path, desde: int) -> list:
    with open(registro, "rb") as f:
        f.seek(desde)
        return [json.loads(l) for l in f.read().splitlines() if l.strip()]


# --- misión de código (spec 2026-09-28 v1.3) ------------------------------------------------

VARIABLE_TOKEN = "JAX_GITHUB_TOKEN"
API_GITHUB = "https://api.github.com"
HERRAMIENTAS_SERVIDOR = "Bash,Skill"
HERRAMIENTAS_CODIGO = "Bash,Read,Edit,Write,Glob,Grep,Skill"
#: Valores por omisión de `axioma_config` (spec §4.3: el autor es configurable; §3.3: 5 MB por
#: archivo; MINOR-3 de la auditoría: 100 MB en total). Medido 2026-09-28: ninguna de estas claves
#: existe todavía en producción -- rige el valor por omisión.
AUTOR_POR_OMISION = "Axioma (Ejecutor) <axioma@axioma-ia.io>"
TOPE_BYTES_POR_OMISION = 5 * 1024 * 1024
TOPE_TOTAL_BYTES_POR_OMISION = 100 * 1024 * 1024
CLAVES_CODIGO = ("ejecutor.codigo.autor", "ejecutor.codigo.tope_bytes", "ejecutor.codigo.tope_total_bytes")
SQL_CONFIG_CODIGO = ("SELECT config_key, config_value FROM axioma_config WHERE config_key IN "
                     f"({', '.join(['%s'] * len(CLAVES_CODIGO))})")


@dataclass(frozen=True)
class ConfigCodigo:
    autor: str
    tope_bytes: int
    tope_total_bytes: int = TOPE_TOTAL_BYTES_POR_OMISION


def _entero_positivo(filas: dict, clave: str, por_omision: int) -> int:
    texto = (filas.get(clave) or str(por_omision)).strip()
    try:
        valor = int(texto)
    except ValueError:
        raise ValueError("config_codigo_invalida", clave) from None
    if valor <= 0:
        raise ValueError("config_codigo_invalida", clave)
    return valor


def config_codigo_desde_filas(filas: dict) -> ConfigCodigo:
    """Fail-closed: una clave PRESENTE con un valor inválido es error, no el valor por omisión."""
    autor = (filas.get("ejecutor.codigo.autor") or AUTOR_POR_OMISION).strip()
    P.parsear_autor(autor)  # ValueError si no es 'Nombre <correo>'
    return ConfigCodigo(autor, _entero_positivo(filas, "ejecutor.codigo.tope_bytes", TOPE_BYTES_POR_OMISION),
                        _entero_positivo(filas, "ejecutor.codigo.tope_total_bytes", TOPE_TOTAL_BYTES_POR_OMISION))


async def leer_config_codigo() -> ConfigCodigo:
    from jacobs.store import conexion
    async with conexion(desechable=True) as conn:
        async with conn.cursor() as cur:
            await cur.execute(SQL_CONFIG_CODIGO, CLAVES_CODIGO)
            return config_codigo_desde_filas(dict(await cur.fetchall()))


@dataclass(frozen=True)
class _ClienteGithub:
    """Adaptador sobre el cliente HTTP COMPARTIDO (E-24, Política 2 de LAS CUATRO DEL
    RENDIMIENTO): `preparar.py`/`entrega.py` reciben algo con `.get`/`.post`/`.patch` que
    aceptan la ruta RELATIVA de la API (`/repos/...`) -- así los siguen probando con un
    MockTransport propio, con `base_url` fijo, sin tocar esos módulos. Este adaptador no
    construye un `httpx.AsyncClient`: le agrega a cada pedido la base, el token (nunca
    guardado en el cliente compartido) y un timeout -- el cliente compartido en sí no
    lleva ninguno de los dos."""
    token: str

    async def _pedir(self, metodo: str, ruta: str, **kwargs):
        cabeceras = {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json"}
        return await getattr(obtener_cliente_http(), metodo)(f"{API_GITHUB}{ruta}", headers=cabeceras,
                                                             timeout=30, **kwargs)

    async def get(self, ruta: str, **kwargs):
        return await self._pedir("get", ruta, **kwargs)

    async def post(self, ruta: str, **kwargs):
        return await self._pedir("post", ruta, **kwargs)

    async def patch(self, ruta: str, **kwargs):
        return await self._pedir("patch", ruta, **kwargs)


def informe_c5(entrega, faceta_auditor: str | None) -> str:
    """El cuerpo del PR (antes del pie `Hecho-por:`): quién auditó y cada afirmación respaldada
    tal como la presenta `cita.presentar` -- valores con `repr`, sin saltos de línea crudos --,
    una por línea dentro de un bloque de código SANGRADO: nada de lo que dijo el modelo puede
    salirse del bloque y convertirse en Markdown del PR."""
    def campos(a) -> str:
        p = cita.presentar(a)
        return " ".join(f"{campo}={getattr(p, campo)}" for campo in cita.CAMPOS_PRESENTACION)

    # MAJOR-2 (DC5: Fernando revisa): lo descartado también se ve, con su estado y su motivo (códigos
    # estables, sin prosa). MINOR-B: va PRIMERO -- el tope del cuerpo del PR corta la cola del
    # informe, así que lo que se pierde primero son las respaldadas, nunca lo descartado.
    lineas = [f"C5: {faceta_auditor or '?'}", "", "Descartadas por el verificador/C5", ""]
    lineas += [f"    estado={d.estado} motivo={d.motivo.codigo} " + campos(d.afirmacion)
               for d in entrega.descartadas] or ["    -"]
    lineas += ["", "Respaldadas", ""]
    lineas += ["    " + campos(a) for a in entrega.respaldadas] or ["    -"]
    return "\n".join(lineas)


def dependencias_reales(env, turno: M.Turno, *, tope_s: float, espera_s: float) -> M.Dependencias:
    es_codigo = turno.tipo == "codigo"
    # El token se lee UNA vez, aquí; nunca entra a la jaula ni a un evento (spec §4).
    token = (env.get(VARIABLE_TOKEN) or "").strip() if es_codigo else ""
    if es_codigo and not token:
        raise SinConfigurar(VARIABLE_TOKEN)
    # Estado del turno: el clon que preparó `preparar_codigo` (el cerebro trabaja ahí), la faceta
    # que auditó (va en el informe) y la configuración de código (una lectura por turno).
    estado: dict = {}

    async def config_codigo() -> ConfigCodigo:
        if "config" not in estado:
            estado["config"] = await leer_config_codigo()
        return estado["config"]

    async def contexto():
        return arranque.contexto_desde_entorno(env, turno.hosts, tipo=turno.tipo)

    async def hosts(ctx):
        doc = json.loads(await asyncio.to_thread(ctx.cuenta.politica.read_bytes))
        return politica.validar(doc).hosts

    async def tamano_registro(ctx):
        return (await asyncio.to_thread(os.stat, ctx.registro)).st_size

    async def vigia(ctx, id_vigia, texto, maquinas):
        from jacobs.store import conexion
        from jax.ejecutor.contratos import eleccion_c5
        # El presupuesto de cierre sale del MISMO plazo que usa el auditor (cfg.tope_s): si no se
        # puede leer, no se abre el vigia (falla cerrado).
        async with conexion(desechable=True) as conn:
            cfg = await eleccion_c5.leer_config(conn)
        return await abrir_vigia(env["JAX_EJECUTOR_MISIONES"], id_vigia, texto, maquinas, tipo=turno.tipo,
                                 cierre_s=presupuesto_cierre_s(cfg.tope_s, len(maquinas)))

    async def poner_pausa(ctx, motivo):
        # El vigia no cerro: el ultimo lote pudo quedar sin auditar. Lo estricto es frenar.
        await asyncio.to_thread(pausa.poner_pausa, ctx.pausa, {"origen": "mision", "motivo": motivo, "paso": None})

    async def latido(ctx):
        return await asyncio.to_thread(pausa.latido_fresco, ctx.latido, ctx.latido_max_s)

    async def cerebro(ctx, prompt, sesion, reanudar):
        # B-1/M-4 (ronda 3): el directorio de "$HOME/.claude/projects" es POR MISIÓN, no
        # por turno -- mismo directorio en todos los turnos de `turno.mision_id`, así
        # "--resume" encuentra la sesión que el turno anterior dejó. Se prepara (dueño el
        # proceso, ACL para axioma -- 2026-09-28, Task 0: `jaxsvc` no tiene sudo) ANTES de
        # cada turno: barato si ya existe (`mkdir`+`setfacl` son idempotentes) y así no
        # hace falta un paso previo separado que pueda quedar desincronizado.
        directorio_projects = cuenta_axioma.ruta_projects_de_la_mision(
            Path(env["JAX_EJECUTOR_MISIONES"]), turno.mision_id)
        de_codigo = {}
        if es_codigo:
            if "clon" not in estado:  # fail-closed: sin clon preparado, el cerebro no corre en ningún lado
                raise RuntimeError("codigo_sin_clon")
            de_codigo = {"directorio_trabajo": estado["clon"].ruta}
        await cuenta_axioma.preparar_directorio_projects(ctx.cuenta, directorio_projects)
        remoto = cuenta_axioma.remoto_claude(
            ctx.cuenta, base_url=f"http://127.0.0.1:{ctx.puerto_proxy}", modelo=env["JAX_PROXY_CARRIL_MODELO"],
            prompt=prompt, herramientas=HERRAMIENTAS_CODIGO if es_codigo else HERRAMIENTAS_SERVIDOR,
            max_salida_tokens=int(env["JAX_PROXY_CARRIL_MAX_SALIDA_TOKENS"]),
            sesion=sesion, reanudar=reanudar, directorio_projects=directorio_projects, **de_codigo)
        rc, crudo, _ = await cuenta_axioma.correr_en_la_cuenta(ctx.cuenta, remoto, entrada=b"sin-clave\n",
                                                               tope_s=tope_s)
        return rc, crudo

    async def eventos(ctx, desde):
        return await asyncio.to_thread(_eventos_desde, ctx.registro, desde)

    async def auditar(texto, entrega, maquinas):
        from facet_resolver import resolve_facet
        from jacobs.store import conexion
        from jax.ejecutor.contratos import auditor_cliente, c3_control, eleccion_c5
        # Spec 2026-09-18-auditor-local-opcion.md §4: `turno.hosts`, no `maquinas` (que ya
        # perdió el nombre plano por A.maquinas_de) -- son las mismas máquinas de la misión,
        # y elegir_y_resolver_auditor necesita nombres para consultar ejecutor_host.
        async with conexion(desechable=True) as conn:
            cfg = await eleccion_c5.leer_config(conn)
            faceta, fallos, modo = await arranque.eleccion_del_auditor(
                conn, hosts_mision=turno.hosts, cfg=cfg, resolve_facet=resolve_facet)
            auditor_local = await eleccion_c5.es_local(conn, faceta.provider_id)
        if fallos:
            raise arranque.ContratosNoVerificados(fallos)
        await c3_control.registrar_auditor_c5(
            mision_id=turno.mision_id, faceta=faceta.key, proveedor_id=faceta.provider_id,
            local=auditor_local, modo=modo, config_sha256=eleccion_c5.huella_config(cfg))
        estado["faceta_auditor"] = faceta
        if modo == "SOLO_ORDENES":
            # El vigía ya auditó cada orden desde C3. Esta revisión final antes juzgaba claims;
            # en SOLO_ORDENES no se reenvía un lote vacío o datos de claims a la nube.
            ids = frozenset(a.id for a in A.afirmaciones_auditables(entrega))
            return A.Revision(False, None, None, (), frozenset(), ids, modo, faceta.key,
                              faceta.provider_id, auditor_local)
        try:
            revision = await auditor_cliente.auditar(
                A.Lote(texto, (), A.afirmaciones_auditables(entrega), maquinas), faceta=faceta,
                max_tokens=cfg.max_tokens, tope_s=cfg.tope_s, modo=modo)
        except A.AuditorIlegible as exc:
            exc.proveedor_id = faceta.provider_id
            exc.local = auditor_local
            raise
        return replace(revision, proveedor_id=faceta.provider_id, local=auditor_local)

    async def cadena(ctx):
        return (await asyncio.to_thread(verificar_cadena, ctx.registro)).ok

    async def pausa_leida(ctx):
        return await asyncio.to_thread(leer_pausa, ctx.pausa)

    async def preparar_codigo(ctx):
        cfg = await config_codigo()
        raiz = Path(env["JAX_EJECUTOR_MISIONES"])
        base = await P.rama_guardada(raiz, turno.mision_id)
        if base is None:  # turno 1: todavía no hay preparado.json con la base guardada
            base = await P.rama_por_omision(_ClienteGithub(token), turno.repo["owner_repo"])
        clon = await P.preparar(P.Repo(turno.repo["owner_repo"], tuple(turno.repo["comandos_prueba"])),
                                mision_id=turno.mision_id, raiz=raiz,
                                rama_por_omision=base, token=token, autor=cfg.autor,
                                accesos=P.accesos_de_la_cuenta(ctx.cuenta), node_bin=ctx.cuenta.node_bin)
        estado["clon"] = clon
        return clon

    async def entregar_codigo(ctx, clon, entrega, auditor_legible):
        cfg = await config_codigo()

        async def pausa_ahora() -> bool:
            # MINOR-1: la entrega la vuelve a leer justo antes de empujar y antes del PR. `leer_pausa`
            # ya falla cerrado (ilegible = puesta).
            return bool((await asyncio.to_thread(leer_pausa, ctx.pausa))["puesta"])
        return await MC.entregar(clon, mision_id=turno.mision_id, repo=turno.repo["owner_repo"],
                                 revision_legible=auditor_legible,
                                 informe=informe_c5(entrega, estado.get("faceta_auditor")), token=token,
                                 cliente=_ClienteGithub(token), tope_bytes=cfg.tope_bytes,
                                 tope_total_bytes=cfg.tope_total_bytes, pausa_puesta=pausa_ahora,
                                 modelo=env["JAX_PROXY_CARRIL_MODELO"], autor=cfg.autor,
                                 upload_pack=E.upload_pack_por_ssh(ctx.cuenta))

    return M.Dependencias(contexto=contexto, hosts=hosts, exigir=arranque.exigir_contratos,
                          tamano_registro=tamano_registro, abrir_vigia=vigia, latido_fresco=latido,
                          correr_cerebro=cerebro, eventos_desde=eventos, auditar=auditar, cadena_ok=cadena,
                          leer_pausa=pausa_leida, espera_latido_s=espera_s, poner_pausa=poner_pausa,
                          preparar_codigo=preparar_codigo if es_codigo else None,
                          entregar_codigo=entregar_codigo if es_codigo else None)


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
