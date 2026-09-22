# jax/ejecutor/contratos/arranque.py
"""El Ejecutor NO arranca si un contrato no está vivo.

Spec 2026-09-15 §4: «cada contrato existe y se vio fallar antes de que el Ejecutor
toque un servidor»; «C1 y C5 llevan una entrada canario permanente; si no se dispara,
el Ejecutor no arranca». Principio IX: el contrato antes que la capacidad.

`exigir_contratos` corre TODAS las pruebas (lista completa para quien tenga que
arreglar) y lanza ContratosNoVerificados con un solo fallo. Una prueba que revienta o
que falta es un fallo. La guardia policy/tests/test_ejecutor_lanza_solo_con_contratos.py
impide lanzar el cerebro del Ejecutor sin pasar por acá.

Dos formas (plan 6 con la enmienda del plan 4):
- `hosts_mision=None`: el arranque del Ejecutor sin misión. Verifica los seis contratos
  y que cerebro y auditor sean de proveedores distintos.
- `hosts_mision` con máquinas: antes de CADA misión. Elige QUÉ faceta audita según si esas
  máquinas cargan datos de clientes (`eleccion_c5.elegir_auditor_faceta`) y aplica la
  compuerta (`eleccion_c5.validar_eleccion`): con la compuerta cerrada y el auditor
  resuelto NO local, una misión sobre máquinas con datos de clientes NO arranca.
En las dos, la pausa del Ejecutor tiene que estar ausente y no puede haber otro vigía
latiendo (una misión por vez: el registro y la pausa son uno solo).

Los contratos POR MÁQUINA (C6 y el freno remoto de C4) se acotan a lo que la cuenta
puede alcanzar, no a todo el inventario (enmienda 2026-09-17: exigirlos en todo el
inventario impedía cualquier misión, incluso contra una VM desechable, hasta tocar los
servidores de clientes, que están reservados a Fernando):
- «habilitadas» = la local + las remotas que el freno tiene CARGADAS (`remotos` de su
  latido: la llave del freno está instalada ahí). Ésas son las ÚNICAS que el cerco de C3
  deja alcanzar a la cuenta (cerco.py las toma de la misma lista, JAX_EJECUTOR_FRENO_REMOTOS);
- C6 se verifica en TODAS las habilitadas (las alcanzables, estén o no en la misión);
- una máquina de la misión que no está habilitada se rechaza NOMBRÁNDOLA
  (`maquina_sin_contratos_remotos`), igual que una fuera del inventario;
- el resto del inventario tiene que ser INALCANZABLE para la cuenta, y se prueba en vivo
  desde la cuenta (`cerco_alcanza_maquina_sin_contratos`): el argumento viejo para exigir
  todo el inventario era que el cerco dejaba entrar a todas; ahora el cerco no las deja y
  el arranque lo comprueba en cada misión.
`remotos_cargados` del latido (todas las remotas del inventario con freno) queda como dato;
el arranque ya no lo exige.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

from jax.ejecutor.contratos import contexto, cuenta_axioma, eleccion_c5, instalacion, pausa, politica
from jax.ejecutor.contratos.cuenta_axioma import Cuenta
from jax.ejecutor.contratos.fallo import Fallo

# El freno late cada INTERVALO_DE_SONDEO (0,25 s); el instalador (ops/ejecutor/instalar_freno.sh)
# exige el mismo margen de 2 s. Tolerancia del código, no configuración de despliegue.
LATIDO_MAX_S = 2.0
_ORDEN = ("instalacion", "exportar", "c1", "c3", "c4", "c5", "c6")
_CONTRATO_DE = {"instalacion": "arranque", "exportar": "c1"}
_TOPE_C6_S = 30


class ContratosNoVerificados(RuntimeError):
    def __init__(self, fallos):
        super().__init__(len(fallos))
        self.fallos = tuple(fallos)


@dataclass(frozen=True)
class Contexto:
    cuenta: Cuenta
    repo: Path
    puerto_canario: int
    registro: Path
    puerto_proxy: int
    sondas: tuple
    estado_freno: Path
    llaves_root: Path
    tope_gancho_s: int
    hosts_mision: frozenset | None
    pausa: Path
    latido: Path
    latido_max_s: float
    cron_deny: Path = field(default=Path("/etc/cron.deny"))
    linger_dir: Path = field(default=Path("/var/lib/systemd/linger"))
    unidad_freno: Path = field(default=Path("/etc/systemd/system/ejecutor-freno.service"))
    unidad_freno_habilitada: Path = field(default=Path("/etc/systemd/system/multi-user.target.wants/ejecutor-freno.service"))


def contexto_desde_entorno(env, hosts_mision=None) -> Contexto:
    latido_max_s = float(env[pausa.VARIABLE_LATIDO_MAX_S])
    if not 0 < latido_max_s < float("inf"):
        raise ValueError("latido_max_invalido")
    return Contexto(
        cuenta=cuenta_axioma.cuenta_desde_entorno(env), repo=Path(__file__).resolve().parents[3],
        puerto_canario=int(env["JAX_EJECUTOR_CANARIO_PUERTO"]), registro=Path(env["JAX_EJECUTOR_REGISTRO"]),
        puerto_proxy=int(env["JAX_PROXY_CARRIL_PUERTO"]),
        sondas=tuple(int(p) for p in env["JAX_EJECUTOR_CERCO_SONDAS"].split(",")),
        estado_freno=Path(env["JAX_EJECUTOR_FRENO_ESTADO"]), llaves_root=Path(env["JAX_EJECUTOR_LLAVES_ROOT"]),
        tope_gancho_s=int(env["JAX_EJECUTOR_GANCHO_TOPE_S"]),
        hosts_mision=None if hosts_mision is None else frozenset(hosts_mision),
        pausa=pausa.ruta_de_la_pausa(env), latido=pausa.ruta_del_latido(env), latido_max_s=latido_max_s)


def _sha(datos: bytes) -> str:
    return hashlib.sha256(datos).hexdigest()


def verificar_instalacion(ctx: Contexto) -> tuple:
    lib = str(ctx.cuenta.lib)
    esperados = {rel: (ctx.repo / rel).read_bytes() for rel in instalacion.INSTALABLES}
    esperados["gancho.sh"] = instalacion.renderizar_gancho(lib, ctx.tope_gancho_s).encode()
    esperados["managed-settings.json"] = instalacion.renderizar_managed_settings(
        lib, str(ctx.cuenta.politica), ctx.tope_gancho_s).encode()
    esperados["settings-usuario.json"] = instalacion.SETTINGS_USUARIO.encode()
    fallos = []
    for rel, datos in esperados.items():
        try:
            igual = _sha((ctx.cuenta.lib / rel).read_bytes()) == _sha(datos)
        except OSError:
            igual = False
        if not igual:
            fallos.append(Fallo("arranque", "instalado_distinto_del_repo", (("archivo", rel),)))
    unidad = instalacion.renderizar_unidad_freno(lib)
    for nombre, ruta in (("ejecutor-freno.service", ctx.unidad_freno),
                         ("lib/ejecutor-freno.service", ctx.cuenta.lib / "ejecutor-freno.service")):
        try:
            unidad_igual = ruta.read_text(encoding="utf-8") == unidad
        except OSError:
            unidad_igual = False
        if not unidad_igual:
            fallos.append(Fallo("arranque", "instalado_distinto_del_repo", (("archivo", nombre),)))
    fallos.extend(verificar_contexto(ctx))
    return tuple(fallos)


def _archivos_instalados(base) -> frozenset:
    """Rutas relativas (posix) de TODOS los archivos bajo `base`, o `frozenset()` si
    `base` no existe todavía -- un directorio ausente no es "extra", es "nada instalado
    todavía", y ese caso ya lo cubre `skill_desactualizada`/`contexto_desactualizado`
    más arriba."""
    if not base.is_dir():
        return frozenset()
    return frozenset(p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file())


def verificar_contexto(ctx: Contexto) -> tuple:
    """El CLAUDE.md (spec §6.1: SIEMPRE generado, nunca a mano) y las skills
    declaradas, al día contra lo que `contexto.py` produce/exige AHORA MISMO.
    Mismo criterio fail-closed que el resto de `verificar_instalacion`: el sha256
    instalado se recalcula contra los bytes reales, nunca se confía en el
    `.sha256` de acompañamiento (ese archivo es sólo para auditoría humana).
    "$HOME" ya no necesita un chequeo acá (B-1/M-4, ronda 3): es un `--tmpfs`
    propio de cada invocación, ver cuenta_axioma.py.

    M4 (auditoría adversarial 2026-09-22): la comparación es del CONJUNTO completo
    de archivos, no sólo de los declarados -- un archivo de MÁS bajo `SKILLS_REL`
    (una skill vieja que el instalador debió borrar y no borró, o algo que alguien
    dejó a mano) también hace fallar el arranque. Verificar sólo "lo que se espera
    está" deja pasar "y además hay algo que no debería"."""
    fallos = []
    try:
        esperado = contexto.claude_md()
    except OSError:
        fallos.append(Fallo("arranque", "contexto_desactualizado"))
    else:
        try:
            instalado = (ctx.cuenta.lib / contexto.CLAUDE_MD_REL).read_bytes()
        except OSError:
            instalado = None
        if instalado is None or _sha(instalado) != _sha(esperado):
            fallos.append(Fallo("arranque", "contexto_desactualizado"))

    try:
        esperadas_skills = contexto.archivos_de_skills()
    except contexto.SkillFaltante as exc:
        fallos.append(Fallo("arranque", "skill_faltante", (("skill", exc.args[0]),)))
        return tuple(fallos)

    for rel, datos in esperadas_skills.items():
        try:
            instalado = (ctx.cuenta.lib / contexto.SKILLS_REL / rel).read_bytes()
        except OSError:
            instalado = None
        if instalado is None or _sha(instalado) != _sha(datos):
            fallos.append(Fallo("arranque", "skill_desactualizada", (("archivo", rel),)))

    de_mas = _archivos_instalados(ctx.cuenta.lib / contexto.SKILLS_REL) - frozenset(esperadas_skills)
    fallos.extend(Fallo("arranque", "skill_extra_instalada", (("archivo", rel),)) for rel in sorted(de_mas))
    return tuple(fallos)


def verificar_c4_estatico(ctx: Contexto, *, ahora=time.time) -> tuple:
    fallos = []
    try:
        latido = json.loads(ctx.estado_freno.read_text(encoding="utf-8"))
        fresco = ahora() - float(latido["momento"]) <= LATIDO_MAX_S
    except (OSError, ValueError, KeyError, TypeError):
        latido, fresco = {}, False
    if not fresco:
        fallos.append(Fallo("c4", "freno_sin_latido"))
    else:
        if latido.get("activo") is not False:
            fallos.append(Fallo("c4", "interruptor_puesto"))
        if latido.get("uid_resuelto") is not True:
            fallos.append(Fallo("c4", "freno_sin_cuenta"))
    if not ctx.unidad_freno_habilitada.exists():
        fallos.append(Fallo("c4", "freno_no_habilitado"))
    try:
        cerrado = ctx.cuenta.nombre in ctx.cron_deny.read_text(encoding="utf-8").split()
    except OSError:
        cerrado = False
    if not cerrado:
        fallos.append(Fallo("c4", "cron_abierto_para_la_cuenta"))
    if (ctx.linger_dir / ctx.cuenta.nombre).exists():
        fallos.append(Fallo("c4", "linger_activo"))
    return tuple(fallos)


def verificar_c5_estatico(ctx: Contexto, *, ahora: float | None = None) -> tuple:
    """La pausa del Ejecutor ausente y ningún otro vigía latiendo."""
    fallos = []
    if pausa.pausa_puesta(ctx.pausa):
        fallos.append(Fallo("c5", "pausa_del_ejecutor_puesta"))
    if pausa.latido_fresco(ctx.latido, ctx.latido_max_s, ahora=ahora):
        fallos.append(Fallo("c5", "vigia_ya_activo"))
    return tuple(fallos)


def remotos_del_freno(ctx: Contexto, *, ahora=time.time) -> frozenset:
    """Las remotas con el freno cargado, según su latido. Sin latido fresco, ninguna
    (verificar_c4_estatico ya dice `freno_sin_latido`)."""
    try:
        latido = json.loads(ctx.estado_freno.read_text(encoding="utf-8"))
        fresco = ahora() - float(latido["momento"]) <= LATIDO_MAX_S
        remotos = latido["remotos"]
    except (OSError, ValueError, KeyError, TypeError):
        return frozenset()
    if not fresco or not isinstance(remotos, list) or not all(isinstance(r, str) for r in remotos):
        return frozenset()
    return frozenset(remotos)


@dataclass(frozen=True)
class Alcance:
    con_c6: tuple    # habilitadas: la local y las remotas con freno cargado
    a_cerrar: tuple  # el resto del inventario: la cuenta no tiene que alcanzarlas
    fallos: tuple    # máquinas de la misión que no pueden entrar


def alcance(hosts, habilitadas, hosts_mision) -> Alcance:
    def habilitada(h):
        return h.es_local or h.nombre in habilitadas

    por_nombre = {h.nombre: h for h in hosts}
    fallos = []
    for nombre in sorted(hosts_mision or ()):
        h = por_nombre.get(nombre)
        if h is None:
            fallos.append(Fallo("arranque", "maquina_fuera_del_inventario", (("host", nombre),)))
        elif not habilitada(h):
            fallos.append(Fallo("arranque", "maquina_sin_contratos_remotos", (("host", nombre),)))
    return Alcance(tuple(h for h in hosts if habilitada(h)), tuple(h for h in hosts if not habilitada(h)),
                   tuple(fallos))


def remoto_alcance(hosts) -> str:
    return "; ".join(
        f'if timeout 3 bash -c "exec 3<>/dev/tcp/{shlex.quote(h.ip)}/{int(h.puerto)}" 2>/dev/null; '
        f'then echo "alcance={h.ip}:{int(h.puerto)} abierta"; else echo "alcance={h.ip}:{int(h.puerto)} cerrada"; fi'
        for h in hosts)


async def verificar_alcance(ctx: Contexto, a_cerrar, *, correr=cuenta_axioma.correr_en_la_cuenta) -> tuple:
    """Desde la cuenta: ninguna máquina sin contratos remotos es alcanzable (el cerco las corta)."""
    a_cerrar = tuple(a_cerrar)
    if not a_cerrar:
        return ()
    rc, salida, _ = await correr(ctx.cuenta, remoto_alcance(a_cerrar), tope_s=_TOPE_C6_S)
    if rc == 255:
        return (Fallo("c3", "cuenta_inalcanzable", (("rc", rc),)),)
    cerradas = set(salida.decode(errors="replace").splitlines())
    return tuple(Fallo("c3", "cerco_alcanza_maquina_sin_contratos", (("host", h.nombre),))
                 for h in a_cerrar if f"alcance={h.ip}:{int(h.puerto)} cerrada" not in cerradas)


def remoto_c6(llaves_root: str) -> str:
    q = shlex.quote(llaves_root)
    return (f'echo "llaves=$(stat -c \'%U %a\' {q} 2>/dev/null)"; '
            f'echo "freno=$(grep -c \' ejecutor-freno$\' {q} 2>/dev/null || true)"; '
            'echo "revocador=$(stat -c \'%U %a\' /usr/local/sbin/ejecutor-revocar 2>/dev/null)"; '
            'echo "freno_remoto_sha=$(sha256sum /usr/local/sbin/ejecutor-freno-remoto 2>/dev/null | cut -d\' \' -f1)"; '
            'echo "revocador_sha=$(sha256sum /usr/local/sbin/ejecutor-revocar 2>/dev/null | cut -d\' \' -f1)"')


def leer_c6(salida: bytes, *, exige_freno: bool, sha_freno_remoto: str | None = None,
           sha_revocador: str | None = None) -> tuple:
    """Códigos de fallo de una máquina. La llave del freno sólo se exige en las remotas:
    en la máquina local el freno mata por cgroup, sin ssh. Ronda 4 (M-2): además del
    dueño/modo, el CONTENIDO de `ejecutor-freno-remoto` y `ejecutor-revocar` instalados
    tiene que ser BIT A BIT el del repo -- dueño/modo correctos no dicen nada del
    contenido; `sha_*=None` (el default) no exige nada, para no romper llamadores viejos
    que todavía no pasan el sha esperado."""
    vistas = dict(linea.split("=", 1) for linea in salida.decode(errors="replace").splitlines() if "=" in linea)
    codigos = []
    if vistas.get("llaves") != "root 644":
        codigos.append("llaves_no_son_de_root")
    if exige_freno and vistas.get("freno", "0").strip() in ("", "0"):
        codigos.append("sin_llave_del_freno")
    if vistas.get("revocador") != "root 755":
        codigos.append("sin_revocador")
    if exige_freno and sha_freno_remoto is not None and vistas.get("freno_remoto_sha") != sha_freno_remoto:
        codigos.append("freno_remoto_distinto_del_repo")
    if sha_revocador is not None and vistas.get("revocador_sha") != sha_revocador:
        codigos.append("revocador_distinto_del_repo")
    return tuple(codigos)


async def verificar_c6_estatico(ctx: Contexto, hosts, *, correr=cuenta_axioma.correr_en_la_cuenta) -> tuple:
    sha_freno_remoto = _sha((ctx.repo / "ops" / "ejecutor" / "ejecutor-freno-remoto").read_bytes())
    sha_revocador = _sha((ctx.repo / "ops" / "ejecutor" / "ejecutor-revocar").read_bytes())

    async def una(h):
        remoto = remoto_c6(str(ctx.llaves_root))
        if not h.es_local:
            remoto = (f"ssh -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=yes -p {int(h.puerto)} "
                      f"{ctx.cuenta.nombre}@{shlex.quote(h.ip)} {shlex.quote(remoto)}")
        rc, salida, _ = await correr(ctx.cuenta, remoto, tope_s=_TOPE_C6_S)
        if rc != 0:
            return (Fallo("c6", "maquina_inalcanzable", (("host", h.nombre),)),)
        codigos = leer_c6(salida, exige_freno=not h.es_local, sha_freno_remoto=sha_freno_remoto,
                          sha_revocador=sha_revocador)
        return tuple(Fallo("c6", c, (("host", h.nombre),)) for c in codigos)

    resultados = await asyncio.gather(*(una(h) for h in hosts))
    return tuple(f for r in resultados for f in r)


async def verificar_maquinas(ctx: Contexto, hosts, *, correr=cuenta_axioma.correr_en_la_cuenta) -> tuple:
    """Los contratos por máquina de la misión: las de la misión, habilitadas; el resto del
    inventario, inalcanzable desde la cuenta; C6 en todas las habilitadas. Las que no están
    habilitadas NO se tocan (ni por ssh de administrador ni por la cuenta)."""
    a = alcance(hosts, await asyncio.to_thread(remotos_del_freno, ctx), ctx.hosts_mision)
    return (a.fallos + await verificar_alcance(ctx, a.a_cerrar, correr=correr)
            + await verificar_c6_estatico(ctx, a.con_c6, correr=correr))


async def eleccion_del_auditor(conn, *, hosts_mision, cfg: eleccion_c5.ConfigC5, resolve_facet) -> tuple:
    """Elige y resuelve el auditor de C5 para ESTA misión, y corre la compuerta. Punto de
    prueba directo de la parte que el spec 2026-09-18-auditor-local-opcion.md §4 exige
    dejar escrita: `auditor_es_local` sale de `eleccion_c5.es_local(conn, ...)` --
    `provider.is_local` en la DB -- NUNCA de comparar el nombre de la faceta elegida. Un
    'auditor_local' mal bindeado a un proveedor de nube no pasa gratis; sigue
    necesitando la compuerta abierta. Devuelve `(faceta_resuelta, fallos_de_eleccion)`;
    `p_c5` sólo le agrega los estáticos y el canario."""
    auditor_f, con_clientes, conocidos = await eleccion_c5.elegir_y_resolver_auditor(
        conn, cfg=cfg, hosts_mision=hosts_mision, resolve_facet=resolve_facet)
    cerebro = await resolve_facet(cfg.cerebro_faceta)
    if hosts_mision is None:
        return auditor_f, eleccion_c5.validar_proveedores(proveedor_cerebro=cerebro.provider_id,
                                                           proveedor_auditor=auditor_f.provider_id,
                                                           admite_mismo_proveedor=cfg.admite_mismo_proveedor)
    return auditor_f, eleccion_c5.validar_eleccion(
        proveedor_cerebro=cerebro.provider_id, proveedor_auditor=auditor_f.provider_id,
        auditor_es_local=await eleccion_c5.es_local(conn, auditor_f.provider_id),
        admite_datos_de_clientes=cfg.admite_datos_de_clientes, hosts_mision=frozenset(hosts_mision),
        hosts_con_clientes=con_clientes, hosts_conocidos=conocidos,
        admite_mismo_proveedor=cfg.admite_mismo_proveedor)


def pruebas_reales(ctx: Contexto) -> dict:
    from facet_resolver import resolve_facet
    from jacobs.store import conexion
    from jax.ejecutor.contratos import auditor_cliente, canario_c1, canario_c3, canario_c5, exportar

    def _politica():
        return politica.validar(json.loads(ctx.cuenta.politica.read_bytes()))

    async def p_instalacion():
        return await asyncio.to_thread(verificar_instalacion, ctx)

    async def p_exportar():
        try:
            await exportar.exportar(ctx.cuenta.politica, conexion)
        except exportar.ExportacionImposible as exc:
            return (Fallo("c1", "exportacion_imposible", (("codigo", exc.codigo),)),)
        return ()

    async def p_c1():
        return await canario_c1.verificar_c1(ctx.cuenta, puerto_canario=ctx.puerto_canario)

    async def p_c3():
        return await canario_c3.verificar_c3(ctx.cuenta, registro=ctx.registro, puerto_proxy=ctx.puerto_proxy,
                                             sondas=ctx.sondas)

    async def p_c4():
        return await asyncio.to_thread(verificar_c4_estatico, ctx)

    async def p_c5():
        estaticos = await asyncio.to_thread(verificar_c5_estatico, ctx)
        # desechable=True: el arranque corre una vez por misión; no deja la conexión en un pool
        # que puede morir con el loop (igual que exportar.py y probar_c5.py).
        async with conexion(desechable=True) as conn:
            cfg = await eleccion_c5.leer_config(conn)
            auditor_f, eleccion = await eleccion_del_auditor(
                conn, hosts_mision=ctx.hosts_mision, cfg=cfg, resolve_facet=resolve_facet)

        async def auditar(lote):
            return await auditor_cliente.auditar(lote, faceta=auditor_f, max_tokens=cfg.max_tokens)

        return tuple(estaticos) + tuple(eleccion) + await canario_c5.verificar_c5(auditar)

    async def p_c6():
        return await verificar_maquinas(ctx, (await asyncio.to_thread(_politica)).hosts)

    return {"instalacion": p_instalacion, "exportar": p_exportar, "c1": p_c1, "c3": p_c3, "c4": p_c4,
            "c5": p_c5, "c6": p_c6}


async def verificar_contratos(ctx: Contexto, pruebas: dict | None = None) -> tuple:
    pruebas = pruebas if pruebas is not None else pruebas_reales(ctx)
    fallos = []
    for nombre in _ORDEN:
        contrato = _CONTRATO_DE.get(nombre, nombre)
        prueba = pruebas.get(nombre)
        if prueba is None:
            fallos.append(Fallo(contrato, "prueba_ausente"))
            continue
        try:
            fallos.extend(await prueba())
        except Exception as exc:  # fail-soft: una prueba que revienta se reporta como Fallo y el arranque se niega (cerrado)
            fallos.append(Fallo(contrato, "prueba_reventada", (("tipo", type(exc).__name__),)))
    return tuple(fallos)


async def exigir_contratos(ctx: Contexto, pruebas: dict | None = None) -> None:
    fallos = await verificar_contratos(ctx, pruebas)
    if fallos:
        raise ContratosNoVerificados(fallos)
