# tests/test_ejecutor_contratos_arranque.py
"""El arranque del Ejecutor se niega si un contrato no está vivo. Pruebas falsas,
archivos reales en tmp_path."""
import asyncio
import json
import os
import re
import time
from pathlib import Path

import pytest

from jax.ejecutor.contratos import arranque as AR
from jax.ejecutor.contratos import contexto, instalacion
from jax.ejecutor.contratos.cuenta_axioma import Cuenta
from jax.ejecutor.contratos.destinos import Host
from jax.ejecutor.contratos.fallo import Fallo

# B3/M1 (auditoría adversarial 2026-09-22): las pruebas de instalación de acá abajo
# usaban la constitución REAL de esta máquina (host-bound, Fase 0) y se saltaban en
# cualquier runner sin /home/fruiz/claude-skills -- CI, siempre. Ninguna necesita el
# CONTENIDO real: comparan "lo instalado" contra "lo que `contexto.claude_md()`
# produce ahora", y para eso alcanza un DOBLE (`_DOBLE_CONSTITUCION`, más abajo,
# inyectado vía `contexto.constitucion_fuente()` -- el seam). Ya no hace falta
# saltarlas; si algún día una prueba necesita la constitución REAL en particular
# (no es el caso de ninguna de éstas), puede volver a definir este marcador.


def _ctx(tmp_path, **cambios):
    base = dict(cuenta=Cuenta("axioma", 58291, Path("/k"), Path("/n"), tmp_path / "lib", tmp_path / "politica.json"),
                repo=Path(__file__).resolve().parents[1], puerto_canario=18436, registro=tmp_path / "registro.jsonl",
                puerto_proxy=18435, sondas=(7777,), estado_freno=tmp_path / "estado.json",
                llaves_root=Path("/etc/ssh/authorized_keys.d/axioma"), tope_gancho_s=10,
                hosts_mision=frozenset({"hall9000"}), pausa=tmp_path / "EJECUTOR_PAUSA",
                latido=tmp_path / "vigia.latido", latido_max_s=30.0,
                cron_deny=tmp_path / "cron.deny", linger_dir=tmp_path / "linger", unidad_freno=tmp_path / "ejecutor-freno.service",
                unidad_freno_habilitada=tmp_path / "wants" / "ejecutor-freno.service")
    base.update(cambios)
    return AR.Contexto(**base)


def _vivas(**rotas):
    async def ok():
        return ()

    pruebas = {n: ok for n in ("instalacion", "exportar", "c1", "c3", "c4", "c5", "c6")}
    pruebas.update(rotas)
    return pruebas


def test_todo_vivo_arranca(tmp_path):
    asyncio.run(AR.exigir_contratos(_ctx(tmp_path), _vivas()))


def test_un_contrato_muerto_no_arranca_y_se_listan_todos(tmp_path):
    async def c1():
        return (Fallo("c1", "canario_no_bloqueado"),)

    async def c5():
        return (Fallo("c5", "canario_no_disparado"),)

    with pytest.raises(AR.ContratosNoVerificados) as e:
        asyncio.run(AR.exigir_contratos(_ctx(tmp_path), _vivas(c1=c1, c5=c5)))
    assert {f.codigo for f in e.value.fallos} == {"canario_no_bloqueado", "canario_no_disparado"}


@pytest.mark.parametrize("nombre", ["instalacion", "exportar", "c1", "c3", "c4", "c5", "c6"])
def test_cada_prueba_sola_niega_el_arranque(tmp_path, nombre):
    async def rota():
        return (Fallo("x", f"roto_{nombre}"),)

    with pytest.raises(AR.ContratosNoVerificados) as e:
        asyncio.run(AR.exigir_contratos(_ctx(tmp_path), _vivas(**{nombre: rota})))
    assert [f.codigo for f in e.value.fallos] == [f"roto_{nombre}"]


def test_una_prueba_que_revienta_es_un_fallo(tmp_path):
    async def c3():
        raise OSError("sin red")

    fallos = asyncio.run(AR.verificar_contratos(_ctx(tmp_path), _vivas(c3=c3)))
    assert fallos == (Fallo("c3", "prueba_reventada", (("tipo", "OSError"),)),)


def test_faltar_una_prueba_es_un_fallo(tmp_path):
    pruebas = _vivas()
    del pruebas["c6"]
    assert asyncio.run(AR.verificar_contratos(_ctx(tmp_path), pruebas)) == (Fallo("c6", "prueba_ausente"),)


def test_corre_todas_en_orden_aunque_falle_la_primera(tmp_path):
    vistas = []

    def prueba(nombre, fallo=None):
        async def p():
            vistas.append(nombre)
            return (fallo,) if fallo else ()
        return p

    pruebas = {n: prueba(n) for n in AR._ORDEN}
    pruebas["instalacion"] = prueba("instalacion", Fallo("arranque", "instalado_distinto_del_repo"))
    asyncio.run(AR.verificar_contratos(_ctx(tmp_path), pruebas))
    assert vistas == list(AR._ORDEN)


# --- contexto ----------------------------------------------------------------

def _entorno(tmp_path):
    return {"JAX_EJECUTOR_CUENTA": "axioma", "JAX_EJECUTOR_SSH_PUERTO": "58291",
            "JAX_EJECUTOR_CONTROLADOR_LLAVE": "/k", "JAX_EJECUTOR_NODE_BIN": "/n", "JAX_EJECUTOR_LIB": "/opt/lib",
            "JAX_EJECUTOR_POLITICA": "/etc/p.json", "JAX_EJECUTOR_CANARIO_PUERTO": "18436",
            "JAX_EJECUTOR_REGISTRO": "/var/log/r.jsonl", "JAX_PROXY_CARRIL_PUERTO": "18435",
            "JAX_EJECUTOR_CERCO_SONDAS": "7777,11434", "JAX_EJECUTOR_FRENO_ESTADO": "/run/e.json",
            "JAX_EJECUTOR_LLAVES_ROOT": "/etc/ssh/authorized_keys.d/axioma", "JAX_EJECUTOR_GANCHO_TOPE_S": "10",
            "JAX_EJECUTOR_PAUSA": "/etc/jax/interruptor/EJECUTOR_PAUSA",
            "JAX_EJECUTOR_VIGIA_LATIDO": "/var/lib/v.latido", "JAX_EJECUTOR_VIGIA_LATIDO_MAX_S": "30"}


def test_contexto_desde_entorno(tmp_path):
    ctx = AR.contexto_desde_entorno(_entorno(tmp_path), ["hall9000"])
    assert (ctx.sondas, ctx.hosts_mision, ctx.pausa, ctx.latido_max_s) == (
        (7777, 11434), frozenset({"hall9000"}), Path("/etc/jax/interruptor/EJECUTOR_PAUSA"), 30.0)
    assert AR.contexto_desde_entorno(_entorno(tmp_path)).hosts_mision is None


@pytest.mark.parametrize("variable", ["JAX_EJECUTOR_PAUSA", "JAX_EJECUTOR_VIGIA_LATIDO",
                                      "JAX_EJECUTOR_VIGIA_LATIDO_MAX_S", "JAX_EJECUTOR_LLAVES_ROOT"])
def test_contexto_sin_una_variable_no_se_arma(tmp_path, variable):
    env = _entorno(tmp_path)
    del env[variable]
    with pytest.raises(Exception):
        AR.contexto_desde_entorno(env)


# --- C4 estático -------------------------------------------------------------

def _c4_sano(ctx):
    ctx.estado_freno.write_text(json.dumps({"momento": time.time(), "activo": False, "remotos_cargados": True,
                                            "uid_resuelto": True}))
    ctx.unidad_freno_habilitada.parent.mkdir(parents=True, exist_ok=True)
    ctx.unidad_freno_habilitada.write_text("")
    ctx.cron_deny.write_text("otro\naxioma\n")
    ctx.linger_dir.mkdir(exist_ok=True)


def test_c4_estatico_sano(tmp_path):
    ctx = _ctx(tmp_path)
    _c4_sano(ctx)
    assert AR.verificar_c4_estatico(ctx) == ()


@pytest.mark.parametrize("romper, codigo", [
    (lambda c: c.estado_freno.unlink(), "freno_sin_latido"),
    (lambda c: c.estado_freno.write_text(json.dumps({"momento": time.time() - 10, "activo": False,
                                                     "remotos_cargados": True, "uid_resuelto": True})), "freno_sin_latido"),
    (lambda c: c.estado_freno.write_text("{"), "freno_sin_latido"),
    (lambda c: c.estado_freno.write_text(json.dumps({"momento": time.time(), "activo": True,
                                                     "remotos_cargados": True, "uid_resuelto": True})), "interruptor_puesto"),
    (lambda c: c.estado_freno.write_text(json.dumps({"momento": time.time(), "activo": False,
                                                     "remotos_cargados": True, "uid_resuelto": False})), "freno_sin_cuenta"),
    (lambda c: c.unidad_freno_habilitada.unlink(), "freno_no_habilitado"),
    (lambda c: c.cron_deny.write_text("axiomas\n"), "cron_abierto_para_la_cuenta"),
    (lambda c: c.cron_deny.unlink(), "cron_abierto_para_la_cuenta"),
    (lambda c: (c.linger_dir / "axioma").write_text(""), "linger_activo"),
])
def test_c4_estatico_roto(tmp_path, romper, codigo):
    ctx = _ctx(tmp_path)
    _c4_sano(ctx)
    romper(ctx)
    assert codigo in [f.codigo for f in AR.verificar_c4_estatico(ctx)]


def test_c4_estatico_no_exige_remotas_de_todo_el_inventario(tmp_path):
    """Las remotas se exigen por misión (verificar_maquinas), no en todo el inventario."""
    ctx = _ctx(tmp_path)
    _c4_sano(ctx)
    ctx.estado_freno.write_text(json.dumps({"momento": time.time(), "activo": False, "remotos_cargados": False,
                                            "remotos": [], "uid_resuelto": True}))
    assert AR.verificar_c4_estatico(ctx) == ()


# --- C5 estático -------------------------------------------------------------

def test_c5_estatico_sano(tmp_path):
    assert AR.verificar_c5_estatico(_ctx(tmp_path)) == ()


def test_c5_estatico_con_pausa_puesta(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.pausa.write_text("{}")
    assert AR.verificar_c5_estatico(ctx) == (Fallo("c5", "pausa_del_ejecutor_puesta"),)


def test_c5_estatico_con_otro_vigia_latiendo(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.latido.write_text("")
    assert AR.verificar_c5_estatico(ctx) == (Fallo("c5", "vigia_ya_activo"),)
    viejo = time.time() - 60
    os.utime(ctx.latido, (viejo, viejo))
    assert AR.verificar_c5_estatico(ctx) == ()


# --- C6 estático -------------------------------------------------------------

@pytest.mark.parametrize("salida, exige_freno, codigos", [
    (b"llaves=root 644\nfreno=1\nrevocador=root 755\n", True, ()),
    (b"llaves=root 644\nfreno=0\nrevocador=root 755\n", False, ()),
    (b"llaves=axioma 600\nfreno=1\nrevocador=root 755\n", True, ("llaves_no_son_de_root",)),
    (b"llaves=root 666\nfreno=1\nrevocador=root 755\n", True, ("llaves_no_son_de_root",)),
    (b"llaves=root 644\nfreno=0\nrevocador=root 755\n", True, ("sin_llave_del_freno",)),
    (b"llaves=root 644\nfreno=1\nrevocador=\n", True, ("sin_revocador",)),
    (b"", True, ("llaves_no_son_de_root", "sin_llave_del_freno", "sin_revocador")),
    (b"", False, ("llaves_no_son_de_root", "sin_revocador")),
])
def test_leer_c6(salida, exige_freno, codigos):
    assert AR.leer_c6(salida, exige_freno=exige_freno) == codigos


def test_c6_estatico_por_maquina_local_y_remota(tmp_path):
    hosts = (Host("hall9000", "127.0.0.1", 58291, "hypervisor", True),
             Host("bridge", "192.0.2.20", 58291, "clientes", False),
             Host("atemai", "192.0.2.11", 58291, "desarrollo", False))
    vistos = []

    async def correr(c, remoto, *, entrada=b"", tope_s):
        vistos.append(remoto)
        if "192.0.2.20" in remoto:
            return 255, b"", b"Connection timed out"
        if "192.0.2.11" in remoto:
            return 0, b"llaves=root 644\nfreno=0\nrevocador=root 755\n", b""
        return 0, b"llaves=root 644\nfreno=0\nrevocador=root 755\n", b""

    fallos = asyncio.run(AR.verificar_c6_estatico(_ctx(tmp_path), hosts, correr=correr))
    assert fallos == (Fallo("c6", "maquina_inalcanzable", (("host", "bridge"),)),
                      Fallo("c6", "sin_llave_del_freno", (("host", "atemai"),)))
    assert vistos[0] == AR.remoto_c6("/etc/ssh/authorized_keys.d/axioma")
    assert vistos[1].startswith("ssh -o BatchMode=yes") and "axioma@192.0.2.20" in vistos[1]


# --- instalación ---------------------------------------------------------------

def _instalar_copia(ctx):
    for rel in instalacion.INSTALABLES:
        destino = ctx.cuenta.lib / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes((ctx.repo / rel).read_bytes())
    lib = str(ctx.cuenta.lib)
    (ctx.cuenta.lib / "gancho.sh").write_text(instalacion.renderizar_gancho(lib, ctx.tope_gancho_s))
    (ctx.cuenta.lib / "managed-settings.json").write_text(
        instalacion.renderizar_managed_settings(lib, str(ctx.cuenta.politica), ctx.tope_gancho_s))
    (ctx.cuenta.lib / "settings-usuario.json").write_text(instalacion.SETTINGS_USUARIO)
    ctx.unidad_freno.write_text(instalacion.renderizar_unidad_freno(lib))
    (ctx.cuenta.lib / "ejecutor-freno.service").write_text(instalacion.renderizar_unidad_freno(lib))


def test_instalacion_identica_al_repo(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    _instalar_contexto(ctx, monkeypatch, fuente_skills=_fuente_skills_de_prueba(tmp_path))
    assert AR.verificar_instalacion(ctx) == ()


def test_instalacion_con_un_byte_distinto(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    _instalar_contexto(ctx, monkeypatch, fuente_skills=_fuente_skills_de_prueba(tmp_path))
    with open(ctx.cuenta.lib / "jax/ejecutor/contratos/politica.py", "a") as f:
        f.write("\n")
    (ctx.cuenta.lib / "gancho.sh").unlink()
    assert AR.verificar_instalacion(ctx) == (
        Fallo("arranque", "instalado_distinto_del_repo", (("archivo", "jax/ejecutor/contratos/politica.py"),)),
        Fallo("arranque", "instalado_distinto_del_repo", (("archivo", "gancho.sh"),)),
    )


def test_instalacion_con_la_unidad_del_freno_cambiada(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    _instalar_contexto(ctx, monkeypatch, fuente_skills=_fuente_skills_de_prueba(tmp_path))
    ctx.unidad_freno.write_text(ctx.unidad_freno.read_text().replace("Restart=always", "Restart=no"))
    assert AR.verificar_instalacion(ctx) == (
        Fallo("arranque", "instalado_distinto_del_repo", (("archivo", "ejecutor-freno.service"),)),)


def test_pruebas_reales_cubren_el_orden(tmp_path):
    assert set(AR.pruebas_reales(_ctx(tmp_path))) == set(AR._ORDEN)


# --- instalación: CLAUDE.md y skills (2026-09-22) -----------------------------

def _fuente_skills_de_prueba(base: Path) -> Path:
    fuente = base / "skills-fuente"
    for nombre in contexto.skills_declaradas():
        (fuente / nombre).mkdir(parents=True)
        (fuente / nombre / "SKILL.md").write_text(f"skill de prueba: {nombre}")
    return fuente


# B3/M1 (auditoría adversarial 2026-09-22): un DOBLE hermético de la constitución, no
# la real de esta máquina -- estas pruebas comparan "lo instalado" contra "lo que
# `contexto.claude_md()` produce AHORA MISMO", y esa comparación es la misma sea cual
# sea el contenido de la fuente. Con esto, las 11 pruebas de instalación de acá abajo
# corren en CUALQUIER runner (antes se saltaban en CI, que no tiene
# /home/fruiz/claude-skills).
_DOBLE_CONSTITUCION = ("## LAS POLÍTICAS DE MARINA\n\nx\n\n## LA REGLA ABSOLUTA\n\nx\n\n"
                      "## LOS NUEVE PRINCIPIOS OPERATIVOS\n\nx\n\n## LOS SEIS IMPOSIBLES\n\nx\n\n"
                      "## JERARQUÍA DE AUTORIDAD\n\nx\n\n## HONOR\n\nx\n")


def _instalar_contexto(ctx, monkeypatch, *, fuente_skills=None):
    """Instala CLAUDE.md + skills en ctx.cuenta.lib, además de lo que ya deja
    `_instalar_copia`. Siempre contra un DOBLE de la constitución (ver arriba):
    `contexto.constitucion_fuente()` (el seam) apunta a un archivo hermético escrito
    en `ctx.cuenta.lib.parent`, no a la ruta real de esta máquina. `fuente_skills=None`
    usa la fuente real de skills declarada en cerebros.toml; pasarla apunta
    `contexto.skills_fuente()` a una fuente de prueba hermética también."""
    doble = ctx.cuenta.lib.parent / "constitucion-doble" / "CLAUDE.md.core"
    doble.parent.mkdir(parents=True, exist_ok=True)
    doble.write_text(_DOBLE_CONSTITUCION)
    monkeypatch.setattr(contexto, "constitucion_fuente", lambda: doble)
    if fuente_skills is not None:
        monkeypatch.setattr(contexto, "skills_fuente", lambda: fuente_skills)
    (ctx.cuenta.lib / contexto.CLAUDE_MD_REL).parent.mkdir(parents=True, exist_ok=True)
    (ctx.cuenta.lib / contexto.CLAUDE_MD_REL).write_bytes(contexto.claude_md())
    (ctx.cuenta.lib / contexto.CLAUDE_MD_SHA256_REL).write_text(contexto.sha256_claude_md())
    (ctx.cuenta.lib / contexto.CLAUDE_MD_HOME_VACIO_REL).write_bytes(b"")
    for rel, datos in contexto.archivos_de_skills().items():
        destino = ctx.cuenta.lib / contexto.SKILLS_REL / rel
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(datos)


def test_instalacion_con_contexto_al_dia_no_falla(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    _instalar_contexto(ctx, monkeypatch, fuente_skills=_fuente_skills_de_prueba(tmp_path))
    assert AR.verificar_instalacion(ctx) == ()


def test_claude_md_desactualizado_no_arranca(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    _instalar_contexto(ctx, monkeypatch, fuente_skills=_fuente_skills_de_prueba(tmp_path))
    (ctx.cuenta.lib / contexto.CLAUDE_MD_REL).write_bytes(b"# version vieja, escrita a mano\n")
    assert AR.verificar_instalacion(ctx) == (Fallo("arranque", "contexto_desactualizado"),)


def test_claude_md_ausente_no_arranca(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    _instalar_contexto(ctx, monkeypatch, fuente_skills=_fuente_skills_de_prueba(tmp_path))
    (ctx.cuenta.lib / contexto.CLAUDE_MD_REL).unlink()
    assert AR.verificar_instalacion(ctx) == (Fallo("arranque", "contexto_desactualizado"),)


def test_skill_faltante_en_la_fuente_no_arranca(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    fuente = tmp_path / "skills-incompleta"
    (fuente / "migrando-sin-romper").mkdir(parents=True)
    (fuente / "migrando-sin-romper" / "SKILL.md").write_text("m")
    # "desde-la-fuente" y "endureciendo" NO existen en la fuente: mismo defecto que
    # test_las_tres_skills_reales_de_claude_skills_hoy_dan_skillfaltante documenta
    # contra la fuente real (test_ejecutor_contratos_contexto.py) -- acá es a
    # propósito, con una fuente de prueba hermética.
    monkeypatch.setattr(contexto, "skills_fuente", lambda: fuente)
    doble = tmp_path / "constitucion-doble" / "CLAUDE.md.core"
    doble.parent.mkdir(parents=True, exist_ok=True)
    doble.write_text(_DOBLE_CONSTITUCION)
    monkeypatch.setattr(contexto, "constitucion_fuente", lambda: doble)
    (ctx.cuenta.lib / contexto.CLAUDE_MD_REL).parent.mkdir(parents=True, exist_ok=True)
    (ctx.cuenta.lib / contexto.CLAUDE_MD_REL).write_bytes(contexto.claude_md())
    (ctx.cuenta.lib / contexto.CLAUDE_MD_HOME_VACIO_REL).write_bytes(b"")
    assert AR.verificar_instalacion(ctx) == (
        Fallo("arranque", "skill_faltante", (("skill", "desde-la-fuente"),)),)


def test_skill_instalada_desactualizada_no_arranca(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    fuente = _fuente_skills_de_prueba(tmp_path)
    _instalar_contexto(ctx, monkeypatch, fuente_skills=fuente)
    (ctx.cuenta.lib / contexto.SKILLS_REL / "endureciendo" / "SKILL.md").write_text("vieja, a mano")
    assert AR.verificar_instalacion(ctx) == (
        Fallo("arranque", "skill_desactualizada", (("archivo", "endureciendo/SKILL.md"),)),)


def test_un_archivo_de_mas_en_skills_no_arranca(tmp_path, monkeypatch):
    """M4 (auditoría adversarial 2026-09-22): el arranque compara el CONJUNTO completo
    de archivos instalados contra lo declarado -- una skill vieja que el instalador
    dejó atrás (o cualquier archivo agregado a mano) también falla cerrado, aunque
    todos los archivos DECLARADOS estén al día."""
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    fuente = _fuente_skills_de_prueba(tmp_path)
    _instalar_contexto(ctx, monkeypatch, fuente_skills=fuente)
    (ctx.cuenta.lib / contexto.SKILLS_REL / "migrando-sin-romper" / "vieja.md").write_text("sobra")
    assert AR.verificar_instalacion(ctx) == (
        Fallo("arranque", "skill_extra_instalada", (("archivo", "migrando-sin-romper/vieja.md"),)),)


def test_una_skill_entera_de_mas_no_arranca(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    fuente = _fuente_skills_de_prueba(tmp_path)
    _instalar_contexto(ctx, monkeypatch, fuente_skills=fuente)
    (ctx.cuenta.lib / contexto.SKILLS_REL / "retirada-hace-meses").mkdir()
    (ctx.cuenta.lib / contexto.SKILLS_REL / "retirada-hace-meses" / "SKILL.md").write_text("vieja")
    assert AR.verificar_instalacion(ctx) == (
        Fallo("arranque", "skill_extra_instalada", (("archivo", "retirada-hace-meses/SKILL.md"),)),)


def test_claude_md_de_home_escrito_a_mano_no_arranca(tmp_path, monkeypatch):
    """M3: el instalable siempre tiene que ser un archivo VACÍO -- si algo lo
    sobrescribió con contenido, el arranque lo trata igual que un CLAUDE.md
    desactualizado."""
    ctx = _ctx(tmp_path)
    _instalar_copia(ctx)
    _instalar_contexto(ctx, monkeypatch, fuente_skills=_fuente_skills_de_prueba(tmp_path))
    (ctx.cuenta.lib / contexto.CLAUDE_MD_HOME_VACIO_REL).write_bytes(b"escrito a mano\n")
    assert AR.verificar_instalacion(ctx) == (
        Fallo("arranque", "contexto_desactualizado", (("archivo", contexto.CLAUDE_MD_HOME_VACIO_REL),)),)


# --- alcance: los contratos por máquina, acotados a la misión ---------------------

HOSTS = (Host("hall9000", "192.0.2.5", 58291, "hypervisor", True),
         Host("ejecutor-prueba", "192.0.2.50", 58291, "desarrollo", False),
         Host("atemai", "192.0.2.11", 58291, "desarrollo", False),
         Host("bridge", "192.0.2.20", 58291, "clientes", False))


def _latido_con_remotos(ctx, remotos, momento=None):
    ctx.estado_freno.write_text(json.dumps({"momento": time.time() if momento is None else momento,
                                            "activo": False, "remotos_cargados": False, "remotos": remotos,
                                            "uid_resuelto": True}))


def test_remotos_del_freno_solo_con_latido_fresco(tmp_path):
    ctx = _ctx(tmp_path)
    assert AR.remotos_del_freno(ctx) == frozenset()
    _latido_con_remotos(ctx, ["ejecutor-prueba"])
    assert AR.remotos_del_freno(ctx) == frozenset({"ejecutor-prueba"})
    _latido_con_remotos(ctx, ["ejecutor-prueba"], momento=time.time() - 10)
    assert AR.remotos_del_freno(ctx) == frozenset()
    ctx.estado_freno.write_text(json.dumps({"momento": time.time(), "remotos": "ejecutor-prueba"}))
    assert AR.remotos_del_freno(ctx) == frozenset()


def test_alcance_mision_solo_a_la_vm():
    a = AR.alcance(HOSTS, frozenset({"ejecutor-prueba"}), frozenset({"ejecutor-prueba"}))
    assert [h.nombre for h in a.con_c6] == ["hall9000", "ejecutor-prueba"]
    assert [h.nombre for h in a.a_cerrar] == ["atemai", "bridge"]
    assert a.fallos == ()


def test_alcance_mision_con_atemai_se_rechaza_nombrandola():
    a = AR.alcance(HOSTS, frozenset({"ejecutor-prueba"}), frozenset({"ejecutor-prueba", "atemai"}))
    assert a.fallos == (Fallo("arranque", "maquina_sin_contratos_remotos", (("host", "atemai"),)),)


def test_alcance_maquina_fuera_del_inventario_y_sin_mision():
    a = AR.alcance(HOSTS, frozenset(), frozenset({"otra"}))
    assert a.fallos == (Fallo("arranque", "maquina_fuera_del_inventario", (("host", "otra"),)),)
    assert AR.alcance(HOSTS, frozenset(), None).fallos == ()
    assert [h.nombre for h in AR.alcance(HOSTS, frozenset(), None).con_c6] == ["hall9000"]


def _correr_solo_vm_y_local(vistos):
    """hall9000 y la VM tienen C6; atemai y bridge NO (y la cuenta no las alcanza)."""
    async def correr(c, remoto, *, entrada=b"", tope_s):
        vistos.append(remoto)
        if "/dev/tcp/" in remoto:  # el cerco corta todo lo que se sondea
            ips = re.findall(r"/dev/tcp/([0-9.]+)/(\d+)", remoto)
            return 0, "".join(f"alcance={ip}:{pt} cerrada\n" for ip, pt in ips).encode(), b""
        if "192.0.2.11" in remoto or "192.0.2.20" in remoto:
            return 255, b"", b"Connection refused"
        if remoto.startswith("ssh "):
            return 0, b"llaves=root 644\nfreno=1\nrevocador=root 755\n", b""
        return 0, b"llaves=root 644\nfreno=0\nrevocador=root 755\n", b""
    return correr


def test_mision_solo_a_la_vm_arranca_sin_tocar_atemai(tmp_path):
    ctx = _ctx(tmp_path, hosts_mision=frozenset({"ejecutor-prueba"}))
    _latido_con_remotos(ctx, ["ejecutor-prueba"])
    vistos = []
    assert asyncio.run(AR.verificar_maquinas(ctx, HOSTS, correr=_correr_solo_vm_y_local(vistos))) == ()
    assert not any(v.startswith("ssh ") and ("192.0.2.11" in v or "192.0.2.20" in v) for v in vistos)
    assert any("/dev/tcp/192.0.2.11/58291" in v and "/dev/tcp/192.0.2.20/58291" in v for v in vistos)


def test_mision_a_la_vm_con_el_cerco_abierto_a_atemai_no_arranca(tmp_path):
    ctx = _ctx(tmp_path, hosts_mision=frozenset({"ejecutor-prueba"}))
    _latido_con_remotos(ctx, ["ejecutor-prueba"])
    normal = _correr_solo_vm_y_local([])

    async def correr(c, remoto, *, entrada=b"", tope_s):
        if "/dev/tcp/" in remoto:
            return 0, b"alcance=192.0.2.11:58291 abierta\nalcance=192.0.2.20:58291 cerrada\n", b""
        return await normal(c, remoto, tope_s=tope_s)

    assert asyncio.run(AR.verificar_maquinas(ctx, HOSTS, correr=correr)) == (
        Fallo("c3", "cerco_alcanza_maquina_sin_contratos", (("host", "atemai"),)),)


def test_mision_con_atemai_se_rechaza_con_el_motivo(tmp_path):
    ctx = _ctx(tmp_path, hosts_mision=frozenset({"ejecutor-prueba", "atemai"}))
    _latido_con_remotos(ctx, ["ejecutor-prueba"])
    fallos = asyncio.run(AR.verificar_maquinas(ctx, HOSTS, correr=_correr_solo_vm_y_local([])))
    assert fallos == (Fallo("arranque", "maquina_sin_contratos_remotos", (("host", "atemai"),)),)


def test_vm_sin_freno_cargado_no_arranca(tmp_path):
    ctx = _ctx(tmp_path, hosts_mision=frozenset({"ejecutor-prueba"}))
    _latido_con_remotos(ctx, [])
    fallos = asyncio.run(AR.verificar_maquinas(ctx, HOSTS, correr=_correr_solo_vm_y_local([])))
    assert fallos == (Fallo("arranque", "maquina_sin_contratos_remotos", (("host", "ejecutor-prueba"),)),)


def test_vm_con_c6_roto_no_arranca(tmp_path):
    ctx = _ctx(tmp_path, hosts_mision=frozenset({"ejecutor-prueba"}))
    _latido_con_remotos(ctx, ["ejecutor-prueba"])

    async def correr(c, remoto, *, entrada=b"", tope_s):
        return 0, b"llaves=axioma 600\nfreno=1\nrevocador=root 755\n", b""

    fallos = asyncio.run(AR.verificar_maquinas(ctx, HOSTS, correr=correr))
    assert Fallo("c6", "llaves_no_son_de_root", (("host", "ejecutor-prueba"),)) in fallos


@pytest.mark.parametrize("salida, rc, esperado", [
    (b"alcance=192.0.2.11:58291 cerrada\nalcance=192.0.2.20:58291 cerrada\n", 0, ()),
    (b"alcance=192.0.2.11:58291 abierta\nalcance=192.0.2.20:58291 cerrada\n", 0,
     (Fallo("c3", "cerco_alcanza_maquina_sin_contratos", (("host", "atemai"),)),)),
    (b"alcance=192.0.2.11:58291 cerrada\n", 0,
     (Fallo("c3", "cerco_alcanza_maquina_sin_contratos", (("host", "bridge"),)),)),
    (b"", 255, (Fallo("c3", "cuenta_inalcanzable", (("rc", 255),)),)),
])
def test_alcance_del_cerco(tmp_path, salida, rc, esperado):
    vistos = []

    async def correr(c, remoto, *, entrada=b"", tope_s):
        vistos.append(remoto)
        return rc, salida, b""

    a_cerrar = HOSTS[2:]
    assert asyncio.run(AR.verificar_alcance(_ctx(tmp_path), a_cerrar, correr=correr)) == esperado
    assert "/dev/tcp/192.0.2.11/58291" in vistos[0] and "/dev/tcp/192.0.2.20/58291" in vistos[0]


def test_alcance_sin_maquinas_que_cerrar_no_entra_a_la_cuenta(tmp_path):
    async def correr(c, remoto, *, entrada=b"", tope_s):
        raise AssertionError("no tenía que entrar")

    assert asyncio.run(AR.verificar_alcance(_ctx(tmp_path), (), correr=correr)) == ()
