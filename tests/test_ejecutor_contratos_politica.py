# tests/test_ejecutor_contratos_politica.py
"""Política del Ejecutor (C1 prohibiciones, C2 respaldo antes de destruir).

Lo que DEBE bloquear se prueba bloqueando; lo ilegible bloquea todo."""
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from jax.ejecutor.contratos import politica as P

AHORA = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
OTRO_UID = os.getuid() + 1


def _bash(comando):
    return {"tool_name": "Bash", "tool_input": {"command": comando}}


def doc_base(**cambios):
    doc = {
        "version": 1, "generada_at": "2026-09-17T11:59:00+00:00",
        "hosts": [
            {"nombre": "hall9000", "ip": "192.0.2.5", "puerto": 58291, "rol": "hypervisor", "es_local": True},
            {"nombre": "bridge", "ip": "192.0.2.20", "puerto": 58291, "rol": "clientes", "es_local": False},
            {"nombre": "atemai", "ip": "192.0.2.11", "puerto": 58291, "rol": "desarrollo", "es_local": False},
        ],
        "reglas": [
            {"id": 1, "codigo": "canario_c1", "tipo": "prohibido", "herramientas": "Bash", "campo": "command",
             "patron": "ejecutor-canario-c1", "ambito_hosts": [], "ambito_roles": [], "es_canario": True,
             "ejemplos_coincide": [_bash("echo ejecutor-canario-c1")], "ejemplos_no_coincide": [_bash("echo x")]},
            {"id": 2, "codigo": "apt_full_upgrade_bridge", "tipo": "prohibido", "herramientas": "Bash",
             "campo": "command", "patron": r"\bapt\s+full-upgrade\b", "ambito_hosts": ["bridge"], "ambito_roles": [],
             "es_canario": False, "ejemplos_coincide": [_bash("ssh -tt axioma@bridge 'apt full-upgrade'")],
             "ejemplos_no_coincide": [_bash("ssh -tt axioma@atemai 'apt full-upgrade'")]},
            {"id": 3, "codigo": "borrar_archivos", "tipo": "destructivo", "herramientas": "Bash", "campo": "command",
             "patron": r"\brm\s", "ambito_hosts": [], "ambito_roles": [], "es_canario": False,
             "ejemplos_coincide": [_bash("rm -rf /tmp/x")], "ejemplos_no_coincide": [_bash("ls /tmp")]},
            {"id": 4, "codigo": "env_por_ruta", "tipo": "prohibido", "herramientas": "Write|Edit", "campo": "file_path",
             "patron": r"\.env$", "ambito_hosts": [], "ambito_roles": ["clientes", "hypervisor"], "es_canario": False,
             "ejemplos_coincide": [{"tool_name": "Write", "tool_input": {"file_path": "/a/.env", "content": ""}}],
             "ejemplos_no_coincide": [{"tool_name": "Read", "tool_input": {"file_path": "/a/.env"}}]},
        ],
        "respaldos": {"bridge": "2026-09-17T06:00:00+00:00"},
        "c2_edad_max_s": 86400,
    }
    doc.update(cambios)
    return P.firmar(doc)


def escribir(tmp_path, doc, modo=0o644):
    ruta = tmp_path / "politica.json"
    ruta.write_text(json.dumps(doc))
    ruta.chmod(modo)
    return ruta


def cargada(tmp_path, **cambios):
    return P.cargar(escribir(tmp_path, doc_base(**cambios)), uid_de_la_cuenta=OTRO_UID)


# --- cargar: fail-closed ------------------------------------------------------

def test_carga_una_politica_sana(tmp_path):
    p = cargada(tmp_path)
    # Las de la DB, en orden, seguidas por el NÚCLEO (LÍMITE, código -- nunca DB; ver
    # su bloque en politica.py): SIEMPRE presentes, aunque el doc no las declare.
    codigos = [r.codigo for r in p.reglas]
    assert codigos[:4] == ["canario_c1", "apt_full_upgrade_bridge", "borrar_archivos", "env_por_ruta"]
    assert set(codigos[4:]) == {r.codigo for r in P.NUCLEO_REGLAS}
    assert len(codigos[4:]) == len(P.NUCLEO_REGLAS)


def test_archivo_ausente(tmp_path):
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(tmp_path / "no-existe.json", uid_de_la_cuenta=OTRO_UID)
    assert e.value.codigo == "no_se_puede_leer"


def test_archivo_de_la_propia_cuenta(tmp_path):
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(escribir(tmp_path, doc_base()), uid_de_la_cuenta=os.getuid())
    assert e.value.codigo == "duenio_es_la_cuenta"


def test_archivo_escribible_por_el_grupo(tmp_path):
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(escribir(tmp_path, doc_base(), modo=0o664), uid_de_la_cuenta=OTRO_UID)
    assert e.value.codigo == "escribible_por_otros"


def test_un_byte_cambiado_rompe_el_sha256(tmp_path):
    doc = doc_base()
    doc["c2_edad_max_s"] = 999999
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(escribir(tmp_path, doc), uid_de_la_cuenta=OTRO_UID)
    assert e.value.codigo == "sha256_no_cuadra"


@pytest.mark.parametrize("contenido, codigo", [(b"{no json", "json_invalido"), (b"[]", "json_invalido")])
def test_json_roto(tmp_path, contenido, codigo):
    ruta = tmp_path / "politica.json"
    ruta.write_bytes(contenido)
    ruta.chmod(0o644)  # con umask 002 nacería escribible por el grupo y caería antes, por otra causa
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(ruta, uid_de_la_cuenta=OTRO_UID)
    assert e.value.codigo == codigo


def _sin(doc, codigo):
    return [r for r in doc["reglas"] if r["codigo"] != codigo]


@pytest.mark.parametrize("mutar, codigo", [
    (lambda d: d.update(version=2), "version_desconocida"),
    (lambda d: d.update(reglas=_sin(d, "canario_c1")), "canario_ausente_o_multiple"),
    (lambda d: d["reglas"][1].update(es_canario=True), "canario_ausente_o_multiple"),
    (lambda d: d["reglas"][2].update(ejemplos_coincide=[]), "regla_sin_ejemplo_que_coincida"),
    (lambda d: d["reglas"][2].update(patron="(sin cerrar"), "regex_invalida"),
    (lambda d: d["reglas"][1].update(ambito_hosts=["no-esta"]), "ambito_desconocido"),
    (lambda d: d["reglas"][1].update(codigo="canario_c1"), "regla_duplicada"),
    (lambda d: d["hosts"][1].update(es_local=True), "inventario_sin_una_local"),
    (lambda d: d.update(c2_edad_max_s=0), "campo_invalido"),
    (lambda d: d.update(respaldos={"bridge": "2026-09-17T06:00:00"}), "respaldo_invalido"),
    (lambda d: d.update(respaldos={"no-esta": "2026-09-17T06:00:00+00:00"}), "respaldo_invalido"),
])
def test_cualquier_parte_ilegible_invalida_todo(tmp_path, mutar, codigo):
    doc = {k: v for k, v in doc_base().items() if k != "sha256"}
    doc = json.loads(json.dumps(doc))
    mutar(doc)
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(escribir(tmp_path, P.firmar(doc)), uid_de_la_cuenta=OTRO_UID)
    assert e.value.codigo == codigo


# --- evaluar ------------------------------------------------------------------

def test_canario_se_bloquea(tmp_path):
    d = P.evaluar(cargada(tmp_path), "Bash", {"command": "touch x # ejecutor-canario-c1"}, AHORA)
    assert (d.permitir, d.codigo, d.regla, d.hosts) == (False, P.PROHIBIDO, "canario_c1", ("hall9000",))


def test_prohibido_con_ambito_de_host(tmp_path):
    p = cargada(tmp_path)
    assert P.evaluar(p, "Bash", {"command": "ssh -tt axioma@bridge 'apt full-upgrade'"}, AHORA).codigo == P.PROHIBIDO
    assert P.evaluar(p, "Bash", {"command": "ssh -tt axioma@atemai 'apt full-upgrade'"}, AHORA).permitir is True


def test_prohibido_por_ruta_con_ambito_de_rol(tmp_path):
    p = cargada(tmp_path)
    assert P.evaluar(p, "Write", {"file_path": "/www/.env", "content": ""}, AHORA).regla == "env_por_ruta"
    assert P.evaluar(p, "Read", {"file_path": "/www/.env"}, AHORA).permitir is True


def test_destructivo_sin_respaldo_se_bloquea_por_maquina(tmp_path):
    p = cargada(tmp_path)
    local = P.evaluar(p, "Bash", {"command": "rm -rf /tmp/x"}, AHORA)
    assert (local.permitir, local.codigo, local.hosts) == (False, P.DESTRUCTIVO_SIN_RESPALDO, ("hall9000",))
    remoto = P.evaluar(p, "Bash", {"command": "ssh -tt axioma@bridge 'rm -rf /tmp/x'"}, AHORA)
    assert remoto.permitir is True


def test_respaldo_viejo_o_del_futuro_no_cuenta(tmp_path):
    comando = {"command": "ssh -tt axioma@bridge 'rm -rf /tmp/x'"}
    viejo = cargada(tmp_path, respaldos={"bridge": (AHORA - timedelta(days=2)).isoformat()})
    assert P.evaluar(viejo, "Bash", comando, AHORA).codigo == P.DESTRUCTIVO_SIN_RESPALDO
    futuro = cargada(tmp_path, respaldos={"bridge": (AHORA + timedelta(minutes=1)).isoformat()})
    assert P.evaluar(futuro, "Bash", comando, AHORA).codigo == P.DESTRUCTIVO_SIN_RESPALDO


@pytest.mark.parametrize("tool_name, tool_input, codigo", [
    ("Bash", {"command": "ssh -tt axioma@192.0.2.99 uptime"}, P.HOST_DESCONOCIDO),
    ("Bash", {"command": "echo 'sin cerrar"}, P.COMANDO_ILEGIBLE),
    ("Bash", {}, P.ENTRADA_ILEGIBLE),
    ("Bash", "no-es-dict", P.ENTRADA_ILEGIBLE),
    (None, {"command": "uptime"}, P.ENTRADA_ILEGIBLE),
])
def test_lo_que_no_se_entiende_se_bloquea(tmp_path, tool_name, tool_input, codigo):
    d = P.evaluar(cargada(tmp_path), tool_name, tool_input, AHORA)
    assert (d.permitir, d.codigo) == (False, codigo)


def test_lo_inocuo_pasa(tmp_path):
    assert P.evaluar(cargada(tmp_path), "Bash", {"command": "uptime"}, AHORA) == \
        P.Decision(True, P.PERMITIDO, None, ("hall9000",))


# --- NÚCLEO: la propia protección del Ejecutor (LÍMITE, auditoría 2026-09-22) --------
#
# Con root en una máquina, axioma PODRÍA anular C3/C4/C6 ahí -- eso no lo cierra
# ninguna regla. Lo que SÍ se puede probar es que tocar sus propios controles queda
# PROHIBIDO aunque la política de la DB no diga una palabra sobre eso (doc_base() no
# declara ninguna regla "nucleo_*"): el núcleo se agrega SIEMPRE, en `validar()`.

_SSH = "ssh -tt -p 58291 axioma@atemai "
NUCLEO_CASOS = {
    _SSH + "'rm -f /etc/sudoers.d/50-ejecutor-axioma-registro'": "nucleo_sudoers_ejecutor",
    _SSH + "'truncate -s0 /etc/ssh/authorized_keys.d/axioma'": "nucleo_authorized_keys",
    _SSH + "'rm -f /usr/local/sbin/ejecutor-freno-remoto'": "nucleo_freno_remoto",
    _SSH + "'chmod 000 /usr/local/sbin/ejecutor-revocar'": "nucleo_revocar",
    _SSH + "'rm -f /var/log/sudo-axioma.log'": "nucleo_sudo_log",
    _SSH + "'rm -rf /var/log/sudo-io/axioma'": "nucleo_sudo_log",
    _SSH + "'chattr -a /opt/ejecutor/lib/registro.jsonl'": "nucleo_chattr",
    _SSH + "'crontab -e'": "nucleo_cron_editar",
    _SSH + "\"echo '* * * * * root touch /tmp/x' > /etc/cron.d/persistencia\"": "nucleo_cron_archivo",
    _SSH + "'systemctl enable mio.service'": "nucleo_systemd_habilitar",
    _SSH + "'rm -f /etc/systemd/system/mio.service'": "nucleo_systemd_archivo",
}


@pytest.mark.parametrize("comando, codigo_esperado", list(NUCLEO_CASOS.items()))
def test_nucleo_protege_sus_controles_aunque_la_db_no_diga_nada(tmp_path, comando, codigo_esperado):
    """`doc_base()` no declara NINGUNA regla `nucleo_*`: si esto pasa, es porque
    `validar()` las agrega solo, no porque la política de prueba las trajera."""
    assert not any(r["codigo"].startswith("nucleo_") for r in doc_base()["reglas"])
    p = cargada(tmp_path)
    d = P.evaluar(p, "Bash", {"command": comando}, AHORA)
    assert (d.permitir, d.codigo, d.regla) == (False, P.PROHIBIDO, codigo_esperado)


NUCLEO_CONTROLES = {
    _SSH + "'cat /etc/sudoers.d/50-ejecutor-axioma-registro'",   # leer, no tocar
    _SSH + "'cat /etc/ssh/authorized_keys.d/axioma'",
    _SSH + "'ls -la /usr/local/sbin/ejecutor-freno-remoto'",
    _SSH + "'stat /usr/local/sbin/ejecutor-revocar'",
    _SSH + "'cat /var/log/sudo-axioma.log'",
    _SSH + "'lsattr /opt/ejecutor/lib/registro.jsonl'",
    _SSH + "'crontab -l'",
    _SSH + "'systemctl status ejecutor-freno.service'",
    _SSH + "'cat /etc/systemd/system/ejecutor-freno.service'",
    _SSH + "'uptime'",
}


@pytest.mark.parametrize("comando", sorted(NUCLEO_CONTROLES))
def test_nucleo_no_bloquea_lectura_ni_lo_inocuo(tmp_path, comando):
    """El núcleo protege contra TOCAR sus propios controles, no contra mirarlos: un
    diagnóstico legítimo (¿existe el revocador? ¿qué dice el log?) tiene que seguir
    pasando, o el núcleo se vuelve una regla que bloquea trabajo real."""
    d = P.evaluar(cargada(tmp_path), "Bash", {"command": comando}, AHORA)
    assert d.permitir is True, (comando, d)


def test_nucleo_es_prohibido_y_gana_sobre_destructivo(tmp_path):
    """`prohibido` siempre gana sobre `destructivo_sin_respaldo` (evaluar(), orden ya
    establecido) -- el núcleo es `tipo="prohibido"`, así que ni con un respaldo
    vigente de esa máquina se puede tocar un control propio."""
    p = cargada(tmp_path, respaldos={"atemai": AHORA.isoformat()})
    d = P.evaluar(p, "Bash", {"command": _SSH + "'rm -f /usr/local/sbin/ejecutor-revocar'"}, AHORA)
    assert (d.permitir, d.codigo) == (False, P.PROHIBIDO)


def test_nucleo_no_se_puede_apagar_declarandolo_en_la_db(tmp_path):
    """Ni siquiera una regla `nucleo_*` DECLARADA en la DB (con un patrón manso, "que
    nunca coincide con nada") lo desactiva -- `validar()` rechaza el choque de código
    en vez de dejar que la última declaración gane."""
    doc = {k: v for k, v in doc_base().items() if k != "sha256"}
    doc["reglas"].append({
        "id": 999, "codigo": "nucleo_chattr", "tipo": "destructivo", "herramientas": "Bash", "campo": "command",
        "patron": "esto-no-coincide-nunca-con-nada-real", "ambito_hosts": [], "ambito_roles": [], "es_canario": False,
        "ejemplos_coincide": [_bash("esto-no-coincide-nunca-con-nada-real")], "ejemplos_no_coincide": [],
    })
    with pytest.raises(P.PoliticaIlegible) as e:
        P.cargar(escribir(tmp_path, P.firmar(doc)), uid_de_la_cuenta=OTRO_UID)
    assert e.value.codigo == "regla_duplicada"


# --- autoprueba ---------------------------------------------------------------

def test_autoprueba_sana_no_devuelve_fallos(tmp_path):
    assert P.autoprueba(cargada(tmp_path)) == ()


def test_autoprueba_ve_un_ejemplo_que_miente(tmp_path):
    doc = {k: v for k, v in doc_base().items() if k != "sha256"}
    doc["reglas"][2]["ejemplos_coincide"] = [_bash("ls /tmp")]
    p = P.cargar(escribir(tmp_path, P.firmar(doc)), uid_de_la_cuenta=OTRO_UID)
    assert P.autoprueba(p) == (P.FalloDeEjemplo("borrar_archivos", 0, "coincide", ()),)


def test_autoprueba_cuenta_un_error_de_destino_como_fallo(tmp_path):
    doc = {k: v for k, v in doc_base().items() if k != "sha256"}
    doc["reglas"][2]["ejemplos_no_coincide"] = [_bash("ssh -tt axioma@192.0.2.99 ls")]
    p = P.cargar(escribir(tmp_path, P.firmar(doc)), uid_de_la_cuenta=OTRO_UID)
    assert P.autoprueba(p) == (P.FalloDeEjemplo("borrar_archivos", 0, "no_coincide", (P.HOST_DESCONOCIDO,)),)
