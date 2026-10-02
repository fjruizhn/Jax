"""El Faro (auditoria de 0.3bc, ABIERTO del uid de la jaula): ACL por ejecucion, puesta por `faro` sin root.

Un bwrap que corre con el uid de la jaula tiene que poder ABRIR las dos fuentes de sus binds dentro del
directorio de sockets (0700 de `faro`) sin poder LISTARLO ni tocar lo de otra ejecucion. Se resuelve con ACL
POSIX con nombre, que puede poner el dueño del archivo:

    directorio  0700 + user:<uid>:--x         (se puede atravesar, no listar)
    socket      0600 + user:<uid>:rw-
    token       0400 + user:<uid>:r--

Se quitan al cerrar la ejecucion. El Puerto (`ServidorPuerto`) solo acepta un directorio con entradas ACL de
usuario con nombre si son de uids de jaulas VIVAS. Via de implementacion: el xattr `system.posix_acl_access`
con `os.setxattr` (sin binario `setfacl` ni libacl); aqui se contrasta contra `getfacl` cuando existe.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import stat
import subprocess

import pytest

from jax.faro import acl
from jax.faro.config import ConfigFaroInvalida, ConfigPuerto
from jax.faro.transporte import ServidorPuerto
from jax.faro.bitacora import Bitacora
from tests._faro_utils import corre, ejecucion, paquete_listo

JAULA, OTRA = 50001, 50002


def _modo(p):
    return stat.S_IMODE(os.lstat(p).st_mode)


def _getfacl(p) -> str:
    """Salida de `getfacl` si existe (contraste independiente de nuestro codificador); '' si no esta."""
    if shutil.which("getfacl") is None:
        return ""
    return subprocess.run(["getfacl", "-cn", str(p)], capture_output=True, text=True).stdout


@pytest.fixture
def d(tmp_path):
    x = tmp_path / "dir"
    x.mkdir(mode=0o700)
    return x


@pytest.fixture
def f(tmp_path):
    x = tmp_path / "token"
    x.write_text("t")
    x.chmod(0o400)
    return x


# --------------------------------------------------------------------------- #
# el codificador                                                              #
# --------------------------------------------------------------------------- #

def test_conceder_pone_la_entrada_con_nombre_y_la_mascara_aparece_en_los_bits_de_grupo(d):
    acl.conceder(d, JAULA, acl.X)
    assert acl.usuarios(d) == {JAULA: acl.X} and _modo(d) == 0o710
    salida = _getfacl(d)
    if salida:
        assert f"user:{JAULA}:--x" in salida and "mask::--x" in salida and "group::---" in salida and "other::---" in salida


def test_el_token_y_el_socket_quedan_con_su_entrada_rw_y_r(f, tmp_path):
    s = tmp_path / "sock"
    s.write_text("")
    s.chmod(0o600)
    acl.conceder(f, JAULA, acl.R)
    acl.conceder(s, JAULA, acl.R | acl.W)
    assert (acl.usuarios(f), _modo(f)) == ({JAULA: 4}, 0o440)
    assert (acl.usuarios(s), _modo(s)) == ({JAULA: 6}, 0o660)
    if _getfacl(f):
        assert f"user:{JAULA}:r--" in _getfacl(f) and f"user:{JAULA}:rw-" in _getfacl(s)


def test_retirar_devuelve_el_modo_original_y_no_deja_acl(d):
    acl.conceder(d, JAULA, acl.X)
    acl.conceder(d, OTRA, acl.X)
    acl.retirar(d, JAULA)
    assert acl.usuarios(d) == {OTRA: 1} and _modo(d) == 0o710
    acl.retirar(d, OTRA)
    assert acl.usuarios(d) == {} and _modo(d) == 0o700
    if _getfacl(d):
        assert "user:" not in _getfacl(d).replace("user::", "")
    acl.retirar(d, OTRA)                                    # retirar lo que no esta no falla


def test_conceder_dos_veces_reemplaza_y_no_duplica(d):
    acl.conceder(d, JAULA, acl.X)
    acl.conceder(d, JAULA, acl.R | acl.X)
    assert acl.usuarios(d) == {JAULA: 5}


@pytest.mark.parametrize("uid", [-1, 2 ** 32 - 1, 2 ** 40, True, "5", None, 1.5])
def test_un_uid_raro_no_se_acepta(d, uid):
    with pytest.raises(ValueError):
        acl.conceder(d, uid, acl.X)


@pytest.mark.parametrize("perm", [0, 8, -1, True, "r"])
def test_un_permiso_raro_no_se_acepta(d, perm):
    with pytest.raises(ValueError):
        acl.conceder(d, JAULA, perm)


def test_un_sistema_de_archivos_sin_acl_falla_cerrado(d, monkeypatch):
    def sin_soporte(*a, **k):
        raise OSError(95, "Operation not supported")
    monkeypatch.setattr(os, "setxattr", sin_soporte)
    with pytest.raises(ConfigFaroInvalida, match="ACL"):
        acl.conceder(d, JAULA, acl.X)


def test_no_sigue_enlaces(tmp_path, d):
    enlace = tmp_path / "enlace"
    enlace.symlink_to(d)
    with pytest.raises((ConfigFaroInvalida, OSError)):
        acl.conceder(enlace, JAULA, acl.X)
    assert acl.usuarios(d) == {}


# --------------------------------------------------------------------------- #
# la validacion del directorio                                                #
# --------------------------------------------------------------------------- #

def test_un_directorio_0700_sin_acl_es_privado(d):
    acl.validar_privado(d, uids_permitidos=())


def test_un_directorio_con_entrada_de_un_uid_vivo_es_privado_y_con_la_de_uno_ajeno_no(d):
    acl.conceder(d, JAULA, acl.X)
    acl.validar_privado(d, uids_permitidos={JAULA})
    for permitidos in ((), {OTRA}):
        with pytest.raises(ConfigFaroInvalida, match="privado|ACL"):
            acl.validar_privado(d, uids_permitidos=permitidos)


def test_la_entrada_de_un_uid_vivo_con_mas_que_x_no_es_privada(d):
    acl.conceder(d, JAULA, acl.R | acl.X)                  # podria LISTAR
    with pytest.raises(ConfigFaroInvalida, match="privado|ACL"):
        acl.validar_privado(d, uids_permitidos={JAULA})


def test_un_grupo_con_nombre_no_es_privado(d):
    acl._escribir(d, [*acl._leer_o_modo(d), (acl.TAG_GROUP, acl.X, 12345)])     # un grupo con nombre, puesto a mano
    with pytest.raises(ConfigFaroInvalida, match="privado|ACL|grupo"):
        acl.validar_privado(d, uids_permitidos={JAULA})


@pytest.mark.parametrize("modo", [0o770, 0o750, 0o707, 0o701, 0o777])
def test_los_bits_de_grupo_u_otros_sin_acl_que_los_explique_no_son_privados(d, modo):
    d.chmod(modo)
    with pytest.raises(ConfigFaroInvalida, match="privado|grupo"):
        acl.validar_privado(d, uids_permitidos={JAULA})


def test_otros_con_acl_tampoco(d):
    acl.conceder(d, JAULA, acl.X)
    d.chmod(0o711)                                          # chmod da otros x; la entrada con nombre sigue
    with pytest.raises(ConfigFaroInvalida, match="privado|otros"):
        acl.validar_privado(d, uids_permitidos={JAULA})


# --------------------------------------------------------------------------- #
# integracion con el Puerto                                                   #
# --------------------------------------------------------------------------- #

@pytest.fixture
def mundo(tmp_path):
    _c, cargado = paquete_listo(tmp_path)
    r = tmp_path / "run"
    r.mkdir(mode=0o700)
    return ConfigPuerto(socket_dir=r), cargado


def _puerto(cfg, cargado, **kw):
    return ServidorPuerto(cfg, ejecucion(**kw), cargado, Bitacora(emisores=[]))


def test_el_puerto_da_a_la_jaula_su_acl_y_la_quita_al_cerrar(mundo):
    cfg, cargado = mundo

    async def caso():
        async with _puerto(cfg, cargado, run_id="run-a", uid_esperado=JAULA) as srv:
            vivo = (acl.usuarios(cfg.socket_dir), acl.usuarios(srv.ruta_socket), acl.usuarios(srv.ruta_token),
                    _modo(cfg.socket_dir), _modo(srv.ruta_socket), _modo(srv.ruta_token))
        return vivo, acl.usuarios(cfg.socket_dir), _modo(cfg.socket_dir)
    vivo, despues, modo_despues = corre(caso())
    assert vivo == ({JAULA: 1}, {JAULA: 6}, {JAULA: 4}, 0o710, 0o660, 0o440)
    assert despues == {} and modo_despues == 0o700


def test_con_el_uid_del_servicio_no_se_pone_ninguna_acl(mundo):
    cfg, cargado = mundo

    async def caso():
        async with ServidorPuerto(cfg, ejecucion(uid_esperado=os.getuid()), cargado, Bitacora(emisores=[]),
                                  solo_pruebas_mismo_uid=True) as srv:
            return acl.usuarios(cfg.socket_dir), _modo(srv.ruta_socket), _modo(srv.ruta_token)
    assert corre(caso()) == ({}, 0o600, 0o400)


def test_cada_jaula_tiene_solo_lo_suyo_y_dos_ejecuciones_conviven_en_el_mismo_directorio(mundo):
    cfg, cargado = mundo

    async def caso():
        async with _puerto(cfg, cargado, run_id="run-a", uid_esperado=JAULA) as a, \
                _puerto(cfg, cargado, run_id="run-b", uid_esperado=OTRA) as b:
            return (acl.usuarios(cfg.socket_dir), acl.usuarios(a.ruta_token), acl.usuarios(b.ruta_token),
                    acl.usuarios(a.ruta_socket), acl.usuarios(b.ruta_socket))
    assert corre(caso()) == ({JAULA: 1, OTRA: 1}, {JAULA: 4}, {OTRA: 4}, {JAULA: 6}, {OTRA: 6})


def test_cerrar_una_ejecucion_no_quita_la_entrada_del_directorio_a_otra_con_el_mismo_uid(mundo):
    cfg, cargado = mundo

    async def caso():
        a = _puerto(cfg, cargado, run_id="run-a", uid_esperado=JAULA)
        b = _puerto(cfg, cargado, run_id="run-b", uid_esperado=JAULA)
        await a.__aenter__()
        await b.__aenter__()
        await a.__aexit__(None, None, None)
        tras_a = acl.usuarios(cfg.socket_dir)
        await b.__aexit__(None, None, None)
        return tras_a, acl.usuarios(cfg.socket_dir)
    assert corre(caso()) == ({JAULA: 1}, {})


def test_una_entrada_ajena_en_el_directorio_impide_abrir_otro_puerto(mundo):
    cfg, cargado = mundo
    acl.conceder(cfg.socket_dir, 777, acl.X)                # alguien dio paso a un uid que no es jaula viva

    async def caso():
        with pytest.raises(ConfigFaroInvalida, match="privado|ACL"):
            async with _puerto(cfg, cargado, uid_esperado=JAULA):
                pass
    corre(caso())


def test_si_no_se_puede_poner_la_acl_el_puerto_no_abre_y_no_deja_archivos(mundo, monkeypatch):
    cfg, cargado = mundo

    def sin_soporte(*a, **k):
        raise OSError(95, "Operation not supported")
    monkeypatch.setattr(os, "setxattr", sin_soporte)

    async def caso():
        with pytest.raises(ConfigFaroInvalida, match="ACL"):
            async with _puerto(cfg, cargado, uid_esperado=JAULA):
                pass
    corre(caso())
    assert list(cfg.socket_dir.iterdir()) == [] and _modo(cfg.socket_dir) == 0o700


def test_el_grupo_propietario_con_permisos_dentro_de_una_acl_no_es_privado(d):
    acl.conceder(d, JAULA, acl.X)
    entradas = [(t, (acl.X if t == acl.TAG_GROUP_OBJ else p), i) for t, p, i in acl._leer_o_modo(d)]
    acl._escribir(d, entradas)                                    # group::--x : cualquier proceso del grupo de faro pasaria
    with pytest.raises(ConfigFaroInvalida, match="privado|ACL"):
        acl.validar_privado(d, uids_permitidos={JAULA})


def test_la_mascara_o_el_permiso_de_un_uid_vivo_por_encima_de_x_se_rechazan_por_separado(d):
    acl.conceder(d, JAULA, acl.R | acl.X)
    with pytest.raises(ConfigFaroInvalida, match="privado|ACL"):
        acl.validar_privado(d, uids_permitidos={JAULA})
    # solo la entrada (con una mascara que no la deja ver): el limite es de la entrada, no solo de la mascara
    entradas = [(t, p, i) for t, p, i in acl._leer_o_modo(d) if t != acl.TAG_MASK]
    acl._escribir(d, entradas)
    assert acl.usuarios(d) == {JAULA: acl.R | acl.X}


# --------------------------------------------------------------------------- #
# reauditoria R-1: la ACL POR DEFECTO (herencia)                              #
# --------------------------------------------------------------------------- #

def _acl_por_defecto(d, uid=33, perm=7):
    """`setfacl -d -m u:<uid>:<perm>`: lo que un archivo nuevo del directorio hereda."""
    entradas = [(acl.TAG_USER_OBJ, 7, 0xFFFFFFFF), (acl.TAG_USER, perm, uid), (acl.TAG_GROUP_OBJ, 0, 0xFFFFFFFF),
                (acl.TAG_MASK, perm, 0xFFFFFFFF), (acl.TAG_OTHER, 0, 0xFFFFFFFF)]
    os.setxattr(d, "system.posix_acl_default", acl._codificar(entradas), follow_symlinks=False)


def test_r1_un_directorio_con_acl_por_defecto_no_es_privado(d):
    _acl_por_defecto(d)
    with pytest.raises(ConfigFaroInvalida, match="defecto|herenc"):
        acl.validar_privado(d, uids_permitidos={33})


def test_r1_tampoco_lo_es_con_una_entrada_valida_mas_la_acl_por_defecto(d):
    acl.conceder(d, JAULA, acl.X)
    _acl_por_defecto(d, uid=33, perm=0)                         # incluso una heredada con efecto ---
    with pytest.raises(ConfigFaroInvalida, match="defecto|herenc"):
        acl.validar_privado(d, uids_permitidos={JAULA, 33})


def test_r1_el_puerto_no_abre_en_un_directorio_con_acl_por_defecto(mundo):
    cfg, cargado = mundo
    _acl_por_defecto(cfg.socket_dir)

    async def caso():
        with pytest.raises(ConfigFaroInvalida, match="defecto|herenc"):
            async with _puerto(cfg, cargado, uid_esperado=JAULA):
                pass
    corre(caso())
    assert list(cfg.socket_dir.iterdir()) == []


def test_r1_conceder_exclusivo_sobre_un_archivo_con_entradas_ajenas_falla_cerrado_y_no_lo_toca(d):
    _acl_por_defecto(d, uid=33, perm=7)
    token = d / "heredado.token"
    token.write_text("t")                                       # hereda user:33:rwx del directorio
    token.chmod(0o400)
    antes = acl.usuarios(token)
    assert 33 in antes                                          # la herencia es real en este sistema de archivos
    with pytest.raises(ConfigFaroInvalida, match="ajena|nombre"):
        acl.conceder(token, JAULA, acl.R, exclusivo=True)
    assert acl.usuarios(token) == antes


def test_r1_conceder_exclusivo_acepta_su_propia_entrada_y_ninguna_otra(f):
    acl.conceder(f, JAULA, acl.R, exclusivo=True)
    acl.conceder(f, JAULA, acl.R | acl.W, exclusivo=True)       # reemplazar la suya vale
    assert acl.usuarios(f) == {JAULA: 6}
    with pytest.raises(ConfigFaroInvalida):
        acl.conceder(f, OTRA, acl.R, exclusivo=True)


def test_r1_el_directorio_si_admite_las_entradas_de_otras_jaulas(d):
    acl.conceder(d, JAULA, acl.X)
    acl.conceder(d, OTRA, acl.X)                                # no exclusivo: es el directorio compartido
    assert acl.usuarios(d) == {JAULA: 1, OTRA: 1}


def test_r1_si_la_herencia_aparece_despues_de_validar_el_puerto_cierra_sin_dar_paso(mundo, monkeypatch):
    """Carrera: la ACL por defecto llega DESPUES de la validacion del directorio. El token nace con la entrada heredada
    y `conceder(exclusivo=True)` falla cerrado: la jaula no recibe paso y no queda nada."""
    cfg, cargado = mundo
    _acl_por_defecto(cfg.socket_dir)
    monkeypatch.setattr(ServidorPuerto, "_validar_entorno", lambda self: None)

    async def caso():
        with pytest.raises(ConfigFaroInvalida, match="ajenas"):
            async with _puerto(cfg, cargado, uid_esperado=JAULA):
                pass
    corre(caso())
    assert list(cfg.socket_dir.iterdir()) == [] and JAULA not in acl.usuarios(cfg.socket_dir)
