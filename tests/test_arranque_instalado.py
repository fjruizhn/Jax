"""Versionado del arranque de systemd que antes sólo vivía en /etc de
hall9000 (decisión de Fernando, 2026-09-25, opción A).

Dos partes:

(a) SIEMPRE corre, también en CI sin ningún host de producción cerca: la
    forma del repo es correcta -- cada archivo del manifiesto existe, los
    drop-ins apuntan a las rutas de producción correctas, y el guion de
    sanidad tiene el bit ejecutable EN GIT (no sólo en el filesystem de
    quien corrió el checkout).

(b) SÓLO en el host de producción (si existen /srv/jax-prod/jax y todas
    las rutas /etc del manifiesto): corre
    ops/verificar-arranque-instalado.sh de verdad y exige código 0. Fuera
    de ese host, skip con el motivo explícito.

El manifiesto es UNA sola lista, ops/manifiesto-arranque-instalado.tsv. La
lectura vive en tests/manifiesto_arranque.py (importada acá) y el propio
guion (ops/verificar-arranque-instalado.sh) la vuelve a leer del mismo
archivo por su cuenta, en shell -- dos listas que pudieran divergir serían
peor que una sola que puede fallar.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from manifiesto_arranque import (
    DUENOS, MANIFIESTO, ROOT, leer_manifiesto as _leer_manifiesto, leer_manifiesto_con_dueno,
)

SCRIPT = ROOT / "ops" / "verificar-arranque-instalado.sh"

RUTA_SANIDAD = "/usr/local/sbin/jax-checkout-de-produccion-sano.sh"
RAIZ_PRODUCCION = "/srv/jax-prod/jax"


def test_el_manifiesto_existe_y_no_esta_vacio():
    assert MANIFIESTO.is_file()
    assert _leer_manifiesto(), "el manifiesto no puede estar vacío"


def test_cada_archivo_del_manifiesto_existe_en_el_repo():
    faltantes = [str(repo_abs) for repo_abs, _ in _leer_manifiesto() if not repo_abs.is_file()]
    assert not faltantes, f"faltan en el repo: {faltantes}"


def test_los_checkout_de_produccion_referencian_el_guion_de_sanidad():
    conf = [repo_abs for repo_abs, _ in _leer_manifiesto() if repo_abs.name == "checkout-de-produccion.conf"]
    assert conf, "se esperaba al menos un checkout-de-produccion.conf en el manifiesto"
    for c in conf:
        contenido = c.read_text(encoding="utf-8")
        assert RUTA_SANIDAD in contenido, f"{c} no referencia {RUTA_SANIDAD}"


def test_los_dropins_de_raiz_referencian_la_raiz_de_produccion():
    """checkout-de-produccion.conf y z-pythonpath.conf fijan rutas contra
    /srv/jax-prod/jax -- cuenta-de-servicio.conf NO (sólo fija User/Group/
    HOME, ver el test de abajo), así que no entra en este chequeo."""
    conf = [
        repo_abs for repo_abs, _ in _leer_manifiesto()
        if repo_abs.name in ("checkout-de-produccion.conf", "z-pythonpath.conf")
    ]
    assert conf, "se esperaban checkout-de-produccion.conf/z-pythonpath.conf en el manifiesto"
    for c in conf:
        contenido = c.read_text(encoding="utf-8")
        assert RAIZ_PRODUCCION in contenido, f"{c} no referencia {RAIZ_PRODUCCION}"


def test_cuenta_de_servicio_fija_jaxsvc_con_hogar_propio():
    conf = [repo_abs for repo_abs, _ in _leer_manifiesto() if repo_abs.name == "cuenta-de-servicio.conf"]
    assert conf, "se esperaba al menos un cuenta-de-servicio.conf en el manifiesto"
    for c in conf:
        contenido = c.read_text(encoding="utf-8")
        assert "User=jaxsvc" in contenido, f"{c} no fija User=jaxsvc"
        assert "Group=jaxsvc" in contenido, f"{c} no fija Group=jaxsvc"
        assert "Environment=HOME=/var/lib/jaxsvc" in contenido, f"{c} no fija HOME propio"


def test_z_pythonpath_fija_la_raiz_del_checkout_de_produccion():
    """Las 4 unidades con z-pythonpath.conf (las-manos, proxy del Ejecutor,
    memory-worker, memory-synthesis) -- auditoría escalón 3, M1: el
    PYTHONPATH efectivo de las últimas 3 apuntaba a /home/fruiz/jax, el
    checkout de TRABAJO de un agente, no el de producción."""
    zpps = [repo_abs for repo_abs, _ in _leer_manifiesto() if repo_abs.name == "z-pythonpath.conf"]
    assert len(zpps) == 4, f"se esperaban 4 z-pythonpath.conf en el manifiesto, hay {len(zpps)}: {zpps}"
    for zpp in zpps:
        contenido = zpp.read_text(encoding="utf-8")
        assert f"PYTHONPATH={RAIZ_PRODUCCION}" in contenido, f"{zpp} no fija PYTHONPATH con {RAIZ_PRODUCCION}"

    proxy_zpp = ROOT / "ops" / "ejecutor" / "jax-ejecutor-proxy.service.d" / "z-pythonpath.conf"
    assert proxy_zpp in zpps
    assert f"PYTHONPATH={RAIZ_PRODUCCION}:{RAIZ_PRODUCCION}/las_manos" in proxy_zpp.read_text(encoding="utf-8"), (
        "el proxy del Ejecutor necesita las_manos/ en el PYTHONPATH (facet_resolver, "
        "credential_resolver, db_connect_config son imports pelados que dependen de eso)"
    )


def test_el_guion_de_sanidad_tiene_bit_ejecutable_en_git():
    salida = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-s", "ops/sbin/jax-checkout-de-produccion-sano.sh"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert salida.startswith("100755"), f"modo en git no es ejecutable: {salida!r}"


def test_el_guion_de_espera_de_la_base_tiene_bit_ejecutable_en_git():
    salida = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-s", "ops/sbin/jax-db-esperar"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert salida.startswith("100755"), f"modo en git no es ejecutable: {salida!r}"


def test_las_dependencias_de_esperar_db_estan_en_el_manifiesto():
    """m2 de la auditoria de #356: esperar-db.conf llama a jax-db-esperar y engancha
    OnFailure=aviso-fallo@%n.service; sin esas dos piezas instaladas la unidad falla."""
    instaladas = {str(i) for _, i in _leer_manifiesto()}
    assert "/usr/local/sbin/jax-db-esperar" in instaladas
    assert "/etc/systemd/system/aviso-fallo@.service" in instaladas


def test_guion_de_verificacion_tiene_bit_ejecutable_en_git():
    salida = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-s", "ops/verificar-arranque-instalado.sh"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert salida.startswith("100755"), f"modo en git no es ejecutable: {salida!r}"


def _motivo_de_skip_fuera_de_produccion() -> str | None:
    """None si ESTA máquina es el host de producción de jax; si no, el
    motivo del skip.

    La máquina es producción SÍ Y SÓLO SI existe /srv/jax-prod/jax --
    auditoría escalón 3, M3: la versión anterior TAMBIÉN saltaba si faltaba
    algún archivo instalado del manifiesto, lo que le daba a cualquier
    drop-in sin instalar un skip silencioso en vez de un rojo, incluso en
    el host de producción real. Ser producción no depende de tener el
    arranque completo instalado -- es al revés: si sos producción y falta
    algo, ESO es lo que este test tiene que gritar."""
    if not Path(RAIZ_PRODUCCION).is_dir():
        return f"esta máquina no tiene {RAIZ_PRODUCCION} -- no es el host de producción de jax"
    return None


def test_verificar_arranque_instalado_da_cero_en_produccion():
    motivo = _motivo_de_skip_fuera_de_produccion()
    if motivo:
        pytest.skip(motivo)
    resultado = subprocess.run([str(SCRIPT)], capture_output=True, text=True)
    assert resultado.returncode == 0, (
        f"ops/verificar-arranque-instalado.sh salió {resultado.returncode}:\n"
        f"stdout: {resultado.stdout}\nstderr: {resultado.stderr}"
    )


def test_env_de_produccion_no_tiene_clave_pythonpath():
    """Auditoría escalón 3, ronda 2, MAJOR-2 (2) -- justificación corregida
    en ronda 3, MINOR-1: PYTHONPATH tiene que llegar SIEMPRE por el
    z-pythonpath.conf de CADA unidad -- nunca por
    `EnvironmentFile=/etc/jax/.env`, que las 4 unidades comparten. Una
    clave PYTHONPATH ahí NO sería inofensiva: systemd.exec(5) es explícito
    -- "Settings from these files [EnvironmentFile=] override settings
    made with Environment=" (man systemd.exec, la sección de
    EnvironmentFile=) -- es decir, EnvironmentFile= se aplica DESPUÉS y
    PISA lo que haya puesto el Environment= de z-pythonpath.conf, no al
    revés. Una clave PYTHONPATH en /etc/jax/.env rompería el PYTHONPATH
    por-unidad de las 4 unidades, silenciosamente.

    Sólo el NOMBRE de la clave, nunca su valor -- ni siquiera para esta
    aserción hace falta leerlo, y el resto de las claves de /etc/jax/.env
    con valores de /home/fruiz (ver
    test_los_archivos_de_unidad_no_apuntan_al_checkout_de_trabajo) es una
    tarea aparte, no de este test."""
    motivo = _motivo_de_skip_fuera_de_produccion()
    if motivo:
        pytest.skip(motivo)
    # MINOR-1: la regex acepta espacios/tabs delante de la clave y un
    # `export ` opcional -- formas reales de escribir un .env que
    # `^[A-Za-z_]...` sin más se perdería.
    resultado = subprocess.run(
        ["sudo", "-n", "grep", "-oE", "^[[:space:]]*(export[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*=", "/etc/jax/.env"],
        capture_output=True, text=True,
    )
    assert resultado.returncode == 0, (
        f"no se pudo leer /etc/jax/.env con sudo -n (código {resultado.returncode}): {resultado.stderr}"
    )
    claves = {
        linea.rstrip("=").split()[-1]
        for linea in resultado.stdout.splitlines() if linea.strip()
    }
    assert "PYTHONPATH" not in claves, (
        "/etc/jax/.env tiene una clave PYTHONPATH -- pisaría (o duplicaría) el "
        "z-pythonpath.conf de cada unidad, la única fuente de verdad que este árbol versiona"
    )


def test_git_en_srv_jax_prod_necesita_safe_directory():
    """Auditoría escalón 3, ronda 3, MAJOR-2: /srv/jax-prod/jax es
    jaxsvc:jaxsvc -- `git rev-parse`/`status` ahí dan "detected dubious
    ownership" (rc=128) sin `-c safe.directory=...`, tanto sin sudo como
    con `sudo -n`. Con el flag, sólo lectura, funcionan -- esto es lo que
    config/systemd/install-memory-scope.sh, ops/ejecutor/instalar_registro_y_cerco.sh
    y ops/instalar-dropins-de-servicio.sh (instalación real, sin DESTDIR)
    usan para poder derivar REPO y comprobar rama/limpieza contra ese
    checkout."""
    motivo = _motivo_de_skip_fuera_de_produccion()
    if motivo:
        pytest.skip(motivo)

    sin_flag = subprocess.run(
        ["git", "-C", RAIZ_PRODUCCION, "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    assert sin_flag.returncode == 128, f"se esperaba 'dubious ownership' sin safe.directory: {sin_flag!r}"
    assert "dubious ownership" in sin_flag.stderr

    con_flag = subprocess.run(
        ["git", "-c", f"safe.directory={RAIZ_PRODUCCION}", "-C", RAIZ_PRODUCCION, "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    assert con_flag.returncode == 0, con_flag.stderr
    assert con_flag.stdout.strip() == RAIZ_PRODUCCION

    con_flag_root = subprocess.run(
        ["sudo", "-n", "git", "-c", f"safe.directory={RAIZ_PRODUCCION}", "-C", RAIZ_PRODUCCION, "status", "--porcelain"],
        capture_output=True, text=True,
    )
    assert con_flag_root.returncode == 0, (
        f"sudo -n git -c safe.directory status debería dar 0: {con_flag_root!r}"
    )


# --- Dueño de cada fila (M1 de la re-auditoria de #356) -------------------------------------
# Tres archivos del manifiesto son ESPEJOS de otro repo: nada lo declaraba y alguien podia editar
# la copia de aqui. Ahora cada fila lleva su dueño y la ruta en el dueño.

_ESPEJOS = {
    "/etc/systemd/system/jax-las-manos.service.d/esperar-db.conf":
        ("jax-platform", "ops/mariadb-12.3/systemd/esperar-db.conf"),
    "/usr/local/sbin/jax-db-esperar":
        ("jax-platform", "ops/mariadb-12.3/systemd/jax-db-esperar"),
    "/etc/systemd/system/aviso-fallo@.service":
        ("claude-skills", "systemd/hall9000/sistema/aviso-fallo@.service"),
}
_REPOS_DUENOS = {
    "jax-platform": ("JAX_PLATFORM_REPO", "/home/fruiz/jax-platform"),
    "claude-skills": ("CLAUDE_SKILLS_REPO", "/home/fruiz/claude-skills"),
}


def test_cada_fila_declara_su_dueno_y_la_ruta_en_el_dueno():
    for repo_abs, instalada, dueno, ruta_dueno in leer_manifiesto_con_dueno():
        assert dueno in DUENOS, f"{instalada}: dueño {dueno!r} no es uno de {DUENOS}"
        if dueno == "jax":
            assert ruta_dueno == "-", f"{instalada}: una fila de jax no lleva ruta en otro repo (usa '-')"
        else:
            assert ruta_dueno != "-" and not ruta_dueno.startswith("/"), f"{instalada}: falta la ruta (relativa) en {dueno}"


def test_los_espejos_declaran_su_repo_dueno_y_todo_lo_demas_es_de_jax():
    obtenido = {str(i): (d, r) for _, i, d, r in leer_manifiesto_con_dueno() if d != "jax"}
    assert obtenido == _ESPEJOS


@pytest.mark.parametrize("instalada", sorted(_ESPEJOS))
def test_cada_espejo_coincide_con_el_archivo_del_repo_dueno(instalada):
    """El espejo se compara con la fuente cuando ese checkout existe (rutas por env:
    JAX_PLATFORM_REPO y CLAUDE_SKILLS_REPO; por defecto las de hall9000). Si no existe el
    checkout o el archivo en su rama actual, skip con el motivo -- no se inventa el resultado."""
    dueno, ruta = _ESPEJOS[instalada]
    variable, defecto = _REPOS_DUENOS[dueno]
    raiz = Path(os.environ.get(variable, defecto))
    fuente = raiz / ruta
    if not fuente.is_file():
        pytest.skip(f"no hay {fuente} (checkout de {dueno} ausente o en otra rama; ajusta {variable})")
    espejo = next(r for r, i, _, _ in leer_manifiesto_con_dueno() if str(i) == instalada)
    assert espejo.read_bytes() == fuente.read_bytes(), (
        f"{espejo} ya no coincide con {dueno}:{ruta}; el dueño es {dueno}, no se edita aqui: "
        f"cambia la fuente y vuelve a copiar"
    )


def test_el_instalador_del_proxy_delega_el_respaldo_y_lo_hace_antes_de_pisar_la_base():
    """m3: la linea que pisa la base del proxy va precedida de la llamada al guion de respaldo."""
    lineas = (ROOT / "ops" / "ejecutor" / "instalar_registro_y_cerco.sh").read_text(encoding="utf-8").splitlines()
    codigo = [l for l in lineas if not l.lstrip().startswith("#")]
    pisa = next(i for i, l in enumerate(codigo) if 'jax-ejecutor-proxy.service" /etc/systemd/system/' in l)
    llamada = [i for i, l in enumerate(codigo) if "respaldar-base-del-proxy.sh" in l]
    assert llamada and llamada[0] < pisa, "el instalador no respalda la base del proxy antes de pisarla"


def test_el_respaldo_de_la_base_del_proxy_copia_el_archivo_original_sin_sudo(tmp_path):
    """m3 (ronda 3): se EJECUTA el guion de respaldo sobre un arbol falso (DESTDIR, sin sudo)."""
    guion = ROOT / "ops" / "ejecutor" / "respaldar-base-del-proxy.sh"
    base = tmp_path / "etc" / "systemd" / "system" / "jax-ejecutor-proxy.service"
    base.parent.mkdir(parents=True)
    original = "[Service]\nUser=fruiz\n"
    base.write_text(original, encoding="utf-8")
    r = subprocess.run([str(guion), "20261006-120000", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    copia = tmp_path / "etc" / "jax-ejecutor-cerco" / "respaldos" / "jax-ejecutor-proxy.service.20261006-120000"
    assert copia.is_file() and copia.read_text(encoding="utf-8") == original
    assert base.read_text(encoding="utf-8") == original, "el respaldo no debe tocar la base"
    assert "Para revertir" in r.stdout and "respaldos/jax-ejecutor-proxy.service.20261006-120000" in r.stdout


def test_el_respaldo_sin_base_instalada_no_falla_ni_crea_nada(tmp_path):
    guion = ROOT / "ops" / "ejecutor" / "respaldar-base-del-proxy.sh"
    r = subprocess.run([str(guion), "20261006-120000", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "etc").exists()


@pytest.mark.parametrize("marca", ["", "../x", "a b", "2026;rm"])
def test_el_respaldo_rechaza_una_marca_invalida(tmp_path, marca):
    guion = ROOT / "ops" / "ejecutor" / "respaldar-base-del-proxy.sh"
    r = subprocess.run([str(guion), marca, str(tmp_path)], capture_output=True, text=True)
    assert r.returncode != 0
