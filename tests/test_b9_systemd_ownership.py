from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "config" / "systemd"


def test_every_b9_scheduled_job_has_a_systemd_owner():
    expected = {
        "jax-memory-worker": "jax.memory.worker",
        "jax-memory-embedding": "jax.memory.embedding_worker",
        "jax-memory-synthesis": "jax.memory.synthesis_worker",
        "jax-memory-lifecycle": "jax.memory.lifecycle_worker",
        "jax-memory-vector-health": "jax.memory.embedding_worker --health",
    }
    for name, module in expected.items():
        service = (ROOT / f"{name}.service").read_text(encoding="utf-8")
        timer = (ROOT / f"{name}.timer").read_text(encoding="utf-8")
        assert f"-m {module}" in service
        assert f"Unit={name}.service" in timer


# --- Auditoria 2026-10-05, MAJOR-2 y menores: las unidades del repo = lo que corre en produccion ---

def _opciones(texto: str, seccion: str, clave: str) -> list[str]:
    """Valores de `clave` dentro de `[seccion]` (systemd admite claves repetidas)."""
    actual, valores = None, []
    for linea in texto.splitlines():
        linea = linea.strip()
        if linea.startswith("[") and linea.endswith("]"):
            actual = linea[1:-1]
        elif actual == seccion and linea.startswith(f"{clave}="):
            valores.append(linea.split("=", 1)[1])
    return valores


def _unidades_memoria() -> list[Path]:
    return sorted(ROOT.glob("jax-memory-*.service"))


def test_hay_cinco_unidades_de_memoria():
    # Si alguien agrega o quita una, que las pruebas de abajo se enteren y se revisen.
    assert [p.stem for p in _unidades_memoria()] == [
        "jax-memory-embedding", "jax-memory-lifecycle", "jax-memory-synthesis",
        "jax-memory-vector-health", "jax-memory-worker"]


def test_toda_unidad_jax_memory_avisa_si_falla():
    """Sin OnFailure= un timer que falla cada 20 minutos no se entera nadie (MAJOR-2)."""
    for unit in _unidades_memoria():
        assert _opciones(unit.read_text(encoding="utf-8"), "Unit", "OnFailure") == ["aviso-fallo@%n.service"], unit.name


def test_embedding_lifecycle_y_vector_health_validan_el_checkout_como_worker_y_synthesis():
    pre = "/usr/local/sbin/jax-checkout-de-produccion-sano.sh /srv/jax-prod/jax"
    for name in ("jax-memory-embedding", "jax-memory-lifecycle", "jax-memory-vector-health"):
        texto = (ROOT / f"{name}.service").read_text(encoding="utf-8")
        assert _opciones(texto, "Service", "ExecStartPre") == [pre], name


def test_el_timer_de_synthesis_corre_a_las_0430_una_vez_al_dia():
    """DeepSeek se paga por corrida: lo instalado en produccion es 04:30 diario, no cada hora."""
    timer = (ROOT / "jax-memory-synthesis.timer").read_text(encoding="utf-8")
    assert _opciones(timer, "Timer", "OnCalendar") == ["*-*-* 04:30:00"]


# --- Drop-ins de worker y synthesis (2026-10-05): el repo reproduce lo instalado en /etc/systemd/system ---
#
# Los dos llevan en produccion tres drop-ins fuera del repo: el checkout de produccion con su freno
# (`jax-checkout-de-produccion-sano.sh`), la cuenta de servicio y un PYTHONPATH que apunta a
# /srv/jax-prod/jax (el 2026-09-25 apuntaba al checkout de TRABAJO de un agente, jax#274). Sin versionarlos,
# reinstalar desde el repo los pierde en silencio. Copiados byte a byte de /etc/systemd/system.

import subprocess  # noqa: E402

import pytest  # noqa: E402

_CON_DROPINS = {
    "jax-memory-worker": "jax.memory.worker",
    "jax-memory-synthesis": "jax.memory.synthesis_worker",
}
_DROPINS = ("checkout-de-produccion.conf", "cuenta-de-servicio.conf", "z-pythonpath.conf")
_RAIZ_DE_PRODUCCION = "/srv/jax-prod/jax"


def _dropin(unidad: str, nombre: str) -> str:
    ruta = ROOT / f"{unidad}.service.d" / nombre
    assert ruta.is_file(), f"falta {ruta.relative_to(ROOT.parents[1])}: la instalacion desde el repo lo perderia"
    return ruta.read_text(encoding="utf-8")


@pytest.mark.parametrize("unidad", sorted(_CON_DROPINS))
def test_el_dropin_del_checkout_fija_directorio_freno_y_ejecutable_de_produccion(unidad):
    texto = _dropin(unidad, "checkout-de-produccion.conf")
    assert _opciones(texto, "Service", "WorkingDirectory") == [_RAIZ_DE_PRODUCCION]
    assert _opciones(texto, "Service", "ExecStartPre") == [
        f"/usr/local/sbin/jax-checkout-de-produccion-sano.sh {_RAIZ_DE_PRODUCCION}"]
    # `ExecStart=` vacio primero: resetea el de la unidad, si no systemd rechaza dos ExecStart en un oneshot.
    assert _opciones(texto, "Service", "ExecStart") == [
        "", f"{_RAIZ_DE_PRODUCCION}/.venv/bin/python -m {_CON_DROPINS[unidad]}"]


@pytest.mark.parametrize("unidad", sorted(_CON_DROPINS))
def test_el_dropin_de_la_cuenta_corre_como_jaxsvc_con_su_home(unidad):
    texto = _dropin(unidad, "cuenta-de-servicio.conf")
    assert _opciones(texto, "Service", "User") == ["jaxsvc"]
    assert _opciones(texto, "Service", "Group") == ["jaxsvc"]
    home = _opciones(texto, "Service", "Environment")
    assert len(home) == 1 and home[0].startswith("HOME=/") and "/home/" not in home[0]


@pytest.mark.parametrize("unidad", sorted(_CON_DROPINS))
def test_el_pythonpath_apunta_al_checkout_de_produccion_y_nunca_al_de_trabajo(unidad):
    texto = _dropin(unidad, "z-pythonpath.conf")
    assert _opciones(texto, "Service", "Environment") == [f"PYTHONPATH={_RAIZ_DE_PRODUCCION}"]
    # `z-` ordena el archivo ultimo: gana sobre cualquier PYTHONPATH de los demas drop-ins.
    assert sorted(p.name for p in (ROOT / f"{unidad}.service.d").glob("*.conf"))[-1] == "z-pythonpath.conf"


def test_install_memory_scope_instala_los_dropins_de_worker_y_synthesis(tmp_path):
    """Se ejecuta el instalador de verdad contra un destino de prueba (nunca /etc)."""
    destino = tmp_path / "etc-systemd-system"
    destino.mkdir()
    r = subprocess.run(
        ["bash", str(ROOT / "install-memory-scope.sh"), "--instalar-obsoleto"],
        env={"PATH": "/usr/bin:/bin", "JAX_SYSTEMD_SRC": str(ROOT), "JAX_SYSTEMD_DEST": str(destino)},
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    for unidad in _CON_DROPINS:
        for nombre in _DROPINS:
            instalado = destino / f"{unidad}.service.d" / nombre
            assert instalado.is_file(), f"el instalador no dejo {instalado.name} de {unidad}"
            assert instalado.read_bytes() == (ROOT / f"{unidad}.service.d" / nombre).read_bytes()
        assert (destino / f"{unidad}.service").is_file() and (destino / f"{unidad}.timer").is_file()
    # No habilita ni arranca nada: solo copia.
    assert not list(destino.rglob("*.wants"))


def test_install_memory_scope_sin_bandera_no_hace_nada(tmp_path):
    destino = tmp_path / "etc"
    destino.mkdir()
    r = subprocess.run(["bash", str(ROOT / "install-memory-scope.sh")],
                       env={"PATH": "/usr/bin:/bin", "JAX_SYSTEMD_SRC": str(ROOT), "JAX_SYSTEMD_DEST": str(destino)},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and not list(destino.iterdir())


# --- Auditoria Jax#355, MINOR 5: el instalador con un systemctl de mentira ---

def _instalar(tmp_path, destino, fuente=ROOT, real=None):
    """Corre el instalador con un `systemctl` falso en el PATH que solo anota sus argumentos.
    Cada llamada usa su propio directorio de trabajo: el registro es solo el de esa corrida."""
    n = len(list(tmp_path.glob("corrida-*")))
    base = tmp_path / f"corrida-{n}"
    bin_falso = base / "bin"
    bin_falso.mkdir(parents=True)
    registro = base / "llamadas.txt"
    falso = bin_falso / "systemctl"
    falso.write_text(f'#!/bin/sh\necho "$@" >> "{registro}"\n', encoding="utf-8")
    falso.chmod(0o755)
    env = {"PATH": f"{bin_falso}:/usr/bin:/bin", "JAX_SYSTEMD_SRC": str(fuente), "JAX_SYSTEMD_DEST": str(destino)}
    if real is not None:
        env["JAX_SYSTEMD_REAL"] = str(real)
    r = subprocess.run(["bash", str(ROOT / "install-memory-scope.sh"), "--instalar-obsoleto"],
                       env=env, capture_output=True, text=True, timeout=60)
    llamadas = registro.read_text(encoding="utf-8").splitlines() if registro.exists() else []
    return r, llamadas


def test_el_instalador_nunca_habilita_ni_arranca_nada(tmp_path):
    real = tmp_path / "etc"
    real.mkdir()
    prueba = tmp_path / "otro"
    prueba.mkdir()
    for destino, esperado in ((real, ["daemon-reload"]), (prueba, [])):
        r, llamadas = _instalar(tmp_path, destino, real=real)
        assert r.returncode == 0, r.stderr
        # Solo se recarga, y solo en el destino real; jamas enable/start/restart.
        assert llamadas == esperado, llamadas


def test_una_barra_final_en_el_destino_no_salta_el_daemon_reload(tmp_path):
    real = tmp_path / "etc"
    real.mkdir()
    for forma in (f"{real}/", f"{real}//", f"{real}/./"):
        r, llamadas = _instalar(tmp_path, forma, real=real)
        assert r.returncode == 0, r.stderr
        assert llamadas == ["daemon-reload"], (forma, llamadas)


def test_el_origen_por_defecto_es_produccion_y_no_el_checkout_de_trabajo():
    """Se ejecuta el guion con `--print-origen` (no copia nada) y se lee el valor RESUELTO: un
    texto igual en un comentario del archivo no puede hacer pasar esta prueba."""
    def _resuelto(**entorno):
        r = subprocess.run(["bash", str(ROOT / "install-memory-scope.sh"), "--instalar-obsoleto", "--print-origen"],
                           env={"PATH": "/usr/bin:/bin", **entorno}, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        return dict(l.split("=", 1) for l in r.stdout.splitlines())

    por_defecto = _resuelto()
    assert por_defecto["origen"] == "/srv/jax-prod/jax/config/systemd"
    assert por_defecto["destino"] == "/etc/systemd/system"
    explicito = _resuelto(JAX_SYSTEMD_SRC="/otra/ruta", JAX_SYSTEMD_DEST="/tmp/x//")
    assert explicito == {"origen": "/otra/ruta", "destino": "/tmp/x"}


def test_retira_solo_los_dropins_que_instalo_el_y_ya_no_estan_en_el_repo(tmp_path):
    import shutil
    fuente = tmp_path / "fuente"
    shutil.copytree(ROOT, fuente)
    destino = tmp_path / "etc"
    destino.mkdir()
    r, _ = _instalar(tmp_path, destino, fuente=fuente)
    assert r.returncode == 0, r.stderr
    carpeta = destino / "jax-memory-worker.service.d"
    ajeno = carpeta / "99-del-operador.conf"
    ajeno.write_text("[Service]\n", encoding="utf-8")
    (fuente / "jax-memory-worker.service.d" / "z-pythonpath.conf").unlink()
    r, _ = _instalar(tmp_path, destino, fuente=fuente)
    assert r.returncode == 0, r.stderr
    assert not (carpeta / "z-pythonpath.conf").exists(), "un drop-in que ya no esta en el repo debe retirarse"
    assert ajeno.exists(), "un .conf ajeno al guion no se toca nunca"
    assert (carpeta / "checkout-de-produccion.conf").exists()
    assert (destino / "jax-memory-synthesis.service.d" / "z-pythonpath.conf").exists()   # la otra unidad no se afecta


@pytest.mark.parametrize("extra", [
    ["--print-origin"],                 # errata de --print-origen
    ["--print-origen", "sobra"],        # un tercer argumento
    ["--instalar-obsoleto"],            # repetido
    ["x"],
    [""],
])
def test_un_argumento_desconocido_se_rechaza_antes_de_instalar_nada(tmp_path, extra):
    """Antes, cualquier segundo argumento distinto de --print-origen se ignoraba y el guion instalaba."""
    destino = tmp_path / "etc"
    destino.mkdir()
    r = subprocess.run(["bash", str(ROOT / "install-memory-scope.sh"), "--instalar-obsoleto", *extra],
                       env={"PATH": "/usr/bin:/bin", "JAX_SYSTEMD_SRC": str(ROOT), "JAX_SYSTEMD_DEST": str(destino)},
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
    assert "argumento no reconocido" in r.stderr
    assert not list(destino.iterdir()), "no debe copiar nada"
    assert r.stdout == ""
