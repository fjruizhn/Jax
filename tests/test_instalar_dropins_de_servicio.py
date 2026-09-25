"""ops/instalar-dropins-de-servicio.sh contra un DESTDIR temporal
(auditoría escalón 3, M4): es el ÚNICO guion de instalación de este árbol
que acepta DESTDIR -- config/systemd/install-memory-scope.sh y
ops/ejecutor/instalar_registro_y_cerco.sh tocan el sistema real en otras
líneas (nftables, /etc/jax/.env, systemctl restart de servicios reales) y
un DESTDIR ahí sería engañoso, así que NO se prueban de punta a punta acá
-- sólo este guion, que sólo copia archivos, se puede ejercitar
completo sin tocar el /etc real.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from manifiesto_arranque import ROOT, leer_manifiesto

SCRIPT = ROOT / "ops" / "instalar-dropins-de-servicio.sh"


@pytest.fixture(autouse=True)
def _limpiar_root_de_destdir(tmp_path):
    """El guion instala con `sudo install -o root -g root` -- tmp_path
    termina con archivos root:root que pytest (corriendo como usuario común)
    no puede borrar solo: sin esto, cada corrida deja basura root-owned en
    /tmp (PermissionError silencioso dentro de un warning, no un fallo,
    pero basura real igual). Se limpia ACÁ, con sudo, apenas termina cada
    test -- antes de que la limpieza automática de pytest lo intente."""
    yield
    subprocess.run(["sudo", "rm", "-rf", str(tmp_path)], check=False)


def _correr(patron: str, destdir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCRIPT), patron, str(ROOT), str(destdir)],
        capture_output=True, text=True,
    )


def test_instala_los_dropins_de_una_unidad_byte_a_byte_y_modo_644(tmp_path):
    resultado = _correr("jax-las-manos.service.d", tmp_path)
    assert resultado.returncode == 0, resultado.stderr

    esperados = [
        (repo_abs, instalada) for repo_abs, instalada in leer_manifiesto()
        if str(instalada).startswith("/etc/systemd/system/jax-las-manos.service.d/")
    ]
    assert esperados, "el manifiesto no tiene drop-ins de jax-las-manos.service.d"
    for repo_abs, instalada in esperados:
        destino = tmp_path / str(instalada).lstrip("/")
        assert destino.is_file(), f"no se instaló: {destino}"
        assert destino.read_bytes() == repo_abs.read_bytes(), f"{destino} no coincide byte a byte con {repo_abs}"
        assert oct(destino.stat().st_mode)[-3:] == "644", f"{destino}: modo incorrecto"


def test_instala_el_guion_de_sanidad_con_modo_755(tmp_path):
    resultado = _correr("/usr/local/sbin/jax-checkout-de-produccion-sano.sh", tmp_path)
    assert resultado.returncode == 0, resultado.stderr

    destino = tmp_path / "usr/local/sbin/jax-checkout-de-produccion-sano.sh"
    assert destino.is_file()
    assert oct(destino.stat().st_mode)[-3:] == "755"

    repo_abs = ROOT / "ops" / "sbin" / "jax-checkout-de-produccion-sano.sh"
    assert destino.read_bytes() == repo_abs.read_bytes()


def test_falla_si_la_unidad_no_esta_en_el_manifiesto(tmp_path):
    resultado = _correr("jax-no-existe-de-mentira.service.d", tmp_path)
    assert resultado.returncode != 0
    assert "ninguna fila coincide" in resultado.stderr


def test_falla_si_la_ruta_absoluta_no_esta_en_el_manifiesto(tmp_path):
    resultado = _correr("/usr/local/sbin/no-existe-de-mentira.sh", tmp_path)
    assert resultado.returncode != 0
    assert "ninguna fila coincide" in resultado.stderr


def _repo_adulterado(tmp_path: Path, repo_rel: str, instalada: str, contenido: str = "[Service]\n") -> Path:
    """Un REPO de mentira, completo, con su propio
    ops/manifiesto-arranque-instalado.tsv de UNA fila -- para probar los
    rechazos de m2 (auditoría escalón 3, MINOR-3) contra un manifiesto
    ADULTERADO, sin tocar jamás el manifiesto real de este árbol."""
    repo = tmp_path / "repo"
    (repo / "ops").mkdir(parents=True)
    (repo / "ops" / "manifiesto-arranque-instalado.tsv").write_text(
        f"{repo_rel}\t{instalada}\n", encoding="utf-8"
    )
    archivo = repo / repo_rel
    archivo.parent.mkdir(parents=True, exist_ok=True)
    archivo.write_text(contenido, encoding="utf-8")
    return repo


def test_rechaza_fila_con_puntos_dobles(tmp_path):
    repo = _repo_adulterado(
        tmp_path, "config/z-pythonpath.conf",
        "/etc/systemd/system/jax-malo.service.d/../../../tmp/evil.conf",
    )
    resultado = subprocess.run(
        [str(SCRIPT), "jax-malo.service.d", str(repo), str(tmp_path / "destdir")],
        capture_output=True, text=True,
    )
    assert resultado.returncode != 0
    assert "rechazada" in resultado.stderr, resultado.stderr


def test_rechaza_nombre_de_archivo_invalido(tmp_path):
    repo = _repo_adulterado(
        tmp_path, "config/z-pythonpath.conf",
        "/etc/systemd/system/jax-malo2.service.d/evil;rm.conf",
    )
    resultado = subprocess.run(
        [str(SCRIPT), "jax-malo2.service.d", str(repo), str(tmp_path / "destdir")],
        capture_output=True, text=True,
    )
    assert resultado.returncode != 0
    assert "inválido" in resultado.stderr, resultado.stderr


def test_rechaza_archivo_de_repo_que_resuelve_fuera_del_repo(tmp_path):
    fuera = tmp_path / "fuera-del-repo.conf"
    fuera.write_text("[Service]\n", encoding="utf-8")
    repo = tmp_path / "repo"
    (repo / "ops").mkdir(parents=True)
    (repo / "ops" / "manifiesto-arranque-instalado.tsv").write_text(
        "escape.conf\t/etc/systemd/system/jax-malo3.service.d/escape.conf\n", encoding="utf-8"
    )
    (repo / "escape.conf").symlink_to(fuera)

    resultado = subprocess.run(
        [str(SCRIPT), "jax-malo3.service.d", str(repo), str(tmp_path / "destdir")],
        capture_output=True, text=True,
    )
    assert resultado.returncode != 0
    assert "fuera del repo" in resultado.stderr, resultado.stderr


def test_todo_bajo_destdir_nunca_fuera(tmp_path):
    """Instala las 4 unidades y el guion de sanidad: TODO lo escrito queda
    bajo tmp_path, nada se sale -- si algo colgara de una ruta absoluta sin
    pasar por DESTDIR, este glob lo encontraría vacío o el test de arriba
    ya habría fallado contra el /etc real (que en CI ni existe)."""
    for patron in (
        "jax-las-manos.service.d",
        "jax-memory-worker.service.d",
        "jax-memory-synthesis.service.d",
        "jax-ejecutor-proxy.service.d",
        "/usr/local/sbin/jax-checkout-de-produccion-sano.sh",
    ):
        resultado = _correr(patron, tmp_path)
        assert resultado.returncode == 0, f"{patron}: {resultado.stderr}"
    instalados = list(tmp_path.rglob("*"))
    archivos = [p for p in instalados if p.is_file()]
    assert len(archivos) == 13, f"se esperaban 13 archivos instalados (12 drop-ins + el guion), hubo {len(archivos)}: {archivos}"
