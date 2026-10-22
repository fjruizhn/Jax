"""La Reversión de docs/runbooks/rutas-de-produccion.md, ejecutada de verdad.

Corre el preámbulo, R2, R3 y R4 (los bloques ```bash del runbook, tal como
están escritos) en un directorio temporal, sin root:

  - TODA ruta real se reescribe al sandbox, y un bloque al que le quede alguna
    ruta real (/etc/jax, /srv/jax-data, /srv/jax-prod, /var/log/jax,
    /home/fruiz) NO se corre: la prueba falla antes.
  - `sudo` es una función que ejecuta sin privilegios, con dos cosas modeladas
    del host real: sin sudo, el directorio jaxsvc:jaxsvc 0750 del audit log
    nuevo NO se ve (fruiz no está en el grupo), y `stat` devuelve el modo real.
  - R1 y R5 (systemctl) nunca se ejecutan.

Nació de la auditoría de escalón 3 de jax#277 (2026-10-21). Las marcas del
corte se movieron de carpeta y R2/R3 se saltaban su trabajo en silencio
(MAJOR-1). R4 pisaba el .env entero (MAJOR-2). R2 comprobaba con `[ -f ]` sin
sudo un archivo que fruiz no ve (MAJOR-1'). Cada caso de abajo falla contra la
versión del runbook que tenía ese defecto.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

RUNBOOK = Path(__file__).resolve().parents[1] / "docs" / "runbooks" / "rutas-de-produccion.md"
MARCAS_REAL = "/home/fruiz/admin-ecosistema/jax-rutas-produccion"
REALES = ["/etc/jax", "/srv/jax-data", "/srv/jax-prod", "/var/log/jax", "/home/fruiz"]

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("rsync") is None,
    reason="hace falta bash y rsync")


def _bloques():
    texto = RUNBOOK.read_text(encoding="utf-8")
    rev = texto[texto.index("## Reversión"):]
    bloques = re.findall(r"```bash\n(.*?)```", rev, flags=re.S)
    pre, r1, r2, r3, r4, r5 = bloques[:6]
    assert "systemctl stop" in r1 and "systemctl start" in r5, "cambió el orden de los bloques de la Reversión"
    return {"PRE": pre, "R2": r2, "R3": r3, "R4": r4}


BLOQUES = _bloques()

SUDO = r'''
PROTEGIDOS="$SB/var-log-jax/las_manos"
_protegido() { local a; for a in "$@"; do case "$a" in "$PROTEGIDOS"|"$PROTEGIDOS"/*) return 0;; esac; done; return 1; }
# Sin sudo, una ruta protegida se evalúa como si NO existiera (lo que ve fruiz):
# se reemplaza por una ruta inexistente y se deja que test decida, `!` incluido.
_como_fruiz() { local a out=(); for a in "$@"; do if builtin test -z "${SUDO_ACTIVO:-}" && _protegido "$a"; then out+=("/nonexistent-para-fruiz"); else out+=("$a"); fi; done; printf '%s\0' "${out[@]}"; }
[()   { local args; mapfile -d '' args < <(_como_fruiz "$@"); builtin [ "${args[@]}"; }
test() { local args; mapfile -d '' args < <(_como_fruiz "$@"); builtin test "${args[@]}"; }
sudo() {
  case "$1" in
    chown|chattr) return 0 ;;
    stat) shift; local f="${!#}"; echo "root:jaxsvc $(command stat -c %a "$f")" ;;
    find) if builtin test -n "${FIND_FALLA:-}"; then echo "find: falla simulada" >&2; return 1; fi; SUDO_ACTIVO=1 "$@" ;;
    *) SUDO_ACTIVO=1 "$@" ;;
  esac
}
'''


def _reescribir(bloque: str, sb: str) -> str:
    b = bloque.replace(MARCAS_REAL, f"{sb}/marcas")
    b = b.replace("/home/fruiz/jax/", f"{sb}/home-jax/")
    b = b.replace("/etc/jax/", f"{sb}/etc-jax/")
    b = b.replace("/srv/jax-data/", f"{sb}/srv-jax-data/").replace("/srv/jax-data", f"{sb}/srv-jax-data")
    b = b.replace("/var/log/jax/", f"{sb}/var-log-jax/")
    resto = b.replace(sb, "")
    for r in REALES:
        assert r not in resto, f"queda una ruta real {r} en el bloque: no se corre"
    return b


def _correr(nombre: str, sb: Path, **extra) -> tuple[int, str]:
    script = f"SB={sb}\n" + SUDO + _reescribir(BLOQUES[nombre], str(sb))
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                       env={"PATH": os.environ["PATH"], "HOME": str(sb), **extra})
    return r.returncode, (r.stdout + r.stderr).strip()


@pytest.fixture
def sb(tmp_path):
    for d in ("marcas", "home-jax/las_manos/logs", "home-jax/repo", "etc-jax",
              "srv-jax-data/repo/documents", "var-log-jax/las_manos"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "marcas/rutas-prod.TS").write_text("20260925-054909\n")
    (tmp_path / "marcas/rutas-prod.N0").write_text("3\n")
    (tmp_path / "marcas/rutas-prod.paso4-completo").write_text("x\n")
    (tmp_path / "home-jax/las_manos/logs/audit.jsonl").write_text("a\nb\nc\n")
    (tmp_path / "var-log-jax/las_manos/audit.jsonl").write_text("a\nb\nc\nd\ne\n")
    (tmp_path / "srv-jax-data/repo/documents/nuevo.md").write_text("nuevo\n")
    (tmp_path / "etc-jax/.env.backup-pre-rutas-de-produccion-20260925-054909").write_text(
        "JAX_DB_PASSWORD=vieja\n"
        "JAX_CONFIG_PATH=/home/fruiz/jax/config/config.toml\n"
        "JAX_AUDIT_LOG_PATH=/home/fruiz/jax/las_manos/logs/audit.jsonl\n"
        "JAX_REPO_BASE=/home/fruiz/jax/repo\n")
    (tmp_path / "etc-jax/.env").write_text(
        "JAX_DB_PASSWORD=rotada\n"
        "JAX_CONFIG_PATH=/srv/jax-prod/jax/config/config.toml\n"
        "JAX_AUDIT_LOG_PATH=/var/log/jax/las_manos/audit.jsonl\n"
        "JAX_REPO_BASE=/srv/jax-data/repo\n"
        "JAX_CLAVE_NUEVA=despues-del-corte\n")
    return tmp_path


# --- preámbulo ---------------------------------------------------------------

def test_preambulo_con_todo_en_su_sitio(sb):
    rc, out = _correr("PRE", sb)
    assert rc == 0, out


def test_preambulo_sin_carpeta_de_marcas_aborta(sb):
    shutil.rmtree(sb / "marcas")
    rc, out = _correr("PRE", sb)
    assert rc != 0 and "ABORTAR" in out, out


# --- R2: audit log ------------------------------------------------------------

def test_r2_anexa_lo_nuevo_aunque_fruiz_no_vea_el_directorio(sb):
    """MAJOR-1': el audit nuevo vive en un directorio que fruiz no ve sin sudo."""
    rc, out = _correr("R2", sb)
    assert rc == 0 and "R2 OK" in out, out
    assert (sb / "home-jax/las_manos/logs/audit.jsonl").read_text() == "a\nb\nc\nd\ne\n"


def test_r2_repetido_es_idempotente(sb):
    _correr("R2", sb)
    rc, out = _correr("R2", sb)
    assert rc == 0 and "ya coinciden" in out, out


def test_r2_sin_n0_con_audit_nuevo_aborta(sb):
    (sb / "marcas/rutas-prod.N0").unlink()
    rc, out = _correr("R2", sb)
    assert rc != 0 and "no se sabe" in out, out


def test_r2_sin_n0_y_sin_audit_nuevo_se_salta(sb):
    (sb / "marcas/rutas-prod.N0").unlink()
    (sb / "var-log-jax/las_manos/audit.jsonl").unlink()
    rc, out = _correr("R2", sb)
    assert rc == 0 and "nada que anexar" in out, out


def test_r2_con_el_viejo_alterado_aborta(sb):
    (sb / "home-jax/las_manos/logs/audit.jsonl").write_text("a\nX\nc\n")
    rc, out = _correr("R2", sb)
    assert rc != 0 and "ABORTAR" in out, out


# --- R3: documents ------------------------------------------------------------

def test_r3_trae_lo_nuevo(sb):
    rc, out = _correr("R3", sb)
    assert rc == 0 and (sb / "home-jax/repo/documents/nuevo.md").exists(), out


def test_r3_sin_marca_y_repo_con_datos_aborta(sb):
    (sb / "marcas/rutas-prod.paso4-completo").unlink()
    rc, out = _correr("R3", sb)
    assert rc != 0 and "no se sabe" in out, out


def test_r3_sin_marca_y_repo_vacio_se_salta(sb):
    (sb / "marcas/rutas-prod.paso4-completo").unlink()
    shutil.rmtree(sb / "srv-jax-data/repo/documents")
    rc, out = _correr("R3", sb)
    assert rc == 0 and "nunca corrio" in out, out


def test_r3_con_find_que_falla_aborta(sb):
    (sb / "marcas/rutas-prod.paso4-completo").unlink()
    rc, out = _correr("R3", sb, FIND_FALLA="1")
    assert rc != 0 and "ABORTAR" in out, out


# --- R4: .env ------------------------------------------------------------------

def test_r4_devuelve_solo_las_tres_claves(sb):
    rc, out = _correr("R4", sb)
    env = (sb / "etc-jax/.env").read_text()
    assert rc == 0 and "diff OK" in out, out
    assert "JAX_DB_PASSWORD=rotada" in env and "JAX_CLAVE_NUEVA=despues-del-corte" in env
    assert "JAX_CONFIG_PATH=/home/fruiz/jax/config/config.toml" in env
    assert "JAX_AUDIT_LOG_PATH=/home/fruiz/jax/las_manos/logs/audit.jsonl" in env
    assert "JAX_REPO_BASE=/home/fruiz/jax/repo" in env
    assert not (sb / "etc-jax/.env.tmp").exists()
    assert oct((sb / "etc-jax/.env").stat().st_mode & 0o777) == "0o640"


def test_r4_repetido_aborta(sb):
    _correr("R4", sb)
    rc, out = _correr("R4", sb)
    assert rc != 0 and "ABORTAR" in out, out


def test_r4_con_clave_duplicada_en_el_respaldo_aborta_sin_tocar(sb):
    b = sb / "etc-jax/.env.backup-pre-rutas-de-produccion-20260925-054909"
    b.write_text(b.read_text() + "JAX_REPO_BASE=/otra\n")
    antes = (sb / "etc-jax/.env").read_text()
    rc, out = _correr("R4", sb)
    assert rc != 0 and (sb / "etc-jax/.env").read_text() == antes, out


def test_r4_con_crlf_conserva_los_retornos_y_no_imprime_secretos(sb):
    e = sb / "etc-jax/.env"
    e.write_bytes(e.read_bytes().replace(b"\n", b"\r\n"))
    rc, out = _correr("R4", sb)
    nuevo = e.read_bytes()
    assert rc == 0 and "diff OK" in out, out
    assert nuevo.count(b"\r\n") == 5
    assert b"JAX_DB_PASSWORD=rotada\r\n" in nuevo and b"JAX_REPO_BASE=/home/fruiz/jax/repo\r\n" in nuevo
    assert "rotada" not in out and "vieja" not in out


def test_r4_no_toca_una_clave_con_prefijo_mas_largo(sb):
    e = sb / "etc-jax/.env"
    e.write_text(e.read_text().replace("JAX_DB_PASSWORD=rotada", "JAX_REPO_BASE_EXTRA=no-tocar\nJAX_DB_PASSWORD=rotada"))
    rc, out = _correr("R4", sb)
    assert rc == 0 and "JAX_REPO_BASE_EXTRA=no-tocar" in e.read_text(), out


def test_r4_sin_una_clave_en_el_vivo_aborta_sin_mostrar_secretos(sb):
    """Si una de las 3 claves falta en el .env vivo, aborta antes de escribir
    nada y sin imprimir otras líneas del archivo."""
    e = sb / "etc-jax/.env"
    e.write_text(e.read_text().replace("JAX_REPO_BASE=/srv/jax-data/repo\n", ""))
    rc, out = _correr("R4", sb)
    assert rc != 0 and "ABORTAR" in out, out
    assert "rotada" not in out


def test_r4_con_env_sin_salto_final(sb):
    """MINOR-E: un .env sin salto de línea final no rompe el conteo del diff."""
    e = sb / "etc-jax/.env"
    e.write_text(e.read_text().replace("JAX_CLAVE_NUEVA=despues-del-corte\n", "")
                 .rstrip("\n"))  # la última línea es JAX_REPO_BASE, sin \n
    rc, out = _correr("R4", sb)
    assert rc == 0 and "diff OK" in out, out
    assert e.read_text().endswith("JAX_REPO_BASE=/home/fruiz/jax/repo")


# --- estático: nada de «no se ve» leído como «no existe» -------------------------
# Límite declarado: el modelo de visibilidad solo intercepta `[` y `test`. Un
# cmp/cat/head sin sudo sobre /var/log/jax/las_manos pasaría aquí y fallaría en
# el host; por eso la prueba de abajo exige sudo en todo uso de $NEW.

def test_r2_todo_uso_del_audit_nuevo_va_con_sudo():
    for l in BLOQUES["R2"].splitlines():
        if "$NEW" in l and not l.lstrip().startswith(("#", "echo", "NEW=")) and "sudo" not in l:
            pytest.fail(f"uso de $NEW sin sudo: {l}")

@pytest.mark.parametrize("nombre", ["PRE", "R2", "R3", "R4"])
def test_sin_pruebas_de_existencia_sin_sudo_fuera_de_marcas(nombre):
    malas = [l for l in BLOQUES[nombre].splitlines()
             if not l.lstrip().startswith("#") and re.search(r"\[\[? !? ?-[a-zA-Z] ", l)
             and "$MARCAS/" not in l and "$CONTENIDO" not in l]
    assert not malas, malas


@pytest.mark.parametrize("nombre", ["PRE", "R2", "R3", "R4"])
def test_ningun_error_de_sudo_escondido(nombre):
    assert "2>/dev/null)" not in BLOQUES[nombre]
