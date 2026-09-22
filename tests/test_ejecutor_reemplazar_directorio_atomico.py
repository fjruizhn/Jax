# tests/test_ejecutor_reemplazar_directorio_atomico.py
"""`ops/ejecutor/reemplazar_directorio_atomico.sh` (M4, auditoría adversarial
2026-09-22): el instalador de skills tiene que dejar el destino EXACTAMENTE igual a
lo declarado -- una skill retirada, o cualquier archivo de más, se borra, no se
queda ahí.

Corre el script REAL con un `sudo` de mentira en el PATH: como esta prueba no corre
como root, el `sudo` falso saca `-o`/`-g` de las llamadas a `install` (no puede
cambiar el dueño a root sin serlo) y ejecuta todo lo demás (`rm`, `mv`, `test`) tal
cual -- mismos efectos reales sobre archivos, sin necesitar privilegios."""
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SCRIPT = RAIZ / "ops" / "ejecutor" / "reemplazar_directorio_atomico.sh"

_SUDO_FALSO = """#!/bin/bash
prog="$1"; shift
if [ "$prog" = "install" ]; then
  args=()
  while [ $# -gt 0 ]; do
    case "$1" in
      -o|-g) shift 2 ;;
      *) args+=("$1"); shift ;;
    esac
  done
  exec install "${args[@]}"
else
  exec "$prog" "$@"
fi
"""


def _bin_con_sudo_falso(tmp_path: Path) -> str:
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "sudo").write_text(_SUDO_FALSO)
    (bin_ / "sudo").chmod(0o755)
    return f"{bin_}:/usr/bin:/bin"


def _arbol(base: Path) -> dict:
    return {p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*") if p.is_file()}


def _correr(tmp_path, etapa: Path, destino: Path):
    return subprocess.run([str(SCRIPT), str(etapa), str(destino)], capture_output=True, text=True,
                          env={"PATH": _bin_con_sudo_falso(tmp_path)}, timeout=30)


def test_el_script_pasa_shellcheck_o_al_menos_bash_menos_n():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0


def test_destino_ausente_queda_igual_a_la_etapa(tmp_path):
    etapa = tmp_path / "etapa"
    (etapa / "skill-a").mkdir(parents=True)
    (etapa / "skill-a" / "SKILL.md").write_text("a")
    destino = tmp_path / "destino" / "skills"

    r = _correr(tmp_path, etapa, destino)
    assert r.returncode == 0, r.stderr
    assert _arbol(destino) == _arbol(etapa)


def test_una_skill_retirada_se_borra_no_se_queda_atras(tmp_path):
    """El hallazgo real de M4: `install` uno por uno SOLO agrega -- nunca borra lo que
    ya no está en la etapa. Esto prueba que el reemplazo atómico sí lo hace."""
    etapa = tmp_path / "etapa"
    (etapa / "skill-nueva").mkdir(parents=True)
    (etapa / "skill-nueva" / "SKILL.md").write_text("nueva")

    destino = tmp_path / "destino" / "skills"
    (destino / "skill-retirada").mkdir(parents=True)
    (destino / "skill-retirada" / "SKILL.md").write_text("vieja, ya no declarada")
    (destino / "skill-nueva").mkdir(parents=True)
    (destino / "skill-nueva" / "SKILL.md").write_text("version vieja del contenido")

    r = _correr(tmp_path, etapa, destino)
    assert r.returncode == 0, r.stderr
    assert _arbol(destino) == {"skill-nueva/SKILL.md": b"nueva"}
    assert not (destino / "skill-retirada").exists()


def test_un_archivo_de_mas_dentro_de_una_skill_declarada_tambien_se_borra(tmp_path):
    etapa = tmp_path / "etapa"
    (etapa / "skill-a").mkdir(parents=True)
    (etapa / "skill-a" / "SKILL.md").write_text("a")

    destino = tmp_path / "destino" / "skills"
    (destino / "skill-a").mkdir(parents=True)
    (destino / "skill-a" / "SKILL.md").write_text("a")
    (destino / "skill-a" / "sobra.md").write_text("archivo de mas, dejado a mano")

    r = _correr(tmp_path, etapa, destino)
    assert r.returncode == 0, r.stderr
    assert _arbol(destino) == {"skill-a/SKILL.md": b"a"}


def test_etapa_vacia_deja_el_destino_vacio(tmp_path):
    etapa = tmp_path / "etapa"
    etapa.mkdir()
    destino = tmp_path / "destino" / "skills"
    (destino / "skill-vieja").mkdir(parents=True)
    (destino / "skill-vieja" / "SKILL.md").write_text("x")

    r = _correr(tmp_path, etapa, destino)
    assert r.returncode == 0, r.stderr
    assert _arbol(destino) == {}
    assert destino.is_dir()


# --- control: la copia archivo-por-archivo, sin reemplazo, SÍ deja basura -------------
# (documenta el defecto real que M4 encontró en instalar_contexto.sh -- sin este
# control no se sabe si el test de arriba prueba algo o si cualquier cosa lo pasaría)

def test_control_la_copia_ingenua_archivo_por_archivo_no_borra_lo_que_sobra(tmp_path):
    etapa = tmp_path / "etapa"
    (etapa / "skill-nueva").mkdir(parents=True)
    (etapa / "skill-nueva" / "SKILL.md").write_text("nueva")

    destino = tmp_path / "destino" / "skills"
    (destino / "skill-retirada").mkdir(parents=True)
    (destino / "skill-retirada" / "SKILL.md").write_text("vieja")

    destino.mkdir(parents=True, exist_ok=True)
    for p in etapa.rglob("*"):
        if p.is_file():
            rel = p.relative_to(etapa)
            (destino / rel).parent.mkdir(parents=True, exist_ok=True)
            (destino / rel).write_bytes(p.read_bytes())

    # Justo el defecto que M4 encontró: "skill-retirada" sigue ahí.
    assert (destino / "skill-retirada").exists()
    assert _arbol(destino) != _arbol(etapa)


# --- reemplazo REALMENTE atómico, no dos `mv` (ronda 3, auditoría adversarial 2026-09-22) --

def test_usa_exch_no_dos_mv_separados_cuando_destino_ya_existe():
    """Regresión: la versión anterior hacía `mv DESTINO VIEJO` y DESPUÉS `mv NUEVO
    DESTINO` -- entre esos dos hay una VENTANA real en la que DESTINO no existe. `exch`
    (envuelve `renameat2(RENAME_EXCHANGE)`) intercambia los dos nombres en un solo
    syscall: cero ventana. Este test es el freno contra que alguien vuelva a partir el
    swap en dos pasos sin darse cuenta de por qué importa."""
    texto = SCRIPT.read_text(encoding="utf-8")
    assert "exch " in texto, "el script ya no usa exch -- ¿volvió el doble mv con ventana?"
    # Cuando DESTINO existe, el ÚNICO mv/exch de esa rama tiene que ser el exch -- no un
    # `mv "$DESTINO" "$VIEJO"` seguido de otro `mv` (eso es, literalmente, la ventana).
    rama_destino_existe = texto.split('if sudo test -e "$DESTINO"; then', 1)[1].split("else", 1)[0]
    assert 'mv -T "$DESTINO"' not in rama_destino_existe, "sigue moviendo DESTINO aparte: eso es la ventana"


def test_exch_esta_disponible_en_este_runner():
    """Si `exch` no está, el script se niega explícito (`exch_no_disponible`), no
    calla ni degrada solo a un doble mv con ventana -- lo prueba el propio script:"""
    r = subprocess.run(["bash", "-c", "command -v exch"], capture_output=True, text=True)
    assert r.returncode == 0, "exch (util-linux) no está instalado en este runner"
