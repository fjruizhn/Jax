"""`docker rm -f` sin `-v` deja huérfano el volumen anónimo del contenedor.

La imagen de MariaDB declara `VOLUME /var/lib/mysql`. Un contenedor arrancado con
`--rm` y borrado con `docker rm -f` (sin `-v`) deja su datadir como volumen anónimo:
el 2026-10-04 había 343 así en hall9000 (56 GB, 22 con copias de `jax_memory`).
Medido ese día: `rm -f` → +1 volumen huérfano; `rm -fv` → 0. `-v` solo borra los
anónimos: un volumen con nombre sobrevive (probado en la auditoría del PR #353).

El barrido no depende del orden de las banderas ni de la forma de llamar a docker:
une las líneas partidas con `\\`, entiende `--force`/`--volumes`, `docker container rm`,
`$DOCKER rm`, `"${DOCKER[@]}" rm` y listas de Python en cualquier orden. Barre todo
archivo de texto versionado que sea `.sh`/`.py`/`.bash` o tenga shebang.
"""
import re
import subprocess
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ESTE = "tests/test_docker_rm_sin_fuga_de_volumenes.py"

# Quién llama a docker en shell: `docker`, `docker container`, opciones globales
# (`docker -H x`), o una variable cuyo nombre lo diga ($DOCKER, ${DOCKER_CMD}, "${DOCKER[@]}").
_DOCKER_SHELL = (
    r"""(?:\bdocker\b(?:\s+-{1,2}[\w-]+(?:[=\s]+\S+)?)*?(?:\s+container)?"""
    r"""|["']?\$\{?[A-Za-z_]*[Dd][Oo][Cc][Kk][Ee][Rr][A-Za-z_]*(?:\[@\])?\}?["']?)"""
)
RM_SHELL = re.compile(_DOCKER_SHELL + r"""\s+rm((?:\s+-{1,2}[\w=-]+)*)""")
# Una lista de Python con el elemento "rm": se analizan sus elementos, en cualquier orden.
LISTA_PY = re.compile(r"""\[([^\[\]]*?["']rm["'][^\[\]]*?)\]""", re.S)
CADENA = re.compile(r"""["']([^"']*)["']""")
STAR_DOCKER = re.compile(r"""\*\s*\w*(?:docker|dk)\w*""", re.I)


def _fuerza_sin_volumenes(banderas):
    """True si entre las banderas hay force y no hay volumes (orden libre)."""
    fuerza = volumenes = False
    for b in banderas:
        if b.startswith("--"):
            nombre = b[2:].split("=", 1)
            if nombre[0] == "force":
                fuerza = len(nombre) == 1 or nombre[1].lower() != "false"
            elif nombre[0] == "volumes":
                volumenes = len(nombre) == 1 or nombre[1].lower() != "false"
        elif b.startswith("-"):
            fuerza |= "f" in b[1:]
            volumenes |= "v" in b[1:]
    return fuerza and not volumenes


def culpables_en_texto(texto):
    """Devuelve las apariciones de un `docker rm` forzado sin borrar volúmenes."""
    hallados = []
    unido = re.sub(r"\\\n\s*", " ", texto)  # líneas partidas con barra invertida
    for m in RM_SHELL.finditer(unido):
        if _fuerza_sin_volumenes(m.group(1).split()):
            hallados.append(m.group(0).strip())
    for m in LISTA_PY.finditer(texto):
        cuerpo = m.group(1)
        if not (re.search(r"docker", cuerpo, re.I) or STAR_DOCKER.search(cuerpo)
                or re.search(r"\bdk\b|\bDOCKER\b", texto[max(0, m.start() - 40):m.start()])):
            continue  # un `rm -f` de archivos, no de contenedores
        elementos = CADENA.findall(cuerpo)
        if "rm" not in elementos:
            continue
        tras_rm = elementos[elementos.index("rm") + 1:]
        if _fuerza_sin_volumenes([e for e in tras_rm if e.startswith("-")]):
            hallados.append(" ".join(m.group(0).split()))
    return hallados


def _archivos():
    salida = subprocess.run(["git", "ls-files", "-z"], cwd=RAIZ,
                            capture_output=True, check=True).stdout
    for nombre in salida.decode("utf-8").split("\0"):
        if not nombre or nombre == ESTE:
            continue
        ruta = RAIZ / nombre
        if not ruta.is_file() or ruta.is_symlink():
            continue
        if ruta.suffix in (".sh", ".py", ".bash"):
            yield ruta
            continue
        try:
            with open(ruta, "rb") as f:
                if f.read(2) == b"#!":
                    yield ruta
        except OSError:
            continue


def test_ningun_docker_rm_forzado_sin_borrar_volumenes():
    culpables, ilegibles = [], []
    for ruta in _archivos():
        try:
            texto = ruta.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            ilegibles.append(str(ruta.relative_to(RAIZ)))
            continue
        for hallado in culpables_en_texto(texto):
            culpables.append(f"{ruta.relative_to(RAIZ)}: {hallado}")
    assert not ilegibles, "guiones versionados que no son UTF-8 (no se pueden barrer):\n" + "\n".join(ilegibles)
    assert not culpables, "docker rm forzado sin -v (deja el volumen anónimo):\n" + "\n".join(culpables)


def test_detecta_las_formas_que_fugan():
    fugan = [
        'sudo docker rm -f "$X"',
        "docker rm --force x",
        "docker rm --force=true c",
        "docker container rm -f x",
        "$DOCKER rm -f x",
        '"${DOCKER[@]}" rm -f x',
        "${DOCKER_CMD} rm -f x",
        "docker -H unix:///x rm -f c",
        "sudo -n docker rm \\\n   -f \"$C\"",
        '[*docker, "rm", "-f", nombre]',
        '[*dk, "rm", "-f", n]',
        '[*docker, "rm", n, "-f"]',
        '[*docker, "rm", "--force", n]',
        'DOCKER + ["rm", "-f", n]',
        '[*docker,\n    "rm",\n    "-f",\n    nombre]',
    ]
    for caso in fugan:
        assert culpables_en_texto(caso), f"no detectó: {caso!r}"


def test_deja_pasar_lo_que_no_fuga():
    no_fugan = [
        'sudo docker rm -fv "$X"',
        'sudo docker rm -vf "$X"',
        "docker rm -f -v x",
        "docker rm -v -f x",
        "docker rm -f --volumes x",
        "docker rm x",
        "docker rm --force=false x",
        '[*docker, "rm", "-fv", nombre]',
        '[*docker, "rm", "-f", "-v", n]',
        '["sudo", "-n", "rm", "-f", str(ruta)]',
        'rm -f "$TMP/archivo"',
        "docker stop x",
    ]
    for caso in no_fugan:
        assert not culpables_en_texto(caso), f"falso positivo: {caso!r} -> {culpables_en_texto(caso)}"
