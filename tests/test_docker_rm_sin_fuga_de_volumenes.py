"""`docker rm -f` sin `-v` deja huérfano el volumen anónimo del contenedor.

La imagen de MariaDB declara `VOLUME /var/lib/mysql`. Un contenedor arrancado con
`--rm` y borrado con `docker rm -f` (sin `-v`) deja su datadir como volumen anónimo:
el 2026-10-04 había 343 así en hall9000 (56 GB, 22 con copias de `jax_memory`).
Medido ese día: `rm -f` → +1 volumen huérfano; `rm -fv` → 0. `-v` solo borra los
anónimos: un volumen con nombre sobrevive (probado en la auditoría del PR #353).

Cómo barre (segunda ronda de auditoría del PR #353: los regex por línea eran una
persecución sin fin):
- Python, con `ast`: listas y tuplas con el elemento "rm" (banderas en cualquier
  posición después), concatenaciones `docker + ["rm", ...]`, f-strings y cadenas que
  son un comando de shell. Comentarios y docstrings no cuentan.
- Shell, con un tokenizador: líneas partidas con `\\` unidas, comentarios fuera,
  `docker`/`docker container`/opciones globales/variables con «docker» en el nombre,
  y TODAS las banderas hasta el fin del comando (docker las acepta después del nombre).
- `docker exec|run ... rm -f x` es un `rm` DENTRO del contenedor: no cuenta.

Límites declarados: una variable que no diga «docker» en su nombre (`$D rm -f`) no se
reconoce como docker; tampoco se sigue flujo indirecto de prefijos Python neutrales
(`base = docker; base + ["rm", ...]`). Se prefiere eso a marcar comandos ajenos como
`git + ["rm", ...]` o cada `$SUDO rm -f archivo`.
"""
import ast
import re
import shlex
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
ESTE = "tests/test_docker_rm_sin_fuga_de_volumenes.py"

_ES_DOCKER = re.compile(r"(?:\S*/)?docker|\$\{?\w*docker\w*(?:\[@\])?\}?|\{[^{}]*docker[^{}]*\}|\bdk\b", re.I)
_SEPARADORES = {";", "&&", "||", "|", "&", "(", ")", "`", "$("}
_FALSO = {"false", "0", "f", "no"}
_DOCKER_OPCIONES_SIN_VALOR = {"-D", "--debug", "--tls", "--tlsverify"}
_DOCKER_OPCIONES_CON_VALOR = {
    "-c", "--context", "-H", "--host", "-l", "--log-level", "--config",
    "--tlscacert", "--tlscert", "--tlskey",
}


def _fuerza_sin_volumenes(banderas):
    """True si entre las banderas hay force y no hay volumes (orden libre)."""
    fuerza = volumenes = False
    for b in banderas:
        if b.startswith("--"):
            nombre, _, valor = b[2:].partition("=")
            activa = valor.lower() not in _FALSO if valor else True
            if nombre == "force":
                fuerza = activa
            elif nombre == "volumes":
                volumenes = activa
        elif b.startswith("-") and len(b) > 1:
            fuerza |= "f" in b[1:]
            volumenes |= "v" in b[1:]
    return fuerza and not volumenes


def _tokens_shell(linea):
    lexer = shlex.shlex(linea, posix=True, punctuation_chars=";&|()<>`")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        return list(lexer)
    except ValueError:
        # Malformed quoting should not hide otherwise recognizable text.
        linea = re.sub(r"(^|\s)#.*$", "", linea)
        linea = re.sub(r"&&|\|\||\$\(|[;|`]", r" \g<0> ", linea)
        return linea.split()


def _es_docker_rm(toks, indice_rm):
    """Reconoce docker [opciones] [container] rm, no otros subcomandos rm."""
    inicio = indice_rm - 1
    while inicio >= 0 and toks[inicio] not in _SEPARADORES:
        inicio -= 1
    for j in range(inicio + 1, indice_rm):
        if _ES_DOCKER.fullmatch(toks[j]):
            # docker exec/run ejecuta un comando dentro de un contenedor; no
            # reinterpretar un contenedor llamado "docker" como el ejecutable.
            if j + 1 < indice_rm and toks[j + 1] in ("exec", "run"):
                return False
            if _prefijo_docker_rm(toks[j + 1:indice_rm]):
                return True
    return False


def culpables_shell(texto):
    hallados = []
    unido = re.sub(r"\\\r?\n\s*", " ", texto)
    for linea in unido.splitlines():
        toks = _tokens_shell(linea)
        for i, t in enumerate(toks):
            if t != "rm":
                continue
            if not _es_docker_rm(toks, i):
                continue
            banderas = []
            for siguiente in toks[i + 1:]:
                if siguiente in _SEPARADORES:
                    break
                if siguiente.startswith("-"):
                    banderas.append(siguiente)
            if _fuerza_sin_volumenes(banderas):
                hallados.append(" ".join(toks[max(0, i - 3):i + 1 + len(banderas) + 1]))
    return hallados


def _texto_de(nodo, fuente):
    return ast.get_source_segment(fuente, nodo) or ""


def _prefijo_docker_rm(tokens):
    comando = None
    k = 0
    while k < len(tokens):
        token = tokens[k]
        if token in _DOCKER_OPCIONES_SIN_VALOR or token.startswith("--") and "=" in token:
            k += 1
        elif token in _DOCKER_OPCIONES_CON_VALOR:
            if k + 1 >= len(tokens):
                return False
            k += 2
        elif any(token.startswith(op) and len(token) > len(op) for op in ("-c", "-H", "-l")):
            k += 1
        elif token == "container" and comando is None:
            comando = token
            k += 1
        elif token == "compose" and comando is None:
            comando = token
            k += 1
        elif comando == "compose":
            # Compose acepta opciones globales en evolución, incluidas opciones
            # con argumentos y formas adjuntas. El `rm` encontrado después por
            # el escáner ya fija el subcomando de destino.
            return True
        else:
            return False
    return True


def _es_prefijo_docker_rm(tokens):
    for i, token in enumerate(tokens):
        if not _ES_DOCKER.search(token):
            continue
        if i + 1 < len(tokens) and tokens[i + 1] in ("exec", "run"):
            return False
        if _prefijo_docker_rm(tokens[i + 1:]):
            return True
    return False


def culpables_python(fuente, filename="<string>"):
    arbol = ast.parse(fuente, filename=filename)
    padres = {hijo: nodo for nodo in ast.walk(arbol) for hijo in ast.iter_child_nodes(nodo)}
    docstrings = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Expr) and isinstance(nodo.value, ast.Constant) and isinstance(nodo.value.value, str):
            docstrings.add(nodo.value)
    hallados = []
    for nodo in ast.walk(arbol):
        if isinstance(nodo, (ast.List, ast.Tuple)):
            elts = nodo.elts
            indices = [k for k, e in enumerate(elts) if isinstance(e, ast.Constant) and e.value == "rm"]
            if not indices:
                continue
            i = indices[0]
            antes = elts[:i]
            tokens = [e.value if isinstance(e, ast.Constant) and isinstance(e.value, str)
                      else _texto_de(e, fuente) for e in antes]
            padre = padres.get(nodo)
            if isinstance(padre, ast.BinOp) and isinstance(padre.op, ast.Add) and padre.right is nodo:
                tokens.insert(0, _texto_de(padre.left, fuente))
            if not _es_prefijo_docker_rm(tokens):
                continue
            banderas = [e.value for e in elts[i + 1:]
                        if isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value.startswith("-")]
            if _fuerza_sin_volumenes(banderas):
                hallados.append(" ".join(_texto_de(nodo, fuente).split()))
        elif isinstance(nodo, ast.JoinedStr):
            partes = [p.value if isinstance(p, ast.Constant) else "{" + _texto_de(p.value, fuente) + "}"
                      for p in nodo.values]
            hallados += culpables_shell("".join(str(x) for x in partes))
        elif isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) and nodo not in docstrings \
                and not isinstance(padres.get(nodo), ast.JoinedStr) and " rm" in nodo.value:
            hallados += culpables_shell(nodo.value)
    return hallados


def culpables_en_texto(texto, es_python=None, filename="<string>"):
    if es_python is None:
        es_python = not texto.lstrip().startswith("#!") or "python" in texto.split("\n", 1)[0]
        if es_python:
            return culpables_python(texto, filename=filename)
    return culpables_python(texto, filename=filename) if es_python else culpables_shell(texto)


def _archivos():
    salida = subprocess.run(["git", "ls-files", "-z"], cwd=RAIZ,
                            capture_output=True, check=True).stdout
    for nombre in salida.decode("utf-8").split("\0"):
        if not nombre or nombre == ESTE:
            continue
        ruta = RAIZ / nombre
        if not ruta.is_file() or ruta.is_symlink():
            continue
        if ruta.suffix in (".sh", ".bash"):
            yield ruta, False
            continue
        if ruta.suffix == ".py":
            yield ruta, True
            continue
        try:
            with open(ruta, "rb") as f:
                primera = f.readline(200)
        except OSError:
            continue
        if primera.startswith(b"#!"):
            yield ruta, b"python" in primera


def test_ningun_docker_rm_forzado_sin_borrar_volumenes():
    culpables, ilegibles, errores_sintaxis = [], [], []
    for ruta, es_python in _archivos():
        try:
            texto = ruta.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            ilegibles.append(str(ruta.relative_to(RAIZ)))
            continue
        try:
            hallazgos = culpables_en_texto(texto, es_python, filename=str(ruta))
        except SyntaxError as error:
            relativo = ruta.relative_to(RAIZ)
            errores_sintaxis.append(
                f"{relativo}:{error.lineno or 0}:{error.offset or 0}: {error.msg}"
            )
            continue
        for hallado in hallazgos:
            culpables.append(f"{ruta.relative_to(RAIZ)}: {hallado}")
    assert not ilegibles, "guiones versionados que no son UTF-8 (no se pueden barrer):\n" + "\n".join(ilegibles)
    assert not errores_sintaxis, "archivos Python que no parsean (no se pueden barrer):\n" + "\n".join(errores_sintaxis)
    assert not culpables, "docker rm forzado sin -v (deja el volumen anónimo):\n" + "\n".join(culpables)


FUGAN_SHELL = [
    'sudo docker rm -f "$X"',
    "docker rm --force x",
    "docker rm --force=true c",
    "docker container rm -f x",
    "$DOCKER rm -f x",
    '"${DOCKER[@]}" rm -f x',
    "${DOCKER_CMD} rm -f x",
    "docker -H unix:///x rm -f c",
    "docker --context=x rm -f c",
    "sudo -E docker rm -f c",
    "xargs docker rm -f",
    "sudo -n docker rm \\\n   -f \"$C\"",
    "sudo -n docker rm \\\r\n   -f \"$C\"",
    'docker rm "$C" -f',
    "docker rm x --force",
    '"docker" rm -f c',
    "docker rm -f c --volumes=0",
    "docker rm -f c --volumes=f",
    "docker compose -f x.yml rm -f",
    "docker compose -p proj rm -f svc",
    "docker compose --file x.yml rm -f svc",
    "docker compose --profile prod rm -f svc",
    "docker compose --env-file .env rm -f svc",
    "docker compose -fcompose.yml rm -f svc",
    "docker compose -pproj rm -f svc",
    "docker compose -f a.yml -f b.yml rm -fs svc",
    "docker compose -f a -f b -f c -f d -f e rm -f svc",
    "sudo -u docker docker rm -f x",
    "sudo -g docker docker rm -f x",
    "docker -Htcp://x rm -f x",
    "docker -H=tcp://x rm -f x",
    "docker -l=debug rm -f x",
    "docker -cctx rm -f x",
    "docker rm x 2>&1 -f",
    "docker rm x >&2 --force",
    "docker rm x &>log -f",
    "docker compose rm -f",
    "docker -c ctx rm -f x",
    "docker -D rm -f x",
    "docker -l debug rm -f x",
    "docker --log-level debug rm -f x",
    "docker --tlsverify --tlscacert ca --tlscert cert --tlskey key rm -f x",
    "docker --context ctx rm -f x",
    "docker --context network rm -f x",
    "docker --host unix:///x rm -f x",
    "docker --config /cfg rm -f x",
    "docker --tls rm -f x",
    "docker --tlsverify rm -f x",
    "docker rm -f x -v & docker rm -f y",
    "docker rm -f x -v& docker rm -f y",
]
FUGAN_PYTHON = [
    '["sudo", "docker", "rm", "-f", n]',
    '["docker", "-c", "ctx", "rm", "-f", n]',
    '["docker", "--context", "ctx", "rm", "-f", n]',
    '["docker", "--context", "network", "rm", "-f", n]',
    '["docker", "-H", "unix:///x", "rm", "-f", n]',
    '["docker", "--host", "unix:///x", "rm", "-f", n]',
    '["docker", "-l", "debug", "rm", "-f", n]',
    '["docker", "--log-level", "debug", "rm", "-f", n]',
    '["docker", "--config", "/cfg", "rm", "-f", n]',
    '["docker", "--tlscacert", "ca", "--tlscert", "cert", "--tlskey", "key", "rm", "-f", n]',
    '[*docker, "rm", "-f", nombre]',
    '[*dk, "rm", "-f", n]',
    '[*docker, "rm", n, "-f"]',
    '[*docker, "rm", "--force", n]',
    'DOCKER + ["rm", "-f", n]',
    'docker + ["rm", "-f", nombre]',
    'docker_cmd + ["rm", "-f", n]',
    'DOCKER_CMD + ["rm", "-f", n]',
    'self.docker + ["rm", "-f", n]',
    '[*docker,\n    "rm",\n    "-f",\n    nombre]',
    '[*docker, "rm", "-f", nombres[0]]',
    '[*docker, "rm", "-f", c["nombre"]]',
    'subprocess.run(("docker", "rm", "-f", n))',
    'subprocess.run(f"{docker} rm -f {n}", shell=True)',
    'subprocess.run("docker rm -f x", shell=True)',
    'subprocess.run(f"docker compose -f {yml} rm -f", shell=True)',
    'subprocess.run(f"sudo -u docker docker rm -f {n}", shell=True)',
    '["docker", "compose", "-f", "x.yml", "rm", "-f"]',
    '["docker", "compose", "-p", p, "rm", "-f", "-s"]',
    '["docker", "compose", "--profile", "prod", "rm", "-f"]',
    '["docker", "compose", "--env-file", ".env", "rm", "-f"]',
    '["docker", "compose", "-fcompose.yml", "rm", "-f"]',
    '["docker", "compose", "-pproj", "rm", "-f"]',
    '["docker", "compose", "-f", "a", "-f", "b", "-f", "c", "-f", "d", "-f", "e", "rm", "-f"]',
    '["ssh", "docker-host", "docker", "rm", "-f", n]',
    '["ssh", docker_host, "docker", "rm", "-f", n]',
    '["sudo", "-u", "docker", "docker", "rm", "-f", n]',
    '["env", "DOCKER_HOST=x", "docker", "rm", "-f", n]',
    '[*sudo, "DOCKER_HOST=x", "docker", "rm", "-f", n]',
    '["docker", "-H=tcp://x", "rm", "-f", n]',
    '[*docker, "container", "rm", "-f", n]',
    'docker + ["container", "rm", "-f", n]',
]
NO_FUGAN_SHELL = [
    'sudo docker rm -fv "$X"',
    'sudo docker rm -vf "$X"',
    "docker rm -f -v x",
    "docker rm -v -f x",
    "docker rm -f --volumes x",
    "docker rm x",
    "docker rm --force=false x",
    "docker rm --force=0 x",
    'rm -f "$TMP/archivo"',
    '$SUDO rm -f "$TMP/archivo"',
    "docker stop x",
    "docker exec c rm -f /tmp/x",
    "docker exec docker rm -f /tmp/x",
    "# nunca uses docker rm -f",
    "docker stop x; rm -f /tmp/y",
    "docker volume rm -f x",
    "docker image rm -f x",
    "docker network rm -f x",
    "docker rm x & rm -f /tmp/y",
    "docker rm x& rm -f /tmp/y",
]
NO_FUGAN_PYTHON = [
    '[*docker, "rm", "-fv", nombre]',
    '[*docker, "rm", "-f", "-v", n]',
    '["sudo", "-n", "rm", "-f", str(ruta)]',
    'sudo + ["rm", "-f", p]',
    '[*docker, "exec", c, "rm", "-f", "/tmp/x"]',
    '["docker", "exec", "docker", "rm", "-f", "/tmp/x"]',
    '# nunca uses docker rm -f\nx = 1',
    'def f():\n    """No uses docker rm -f."""\n    return 1',
    'DOCKER = 1\nsubprocess.run(["rm", "-f", p])',
    'git + ["rm", "-f", p]',
    'sdk + ["rm", "-f", p]',
    'base + ["rm", "-f", p]',
    '["docker", "volume", "rm", "-f", v]',
    '[*docker, "network", "rm", "-f", v]',
    '[*docker, "image", "rm", "-f", v]',
    '["docker", "-c", "rm", "-f", v]',
    'subprocess.run("docker rm x & rm -f /tmp/y", shell=True)',
]


@pytest.mark.parametrize("caso", FUGAN_SHELL)
def test_detecta_fugas_shell(caso):
    assert culpables_en_texto(caso, es_python=False), f"no detectó (shell): {caso!r}"


@pytest.mark.parametrize("caso", FUGAN_PYTHON)
def test_detecta_fugas_python(caso):
    assert culpables_en_texto(caso, es_python=True), f"no detectó (python): {caso!r}"


def test_deja_pasar_lo_que_no_fuga():
    for caso in NO_FUGAN_SHELL:
        assert not culpables_en_texto(caso, es_python=False), \
            f"falso positivo (shell): {caso!r} -> {culpables_en_texto(caso, es_python=False)}"
    for caso in NO_FUGAN_PYTHON:
        assert not culpables_en_texto(caso, es_python=True), \
            f"falso positivo (python): {caso!r} -> {culpables_en_texto(caso, es_python=True)}"


def test_error_de_sintaxis_python_se_informa_separado_con_ruta_linea_y_mensaje(monkeypatch, tmp_path):
    roto = tmp_path / "broken.py"
    roto.write_text("def roto(:\n", encoding="utf-8")
    monkeypatch.setattr(__import__(__name__), "_archivos", lambda: iter([(roto, True)]))
    monkeypatch.setattr(__import__(__name__), "RAIZ", tmp_path)
    with pytest.raises(AssertionError) as error:
        test_ningun_docker_rm_forzado_sin_borrar_volumenes()
    mensaje = str(error.value)
    assert "broken.py:1:" in mensaje
    assert "invalid syntax" in mensaje or "expected" in mensaje
    assert "no son UTF-8" not in mensaje


def test_python_que_no_parsea_falla_en_vez_de_degradar_a_shell():
    with pytest.raises(SyntaxError):
        culpables_en_texto("def roto(:\n    docker rm -f x", es_python=True)


def test_python_inferido_que_no_parsea_no_degrada_a_shell():
    with pytest.raises(SyntaxError):
        culpables_en_texto("def roto(:\n    docker rm -f x")


def test_python_ilegible_informa_la_ruta():
    with pytest.raises(SyntaxError) as error:
        culpables_python("def roto(:", filename="ruta/con error.py")
    assert error.value.filename == "ruta/con error.py"
