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
_COMPOSE_OPCIONES_CON_VALOR = {
    "-f", "--file", "-p", "--project-name", "--env-file", "--profile",
    "--parallel", "--progress", "--ansi", "--project-directory",
}


def _compose_exec_o_run(tokens):
    try:
        k = tokens.index("compose") + 1
    except ValueError:
        return False
    while k < len(tokens):
        token = tokens[k]
        if token in _COMPOSE_OPCIONES_CON_VALOR:
            k += 2
        elif token.startswith("--") and "=" in token:
            k += 1
        elif token.startswith(("-f", "-p")) and len(token) > 2:
            k += 1
        elif token.startswith("-"):
            k += 1
        else:
            return token in ("exec", "run")
    return False


def _docker_exec_o_run(tokens):
    k = 0
    opciones = True
    while k < len(tokens):
        token = tokens[k]
        if opciones and token == "--":
            opciones = False
            k += 1
        elif opciones and (token in _DOCKER_OPCIONES_SIN_VALOR or token.startswith("--") and "=" in token):
            k += 1
        elif opciones and token.startswith(("-D=", "-v=")):
            k += 1
        elif opciones and token in _DOCKER_OPCIONES_CON_VALOR:
            k += 2
        elif opciones and any(token.startswith(op) and len(token) > len(op) for op in ("-c", "-H", "-l")):
            k += 1
        elif token == "container":
            k += 1
        else:
            return token in ("exec", "run")
    return False


def _indices_hosts_ssh(tokens, inicio=0):
    """Todos los hosts SSH en un comando y sus comandos remotos anidados."""
    opciones_con_valor = set("bBcDEeF IiJLMlmOoPpQRSWw".replace(" ", ""))
    valores_wrapper = {"-u", "-g", "-h", "-p", "-r", "-t", "-C", "-T", "-D", "-R"}
    hosts = set()
    i = inicio
    while i < len(tokens):
        if tokens[i] != "ssh" or i > inicio and tokens[i - 1] in valores_wrapper:
            i += 1
            continue
        k = i + 1
        while k < len(tokens):
            token = tokens[k]
            if token == "--":
                k += 1
                break
            if not token.startswith("-") or token == "-":
                break
            grupo = token[1:]
            necesita_valor = False
            for posicion, opcion in enumerate(grupo):
                if opcion not in opciones_con_valor:
                    continue
                necesita_valor = True
                k += 1 if posicion < len(grupo) - 1 else 2
                break
            if not necesita_valor:
                k += 1
        if k >= len(tokens):
            break
        hosts.add(k)
        # Continue after the host to discover nested SSH in the remote command.
        i = k + 1
    return hosts


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
            cortas, tiene_valor, valor = b[1:].partition("=")
            activa = valor.lower() not in _FALSO if tiene_valor else True
            for indice, opcion in enumerate(cortas):
                valor_opcion = activa if tiene_valor and indice == len(cortas) - 1 else True
                if opcion == "f":
                    fuerza = valor_opcion
                elif opcion == "v":
                    volumenes = valor_opcion
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


def _es_docker_rm(toks, indice_rm, subcomando):
    """Reconoce docker [opciones] [container] rm, no otros subcomandos rm."""
    inicio = indice_rm - 1
    while inicio >= 0 and toks[inicio] not in _SEPARADORES:
        inicio -= 1
    ssh_hosts = _indices_hosts_ssh(toks, inicio + 1)
    for j in range(inicio + 1, indice_rm):
        if _ES_DOCKER.fullmatch(toks[j]):
            if subcomando == "remove" and "container" not in toks[j + 1:indice_rm]:
                continue
            if j in ssh_hosts:
                # `ssh docker exec docker rm` has a host literally named docker;
                # keep searching for the remote executable anchor.
                continue
            if _compose_exec_o_run(toks[j + 1:indice_rm]):
                return False
            # Docker global options may precede exec/run; do not reinterpret
            # the following container name as another Docker executable.
            if _docker_exec_o_run(toks[j + 1:indice_rm]):
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
            if t not in ("rm", "remove"):
                continue
            if not _es_docker_rm(toks, i, t):
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
    opciones_docker = True
    while k < len(tokens):
        token = tokens[k]
        if opciones_docker and token == "--":
            opciones_docker = False
            k += 1
        elif opciones_docker and (token in _DOCKER_OPCIONES_SIN_VALOR or token.startswith("--") and "=" in token):
            k += 1
        elif opciones_docker and token.startswith(("-D=", "-v=")):
            k += 1
        elif opciones_docker and token in _DOCKER_OPCIONES_CON_VALOR:
            if k + 1 >= len(tokens):
                return False
            k += 2
        elif opciones_docker and any(token.startswith(op) and len(token) > len(op) for op in ("-c", "-H", "-l")):
            k += 1
        elif token == "container" and comando is None:
            comando = token
            k += 1
        elif token == "compose" and comando is None:
            comando = token
            k += 1
        elif comando == "compose":
            # Consumir opciones globales antes de que el llamador confirme que
            # `rm` es el subcomando; un positional como `exec`/`run` significa
            # que el rm encontrado pertenece al comando dentro del servicio.
            if token in _COMPOSE_OPCIONES_CON_VALOR:
                if k + 1 >= len(tokens):
                    return False
                k += 2
            elif token.startswith("--") and "=" in token:
                k += 1
            elif token.startswith(("-f", "-p")) and len(token) > 2:
                k += 1
            elif token.startswith("-"):
                # Las opciones booleanas no tienen argumento posicional.
                k += 1
            else:
                return False
        else:
            return False
    return True


def _es_prefijo_docker_rm(tokens):
    ssh_hosts = _indices_hosts_ssh(tokens)
    for i, token in enumerate(tokens):
        if not _ES_DOCKER.search(token):
            continue
        if i in ssh_hosts:
            continue
        if _compose_exec_o_run(tokens[i + 1:]):
            return False
        if _ES_DOCKER.fullmatch(token) and _docker_exec_o_run(tokens[i + 1:]):
            # `docker-host` can be an SSH destination before the real docker
            # executable (`ssh docker-host exec docker rm ...`). Only an exact
            # executable anchor closes the search as docker exec/run.
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
            indices = [k for k, e in enumerate(elts)
                       if isinstance(e, ast.Constant) and e.value in ("rm", "remove")]
            if not indices:
                continue
            for i in indices:
                antes = elts[:i]
                tokens = [e.value if isinstance(e, ast.Constant) and isinstance(e.value, str)
                          else _texto_de(e, fuente) for e in antes]
                if elts[i].value == "remove" and "container" not in tokens:
                    continue
                padre = padres.get(nodo)
                if isinstance(padre, ast.BinOp) and isinstance(padre.op, ast.Add) and padre.right is nodo:
                    tokens.insert(0, _texto_de(padre.left, fuente))
                if not _es_prefijo_docker_rm(tokens):
                    continue
                banderas = [e.value for e in elts[i + 1:]
                            if isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value.startswith("-")]
                if _fuerza_sin_volumenes(banderas):
                    hallados.append(" ".join(_texto_de(nodo, fuente).split()))
                    break
        elif isinstance(nodo, ast.JoinedStr):
            partes = [p.value if isinstance(p, ast.Constant) else "{" + _texto_de(p.value, fuente) + "}"
                      for p in nodo.values]
            hallados += culpables_shell("".join(str(x) for x in partes))
        elif isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) and nodo not in docstrings \
                and not isinstance(padres.get(nodo), ast.JoinedStr) and (" rm" in nodo.value or " remove" in nodo.value):
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
    "docker -D=false rm -f c",
    "docker -D=0 rm -f c",
    "docker -v=false rm -f c",
    "docker -v=0 rm -f c",
    "docker -- rm -f c",
    "docker -- container rm -f c",
    "docker container remove -f c",
    "ssh docker exec docker rm -f c",
    "ssh -p 22 docker exec docker rm -f c",
    "ssh -o BatchMode=yes docker exec docker rm -f c",
    "ssh rm docker rm -f c",
    "ssh -B lo docker exec docker rm -f c",
    "ssh -I none docker exec docker rm -f c",
    "ssh -P audit docker exec docker rm -f c",
    "ssh -4p 22 docker exec docker rm -f c",
    "ssh -vp 22 docker exec docker rm -f c",
    "ssh -46p 22 docker exec docker rm -f c",
    "ssh gateway ssh docker exec docker rm -f c",
    "ssh gateway exec ssh docker exec docker rm -f c",
    "sudo -u rm docker rm -f c",
    "sudo -E docker rm -f c",
    "xargs docker rm -f",
    "sudo -n docker rm \\\n   -f \"$C\"",
    "sudo -n docker rm \\\r\n   -f \"$C\"",
    'docker rm "$C" -f',
    "docker rm x --force",
    '"docker" rm -f c',
    "docker rm -f c --volumes=0",
    "docker rm -f -v=false c",
    "docker rm -fv=false c",
    "docker rm -f -v=0 c",
    "docker rm -fv=0 c",
    "docker compose rm -f -v=false svc",
    "docker compose rm -f -v=0 svc",
    "docker rm -f c --volumes=f",
    "docker compose -f x.yml rm -f",
    "docker compose -p proj rm -f svc",
    "docker compose --file x.yml rm -f svc",
    "docker compose --profile prod rm -f svc",
    "docker compose --env-file .env rm -f svc",
    "docker compose -fcompose.yml rm -f svc",
    "docker compose -pproj rm -f svc",
    "docker compose --project-directory /tmp/project rm -f svc",
    "docker compose --parallel 4 rm -f svc",
    "docker compose --progress plain rm -f svc",
    "docker compose --ansi never rm -f svc",
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
    '["docker", "-D=false", "rm", "-f", n]',
    '["docker", "-D=0", "rm", "-f", n]',
    '["docker", "-v=false", "rm", "-f", n]',
    '["docker", "-v=0", "rm", "-f", n]',
    '["docker", "--", "rm", "-f", n]',
    '["docker", "--", "container", "rm", "-f", n]',
    '["docker", "container", "remove", "-f", n]',
    '["ssh", "docker", "exec", "docker", "rm", "-f", n]',
    '["ssh", "-p", "22", "docker", "exec", "docker", "rm", "-f", n]',
    '["ssh", "-o", "BatchMode=yes", "docker", "exec", "docker", "rm", "-f", n]',
    '["ssh", "rm", "docker", "rm", "-f", n]',
    '["ssh", "-B", "lo", "docker", "exec", "docker", "rm", "-f", n]',
    '["ssh", "-I", "none", "docker", "exec", "docker", "rm", "-f", n]',
    '["ssh", "-P", "audit", "docker", "exec", "docker", "rm", "-f", n]',
    '["ssh", "-4p", "22", "docker", "exec", "docker", "rm", "-f", n]',
    '["ssh", "-vp", "22", "docker", "exec", "docker", "rm", "-f", n]',
    '["ssh", "-46p", "22", "docker", "exec", "docker", "rm", "-f", n]',
    '["ssh", "gateway", "ssh", "docker", "exec", "docker", "rm", "-f", n]',
    '["ssh", "gateway", "exec", "ssh", "docker", "exec", "docker", "rm", "-f", n]',
    '["sudo", "-u", "rm", "docker", "rm", "-f", n]',
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
    'subprocess.run("docker container remove -f x", shell=True)',
    'cmd = "docker container remove --force x"',
    '["docker", "rm", "-f", "-v=false", n]',
    '["docker", "rm", "-fv=false", n]',
    '["docker", "rm", "-f", "-v=0", n]',
    '["docker", "rm", "-fv=0", n]',
    '["docker", "compose", "rm", "-f", "-v=false", svc]',
    '["docker", "compose", "rm", "-f", "-v=0", svc]',
    '["docker", "compose", "-f", "x.yml", "rm", "-f"]',
    '["docker", "compose", "-p", p, "rm", "-f", "-s"]',
    '["docker", "compose", "--profile", "prod", "rm", "-f"]',
    '["docker", "compose", "--env-file", ".env", "rm", "-f"]',
    '["docker", "compose", "-fcompose.yml", "rm", "-f"]',
    '["docker", "compose", "-pproj", "rm", "-f"]',
    '["docker", "compose", "--project-directory", "/tmp/project", "rm", "-f"]',
    '["docker", "compose", "--parallel", "4", "rm", "-f"]',
    '["docker", "compose", "--progress", "plain", "rm", "-f"]',
    '["docker", "compose", "--ansi", "never", "rm", "-f"]',
    '["docker", "compose", "-f", "a", "-f", "b", "-f", "c", "-f", "d", "-f", "e", "rm", "-f"]',
    '["ssh", "docker-host", "docker", "rm", "-f", n]',
    '["ssh", docker_host, "docker", "rm", "-f", n]',
    '["ssh", "docker-host", "exec", "docker", "rm", "-f", n]',
    '["ssh", docker_host, "exec", "docker", "rm", "-f", n]',
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
    "docker rm -f=false x",
    "docker rm -f -v=true x",
    "docker rm -fv=true x",
    'rm -f "$TMP/archivo"',
    '$SUDO rm -f "$TMP/archivo"',
    "docker stop x",
    "docker exec c rm -f /tmp/x",
    "docker exec docker rm -f /tmp/x",
    "docker container exec docker rm -f /tmp/x",
    "docker -- container exec docker rm -f /tmp/x",
    "docker --context ctx container exec docker rm -f /tmp/x",
    "docker --context ctx exec docker rm -f /tmp/x",
    "docker -H tcp://x exec docker rm -f /tmp/x",
    "docker --context=ctx run docker rm -f /tmp/x",
    "docker -D=false exec docker rm -f /tmp/x",
    "docker -v=false run docker rm -f /tmp/x",
    "docker -- exec docker rm -f /tmp/x",
    "docker -- run docker rm -f /tmp/x",
    "docker compose exec docker rm -f /tmp/x",
    "docker compose run svc docker rm -f /tmp/x",
    "docker compose --profile prod exec docker rm -f /tmp/x",
    "docker compose --env-file .env run svc docker rm -f /tmp/x",
    "# nunca uses docker rm -f",
    "docker stop x; rm -f /tmp/y",
    "docker volume rm -f x",
    "docker remove -f x",
    "docker compose remove -f svc",
    "docker image rm -f x",
    "docker network rm -f x",
    "docker rm x & rm -f /tmp/y",
    "docker rm x& rm -f /tmp/y",
]
NO_FUGAN_PYTHON = [
    '[*docker, "rm", "-fv", nombre]',
    '[*docker, "rm", "-f", "-v", n]',
    '["docker", "rm", "-f", "-v=true", n]',
    '["docker", "rm", "-fv=true", n]',
    '["docker", "rm", "-f=false", n]',
    '["sudo", "-n", "rm", "-f", str(ruta)]',
    'sudo + ["rm", "-f", p]',
    '[*docker, "exec", c, "rm", "-f", "/tmp/x"]',
    '["docker", "exec", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "container", "exec", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "--", "container", "exec", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "--context", "ctx", "container", "exec", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "--context", "ctx", "exec", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "-H", "tcp://x", "exec", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "--context=ctx", "run", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "-D=false", "exec", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "-v=false", "run", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "--", "exec", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "--", "run", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "compose", "exec", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "compose", "run", "svc", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "compose", "--profile", "prod", "exec", "docker", "rm", "-f", "/tmp/x"]',
    '["docker", "compose", "--env-file", ".env", "run", "svc", "docker", "rm", "-f", "/tmp/x"]',
    '# nunca uses docker rm -f\nx = 1',
    'def f():\n    """No uses docker rm -f."""\n    return 1',
    'DOCKER = 1\nsubprocess.run(["rm", "-f", p])',
    'git + ["rm", "-f", p]',
    'sdk + ["rm", "-f", p]',
    'base + ["rm", "-f", p]',
    '["docker", "volume", "rm", "-f", v]',
    '["docker", "remove", "-f", n]',
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
