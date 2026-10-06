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

Límite declarado: una variable que no diga «docker» en su nombre (`$D rm -f`) no se
reconoce como docker. Se prefiere eso a marcar cada `$SUDO rm -f archivo`.
"""
import ast
import re
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
ESTE = "tests/test_docker_rm_sin_fuga_de_volumenes.py"

_ES_DOCKER = re.compile(r"(?:\S*/)?docker|\$\{?\w*docker\w*(?:\[@\])?\}?|\{[^{}]*docker[^{}]*\}|dk", re.I)
_SEPARADORES = {";", "&&", "||", "|", "(", ")", "`", "$("}
_SEPARADORES_LISTA = {";", "&&", "||", "|"}
_FALSO = {"false", "0", "f", "no"}
_SUBCOMANDOS_SIN_VOLUMEN_ANONIMO = {
    "volume", "network", "image", "context", "buildx", "plugin", "secret",
    "config", "node", "service", "stack",
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
    linea = re.sub(r"(^|\s)#.*$", "", linea)          # comentario
    for sep in ("&&", "||", "$(", ";", "|", "(", ")", "`"):
        linea = linea.replace(sep, f" {sep} ")
    return [t.strip("\"'") for t in linea.split()]


def culpables_shell(texto):
    hallados = []
    unido = re.sub(r"\\\r?\n\s*", " ", texto)
    for linea in unido.splitlines():
        toks = _tokens_shell(linea)
        for i, t in enumerate(toks):
            if t != "rm":
                continue
            if i >= 2 and toks[i - 2] == "docker" and toks[i - 1] in _SUBCOMANDOS_SIN_VOLUMEN_ANONIMO:
                continue
            # Hacia atrás, dentro del mismo comando: ¿lo llama docker?
            es_docker = dentro = False
            for j in range(i - 1, -1, -1):
                previo = toks[j]
                if previo in _SEPARADORES:
                    break
                if previo in ("exec", "run"):
                    dentro = True
                if _ES_DOCKER.fullmatch(previo):
                    es_docker = True
                    break
            if not es_docker or dentro:
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


def _comandos_de_lista(elts):
    """Pares (inicio, fin) de cada comando de una lista Python, cortada en los
    separadores de shell que aparecen como elemento literal (`&&`, `;`, `||`, `|`)."""
    tramos, inicio = [], 0
    for k, e in enumerate(elts):
        if isinstance(e, ast.Constant) and e.value in _SEPARADORES_LISTA:
            tramos.append((inicio, k))
            inicio = k + 1
    tramos.append((inicio, len(elts)))
    return tramos


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
            padre = padres.get(nodo)
            marcado = False
            for inicio, fin_cmd in _comandos_de_lista(elts):  # cada comando, por separado
                for i in indices:
                    if not inicio <= i < fin_cmd:
                        continue
                    antes = elts[inicio:i]
                    if i - inicio >= 2 and isinstance(elts[i - 1], ast.Constant) \
                            and elts[i - 1].value in _SUBCOMANDOS_SIN_VOLUMEN_ANONIMO \
                            and isinstance(elts[i - 2], ast.Constant) and elts[i - 2].value == "docker":
                        continue
                    if any(isinstance(e, ast.Constant) and e.value in ("exec", "run") for e in antes):
                        continue
                    es_docker = any(_ES_DOCKER.search(_texto_de(e, fuente)) for e in antes)
                    if not es_docker and i == 0 and isinstance(padre, ast.BinOp) and isinstance(padre.op, ast.Add) \
                            and padre.right is nodo:
                        izquierda = _texto_de(padre.left, fuente)
                        # `algo + ["rm", ...]`: un prefijo de comando. Docker salvo que sea claramente sudo.
                        es_docker = bool(_ES_DOCKER.search(izquierda)) or not re.search(r"sudo", izquierda, re.I)
                    if not es_docker:
                        continue
                    banderas = [e.value for e in elts[i + 1:fin_cmd]
                                if isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value.startswith("-")]
                    if _fuerza_sin_volumenes(banderas):
                        hallados.append(" ".join(_texto_de(nodo, fuente).split()))
                        marcado = True
                        break
                if marcado:
                    break
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
            errores_sintaxis.append(f"{relativo}:{error.lineno or 0}: {error.msg}")
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
]
FUGAN_PYTHON = [
    '[*docker, "rm", "-f", nombre]',
    '[*dk, "rm", "-f", n]',
    '[*docker, "rm", n, "-f"]',
    '[*docker, "rm", "--force", n]',
    'DOCKER + ["rm", "-f", n]',
    'docker + ["rm", "-f", nombre]',
    'docker_cmd + ["rm", "-f", n]',
    'DOCKER_CMD + ["rm", "-f", n]',
    'self.docker + ["rm", "-f", n]',
    'base + ["rm", "-f", n]',
    '[*docker,\n    "rm",\n    "-f",\n    nombre]',
    '[*docker, "rm", "-f", nombres[0]]',
    '[*docker, "rm", "-f", c["nombre"]]',
    'subprocess.run(("docker", "rm", "-f", n))',
    'subprocess.run(f"{docker} rm -f {n}", shell=True)',
    'subprocess.run("docker rm -f x", shell=True)',
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
    "# nunca uses docker rm -f",
    "docker stop x; rm -f /tmp/y",
]
NO_FUGAN_PYTHON = [
    '[*docker, "rm", "-fv", nombre]',
    '[*docker, "rm", "-f", "-v", n]',
    '["sudo", "-n", "rm", "-f", str(ruta)]',
    'sudo + ["rm", "-f", p]',
    '[*docker, "exec", c, "rm", "-f", "/tmp/x"]',
    '# nunca uses docker rm -f\nx = 1',
    'def f():\n    """No uses docker rm -f."""\n    return 1',
    'DOCKER = 1\nsubprocess.run(["rm", "-f", p])',
]


def test_detecta_las_formas_que_fugan():
    for caso in FUGAN_SHELL:
        assert culpables_en_texto(caso, es_python=False), f"no detectó (shell): {caso!r}"
    for caso in FUGAN_PYTHON:
        assert culpables_en_texto(caso, es_python=True), f"no detectó (python): {caso!r}"


def test_deja_pasar_lo_que_no_fuga():
    for caso in NO_FUGAN_SHELL:
        assert not culpables_en_texto(caso, es_python=False), \
            f"falso positivo (shell): {caso!r} -> {culpables_en_texto(caso, es_python=False)}"
    for caso in NO_FUGAN_PYTHON:
        assert not culpables_en_texto(caso, es_python=True), \
            f"falso positivo (python): {caso!r} -> {culpables_en_texto(caso, es_python=True)}"


@pytest.mark.parametrize("subcomando", sorted(_SUBCOMANDOS_SIN_VOLUMEN_ANONIMO))
def test_whitelist_shell_solo_subcomando_literal_inmediato(subcomando):
    assert not culpables_en_texto(f"docker {subcomando} rm -f recurso", es_python=False)


@pytest.mark.parametrize("subcomando", sorted(_SUBCOMANDOS_SIN_VOLUMEN_ANONIMO))
def test_whitelist_python_solo_lista_de_literales(subcomando):
    assert not culpables_en_texto(
        f'["docker", "{subcomando}", "rm", "-f", recurso]', es_python=True
    )


@pytest.mark.parametrize("con_ssh", [False, True], ids=["sin_ssh", "con_ssh"])
@pytest.mark.parametrize("subcomando", sorted(_SUBCOMANDOS_SIN_VOLUMEN_ANONIMO))
def test_whitelist_python_no_exime_un_rm_posterior_de_contenedor(subcomando, con_ssh):
    """Cada `rm` de la lista se evalúa por separado: el de la lista blanca se saltea,
    el `docker rm -f` de contenedor que viene después sigue vigilado."""
    prefijo = '"ssh", h, ' if con_ssh else ""
    fuente = (f'[{prefijo}"docker", "{subcomando}", "rm", "-f", v, "&&", '
              f'"docker", "rm", "-f", c]')
    assert culpables_en_texto(fuente, es_python=True), f"no detectó: {fuente}"


_SEPS_LISTA = ["&&", ";", "||", "|"]


def _lista(*tramos, con_ssh):
    """Arma el texto de una lista Python con los tramos unidos tal cual (ya con comillas)."""
    prefijo = '"ssh", h, ' if con_ssh else ""
    return "[" + prefijo + ", ".join(tramos) + "]"


@pytest.mark.parametrize("con_ssh", [False, True], ids=["sin_ssh", "con_ssh"])
@pytest.mark.parametrize("sep", _SEPS_LISTA)
@pytest.mark.parametrize("rm_docker", ['"docker", "rm", "-fv", c', '"docker", "rm", "-v", c'])
@pytest.mark.parametrize("despues", ['"rm", "-rf", "/tmp/build"', '"sudo", "rm", "-f", lock', '"rm", "-f", p'])
def test_rm_de_host_tras_un_docker_rm_con_v_no_es_falso_positivo(con_ssh, sep, rm_docker, despues):
    """Cada comando de la lista se evalúa solo: las banderas del `rm` de host que viene
    después del separador no son las del `docker rm -v`."""
    fuente = _lista(rm_docker, f'"{sep}"', despues, con_ssh=con_ssh)
    assert not culpables_en_texto(fuente, es_python=True), f"falso positivo: {fuente}"


@pytest.mark.parametrize("con_ssh", [False, True], ids=["sin_ssh", "con_ssh"])
@pytest.mark.parametrize("sep", _SEPS_LISTA)
def test_docker_rm_f_no_se_exime_por_las_banderas_del_docker_rm_fv_siguiente(con_ssh, sep):
    """Las banderas de un comando no se leen hasta el final de la lista: el `-fv` del
    segundo `docker rm` no tapa al primero."""
    fuente = _lista('"docker", "rm", "-f", c1', f'"{sep}"', '"docker", "rm", "-fv", c2', con_ssh=con_ssh)
    assert culpables_en_texto(fuente, es_python=True), f"no detectó: {fuente}"


@pytest.mark.parametrize("con_ssh", [False, True], ids=["sin_ssh", "con_ssh"])
@pytest.mark.parametrize("sep", _SEPS_LISTA)
@pytest.mark.parametrize("antes", ['"docker", "run", img', '"docker", "exec", c, "true"'])
def test_exec_o_run_de_un_comando_no_exime_al_docker_rm_de_otro(con_ssh, sep, antes):
    """`run`/`exec` solo exime a los `rm` de SU comando: el `docker rm -f` que viene
    después del separador es un comando aparte."""
    fuente = _lista(antes, f'"{sep}"', '"docker", "rm", "-f", c', con_ssh=con_ssh)
    assert culpables_en_texto(fuente, es_python=True), f"no detectó: {fuente}"


@pytest.mark.parametrize("sep", _SEPS_LISTA)
def test_exec_rm_dentro_del_contenedor_sigue_sin_contar_con_separadores(sep):
    fuente = _lista('"docker", "exec", c, "rm", "-f", p', f'"{sep}"', '"docker", "rm", "-fv", c', con_ssh=True)
    assert not culpables_en_texto(fuente, es_python=True), f"falso positivo: {fuente}"


def test_whitelist_no_se_extiende_a_prefijos_dinamicos_ni_contextos_cercanos():
    assert culpables_en_texto('[*docker, "volume", "rm", "-f", recurso]', es_python=True)
    assert culpables_en_texto('["docker", *opts, "rm", "-f", recurso]', es_python=True)
    assert culpables_en_texto("docker $SUBCOMMAND rm -f recurso", es_python=False)
    assert culpables_en_texto("docker context export rm -f recurso", es_python=False)
    assert culpables_en_texto(
        "docker compose -f a -f b -f c -f d -f e rm -f svc", es_python=False
    )
    assert culpables_en_texto("docker compose --workdir /x rm -f svc", es_python=False)


def test_python_que_no_parsea_falla_cerrado_con_la_ruta():
    with pytest.raises(SyntaxError) as error:
        culpables_en_texto("def roto(:\n    docker rm -f recurso", es_python=True,
                           filename="scripts/roto.py")
    assert error.value.filename == "scripts/roto.py"


def test_python_inferido_que_no_parsea_no_degrada_a_shell():
    with pytest.raises(SyntaxError):
        culpables_en_texto("def roto(:\n    docker rm -f recurso")
