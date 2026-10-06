"""`docker rm -f` sin `-v` deja huérfano el volumen anónimo del contenedor.

La imagen de MariaDB declara `VOLUME /var/lib/mysql`. Un contenedor arrancado con
`--rm` y borrado con `docker rm -f` (sin `-v`) deja su datadir como volumen anónimo:
el 2026-10-04 había 343 así en hall9000 (56 GB, 22 con copias de `jax_memory`).
Medido ese día: `rm -f` → +1 volumen huérfano; `rm -fv` → 0. `-v` solo borra los
anónimos: un volumen con nombre sobrevive (probado en la auditoría del PR #353).

Cómo barre (segunda ronda de auditoría del PR #353: los regex por línea eran una
persecución sin fin):
- Python, con `ast`: listas y tuplas con el elemento "rm" (o "remove" tras "container",
  el alias de `docker container rm`), concatenaciones `docker + ["rm", ...]`, f-strings y
  cadenas que son un comando de shell. Comentarios y docstrings no cuentan. `*[...]`
  literal se expande en su sitio.
- Shell, con un tokenizador: líneas partidas con `\\` unidas, comentarios fuera,
  `docker`/`docker container`/opciones globales/variables con «docker» en el nombre,
  `&`, `|&`, `&&`, `||`, `;`, `|` como separadores (las redirecciones `2>&1`, `>&2`,
  `<&0`, `&>f` no lo son) y TODAS las banderas hasta el fin del comando (docker las
  acepta después del nombre).
- `docker exec|run ... rm -f x` es un `rm` DENTRO del contenedor: no cuenta.

Listas Python y separadores (rondas 6 a 8). Una lista puede ser argv puro o llevar un
comando de shell partido en elementos, y desde el escáner no se puede saber cuál (la lista
se arma en una variable y se pasa después). Por eso cada lista se lee DOS veces y se marca
si CUALQUIERA de las dos lecturas fuga:
1. argv puro: un solo comando, TODAS las banderas hasta el final de la lista, sin cortar
   (`["docker", "rm", c, "|", "-f"]` fuga: docker acepta banderas después de los argumentos).
2. shell: la lista se corta en los separadores literales (`&&`, `;`, `||`, `|`, `&`, `|&`,
   con espacios normalizados, el salto de línea y el separador pegado a un argumento,
   `"c1;"`), y contexto, banderas, exec/run y lista blanca se leen dentro de cada comando.
Con el OR nunca se marca menos que con la lectura sin cortar de master.

Falsos positivos ACEPTADOS: lo que marca la lectura argv aunque la lista vaya a un shell,
p. ej. `["docker", "rm", "-v", c, "&&", "rm", "-f", p]` (el `-f` del `rm` de host se atribuye
a docker) o `["docker", "rm", c, "&&", "touch", "-f", p]`. Se prefiere marcar de más.

Límites declarados:
- Una variable que no diga «docker» en su nombre (`$D rm -f`) no se reconoce como docker.
  Se prefiere eso a marcar cada `$SUDO rm -f archivo`.
- Un separador metido en una variable (`SEP`) no se reconoce: no corta la lectura de shell,
  y si la lectura argv tampoco marca (por un `-v` posterior) el caso fuga. Igual que master.
- En shell (cadenas), las banderas de un comando terminan en `$(` o `` ` ``:
  `docker rm $(docker ps -aq) -f` no se marca. Igual que master.
"""
import ast
import re
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
ESTE = "tests/test_docker_rm_sin_fuga_de_volumenes.py"

_ES_DOCKER = re.compile(r"(?:\S*/)?docker|\$\{?\w*docker\w*(?:\[@\])?\}?|\{[^{}]*docker[^{}]*\}|dk", re.I)
_SEPARADORES = {";", "&&", "||", "|", "|&", "&", "(", ")", "`", "$("}
_SEPARADORES_LISTA = {";", "&&", "||", "|", "|&", "&"}
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


_REDIRECCION_FD = re.compile(r"\d*[<>]&\d*-?|&>>?")
_CORTE_SHELL = re.compile(r"(&&|\|\||\|&|\$\(|;|\||\(|\)|`|&)")


def _tokens_shell(linea):
    linea = re.sub(r"(^|\s)#.*$", "", linea)          # comentario
    linea = _REDIRECCION_FD.sub(" ", linea)            # `2>&1`, `>&2`, `&>f`: no son `&`
    linea = _CORTE_SHELL.sub(r" \1 ", linea)
    return [t.strip("\"'") for t in linea.split()]


def _es_rm_shell(toks, i):
    """`rm`, o `remove` justo después de `container` (alias de `docker container rm`)."""
    return toks[i] == "rm" or (toks[i] == "remove" and i >= 1 and toks[i - 1] == "container")


def culpables_shell(texto):
    hallados = []
    unido = re.sub(r"\\\r?\n\s*", " ", texto)
    for linea in unido.splitlines():
        toks = _tokens_shell(linea)
        for i, t in enumerate(toks):
            if not _es_rm_shell(toks, i):
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


def _cadena(nodo):
    return nodo.value if isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) else None


def _planas(elts):
    """Los elementos de la lista, con `*[...]` literal expandido en su sitio."""
    salida = []
    for e in elts:
        if isinstance(e, ast.Starred) and isinstance(e.value, (ast.List, ast.Tuple)):
            salida += _planas(e.value.elts)
        else:
            salida.append(e)
    return salida


def _es_separador(nodo):
    """Elemento literal que es un separador de shell, con espacios normalizados
    (`" && "`, `"&& "`, `"\\n"`): lo único que hay en él es el separador o un salto de línea."""
    valor = _cadena(nodo)
    if valor is None:
        return False
    limpio = valor.strip()
    return limpio in _SEPARADORES_LISTA or (not limpio and "\n" in valor)


def _es_separador_pegado(nodo):
    """`"c1;"`, `"c1&&"`: un separador pegado a un argumento. No se sabe dónde parte el
    comando, así que el elemento entero se trata como separador."""
    valor = _cadena(nodo)
    if valor is None or _es_separador(nodo):
        return False
    limpio = valor.strip()
    return bool(limpio) and (limpio[0] in ";&|" or limpio[-1] in ";&|")


def _culpable_lista(nodo, fuente, padres):
    """Una lista se lee de dos maneras y fuga si CUALQUIERA de las dos fuga:
    (1) como argv puro: un solo comando, TODAS las banderas hasta el final, sin cortar;
    (2) como un comando de shell partido en elementos: cortada en los separadores, y cada
        comando por separado (contexto, banderas, exec/run y lista blanca).
    No hace falta saber si la lista llega a un shell, algo que no se puede ver desde aquí."""
    elts = _planas(nodo.elts)
    n = len(elts)

    def es_rm(k):
        valor = _cadena(elts[k])
        return valor == "rm" or (valor == "remove" and k >= 1 and _cadena(elts[k - 1]) == "container")

    indices = [k for k in range(n) if es_rm(k)]
    if not indices:
        return False
    padre = padres.get(nodo)
    cortes = [k for k, e in enumerate(elts) if _es_separador(e) or _es_separador_pegado(e)]
    tramos, inicio = [], 0
    for k in cortes:
        tramos.append((inicio, k))
        inicio = k + 1
    tramos.append((inicio, n))
    for inicio, fin in [(0, n)] + tramos:
        for i in indices:
            if not inicio <= i < fin:
                continue
            antes = elts[inicio:i]
            if i - inicio >= 2 and _cadena(elts[i - 1]) in _SUBCOMANDOS_SIN_VOLUMEN_ANONIMO \
                    and _cadena(elts[i - 2]) == "docker":
                continue
            if any(_cadena(e) in ("exec", "run") for e in antes):
                continue
            es_docker = any(_ES_DOCKER.search(_texto_de(e, fuente)) for e in antes
                            if not isinstance(e, (ast.List, ast.Tuple)))
            if not es_docker and i == 0 and isinstance(padre, ast.BinOp) and isinstance(padre.op, ast.Add) \
                    and padre.right is nodo:
                izquierda = _texto_de(padre.left, fuente)
                # `algo + ["rm", ...]`: un prefijo de comando. Docker salvo que sea claramente sudo.
                es_docker = bool(_ES_DOCKER.search(izquierda)) or not re.search(r"sudo", izquierda, re.I)
            if not es_docker:
                continue
            banderas = [v for v in (_cadena(e) for e in elts[i + 1:fin]) if v is not None and v.startswith("-")]
            if _fuerza_sin_volumenes(banderas):
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
            if _culpable_lista(nodo, fuente, padres):
                hallados.append(" ".join(_texto_de(nodo, fuente).split()))
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
@pytest.mark.parametrize("despues", ['"rm", "-rf", "/tmp/build"', '"sudo", "rm", "-f", lock'])
def test_falso_positivo_aceptado_rm_de_host_tras_un_docker_rm_con_v(con_ssh, sep, rm_docker, despues):
    """Falso positivo ACEPTADO (ronda 8): la lectura argv no corta, así que las banderas del
    `rm` de host que viene tras el separador cuentan para el `rm` de docker. No se sabe
    si la lista llega a un shell; ante la duda se marca (nunca menos que master)."""
    fuente = _lista(rm_docker, f'"{sep}"', despues, con_ssh=con_ssh)
    assert culpables_en_texto(fuente, es_python=True), f"se esperaba la marca: {fuente}"


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


# --- Ronda 7 -------------------------------------------------------------------------
# MINOR-A: un separador literal dentro de una lista argv (sin shell) es un argumento más.
FUGAN_ARGV_CON_SEPARADOR = [
    '["docker", "rm", c, "|", "-f"]',
    '["docker", "rm", c, ";", "--force"]',
    '["docker", "rm", c, "&&", "-f"]',
    '["docker", "rm", c, "||", "-f"]',
    '["docker", "rm", c, "&", "-f"]',
    '["docker", "-H", "|", "rm", "-f", c]',
    '["docker", "--context", ";", "rm", "-f", c]',
    '["docker", "-H", "&&", "rm", "-f", c]',
    '["find", ".", "-exec", "docker", "rm", "-f", "{}", ";"]',
    '["find", ".", "-exec", "docker", "rm", "{}", ";", "-f"]',
    '["xargs", "-d", "|", "docker", "rm", "-f"]',
    '["grep", "|", f, "&&", "docker", "rm", "-f", c]',
    '["echo", ";", "docker", "rm", "-f", c]',
    # La lista se arma en una variable y se pasa después: el escaner no puede saber si va
    # a un shell, así que lee el segundo `rm` como un comando nuevo y marca el primero.
    'cmd = ["docker", "rm", "-f", c1, "&&", "docker", "rm", "-fv", c2]\nsubprocess.run(cmd, shell=True)',
]


@pytest.mark.parametrize("fuente", FUGAN_ARGV_CON_SEPARADOR)
def test_separador_literal_en_lista_argv_no_corta_las_banderas(fuente):
    assert culpables_en_texto(fuente, es_python=True), f"no detectó: {fuente}"


NO_FUGAN_CON_SEPARADOR = [
    '["echo", ";", "&&", "docker", "rm", "-fv", c]',
    '["docker", "exec", c, "rm", "-f", p, "&&", "docker", "rm", "-fv", c]',
]


@pytest.mark.parametrize("fuente", NO_FUGAN_CON_SEPARADOR)
def test_separador_que_ninguna_lectura_marca(fuente):
    assert not culpables_en_texto(fuente, es_python=True), f"falso positivo: {fuente}"


# MINOR-B: más separadores, espacios normalizados y separadores que no se reconocen.
@pytest.mark.parametrize("sep", ["&", "|&", " && ", "&& ", " ; ", "\n", " \n ", "\r\n", "\t|\t"])
def test_separadores_extra_y_con_espacios_cortan_en_listas(sep):
    """La lectura de shell corta en ellos: el `-fv` del segundo comando no tapa al primero."""
    fuga = f'["ssh", h, "docker", "rm", "-f", c1, {sep!r}, "docker", "rm", "-fv", c2]'
    assert culpables_en_texto(fuga, es_python=True), f"no detectó: {fuga}"


@pytest.mark.parametrize("fuente", [
    '["ssh", h, "docker", "rm", "-f", "c1;", "docker", "rm", "-fv", c2]',
    '["ssh", h, "docker", "rm", "-f", "c1&&", "docker", "rm", "-fv", c2]',
    '["docker", "rm", "-f", ";c1", "docker", "rm", "-fv", c2]',
    '[*docker, "rm", *["-f"], c]',
    '["ssh", h, "docker", "rm", "$(", "docker", "ps", "-aq", ")", "-f"]',
    '["ssh", h, "docker", "rm", c, "docker", "-f"]',
    'DOCKER + ["rm", "rm", "-f"]',
    '["docker", "rm", c, "&&", "rm", "-f", p]',
])
def test_separador_pegado_y_lectura_argv_marcan(fuente):
    """Un separador pegado a un argumento (`"c1;"`) corta en la lectura de shell. La lectura
    argv (todas las banderas, sin cortar) marca lo que master ya marcaba."""
    assert culpables_en_texto(fuente, es_python=True), f"no detectó: {fuente}"


def test_limite_declarado_separador_en_variable_no_se_reconoce():
    """Límite declarado: un separador metido en una variable no corta. Igual que master."""
    fuente = '["ssh", h, "docker", "rm", "-f", c1, SEP, "docker", "rm", "-fv", c2]'
    assert not culpables_en_texto(fuente, es_python=True)


def test_limite_declarado_argv_atribuye_las_banderas_de_otro_comando():
    """Falso positivo aceptado: la lectura argv atribuye `-f` de otro comando a docker."""
    assert culpables_en_texto('["docker", "rm", c, "&&", "touch", "-f", p]', es_python=True)
    assert culpables_en_texto('["ssh", h, "docker", "rm", c, "&&", "touch", "-f", p]', es_python=True)


# MINOR-C: `docker container remove` es el alias de `docker container rm`.
@pytest.mark.parametrize("fuente", [
    '["docker", "container", "remove", "-f", c]',
    '["docker", "container", "remove", "--force", c]',
    '["ssh", h, "docker", "container", "remove", "-f", c]',
    '[*docker, "container", "remove", "-f", c]',
    '["docker", "container", "remove", c, "-f"]',
    '["docker", "container", "remove", "-f", c1, "&&", "docker", "container", "remove", "-fv", c2]',
    'subprocess.run("docker container remove -f x", shell=True)',
])
def test_container_remove_se_marca_como_rm(fuente):
    assert culpables_en_texto(fuente, es_python=True), f"no detectó: {fuente}"


@pytest.mark.parametrize("fuente", [
    '["docker", "container", "remove", "-fv", c]',
    '["docker", "remove", "-f", c]',
    '["docker", "compose", "remove", "-f", s]',
    '["docker", "exec", c, "remove", "-f", p]',
])
def test_container_remove_no_marca_lo_que_no_fuga(fuente):
    assert not culpables_en_texto(fuente, es_python=True), f"falso positivo: {fuente}"


@pytest.mark.parametrize("comando", [
    "docker container remove -f x",
    "docker container remove --force x",
    "sudo docker container remove -f x",
    "docker container remove x -f",
    "docker rm -f c1 & docker rm -fv c2",
    "docker rm -f c&echo -v",
    "docker rm -f c1&docker rm -fv c2",
    "docker rm -f c1 |& docker rm -fv c2",
    "docker rm -f c1 &\ndocker rm -fv c2",
    "docker rm -f c 2>&1",
    "docker rm -f c >&2",
    "docker rm -f c &>/dev/null",
    "docker rm c <&0 -f",
    "docker rm c 0<&0 --force",
    "docker run -d img & docker rm -f c",
])
def test_shell_marca_alias_remove_y_cuenta_el_ampersand_como_separador(comando):
    assert culpables_en_texto(comando, es_python=False), f"no detectó: {comando}"


@pytest.mark.parametrize("comando", [
    "docker container remove -fv x",
    "docker rm -v c & rm -f p",
    "docker rm -v c |& rm -f p",
    "docker rm c & rm -f p",
    "docker rm -fv c 2>&1",
    "docker rm -vf c >&2",
    "docker exec c true & rm -f p",
    "docker remove -f x",
])
def test_shell_no_marca_lo_que_no_fuga_con_ampersand_y_remove(comando):
    assert not culpables_en_texto(comando, es_python=False), f"falso positivo: {comando}"


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
