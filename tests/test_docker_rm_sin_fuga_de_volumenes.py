"""`docker rm -f` sin `-v` deja huérfano el volumen anónimo del contenedor.

La imagen de MariaDB declara `VOLUME /var/lib/mysql`. Un contenedor arrancado con
`--rm` y borrado con `docker rm -f` (sin `-v`) deja su datadir como volumen anónimo:
el 2026-10-04 había 343 así en hall9000 (56 GB, 22 con copias de `jax_memory`).
Medido ese día: `rm -f` → +1 volumen huérfano; `rm -fv` → 0. `-v` solo borra los
anónimos: un volumen con nombre sobrevive (probado en la auditoría del PR #353).

La detección final es `master(x) OR nuevo(x)`. `master` es la copia TEXTUAL congelada de las
funciones de detección de master f47820f5 (sufijo `_master`, bloque marcado en el archivo, con
hash y prueba contra `git show`); `nuevo` es lo que se suma. Por construcción nunca se marca
menos que master, con una sola excepción documentada: `docker <volume|network|...> rm -f x`
(la lista blanca, que master marcaba de más) se descarta de los hallazgos de master.
Todo lo que sigue describe el código NUEVO, que solo suma marcas.

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
  acepta después del nombre). La lectura nueva salta un subshell `$(...)` o `` `...` `` hasta
  su cierre balanceado y las banderas que vienen después cuentan (`docker rm $(docker ps -aq)
  -f` fuga); el acento grave que cierra un par corta la ventana; si falta el cierre de un
  subshell, falla cerrado y marca. La lectura que corta en `$(` y en el acento grave es la de
  master, congelada, y se une por OR: si la nueva se extravía (acento grave de más, `)` entre
  comillas), master sigue marcando.
  Las cadenas Python se prefiltran con cualquier espacio en blanco antes de `rm`/`remove`.
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
- Falsos positivos aceptados en shell: `docker rm -f $(docker ps -aq) -v` marca (la lectura
  (1) ve el `-f` y no el `-v`), y un subshell con un paréntesis sin par dentro de comillas
  (`docker rm -v $(docker ps --format "(x" -q)`) se toma por sin cerrar y marca.
- Fugas que master tampoco veía y siguen sin verse: un `)` entre comillas dentro de `$(...)`
  seguido de `-v` y luego `-f` (`docker rm $(echo ")" -v) -f`). El escáner no entiende
  comillas dentro del subshell.
- Una variable que no diga «docker» en su nombre (`$D rm -f`) no se reconoce como docker.
  Se prefiere eso a marcar cada `$SUDO rm -f archivo`.
- Un separador metido en una variable (`SEP`) no se reconoce: no corta la lectura de shell,
  y si la lectura argv tampoco marca (por un `-v` posterior) el caso fuga. Igual que master.
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


def _banderas_shell(toks, i):
    """Banderas del comando cuyo `rm` está en `toks[i]`, hasta el fin del comando. Un subshell
    `$(...)` o `` `...` `` se salta entero y lo que viene después sigue contando; el acento
    grave que CIERRA un par corta la ventana. Devuelve (banderas, balanceado): si falta el
    cierre del subshell no se puede saber dónde sigue el comando (falla cerrado). La lectura
    que corta en `$(` y en el acento grave es la de master, congelada: no se repite aquí."""
    banderas, j = [], i + 1
    while j < len(toks):
        t = toks[j]
        if t == "$(":
            profundidad, j = 1, j + 1
            while j < len(toks) and profundidad:
                if toks[j] in ("$(", "("):
                    profundidad += 1
                elif toks[j] == ")":
                    profundidad -= 1
                j += 1
            if profundidad:
                return banderas, False
            continue
        if t == "`":
            if toks[:i].count("`") % 2:   # el `rm` va DENTRO de un par: este es el que cierra
                break
            try:
                j = toks.index("`", j + 1) + 1
            except ValueError:
                return banderas, False
            continue
        if t in _SEPARADORES:
            break
        if t.startswith("-"):
            banderas.append(t)
        j += 1
    return banderas, True


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
            # Lectura nueva: salta el subshell. La que corta en `$(` y en el acento grave es
            # la de master (`culpables_shell_master`), que se une por OR en `culpables_en_texto`.
            saltada, balanceado = _banderas_shell(toks, i)
            if not balanceado or _fuerza_sin_volumenes(saltada):
                hallados.append(" ".join(toks[max(0, i - 3):i + 1 + len(saltada) + 1]))
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
                and not isinstance(padres.get(nodo), ast.JoinedStr) and re.search(r"\s(?:rm|remove)\b", nodo.value):
            hallados += culpables_shell(nodo.value)
    return hallados


# >>> BLOQUE CONGELADO: detección de master f47820f5af36d7b0e4e0d5b2156ac6275c10462c <<<
# Copia TEXTUAL de las funciones de detección de `tests/test_docker_rm_sin_fuga_de_volumenes.py` en ese SHA
# (shell y listas), solo con el sufijo `_master` en los nombres. NO SE EDITAN: la prueba
# `test_el_bloque_congelado_es_el_de_master` compara su hash y, si el SHA está en el checkout,
# el texto contra `git show`. La detección final es `master(x) OR nuevo(x)`.
_ES_DOCKER_master = re.compile(r"(?:\S*/)?docker|\$\{?\w*docker\w*(?:\[@\])?\}?|\{[^{}]*docker[^{}]*\}|dk", re.I)
_SEPARADORES_master = {";", "&&", "||", "|", "(", ")", "`", "$("}
_FALSO_master = {"false", "0", "f", "no"}


def _fuerza_sin_volumenes_master(banderas):
    """True si entre las banderas hay force y no hay volumes (orden libre)."""
    fuerza = volumenes = False
    for b in banderas:
        if b.startswith("--"):
            nombre, _, valor = b[2:].partition("=")
            activa = valor.lower() not in _FALSO_master if valor else True
            if nombre == "force":
                fuerza = activa
            elif nombre == "volumes":
                volumenes = activa
        elif b.startswith("-") and len(b) > 1:
            fuerza |= "f" in b[1:]
            volumenes |= "v" in b[1:]
    return fuerza and not volumenes


def _tokens_shell_master(linea):
    linea = re.sub(r"(^|\s)#.*$", "", linea)          # comentario
    for sep in ("&&", "||", "$(", ";", "|", "(", ")", "`"):
        linea = linea.replace(sep, f" {sep} ")
    return [t.strip("\"'") for t in linea.split()]


def culpables_shell_master(texto):
    hallados = []
    unido = re.sub(r"\\\r?\n\s*", " ", texto)
    for linea in unido.splitlines():
        toks = _tokens_shell_master(linea)
        for i, t in enumerate(toks):
            if t != "rm":
                continue
            # Hacia atrás, dentro del mismo comando: ¿lo llama docker?
            es_docker = dentro = False
            for j in range(i - 1, max(-1, i - 8), -1):
                previo = toks[j]
                if previo in _SEPARADORES_master:
                    break
                if previo in ("exec", "run"):
                    dentro = True
                if _ES_DOCKER_master.fullmatch(previo):
                    es_docker = True
                    break
            if not es_docker or dentro:
                continue
            banderas = []
            for siguiente in toks[i + 1:]:
                if siguiente in _SEPARADORES_master:
                    break
                if siguiente.startswith("-"):
                    banderas.append(siguiente)
            if _fuerza_sin_volumenes_master(banderas):
                hallados.append(" ".join(toks[max(0, i - 3):i + 1 + len(banderas) + 1]))
    return hallados


def _texto_de_master(nodo, fuente):
    return ast.get_source_segment(fuente, nodo) or ""


def culpables_python_master(fuente):
    try:
        arbol = ast.parse(fuente)
    except SyntaxError:
        return culpables_shell_master(fuente)  # no es Python válido: se barre como texto
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
            if any(isinstance(e, ast.Constant) and e.value in ("exec", "run") for e in antes):
                continue
            es_docker = any(_ES_DOCKER_master.search(_texto_de_master(e, fuente)) for e in antes)
            padre = padres.get(nodo)
            if not es_docker and i == 0 and isinstance(padre, ast.BinOp) and isinstance(padre.op, ast.Add) \
                    and padre.right is nodo:
                izquierda = _texto_de_master(padre.left, fuente)
                # `algo + ["rm", ...]`: un prefijo de comando. Docker salvo que sea claramente sudo.
                es_docker = bool(_ES_DOCKER_master.search(izquierda)) or not re.search(r"sudo", izquierda, re.I)
            if not es_docker:
                continue
            banderas = [e.value for e in elts[i + 1:]
                        if isinstance(e, ast.Constant) and isinstance(e.value, str) and e.value.startswith("-")]
            if _fuerza_sin_volumenes_master(banderas):
                hallados.append(" ".join(_texto_de_master(nodo, fuente).split()))
        elif isinstance(nodo, ast.JoinedStr):
            partes = [p.value if isinstance(p, ast.Constant) else "{" + _texto_de_master(p.value, fuente) + "}"
                      for p in nodo.values]
            hallados += culpables_shell_master("".join(str(x) for x in partes))
        elif isinstance(nodo, ast.Constant) and isinstance(nodo.value, str) and nodo not in docstrings \
                and not isinstance(padres.get(nodo), ast.JoinedStr) and " rm" in nodo.value:
            hallados += culpables_shell_master(nodo.value)
    return hallados
# >>> FIN DEL BLOQUE CONGELADO <<<
_HASH_BLOQUE_MASTER = "07e54de201fc9a4351e9a51a1052e7ad3be4481bb1c6b199e175689314200c76"
_SHA_MASTER = "f47820f5af36d7b0e4e0d5b2156ac6275c10462c"


def _es_rm_de_recurso_con_nombre(hallazgo):
    """True si el `rm` que MARCÓ master en este hallazgo es `docker <volume|network|image|...> rm`.

    Se decide por el token `rm` marcado, no por una búsqueda en la ventana de texto: master no
    corta en un `&` suelto y sigue juntando banderas, así que su ventana puede llegar a un
    `docker volume rm` POSTERIOR que no es el `rm` marcado. Hallazgo de lista de Python: es el
    texto fuente de la lista y el `rm` marcado es el primero. Hallazgo de shell: es la ventana
    `toks[i-3 : i+1+n+1]`, así que el `rm` marcado está en las posiciones 0..3; si hay más de
    una posible, solo se exime cuando TODAS son de un recurso con nombre (ante la duda, se marca)."""
    try:
        nodo = ast.parse(hallazgo, mode="eval").body
    except SyntaxError:
        nodo = None
    if isinstance(nodo, (ast.List, ast.Tuple)):
        cadenas = [e.value if isinstance(e, ast.Constant) else None for e in nodo.elts]
        if "rm" not in cadenas:
            return False
        i = cadenas.index("rm")
        return i >= 2 and cadenas[i - 2] == "docker" and cadenas[i - 1] in _SUBCOMANDOS_SIN_VOLUMEN_ANONIMO
    palabras = hallazgo.split()
    posibles = [p for p in range(min(4, len(palabras))) if palabras[p] == "rm"]
    return bool(posibles) and all(
        p >= 2 and palabras[p - 2] == "docker" and palabras[p - 1] in _SUBCOMANDOS_SIN_VOLUMEN_ANONIMO
        for p in posibles)


def _sin_la_excepcion_de_lista_blanca(hallazgos_master):
    """Único recorte a master: `docker <volume|network|image|...> rm -f x` NO es un contenedor
    (un volumen con nombre no deja huérfano nada) y esta rama lo exime a propósito desde la
    ronda 1; master lo marcaba de más. Se descartan solo los hallazgos de master cuyo `rm`
    marcado es ese (`_es_rm_de_recurso_con_nombre`)."""
    return [h for h in hallazgos_master if not _es_rm_de_recurso_con_nombre(h)]


def culpables_en_texto(texto, es_python=None, filename="<string>"):
    """master(x) OR nuevo(x): lo que marca la detección congelada de master, más lo que suma
    la nueva. Nunca se marca menos que master por construcción, salvo la excepción documentada
    de la lista blanca (`_sin_la_excepcion_de_lista_blanca`)."""
    if es_python is None:
        es_python = not texto.lstrip().startswith("#!") or "python" in texto.split("\n", 1)[0]
    if es_python:
        nuevo = culpables_python(texto, filename=filename)   # falla cerrado (SyntaxError)
        viejo = culpables_python_master(texto)
    else:
        nuevo = culpables_shell(texto)
        viejo = culpables_shell_master(texto)
    return nuevo + [h for h in _sin_la_excepcion_de_lista_blanca(viejo) if h not in nuevo]


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
    "docker rm -v c |& rm -f p",
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


# Ronda 9 (a): el prefiltro de cadenas Python ve cualquier espacio en blanco.
@pytest.mark.parametrize("fuente", [
    'x = "docker\\trm -f c"',
    'x = "docker  rm -f c"',
    'x = "docker container\\tremove -f c"',
    'x = "docker\\t\\trm\\t-f c"',
    'subprocess.run("docker\\trm -f c", shell=True)',
])
def test_cadena_python_con_cualquier_espacio_en_blanco_se_marca(fuente):
    assert culpables_en_texto(fuente, es_python=True), f"no detectó: {fuente}"


@pytest.mark.parametrize("fuente", [
    'x = "docker\\trm -fv c"',
    'x = "docker  rm -v -f c"',
    'x = "docker\\tps -a"',
    'x = "docker\\nrm -f c"',  # el salto de línea separa comandos: `rm` de host
])
def test_cadena_python_con_espacios_raros_no_marca_lo_que_no_fuga(fuente):
    assert not culpables_en_texto(fuente, es_python=True), f"falso positivo: {fuente}"


# Ronda 9 (b): en shell las banderas no cortan en `$(` ni en el acento grave.
@pytest.mark.parametrize("comando", [
    "docker rm $(docker ps -aq) -f",
    "docker rm `docker ps -aq` -f",
    "docker rm $(docker ps -aq) --force",
    'docker rm "$(docker ps -aq)" -f',
    "docker rm $(docker ps -aq -f status=exited) -f",
    "docker rm $(echo $(docker ps -aq) x) -f",
    "docker rm $(echo $(docker ps -aq) $(echo y)) -f",
    "docker rm $( (echo a) ) -f",
    "docker rm $(docker ps -aq) `echo c` -f",
    "docker rm `echo a` $(echo b) -f",
    "sudo docker container remove $(docker ps -aq) -f",
    "docker rm $(docker ps -aq",
    "docker rm $(docker ps -aq -f",
    "docker rm `docker ps -aq",
    "docker rm $(echo $(docker ps -aq) -v",
    # casos del auditor (ronda 9): acentos graves en número par, `)` desbalanceado
    "x=`docker rm -f c`; y=`grep -v z; true`",
    "x=`docker rm -f c` y=`echo -v; true`",
    "echo `docker rm -f c` `grep -v x;`",
    "x=`docker rm -f c`",
    'docker rm -f $(grep "web)" -v ids.txt)',
    "docker rm -f $(grep 'a)' -v ids)",
    "docker rm -f $(echo \\) -v)",
    "docker rm -f $(case $x in a) echo -v;; esac)",
    "docker rm -f $(docker ps -aq | grep -v keep)",
    "docker rm $((1+2)) -f",
    # la lectura de master (la ventana corta en `$(`) sigue marcando: nunca menos que master
    "docker rm -f $(docker ps -aq) -v",
    'docker rm -f $(echo ")") -v',
    "docker rm -f ${x%)} -v",
    "docker rm -f <(echo) -v",
])
def test_shell_las_banderas_posteriores_a_un_subshell_cuentan(comando):
    """Un subshell sin cerrar se marca (falla cerrado)."""
    assert culpables_en_texto(comando, es_python=False), f"no detectó: {comando}"


@pytest.mark.parametrize("comando", [
    "docker rm $(docker ps -aq) -v",
    "docker rm $(docker ps -aq) -fv",
    "docker rm `docker ps -aq` -v -f",
    "docker rm $(echo $(docker ps -aq) x) -vf",
    "docker rm $(docker ps -aq -f status=exited)",
    "docker rm -v $(docker ps -aq -f status=exited)",
    "docker ps -f $(echo x)",
    "x=$(docker rm -v c) -f",
    "x=`docker rm -fv c`",
    "x=`docker rm -v c`; y=`foo`",
])
def test_shell_un_v_posterior_a_un_subshell_exime_y_sus_banderas_no_cuentan(comando):
    assert not culpables_en_texto(comando, es_python=False), f"falso positivo: {comando}"


# Ronda 10: separadores `|` y `||` pegados a un argumento (mata el mutante que quita `|`).
@pytest.mark.parametrize("pegado", ['"c1|"', '"c1||"', '"|c1"', '"||c1"', '"c1;"', '"c1&"'])
def test_separador_pegado_de_cada_caracter_corta_la_lectura_de_shell(pegado):
    fuente = f'["ssh", h, "docker", "rm", "-f", {pegado}, "docker", "rm", "-fv", c2]'
    assert culpables_en_texto(fuente, es_python=True), f"no detectó: {fuente}"


def test_limites_declarados_del_subshell_en_shell():
    """Los dos límites del docstring: un paréntesis sin par entre comillas marca de más, y un
    `)` entrecomillado seguido de `-v` y luego `-f` no se ve (master tampoco lo veía)."""
    assert culpables_en_texto('docker rm -v $(docker ps --format "(x" -q)', es_python=False)
    assert not culpables_en_texto('docker rm $(echo ")" -v) -f', es_python=False)


# --- Ronda 11: la lectura (1) es el código de master, congelado -------------------------------
def _bloque_congelado():
    fuente = Path(__file__).read_text(encoding="utf-8")
    ini = fuente.index("# >>> BLOQUE CONGELADO")
    fin = fuente.index("# >>> FIN DEL BLOQUE CONGELADO <<<")
    cuerpo = fuente[ini:fin].split("\n", 5)[5]            # sin las 5 líneas de cabecera
    return cuerpo


def _texto_de_este_archivo_en_master():
    """Este archivo en el SHA de master. Un checkout superficial (el runner de `tests-puros` hace
    `actions/checkout` sin `fetch-depth`) no lo trae: se pide con `git fetch --depth=1 origin <sha>`
    (GitHub sirve cualquier SHA alcanzable). Si tampoco así está, FALLA: un `return` aquí contaba
    como passed sin haber comparado nada."""
    def mostrar():
        return subprocess.run(["git", "show", f"{_SHA_MASTER}:{ESTE}"], cwd=RAIZ,
                              capture_output=True, text=True)
    git = mostrar()
    if git.returncode != 0:
        pedido = subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", _SHA_MASTER],
                                cwd=RAIZ, capture_output=True, text=True, timeout=120)
        git = mostrar()
        if git.returncode != 0:
            pytest.fail(f"el SHA de master {_SHA_MASTER} no está en el checkout y no se pudo traer "
                        f"(git fetch rc={pedido.returncode}: {pedido.stderr.strip()[:300]}): "
                        "no se puede comparar el bloque congelado con master")
    return git.stdout


def test_el_bloque_congelado_es_el_de_master():
    """El hash guardado es el del bloque tal cual, y el texto tiene que coincidir con `git show`
    del SHA de master (solo cambia el sufijo `_master`)."""
    import hashlib
    cuerpo = _bloque_congelado()
    assert hashlib.sha256(cuerpo.encode()).hexdigest() == _HASH_BLOQUE_MASTER
    maestro = _texto_de_este_archivo_en_master()
    maestro = maestro[maestro.index("_ES_DOCKER = "):maestro.index("def culpables_en_texto(")]
    nombres = ["_ES_DOCKER", "_SEPARADORES", "_FALSO", "_fuerza_sin_volumenes", "_tokens_shell",
               "culpables_shell", "_texto_de", "culpables_python"]
    esperado = re.sub(r"\b(" + "|".join(nombres) + r")\b", lambda x: x.group(1) + "_master", maestro)
    assert cuerpo.rstrip("\n") == esperado.rstrip("\n")


CASOS_DE_MASTER = [
    ("docker -H 'ssh://h?x&y' rm -f c", False),
    ('docker rm "a&b" c1 -f', False),
    ("docker rm ${ids//&/ } -f", False),
    ("docker rm a\\&b -f", False),
    ("docker rm -f c &", False),
    ("docker rm c & -f", False),
    ("docker rm -v c & rm -f p", False),
    ("docker rm c & rm -f p", False),
    ("docker rm -f $(docker ps -aq) -v", False),
    ("x=`docker rm -f c`; y=`grep -v z; true`", False),
    ('["docker", "rm", c, "&&", "rm", "-f", p]', True),
    ('["docker", "rm", c, "|", "-f"]', True),
    ('DOCKER + ["rm", "-f", n]', True),
]


@pytest.mark.parametrize("fuente,es_python", CASOS_DE_MASTER)
def test_nunca_se_marca_menos_que_master(fuente, es_python):
    viejo = culpables_python_master(fuente) if es_python else culpables_shell_master(fuente)
    if viejo:
        assert culpables_en_texto(fuente, es_python=es_python), f"master marca y esto no: {fuente}"


@pytest.mark.parametrize("comando", [
    "docker -H 'ssh://h?x&y' rm -f c",
    'docker rm "a&b" c1 -f',
    "docker rm ${ids//&/ } -f",
])
def test_el_ampersand_entre_comillas_o_en_expansion_no_corta_la_lectura_de_master(comando):
    assert culpables_en_texto(comando, es_python=False), f"no detectó: {comando}"


# La lectura nueva se prueba SOLA (culpables_shell): con el OR, master taparía sus mutantes.
@pytest.mark.parametrize("comando", [
    "x=`docker rm -fv c`; y=`foo`",
    "x=`docker rm -v c` y=`echo hi`",
    "docker `echo x` rm -f c",
    "docker `echo x` rm -f c; true",
])
def test_lectura_nueva_acento_grave_corta_la_paridad_y_es_separador(comando):
    """El acento grave que CIERRA el par corta la ventana (no sigue leyendo hasta el próximo)
    y es un separador hacia atrás: el `rm` tras `docker $(echo x)` no es de docker."""
    assert not culpables_shell(comando), f"falso positivo: {comando}"


def test_lectura_nueva_acento_grave_cerrado_marca_lo_que_sigue_sin_leer_de_mas():
    assert culpables_shell("x=`docker rm -f c` y=`echo -v; true`")
    assert culpables_shell("echo `docker rm -f c` `grep -v x;`")
    # el acento grave que cierra corta: el `-v` de después NO es de este comando
    assert culpables_shell("x=`docker rm -f c` -v")
    assert culpables_shell("x=`docker rm -f c` -v y=`z`")


def test_python_que_no_parsea_falla_cerrado_con_la_ruta():
    with pytest.raises(SyntaxError) as error:
        culpables_en_texto("def roto(:\n    docker rm -f recurso", es_python=True,
                           filename="scripts/roto.py")
    assert error.value.filename == "scripts/roto.py"


def test_python_inferido_que_no_parsea_no_degrada_a_shell():
    with pytest.raises(SyntaxError):
        culpables_en_texto("def roto(:\n    docker rm -f recurso")


# --- Ronda 12: la exención de lista blanca se decide por el `rm` que marcó master -------------
# master no corta en un `&` suelto y su ventana llega a un `docker <volume|image|network> rm`
# POSTERIOR; ese no es el `rm` marcado y el hallazgo (un `rm` de CONTENEDOR) no se descarta.
CASOS_VENTANA_QUE_LLEGA_A_UN_RECURSO_POSTERIOR = [
    "docker -H 'ssh://h?x&y' rm -f c & docker image rm i & docker run -d -p 1:1 -e A=1 --name n img",
    'docker rm "a&b" -f & docker volume rm v & docker run --rm -d -p 80:80 -e A=1 --name n img',
    "docker rm ${ids//&/ } -f & docker volume rm v & docker run -d -p 1:1 -e A=1 --name n -l x img",
    'docker rm "a&b" c1 -f & docker volume rm v -f -f -f -f -f -f',
]


@pytest.mark.parametrize("comando", CASOS_VENTANA_QUE_LLEGA_A_UN_RECURSO_POSTERIOR)
def test_shell_la_ventana_de_master_que_llega_a_un_recurso_posterior_no_exime_el_rm_de_contenedor(comando):
    assert culpables_shell_master(comando), "el caso ya no ejercita a master"
    assert culpables_en_texto(comando, es_python=False), f"se perdió un rm de contenedor: {comando}"


@pytest.mark.parametrize("comando", CASOS_VENTANA_QUE_LLEGA_A_UN_RECURSO_POSTERIOR)
def test_python_la_misma_cadena_dentro_de_subprocess_run_tampoco_se_pierde(comando):
    fuente = f"subprocess.run({comando!r}, shell=True)"
    assert culpables_python_master(fuente), "el caso ya no ejercita a master"
    assert culpables_en_texto(fuente, es_python=True), f"se perdió un rm de contenedor: {fuente}"


# Minor 1: casos de Python que master marca y la lectura nueva SOLA no (fija la unión OR en Python).
@pytest.mark.parametrize("fuente", [
    'subprocess.run("docker rm \\"a&b\\" c1 -f", shell=True)',
    'subprocess.run("docker rm ${ids//&/ } -f", shell=True)',
    "subprocess.run(\"docker -H 'ssh://h?x&y' rm -f c\", shell=True)",
    'os.system("docker rm \\"a&b\\" c1 -f")',
    'x = f"docker rm \\"a&b\\" -f {n}"',
])
def test_python_lo_que_solo_marca_master_sigue_marcado_por_la_union(fuente):
    assert culpables_python_master(fuente), "master ya no marca: el caso no sirve"
    assert not culpables_python(fuente), "la lectura nueva ya lo marca: el caso no fija el OR"
    assert culpables_en_texto(fuente, es_python=True), f"master marca y la unión no: {fuente}"


# La exención en sí, probada SOBRE `_sin_la_excepcion_de_lista_blanca` (con la unión, la lectura
# nueva marca esto por su cuenta y taparía los mutantes): solo cuenta el `rm` marcado, y solo si
# va PEGADO a `docker <subcomando>`.
@pytest.mark.parametrize("comando", [
    "docker volume rm -f v",
    "docker network rm -f n",
    "docker image rm -f i",
    "docker volume rm -f v & docker volume rm -f w",
])
def test_la_exencion_de_recurso_con_nombre_sigue_funcionando_en_shell(comando):
    hallazgos = culpables_shell_master(comando)
    assert hallazgos, "master ya no marca: el caso no ejercita la exencion"
    assert not _sin_la_excepcion_de_lista_blanca(hallazgos), f"falso positivo: {comando}"
    assert not culpables_en_texto(comando, es_python=False)


def test_la_exencion_de_recurso_con_nombre_sigue_funcionando_en_listas():
    for fuente in ['["docker", "volume", "rm", "-f", v]', '("docker", "network", "rm", "-f", n)']:
        hallazgos = culpables_python_master(fuente)
        assert hallazgos, "master ya no marca: el caso no ejercita la exencion"
        assert not _sin_la_excepcion_de_lista_blanca(hallazgos), f"falso positivo: {fuente}"
        assert not culpables_en_texto(fuente, es_python=True)


# Minor 3: la forma de la exención. Ensancharla (no exigir `docker` ni el subcomando PEGADOS al
# `rm` marcado, o exigirlos en cualquier lugar de la ventana) exime de más.
@pytest.mark.parametrize("comando", [
    "docker volume ls rm -f c",
    "docker volume --opt rm -f c",
    "docker x volume rm -f c",
    "docker rm -f volume",
    "docker rm volume rm -f c",
    "docker image ls -f rm -f c",
    "sudo volume rm -f c; docker rm -f c",
    "docker rm volume -f c",
])
def test_la_exencion_exige_docker_subcomando_rm_pegados_en_shell(comando):
    hallazgos = culpables_shell_master(comando)
    assert hallazgos, f"master ya no marca: {comando}"
    assert _sin_la_excepcion_de_lista_blanca(hallazgos), f"se eximió de más: {comando}"


@pytest.mark.parametrize("fuente", [
    '["docker", "volume", "ls", "rm", "-f", c]',
    '["docker", "x", "volume", "rm", "-f", c]',
    '["docker", "rm", "-f", "volume", "rm"]',
    '["docker", "rm", "-f", c, "docker", "volume", "rm"]',
    '[DOCKER, "volume", "rm", "-f", v] + ["rm", "-f", c]',
])
def test_la_exencion_exige_docker_subcomando_rm_pegados_en_listas(fuente):
    hallazgos = culpables_python_master(fuente)
    assert hallazgos, f"master ya no marca: {fuente}"
    assert _sin_la_excepcion_de_lista_blanca(hallazgos), f"se eximió de más: {fuente}"


def test_la_exencion_ante_dos_rm_posibles_marca_si_alguno_no_es_de_recurso_con_nombre():
    """`docker volume rm rm -f`: la ventana tiene dos `rm` posibles y uno no va pegado a
    `docker <subcomando>`. Ante la duda se marca (falso positivo conocido: un volumen llamado `rm`)."""
    hallazgos = culpables_shell_master("docker volume rm rm -f")
    assert hallazgos and _sin_la_excepcion_de_lista_blanca(hallazgos)
    assert _sin_la_excepcion_de_lista_blanca(culpables_shell_master("volume docker x rm -f c"))


def test_la_exencion_en_listas_no_lee_hacia_atras_con_indice_negativo():
    """`rm` en la posición 0 no mira el final de la lista (`elts[-2]`, `elts[-1]`)."""
    hallazgos = culpables_python_master('DOCKER + ["rm", "-f", "docker", "volume"]')
    assert hallazgos and _sin_la_excepcion_de_lista_blanca(hallazgos)


def test_sin_el_sha_de_master_la_comparacion_falla_y_no_vuelve_en_silencio(monkeypatch):
    """El camino del checkout superficial sin red: ni `git show` ni `git fetch` lo traen."""
    def siempre_falla(argv, **_):
        return subprocess.CompletedProcess(argv, 128, stdout="", stderr="fatal: no hay red")
    monkeypatch.setattr(subprocess, "run", siempre_falla)
    with pytest.raises(pytest.fail.Exception, match="no se puede comparar el bloque congelado"):
        _texto_de_este_archivo_en_master()


def test_el_sha_de_master_se_trae_por_fetch_si_el_checkout_es_superficial(monkeypatch):
    """`git show` falla la primera vez, `git fetch` lo trae y la segunda vez funciona."""
    llamadas = []

    def simulado(argv, **_):
        llamadas.append(argv[1])
        if argv[1] == "show":
            return subprocess.CompletedProcess(argv, 0 if "fetch" in llamadas else 128,
                                               stdout="TEXTO", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    monkeypatch.setattr(subprocess, "run", simulado)
    assert _texto_de_este_archivo_en_master() == "TEXTO"
    assert llamadas == ["show", "fetch", "show"]


# --- Ronda 13: el tipo de hallazgo viene de dónde salió, no de re-interpretar su texto ---------
_VENTANA_QUE_PARSEA_COMO_LISTA = '["docker","volume","rm",bin/docker --config if rm "&y" -f else 0] --force --force'


def test_shell_una_ventana_que_parsea_como_lista_de_python_no_toma_el_camino_de_listas():
    assert culpables_shell_master(_VENTANA_QUE_PARSEA_COMO_LISTA), "el caso ya no ejercita a master"
    assert culpables_en_texto(_VENTANA_QUE_PARSEA_COMO_LISTA, es_python=False)


def test_python_la_misma_ventana_dentro_de_una_cadena_no_se_exime_como_lista():
    fuente = f"subprocess.run({_VENTANA_QUE_PARSEA_COMO_LISTA!r}, shell=True)"
    assert culpables_python_master(fuente), "el caso ya no ejercita a master"
    assert culpables_en_texto(fuente, es_python=True)


def test_python_lista_con_comentario_cuyo_texto_aplanado_no_parsea_no_se_exime():
    fuente = 'cmd = [# docker volume rm\n    ["docker"], "rm", "-f", c]\n'
    assert culpables_python_master(fuente), "el caso ya no ejercita a master"
    assert culpables_en_texto(fuente, es_python=True)


# Minor 3 de la ronda 13: la prueba del bloque congelado tiene que FALLAR si el texto de master
# difiere en un byte (mata el mutante que vuelve antes de compararlo).
def test_el_bloque_congelado_falla_si_master_difiere_en_un_byte(monkeypatch):
    real = _texto_de_este_archivo_en_master()
    adulterado = real.replace('_FALSO = {"false"', '_FALSO = {"falsE"', 1)
    assert adulterado != real
    monkeypatch.setitem(globals(), "_texto_de_este_archivo_en_master", lambda: adulterado)
    with pytest.raises(AssertionError):
        test_el_bloque_congelado_es_el_de_master()
