#!/usr/bin/env python3
"""Barrera de contenido SERVER-SIDE: rechaza un PR que introduzca un secreto
ya conocido, en cualquier parte de su diff completo.

POR QUE EXISTE, con lo que la motivo (medido 2026-09-01): los hooks de
pre-commit NO corren en `merge`, `rebase`, `cherry-pick`, `revert` ni `stash`
-- solo en `commit` y `commit --amend`. Y el `pre-push` mira el ref DESTINO,
no el contenido. Consecuencia: un commit hecho con `--no-verify`, o anterior a
la activacion del hook, entraba a `master` por merge SIN pasar por ninguna
revision de contenido. Este check cierra esa clase, porque corre del lado del
servidor sobre el diff `base...head` completo del PR y no lo saltea ningun
flag local.

POR QUE NO USA EL PEPPER: ver la cabecera de ops/ci/known-secrets-ci.txt. Salt
publico por entrada + scrypt de costo alto, en vez de meter el pepper como
secret de Actions y crear superficie nueva.

FAIL-CLOSED (P10): lista ilegible, no parseable o vacia => el check FALLA. Un
comparador sin con que comparar no encontro cero: no comparo.

QUE CUBRE que el hook local no cubre: merge, rebase, cherry-pick, revert, y
cualquier commit que haya entrado con --no-verify.
QUE NO CUBRE, declarado: un secreto NUEVO (no esta en la lista); un secreto
embebido sin separadores alrededor; y codificaciones alternativas (base64,
URL-encode) del valor.
"""
import hashlib
import os
import re
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
LISTA = os.environ.get("JAX_CI_SECRETS", os.path.join(AQUI, "known-secrets-ci.txt"))
MIN_TOKEN = 6
MAX_TOKENS = 40000  # techo de runtime; si se supera, falla en vez de truncar
SEPARADORES = re.compile(r"""[\s"'`,;()\[\]{}<>=:]+""")
RUNS = re.compile(rb"[\x20-\x7e]{%d,}" % MIN_TOKEN)


def fallar(motivo, detalle=""):
    print(f"::error::scan-pr-secrets: {motivo}")
    if detalle:
        print(detalle)
    sys.exit(1)


def git(args, binario=False):
    out = subprocess.run(["git"] + args, capture_output=True, check=False,
                         **({} if binario else {"text": True, "errors": "replace"}))
    if out.returncode != 0:
        err = out.stderr if not binario else out.stderr.decode("utf-8", "replace")
        fallar("git fallo durante el escaneo (FAIL-CLOSED).",
               f"  git {' '.join(args)}\n  {str(err).strip()[:300]}")
    return out.stdout


def cargar():
    try:
        crudo = open(LISTA, encoding="utf-8").read()
    except OSError as exc:
        fallar(f"no se pudo leer la lista (FAIL-CLOSED): {exc.strerror}", f"  {LISTA}")
    entradas = []
    for num, linea in enumerate(crudo.splitlines(), 1):
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        p = linea.split(None, 2)
        if len(p) != 3 or not re.fullmatch(r"[0-9a-f]{32}", p[0]) or not re.fullmatch(r"[0-9a-f]{64}", p[1]):
            fallar("la lista no parsea (FAIL-CLOSED).",
                   f"  {LISTA}:{num} no tiene la forma '<salt-hex> <scrypt-hex> <etiqueta>'")
        entradas.append((bytes.fromhex(p[0]), p[1], p[2]))
    if not entradas:
        fallar("la lista esta vacia (FAIL-CLOSED).",
               "  Un comparador sin nada con que comparar no encontro cero: no comparo.")
    return entradas


def tokens_de(texto):
    vistos = {texto.strip()}
    for t in SEPARADORES.split(texto):
        if t:
            vistos.add(t)
            vistos.add(t.strip("-_./\\"))
    return {t for t in vistos if len(t) >= MIN_TOKEN}


def main():
    if len(sys.argv) != 3:
        fallar("uso: scan-pr-secrets.py <base-sha> <head-sha>")
    base, head = sys.argv[1], sys.argv[2]
    entradas = cargar()

    # Diff COMPLETO del PR (base...head), no el ultimo commit: asi cubre lo que
    # entro por merge, rebase o cherry-pick dentro de la rama.
    archivos = [x for x in git(["diff", "--name-only", "--diff-filter=ACMRT",
                                f"{base}...{head}"]).splitlines() if x]
    diff = git(["diff", "--unified=0", "--no-color", "--diff-filter=ACMRT",
                f"{base}...{head}"])

    cands, con_texto, archivo, en_hunk = set(), set(), None, False
    for linea in diff.splitlines():
        if linea.startswith("diff --git "):
            archivo, en_hunk = None, False
        elif linea.startswith("+++ ") and not en_hunk:
            archivo = linea[6:] if linea.startswith("+++ b/") else linea[4:]
        elif linea.startswith("@@"):
            en_hunk = True
        elif en_hunk and linea.startswith("+"):
            if archivo:
                con_texto.add(archivo)
            for t in tokens_de(linea[1:]):
                cands.add((archivo, t))

    # Binarios y rutas con el diff apagado por .gitattributes: sin lineas '+'.
    # Se leen enteros desde head. Es la clase del .pyc que origino todo esto.
    for a in archivos:
        if a in con_texto:
            continue
        crudo = git(["show", f"{head}:{a}"], binario=True)
        for run in RUNS.findall(crudo):
            for t in tokens_de(run.decode("ascii", "replace")):
                cands.add((a, t))

    unicos = {t for _, t in cands}
    if len(unicos) > MAX_TOKENS:
        fallar(f"el diff supera el techo de {MAX_TOKENS} tokens unicos ({len(unicos)}).",
               "  Se falla en vez de truncar: un escaneo parcial que dice 'limpio' es fail-open.")

    print(f"scan-pr-secrets: {len(archivos)} archivos, {len(unicos)} tokens unicos, "
          f"{len(entradas)} entradas conocidas")

    hallados = []
    for salt, esperado, etiqueta in entradas:
        for a, t in cands:
            if hashlib.scrypt(t.encode("utf-8", "replace"), salt=salt,
                              n=2**14, r=8, p=1, dklen=32).hex() == esperado:
                hallados.append((a, etiqueta))
    if hallados:
        for a, e in sorted(set(hallados)):
            print(f"::error file={a}::secreto conocido '{e}' introducido por este PR")
        fallar(f"{len(set(hallados))} ocurrencia(s) de secreto conocido en el diff del PR.",
               "  Esta barrera corre server-side: no la saltea --no-verify ni un\n"
               "  .gitattributes local. El valor debe salir del diff.")
    print("scan-pr-secrets: sin secretos conocidos en el diff")
    return 0


if __name__ == "__main__":
    sys.exit(main())
