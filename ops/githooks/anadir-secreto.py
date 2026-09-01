#!/usr/bin/env python3
"""Agrega un secreto a ops/githooks/known-secrets.txt sin que su valor toque
el disco ni la pantalla.

  ./ops/githooks/anadir-secreto.py <etiqueta>

Pide el valor por stdin sin eco, lo hashea con el pepper y escribe solo el
HMAC. La etiqueta nombra la cuenta y la epoca -- nunca la forma, el largo ni
el charset del valor, que es justo lo que un atacante necesitaria para
acotar el espacio de busqueda.
"""
import getpass
import hashlib
import hmac
import os
import sys

PEPPER = os.environ.get("JAX_PRECOMMIT_PEPPER", "/etc/jax/precommit-pepper")
LISTA = os.environ.get("JAX_PRECOMMIT_SECRETS", "ops/githooks/known-secrets.txt")

if len(sys.argv) != 2:
    sys.exit(__doc__)
etiqueta = sys.argv[1].strip()
if not etiqueta or " " in etiqueta:
    sys.exit("La etiqueta va sin espacios. Ej: seed-superadmin-2026-06")

try:
    pepper = open(PEPPER, "rb").read().strip()
except OSError as exc:
    sys.exit(f"No se pudo leer el pepper ({PEPPER}): {exc.strerror}")
if not pepper:
    sys.exit(f"El pepper esta vacio ({PEPPER}).")

valor = getpass.getpass("Valor del secreto (no se muestra ni se guarda): ")
if not valor:
    sys.exit("Valor vacio, no se agrego nada.")

h = hmac.new(pepper, valor.encode(), hashlib.sha256).hexdigest()
del valor

with open(LISTA, "r", encoding="utf-8") as fh:
    if h in fh.read():
        sys.exit("Ese secreto ya estaba en la lista. No se agrego nada.")
with open(LISTA, "a", encoding="utf-8") as fh:
    fh.write(f"{h}  {etiqueta}\n")
print(f"Agregado: {h[:12]}...  {etiqueta}")
