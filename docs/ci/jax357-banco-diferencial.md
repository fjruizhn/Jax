# Banco diferencial de JAX#357 (`docker rm` sin `-v`)

**El freno de CI es el archivo de pytest** `tests/test_docker_rm_sin_fuga_de_volumenes.py`
(corre en el job `tests-puros` y barre todo archivo versionado). El banco de esta carpeta
**no lo corre ningún workflow**: es evidencia diferencial, no un freno. Un banco en verde no
reemplaza al pytest, y el pytest no depende del banco.

Qué es: `docs/ci/jax357-differential.py.txt` carga el escáner en dos SHAs y compara qué marca cada uno sobre
un conjunto fijo de casos (los del PR, `jax357-auditor-cases.json` y los propios).
Se declara **fuga** (`LEAK`) si el SHA candidato deja de marcar algo que marcaba el de
referencia, y **marca nueva** (`NEW_MARK`) si marca algo que el de referencia no marcaba y
no está declarado como mejora esperada. Falla con código 1 ante cualquiera de las dos o
ante una mejora esperada que no se cumple.

Cómo se corre, desde la raíz del repo, con ambos SHAs presentes en el checkout local:

```
python3 docs/ci/jax357-differential.py.txt <SHA_MASTER> <SHA_CANDIDATO> > docs/ci/jax357-differential.out
```

Qué SHAs compara el `.out` versionado: referencia = `master` en
`f47820f5af36d7b0e4e0d5b2156ac6275c10462c` (su escáner es el de antes de la rama);
candidato = el commit del escáner de la ronda 6, indicado en la primera línea de
resumen del propio `.out` (`candidate_sha=`). Al cambiar el escáner hay que regenerar el
`.out` con el SHA nuevo.

Desde la ronda 8 el banco no incluye los casos `docker rm -v c SEP rm -f p`: la lectura argv
del escáner (sin cortar) los marca a propósito, y eso es un falso positivo aceptado
(documentado en el docstring del pytest, con pruebas). El banco exige 0 fugas respecto de master.
