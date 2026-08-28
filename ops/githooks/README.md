# Hooks versionados

Barrera **local** contra un push que aterrice en `master`. El control real
es el ruleset del repo (`bypass_mode: pull_request`, que rechaza del lado
del servidor); esto falla antes, dice por qué, y sigue puesto si alguna vez
se afloja el ruleset.

## Activarlos (hace falta una vez por clon)

```bash
git config core.hooksPath ops/githooks
```

Git no puede activar hooks solo desde el repo — sería ejecución de código
arbitrario al clonar. Por eso el paso es manual y explícito. Los hooks
viven en el directorio común, así que **un solo `git config` cubre todos
los worktrees** del clon.

## Verificarlos

```bash
git config core.hooksPath                      # -> ops/githooks
git commit --allow-empty -m "prueba" && git push origin master   # -> RECHAZADO
git reset --hard @~1
```

Un hook que nunca se vio rechazar no es un hook, es una hipótesis.
