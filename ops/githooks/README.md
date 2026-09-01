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

## `pre-commit` — reintroducción de secretos conocidos

Rechaza un commit cuya línea **añadida** contenga un secreto **ya conocido**,
o que toque rutas de la clase donde los secretos aparecieron.

**No es un detector de patrones, y la razón está medida.** El 2026-09-01
`gitleaks` 8.30.1 devolvió cero hallazgos sobre un repo cuya historia contiene
una contraseña de superadmin en texto plano, y sobre el archivo que denunció
GitGuardian dio `no leaks found` incluso servido como texto plano a
`gitleaks dir`. La contraseña real tiene **ocho caracteres**: no hay formato
ni entropía que disparar. Un hook por patrón habría sido ciego al único caso
que este repo ya sufrió.

### Cómo compara

`HMAC-SHA256(pepper, token)` contra `known-secrets.txt`. El **pepper vive
fuera del repo** (`/etc/jax/precommit-pepper`, 0600) porque la lista viaja
versionada en un repo **público**: un `sha256` pelado de una contraseña de 8
caracteres se crackea con wordlist en minutos, y publicarlo sería publicar la
contraseña otra vez. `bcrypt`/`argon2`, que sí serían seguros de publicar,
cuestan ~250 ms por comparación — un commit de 200 líneas contra 5 hashes
tardaría minutos. HMAC conserva las dos propiedades.

### Fail-closed (P10)

Si el pepper o la lista **no se pueden leer, no parsean, o la lista está
vacía**, el hook **rechaza**. Un comparador sin nada con que comparar no
"no encontró nada": no comparó. Es la distinción exacta que separa este hook
de un escáner en verde.

### Rutas

| Clase | Comportamiento |
|---|---|
| `*_result.md` | **bloqueo duro** — la clase donde apareció el único hallazgo real |
| `missions/`, `prompts/`, `*seed*`, `*test*credential*`, `*.pyc` | exigen marca deliberada: `JAX_PRECOMMIT_ALLOW_PATH=1 git commit ...` |

### Agregar un secreto

```bash
./ops/githooks/anadir-secreto.py seed-superadmin-2026-06
```

Pide el valor por stdin sin eco; solo se escribe el HMAC.

### Lo que NO cubre — declarado, no disimulado

- **Un secreto NUEVO pasa.** Este hook previene la **reintroducción**, que es
  lo que de verdad ocurrió: el commit que sacó la contraseña de `seed.py` la
  escribió en el test de regresión que probaba que ya no se usaba. Contra
  secretos nuevos hace falta otra cosa — inyección por env var sin literales
  en tests, o revisión obligatoria de rutas de alto riesgo.
- **`--no-verify` lo saltea.** Es evadible por diseño: un acto explícito, no
  un resbalón. No es un control duro y no se pretende que lo sea.
- Solo existe donde `core.hooksPath` apunte acá.
