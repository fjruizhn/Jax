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

### Canario

La primera entrada de `known-secrets.txt` es `HMAC(pepper, "JAX-PRECOMMIT-CANARY")`.
Si el pepper se rota, regenera o copia con un byte distinto, **todos los HMAC
cambian**: sin canario el hook seguiría arrancando en verde, comparando contra
nada y aprobando todo, con toda la apariencia de estar sano — el modo de falla
invisible exacto que este diseño existe para evitar. El canario lo convierte en
un rechazo ruidoso. **No borrar esa línea.**

### Contenido que el diff no muestra

Todo archivo staged del que el diff **no** produjo ni una línea de texto se
escanea **entero desde el índice**, por corridas imprimibles estilo `strings`.
Cubre tres huecos que eran falsos negativos silenciosos:

- **Binarios.** `git diff` emite `Binary files … differ` y cero líneas `+`.
  Es la clase del `.pyc` que conservó la contraseña real cuando `filter-repo`
  limpió el `.py` — el hallazgo que originó todo esto.
- **`.gitattributes` con `-diff`.** Una línea de aspecto inocente (`*.md -diff`,
  "los diffs de docs son ruidosos") apagaba la revisión de contenido de toda una
  clase de rutas, de forma permanente y sin dejar rastro en el commit ofensivo.
  Era **más barata que `--no-verify`**.
- **Typechange** (`T`) y cualquier caso que el parser de hunks no cubra.

### Lo que NO cubre — declarado, no disimulado

- **Un secreto NUEVO pasa.** Este hook previene la **reintroducción**, que es
  lo que de verdad ocurrió: el commit que sacó la contraseña de `seed.py` la
  escribió en el test de regresión que probaba que ya no se usaba. Contra
  secretos nuevos hace falta otra cosa — inyección por env var sin literales
  en tests, o revisión obligatoria de rutas de alto riesgo.
- **`--no-verify` lo saltea.** Es evadible por diseño: un acto explícito, no
  un resbalón. No es un control duro y no se pretende que lo sea.
- **Un secreto embebido sin separadores alrededor no matchea.** La comparación
  es por token exacto: `pw = "<secreto>"` dispara, `pw = "prefijo<secreto>"` no.
  Es inherente al match por valor y no se arregla bajando el umbral.
- **Un secreto conocido de menos de 6 caracteres, o que contenga un separador
  (espacio, comilla, `= : ; ,` paréntesis, corchetes…), nunca se tokenizaría.**
  `anadir-secreto.py` **rechaza** esos valores en vez de escribir una entrada
  muerta que parecería cobertura y no lo sería. Ojo con el padding `=` de
  base64: es el caso más común de esta trampa.
- **Tres evasiones, no una.** Además de `--no-verify`: apuntar
  `JAX_PRECOMMIT_SECRETS` o `JAX_PRECOMMIT_PEPPER` a archivos propios, y borrar
  una entrada de la lista **en el working tree** sin commitear (la lista se lee
  del árbol, no del índice ni de HEAD). Las tres son actos explícitos.
- **`merge`, `rebase`, `cherry-pick`, `revert` y `stash` NO ejecutan este hook.**
  Solo `commit` y `commit --amend`. Consecuencia concreta: un commit hecho antes
  de activar `core.hooksPath`, o hecho con `--no-verify`, **entra a `master` por
  merge o rebase sin pasar jamás por acá**. El `pre-push` tampoco mira contenido
  —solo el ref destino—, así que **hoy no existe barrera de contenido en el
  camino rama → master**. Queda anotado como deuda, no disimulado.
- Solo existe donde `core.hooksPath` apunte acá.

### Recordatorio operativo (no es una lección nueva, ya está escrita)

**Usar `git -C <ruta>` siempre.** El 2026-09-01, publicando este mismo hook, se
empujó la misma rama dos veces por asumir que un `cd` anterior seguía vigente:
el segundo `git push` corrió en el checkout equivocado. **Ni este hook ni el CI
atajan esa clase de error** — el hook mira contenido, el CI mira el resultado, y
un push al repo equivocado es correcto en ambos. La única defensa es no depender
del cwd. Ver la lección correspondiente en `CONTEXT.md` §7.

**`git reset --hard` no se usa para limpiar durante una tarea con cambios sin
commitear.** Se lleva lo recién aplicado junto con lo que querías descartar. El
2026-09-01, probando este mismo hook, un `--hard` para deshacer un commit de
prueba borró el cambio al hook que se acababa de escribir. **Se detectó porque
se verificó el archivo después, no porque se notara al hacerlo** — un
`reset --hard` no avisa de lo que se llevó puesto. Para limpiar una prueba:
`git reset HEAD <archivo>` y borrar el archivo, sin tocar el árbol. Cubierto
por la lección de verificar el estado real en vez de suponerlo.
