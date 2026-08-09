/using-superpowers
/ruflo

# Misión: purgar frontend/node_modules del historial de git (jax-platform)

## ADVERTENCIA — operación destructiva, leer completo antes de empezar

Reescribe el historial de commits del repo `jax-platform`. Todo hash de
commit después del primero que agregó `frontend/node_modules` va a cambiar.
Cualquier clon de este repo que no sea el de hall9000 queda desincronizado
y NO se puede arreglar con `git pull` — hay que re-clonar desde cero.

Precedente: ya se hizo esta misma operación con `backend/.venv` (ver
memoria/historial — commit de esa purga). Repetir el mismo procedimiento
validado, no improvisar uno nuevo.

NO hacer force-push a origin sin que Fernando confirme explícitamente en
esta sesión, después de ver el resultado de la Fase 1. Gate humano real,
no protocolar.

## Fase 1 — Reconocimiento (read-only, reportar y ESPERAR antes de Fase 2)

```bash
cd /home/fruiz/jax-platform

git remote -v
du -sh .git
cat .gitignore | grep -i node_modules
git log --oneline --diff-filter=A -- 'frontend/node_modules/*' | tail -5
which git-filter-repo || python3 -m pip show git-filter-repo 2>/dev/null
```

`git-filter-repo` probablemente ya está instalado desde la purga de `.venv`
de la sesión anterior (se instaló vía pipx) — confirmar antes de reinstalar.

Reportar todo antes de continuar. Si `git remote -v` muestra un remoto
real, confirmar con Fernando si hay otros clones activos en otro lado
(ya se sabe de hall9000 v1.0/BlackTower como precedente del caso anterior
— confirmar si aplica acá también, no asumir que es lo mismo).

## Fase 2 — Backup completo (obligatorio antes de cualquier rewrite)

```bash
cd /home/fruiz
cp -r jax-platform jax-platform.backup-pre-nodemodules-purge-$(date +%Y%m%d%H%M%S)
```

Confirmar que el backup se hizo y tiene tamaño razonable antes de seguir.

## Fase 3 — Purgar frontend/node_modules del historial

```bash
cd /home/fruiz/jax-platform
git filter-repo --path frontend/node_modules --invert-paths --force
```

## Fase 4 — Verificación

```bash
du -sh .git    # debe ser sensiblemente menor
git log --oneline --diff-filter=A -- 'frontend/node_modules/*'    # vacío
git status
ls frontend/node_modules 2>&1    # NO debe existir — filter-repo lo remueve
                                   # del working tree también
```

## Fase 5 — GATE HUMANO — no cruzar sin confirmación explícita

Reportar estado del remoto (de la Fase 1) y PREGUNTAR a Fernando si hace
`git push --force`. No asumir que sí.

## Fase 6 — Recrear node_modules funcional

El servicio `jax-platform-frontend.service` depende de `node_modules` para
correr `vite dev`. Recrearlo:

```bash
cd /home/fruiz/jax-platform/frontend
npm install
```

Confirmar que `.gitignore` ya tiene `frontend/node_modules/` (o
`node_modules/` genérico) para que no vuelva a trackearse — si no está,
agregarlo ANTES de este paso.

Con el prefijo `!`, pedirle a Fernando que confirme el restart:
```
! sudo systemctl restart jax-platform-frontend.service
```

Verificar después:
```bash
systemctl is-active jax-platform-frontend.service
curl -s https://axioma-ia.io/ | head -5   # debe ser HTML real, no error
```

## 7. Reporte final

Tamaño del repo antes/después, confirmación de que node_modules no aparece
más en `git log`, confirmación de que el frontend sigue funcionando tras
`npm install`, path del backup completo (rollback real), y si se hizo o no
force-push (y por qué).
