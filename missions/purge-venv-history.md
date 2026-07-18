/using-superpowers
/ruflo

# Misión: purgar backend/.venv del historial de git (jax-platform)

## ADVERTENCIA — operación destructiva, leer completo antes de empezar

Esta misión reescribe el historial de commits del repo `jax-platform`. Todo
hash de commit después del primero que agregó `backend/.venv` va a cambiar.
Cualquier clon de este repo que no sea el de hall9000 (otra máquina, un
backup en Sésamo, un fork) queda desincronizado y NO se puede arreglar con
`git pull` — hay que re-clonar desde cero.

NO hacer force-push a ningún remoto sin que Fernando confirme explícitamente
en esta sesión, después de ver el resultado de la Fase 1 (reconocimiento).
Este es un gate humano real, no protocolar — pausar y preguntar.

## Fase 1 — Reconocimiento (read-only, reportar y ESPERAR antes de Fase 2)

```bash
cd /home/fruiz/jax-platform

# ¿Hay remoto configurado? ¿A dónde apunta?
git remote -v

# Tamaño actual del repo (para comparar después)
du -sh .git

# Confirmar que .venv está en .gitignore para que no vuelva a trackearse
cat .gitignore | grep -i venv

# Historial de cuándo se agregó .venv (para entender el alcance del rewrite)
git log --oneline --diff-filter=A -- 'backend/.venv/*' | tail -5

# ¿git filter-repo está instalado? Es la herramienta recomendada (mejor que BFG
# para este caso). Si no está, reportar y no instalar sin confirmar con Fernando
# (viene por pip o por el paquete del sistema, no asumir cuál usar).
which git-filter-repo
python3 -m pip show git-filter-repo 2>/dev/null
```

Reportar TODO esto antes de continuar. En particular: si `git remote -v`
muestra un remoto real (no vacío), preguntar a Fernando si ese remoto tiene
otros clones activos en otro lado (otra máquina, CI, backup) antes de
proponer cualquier rewrite.

## Fase 2 — Backup completo (obligatorio antes de cualquier rewrite)

```bash
cd /home/fruiz
cp -r jax-platform jax-platform.backup-pre-purge-$(date +%Y%m%d%H%M%S)
```

Confirmar que el backup se hizo y tiene el mismo tamaño aproximado que el
original antes de seguir.

## Fase 3 — Purgar backend/.venv del historial

Usando `git filter-repo` (instalar vía pip si no está, PERO confirmar con
Fernando el método de instalación antes — puede que prefiera pipx o un venv
aislado para no ensuciar el Python del sistema):

```bash
git filter-repo --path backend/.venv --invert-paths --force
```

`--force` es necesario porque filter-repo por defecto rechaza correr sobre
un repo que no es un clon fresco — leer su output con cuidado, no ignorar
warnings.

## Fase 4 — Verificación

```bash
du -sh .git    # debe ser sensiblemente menor que el número de la Fase 1
git log --oneline --diff-filter=A -- 'backend/.venv/*'    # debe estar vacío
git status
ls backend/    # confirmar que backend/.venv YA NO EXISTE ni siquiera en working tree
              # (filter-repo lo remueve del tree actual también — si Fernando
              # necesita el venv funcional, hay que recrearlo con
              # python3 -m venv backend/.venv && pip install -r requirements.txt
              # DESPUÉS de este paso, no antes)
```

## Fase 5 — GATE HUMANO — no cruzar sin confirmación explícita

Si en la Fase 1 había un remoto configurado, reportar el estado y PREGUNTAR
a Fernando si querés que hagas `git push --force` a ese remoto. No asumir
que sí. Si no hay remoto, o si Fernando dice que no hay otros clones activos,
proceder es más seguro pero igual anunciarlo antes de ejecutar.

## Fase 6 — Recrear el venv funcional

El servicio `jax-platform.service` (backend) probablemente depende de
`backend/.venv/bin/uvicorn` para arrancar (confirmar con
`systemctl cat jax-platform.service | grep ExecStart`). Como el filter-repo
remueve el venv del working tree, hay que recrearlo:

```bash
cd /home/fruiz/jax-platform/backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # confirmar que este archivo existe
                                              # y tiene las dependencias reales
                                              # (si no hay requirements.txt,
                                              # PARAR y preguntar a Fernando
                                              # cómo reconstruir las deps)
```

Con el prefijo `!`, pedirle a Fernando que confirme el restart del servicio:
```
! sudo systemctl restart jax-platform.service
```

Verificar que el servicio arrancó sano después:
```bash
systemctl is-active jax-platform.service
curl -s http://127.0.0.1:8080/api/health   # o el endpoint de health que exista
```

## 7. Reporte final

Tamaño del repo antes/después, confirmación de que .venv no aparece más en
`git log`, confirmación de que el servicio backend sigue funcionando tras
recrear el venv, path del backup completo de la Fase 2 (rollback real: borrar
el repo actual y restaurar desde ahí si algo salió mal), y si se hizo o no
force-push (y por qué).
