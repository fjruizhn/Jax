faceta: hyde

# Diagnóstico de sys.path REAL del servicio jax-las-manos (instrumentación temporal)

> Objetivo: capturar el sys.path, cwd, y resolución de 'jacobs' DENTRO del proceso
> systemd real (no a mano — el python -c a mano engaña porque el shell mete el cwd).
> Esto es DIAGNÓSTICO, no arreglo. Instrumentación temporal y reversible.
> NO consolidar nada todavía. Solo medir la verdad del proceso.

## Principios (HYDE)
- Backup de server.py antes de tocar. Cambio temporal, se revierte tras leer el log.
- No suponer. Medir. "El que supone se equivoca."

## Paso 1 — Backup
```bash
cp ~/jax/las_manos/server.py ~/jax/las_manos/server.py.backup-syspath-diag-$(date +%Y%m%d-%H%M%S)
```

## Paso 2 — Insertar bloque de debug ANTES del import de jacobs
Ubicar en `~/jax/las_manos/server.py` la línea:
    from jacobs.routes import router as jacobs_router
JUSTO ANTES de esa línea, insertar este bloque (idéntico, sin cambios):

```python
# === DEBUG SYSPATH TEMPORAL — remover tras diagnóstico ===
import os as _os, sys as _sys, importlib.util as _ilu
try:
    with open("/tmp/jax_syspath_debug.log", "w") as _f:
        _f.write(f"PID: {_os.getpid()}\n")
        _f.write(f"CWD: {_os.getcwd()}\n")
        _f.write(f"EXECUTABLE: {_sys.executable}\n")
        _f.write(f"SERVER_FILE: {__file__}\n")
        _f.write(f"PYTHONPATH env: {_os.environ.get('PYTHONPATH')!r}\n")
        _f.write("=== SYS.PATH ===\n")
        for _i, _p in enumerate(_sys.path):
            _f.write(f"  {_i}: {_p!r}\n")
        _spec = _ilu.find_spec("jacobs")
        _f.write("=== find_spec('jacobs') ===\n")
        _f.write(f"  origin: {getattr(_spec, 'origin', None)!r}\n")
        _f.write(f"  search_locations: {getattr(_spec, 'submodule_search_locations', None)!r}\n")
        _f.write("=== EXISTENCIA EN DISCO ===\n")
        for _q in ["/home/fruiz/jax/jacobs/__init__.py",
                   "/home/fruiz/jax/las_manos/jacobs/__init__.py"]:
            _f.write(f"  {_q}: exists={_os.path.exists(_q)}\n")
except Exception as _e:
    with open("/tmp/jax_syspath_debug_err.log", "w") as _f:
        _f.write(repr(_e))
# === FIN DEBUG TEMPORAL ===
```

## Paso 3 — Reiniciar y leer la VERDAD
```bash
sudo systemctl daemon-reload
sudo systemctl restart jax-las-manos
sleep 8
systemctl is-active jax-las-manos
echo "===================== SYS.PATH REAL DEL SERVICIO ====================="
cat /tmp/jax_syspath_debug.log
echo "===================================================================="
```

## Paso 4 — Reporte
Pegar el CONTENIDO COMPLETO de /tmp/jax_syspath_debug.log en el resultado.
Confirmar que el servicio quedó 'active' tras la instrumentación.
NO remover el bloque debug todavía (se remueve después de decidir la consolidación).
NO consolidar A/B en esta misión.

## Criterio de aceptación
- /tmp/jax_syspath_debug.log existe y muestra: sys.path real, find_spec origin, cwd.
- Servicio 'active'.
- Reporte incluye el log completo + ubicación real de donde se importó jacobs.
