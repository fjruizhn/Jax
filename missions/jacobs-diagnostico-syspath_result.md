# Resultado de: jacobs-diagnostico-syspath.md

La verdad medida. Aquí está el reporte completo:

---

## Diagnóstico sys.path — Servicio `jax-las-manos` — 2026-06-29

### Datos crudos del proceso real (PID 20841)

```
CWD:         /home/fruiz/jax/las_manos
EXECUTABLE:  /home/fruiz/jax/las_manos/.venv/bin/python3.12
SERVER_FILE: /home/fruiz/jax/las_manos/server.py
PYTHONPATH:  None (no seteado)
```

### sys.path del proceso

| # | Ruta |
|---|------|
| 0 | `''` ← **CWD = `/home/fruiz/jax/las_manos`** |
| 1 | `/home/fruiz/jax/las_manos/.venv/bin` |
| 2 | `/usr/lib/python312.zip` |
| 3 | `/usr/lib/python3.12` |
| 4 | `/usr/lib/python3.12/lib-dynload` |
| 5 | `/home/fruiz/jax/las_manos/.venv/lib/python3.12/site-packages` |

**`/home/fruiz/jax` nunca aparece en sys.path.**

### Resolución real de `jacobs`

```
origin:           /home/fruiz/jax/las_manos/jacobs/__init__.py
search_locations: ['/home/fruiz/jax/las_manos/jacobs']
```

### Existencia en disco

```
/home/fruiz/jax/jacobs/__init__.py         → existe (canónico)
/home/fruiz/jax/las_manos/jacobs/__init__.py → existe (el que gana)
```

---

### Conclusión — La verdad sin suponer

- El servicio importa `jacobs` desde **`/home/fruiz/jax/las_manos/jacobs/`**, no desde **`/home/fruiz/jax/jacobs/`** (canónico).
- El mecanismo: `sys.path[0] = ''` (CWD) resuelve `''` → `/home/fruiz/jax/las_manos`, y ahí existe un `jacobs/`. Gana sin buscar más.
- `/home/fruiz/jax` nunca entró al path porque `PYTHONPATH` es `None` y el venv es local a `las_manos/`.
- Hay **dos instancias de `jacobs`** en disco. Cualquier cambio en el canónico (`/home/fruiz/jax/jacobs/`) es invisible para el servicio mientras exista la copia local.

### Estado del servicio

`active` — instrumentación temporal en pie, sin remover todavía.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
