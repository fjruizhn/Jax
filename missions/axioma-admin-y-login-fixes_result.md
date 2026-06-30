# Resultado de: axioma-admin-y-login-fixes.md

---

## Resumen de entrega

### PRIORIDAD 0 — Systemd
- ✅ Proceso huérfano (PID 177569) eliminado
- ✅ Backend en puerto 8080 corriendo y respondiendo
- ⚠️ **Pendiente tu parte**: `sudo systemctl start jax-platform` — no puedo ejecutar sudo en esta sesión. El proceso actual es un nohup temporal; systemd no se activa sin terminal interactiva.

### MÓDULO 1 — Login UX y seguridad
- ✅ Toggle show/hide password con ícono SVG
- ✅ Error genérico "Usuario o contraseña incorrectos" (sin revelar cuál falló)
- ✅ Header `X-Attempts-Remaining` + contador visual en frontend
- ✅ A los 5 intentos: bloqueo 15 min, HTTP 423
- ✅ Mensaje con minutos restantes de bloqueo
- ✅ Link "¿Olvidaste tu contraseña?" → pantalla inline
- ✅ `POST /api/auth/forgot-password` + `POST /api/auth/reset-password`
- ✅ Página `/reset-password?token=...`
- ✅ BD: tabla `password_reset_tokens`, columnas `failed_attempts` / `locked_until`
- ✅ SMTP: si no hay config → el link aparece en el log del servidor (`/tmp/uvicorn-jax-platform.log`). Cuando quieras SMTP real, agrega `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` a `/etc/jax/.env`

### MÓDULO 2 — Admin
- ✅ **API Keys**: migradas a BD con cifrado Fernet. `FERNET_KEY` generada en `/etc/jax/.env`. Seed automático desde `.env` al primer acceso. Sync bidireccional BD↔.env.
- ✅ **Usuarios**: tabla muestra estado "bloqueado" en naranja, intentos fallidos, botón "Desbloquear" (`POST /admin/users/{id}/unlock`)
- ✅ **Configuración**: default "Axioma" ya existía y funciona
- ✅ **Dashboard**: usuarios activos, usuarios bloqueados, API Keys configuradas/total, RAM% via psutil
- ✅ **Volver a Axioma**: ya existía en el sidebar, confirmado

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
