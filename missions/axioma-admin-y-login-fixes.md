# MISIÓN: axioma-admin-y-login-fixes-v1

## PRIORIDAD 0 — Estabilizar jax-platform bajo systemd

El uvicorn actual corre como proceso huérfano (nohup, no systemd).
Si muere, Axioma cae sin recuperación automática.

1. Identificar PID del uvicorn huérfano:
   ps aux | grep uvicorn | grep 8080

2. Matar el proceso huérfano:
   kill {PID}

3. Iniciar bajo systemd:
   sudo systemctl start jax-platform
   sudo systemctl status jax-platform

4. Verificar que el backend responde:
   curl -s http://localhost:8080/api/health

5. Verificar login en producción:
   curl -s -X POST https://axioma-ia.io/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"fernando@rich-hn.com","password":"TU_PASSWORD"}' | python3 -m json.tool

## MÓDULO 1 — Login UX y seguridad

Archivo: ~/jax-platform/frontend/src/ (buscar componente Login)
Archivo: ~/jax-platform/backend/api/auth/ (buscar rutas de autenticación)

### 1A — Frontend: mejoras de UX en pantalla de login
- Toggle show/hide password (ojo) en el campo de contraseña
- Mensaje de error claro cuando credenciales son incorrectas:
  "Usuario o contraseña incorrectos" (no revelar cuál de los dos)
- Contador visual de intentos restantes: "Te quedan X intentos"
- Después del intento 5: mensaje "Cuenta bloqueada. Revisa tu correo."
- i18n: agregar claves en es.js y en.js
- dark/light: CSS variables siempre

### 1B — Backend: bloqueo por intentos fallidos
En la tabla users (o crear tabla login_attempts):
  - failed_attempts: INTEGER DEFAULT 0
  - locked_until: DATETIME NULL

Lógica en el endpoint POST /api/auth/login:
  1. Si locked_until > NOW(): retornar 423 con mensaje "Cuenta bloqueada hasta {tiempo}"
  2. Si credenciales incorrectas: incrementar failed_attempts
  3. Si failed_attempts >= 5: establecer locked_until = NOW() + 15 minutos
  4. Si login exitoso: resetear failed_attempts = 0, locked_until = NULL
  5. Registrar cada intento en audit log con IP y timestamp

### 1C — Recuperación de contraseña por correo
Frontend: link "¿Olvidaste tu contraseña?" en la pantalla de login
Backend: 
  - POST /api/auth/forgot-password → recibe email, genera token único (UUID), 
    guarda en tabla password_reset_tokens (token, user_id, expires_at = NOW()+1h, used=false)
    Envía email con link: https://axioma-ia.io/reset-password?token={token}
  - POST /api/auth/reset-password → recibe token + nueva password,
    valida token (existe, no usado, no expirado), actualiza password, marca token usado
Frontend: pantalla /reset-password con formulario nueva contraseña + confirmación

Email: usar SMTP configurado en /etc/jax/.env o ~/jax-platform/backend/.env
  Buscar SMTP_HOST, SMTP_USER, SMTP_PASSWORD, SMTP_FROM

### 1D — Tabla password_reset_tokens (nueva)
  - id: INTEGER PRIMARY KEY
  - user_id: INTEGER FK users(id)
  - token: VARCHAR(36) UNIQUE
  - expires_at: DATETIME
  - used: BOOLEAN DEFAULT FALSE
  - created_at: DATETIME DEFAULT NOW()
  - ip_address: VARCHAR(45)

## MÓDULO 2 — Admin: correcciones pendientes

### 2A — API Keys: BD multi-usuario
(Ver misión axioma-admin-multiuser-foundation-v1 en ~/jax/missions/)
Implementar esquema completo:
  - Tabla user_api_keys con cifrado Fernet
  - Seed desde /etc/jax/.env al superadmin
  - Endpoints GET/POST/PUT/DELETE /api/admin/api-keys
  - Frontend: tabla con preview de keys, modal agregar/editar

### 2B — Usuarios: tabla funcional
  - GET /api/admin/users debe retornar todos los usuarios de la tabla users
  - Agregar columnas: failed_attempts, locked_until al schema
  - Frontend: mostrar estado "Bloqueado" si locked_until > NOW()
  - Botón "Desbloquear" para superadmin

### 2C — Configuración: nombre del sistema
  - Default "Axioma" si no hay valor guardado
  - Tabla system_config con seed inicial

### 2D — Dashboard Admin
  - Total usuarios activos
  - Usuarios bloqueados
  - API Keys configuradas vs faltantes
  - Estado servicios: jax-platform :8080, jax-las-manos :7777
  - RAM usage via psutil

### 2E — Botón "Volver a Axioma"
  - Navegar a "/" con react-router navigate('/')

## Protocolo de entrega
1. Backup: cp -r ~/jax-platform ~/jax-platform-backup-$(date +%Y%m%d-%H%M%S)
2. Orden: PRIORIDAD 0 → MÓDULO 1 → MÓDULO 2
3. Después de cambios backend: sudo systemctl restart jax-platform
4. Después de cambios frontend: 
   cd ~/jax-platform/frontend && npm run build
   rsync -av dist/ /tmp/axioma-dist/
   ssh -p 58291 fruiz@172.16.20.11 "sudo cp -r /tmp/axioma-dist/* /www/wwwroot/axioma-ia.io/ && sudo nginx -s reload"
5. Verificar cada módulo con evidencia antes de marcar completo

## Restricciones ADN
- i18n SIEMPRE: es.js y en.js
- dark/light SIEMPRE: CSS variables
- Sin hardcoding
- Soft delete en usuarios
- NUNCA retornar passwords ni API keys en plano
- Bloqueo de cuenta: 5 intentos, 15 minutos
- Backup antes de modificar BD
