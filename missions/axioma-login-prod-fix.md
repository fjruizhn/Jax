# MISIÓN URGENTE: axioma-login-produccion

## Problema
Login funciona en localhost:5173 pero NO en https://axioma-ia.io
Tengo demo con sponsors en menos de 30 minutos.

## Arquitectura de producción
- Frontend estático: /www/wwwroot/axioma-ia.io/ en atemai-net (<IP interna, ver /etc/jax/.env>)
- Backend: hall9000 (<IP interna, ver /etc/jax/.env>) :8080 — uvicorn PID 177569 (nohup, no systemd)
- Nginx en atemai-net proxy /api → <IP interna, ver /etc/jax/.env>:8080
- Cloudflare proxy activo en axioma-ia.io

## Diagnóstico requerido ANTES de tocar nada
1. curl -X POST https://axioma-ia.io/api/auth/login -H "Content-Type: application/json" -d '{"email":"fernando@rich-hn.com","password":"<contraseña, ver gestor de secretos>"}' -v 2>&1 | head -40
2. curl -X OPTIONS https://axioma-ia.io/api/auth/login -H "Origin: https://axioma-ia.io" -H "Access-Control-Request-Method: POST" -v 2>&1 | head -30
3. ps aux | grep uvicorn

## Fix probable
CORS: verificar que el uvicorn corriendo (PID 177569) tiene FRONTEND_ORIGIN=https://axioma-ia.io en su entorno:
  cat /proc/177569/environ | tr '\0' '\n' | grep FRONTEND

Si no tiene la variable, matar y reiniciar con el env correcto.

## Restricción absoluta
NO tocar jax-platform-frontend ni Nginx ni ningún otro servicio.
Solo el proceso uvicorn del backend y main.py si es necesario.

## Evidencia requerida
curl -X POST https://axioma-ia.io/api/auth/login con respuesta JSON válida (token)
