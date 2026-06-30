# MISIÓN URGENTE: axioma-emergency-production-fix

## Contexto
Axioma Platform dejó de funcionar local y en producción después de cambios de anoche.
Tengo pitch con sponsors en menos de 1 hora. Necesito login funcionando.

## Síntomas
- localhost:5173 no funciona (antes funcionaba)
- https://axioma-ia.io no permite login (CORS)
- Los servicios pueden estar caídos o con configuración rota

## Servicios a verificar y reparar
1. jax-platform (backend FastAPI :8080)
2. jax-las-manos (LAS MANOS :7777)
3. jax-platform-frontend (frontend — fue deshabilitado anoche, reemplazado por Nginx estático)

## Diagnóstico requerido primero
1. sudo systemctl status jax-platform jax-las-manos
2. sudo journalctl -u jax-platform -n 30 --no-pager
3. curl -s http://localhost:8080/api/health

## Fix CORS producción
En ~/jax-platform/backend/main.py, reemplazar ALLOWED_ORIGINS con lista que incluya:
- http://localhost:5173
- http://127.0.0.1:5173  
- https://axioma-ia.io
- https://www.axioma-ia.io
- https://admin.axioma-ia.io
- os.getenv("FRONTEND_ORIGIN", "")

## Fix frontend local
El servicio jax-platform-frontend fue deshabilitado anoche.
Opciones:
A) Re-habilitar el servicio Vite dev para localhost
B) Instalar Nginx en hall9000 y servir dist/ en :5173

Evaluar cuál es más rápido y ejecutar.

## Protocolo
- Diagnosticar ANTES de tocar nada
- Backup de main.py antes de modificar
- Verificar cada fix con evidencia
- Reportar estado final con curl y systemctl status
- TIEMPO LÍMITE: 30 minutos

## Prioridad absoluta
Login funcionando en https://axioma-ia.io para demo con sponsors.
