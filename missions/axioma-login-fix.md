faceta: hyde

# Fix urgente: login no redirige al dashboard en Axioma

## Problema
El login hace POST /api/auth/login → recibe 200 OK con token → pero el frontend no redirige al dashboard. El usuario queda pegado en la pantalla de login.

## Diagnóstico
- useJaxStore.login() setea token y user en el store y localStorage
- App.jsx tiene rutas pero la condición de autenticación no detecta el cambio de estado

## Fix
En App.jsx: la ruta protegida debe leer `token` del store (o localStorage) para decidir si mostrar Dashboard o redirigir a Login. Si token existe → Dashboard. Si no → Login.

En useJaxStore.js: verificar que el estado inicial lee el token de localStorage al arrancar (para que al recargar la página el usuario siga autenticado).

## Verificación
1. Entrar con fernando@rich-hn.com / [contraseña, rotada -- ver gestor de secretos] → redirige al Dashboard
2. Recargar la página → sigue en Dashboard (token persistido)
3. Logout → redirige a Login

Rebuild después del fix.
Escribir resultado en ~/jax/missions/axioma-login-fix_result.md
