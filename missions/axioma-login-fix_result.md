# Resultado de: axioma-login-fix.md

Listo. Dos fixes, build limpio, servicio activo:

- **`Login.jsx`**: agregado `navigate('/')` después del `await login()` — el store ya tenía el token, solo faltaba moverle la pantalla.
- **`useJaxStore.js`**: el estado inicial ahora lee `jax_token` / `jax_user` de localStorage directamente, en lugar de esperar al `useEffect`. Esto elimina el flash de redirección al recargar.

🔧 *Origen de autoridad: conocimiento técnico de Hyde. Los comandos ejecutados o verificados se indican en el cuerpo; el resto es de conocimiento, no verificado en vivo.*
