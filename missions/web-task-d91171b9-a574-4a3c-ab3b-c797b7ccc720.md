---
faceta: hyde
---

Dos fixes en ~/jax-platform/frontend:
1. En src/i18n/es.js y en.js: el botón de login dice "Entrar a JAX" — cambiar la clave loginButton (o como se llame) a "Entrar a Axioma" en es.js y "Enter Axioma" en en.js
2. En index.html: el <title> debe decir "Axioma" — verificar que dice exactamente eso y no "JAX Platform" ni "Axioma Platform"
Rebuild después. Verificar que no quedan residuos de "JAX" en textos visibles al usuario.
