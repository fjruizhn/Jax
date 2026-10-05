# Traspaso — Faro 0.4 `memoria.buscar`

Actualizado: 2026-10-05 · rama `feat/faro-0.4-memoria-buscar`

## Estado

- Reconciliación en curso contra `origin/master` `e53e59e9d86bf55b20ba4f9305846e2839f5af36` mediante merge regular.
- Conflictos resueltos: `.github/workflows/policy.yml` conserva gates SR2/F2-E y documentación B9 de master; `ci/pisos.json` conserva los pisos aditivos y el mínimo B9 `176` de #354, superior al 128 previo de esta rama.
- El perfil de prueba sigue limitado a `jax_test@127.0.0.1:3308/jax_memory_test`; no se conectó B9 de producción ni se cambió configuración.

## Para quien retome

1. Verificar suite Faro focal y el subconjunto B9 compatible con este host; recontar la suite aislada completa en CI sobre el SHA publicado.
2. Si CI mide sobre 176, elevar el piso sin bajar ningún gate SR2/F2-E.
3. Archivar este traspaso en `docs/historia/2026-10-04-faro-memoria-buscar.md`, borrar `TRASPASO.md` y pedir auditoría Tier 3 sobre el SHA exacto.
