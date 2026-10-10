# Faro F1.1: rebase del grant y del revert tras corregir readiness

**Fecha:** 2026-10-10  
**Tipo:** HISTORIA / PENDIENTE  
**Fuente:** PR #392, run de PR `38051651432` (head `597b9a75d32874a84613e1ed7a0de40ea9e620ab`), run post-merge `38052737043`, PRs #387 y #390.

## Hechos

- El guard post-merge de #385, run `38022192036`, devolvió `REVERT_REQUIRED` al fallar el piso MariaDB Rule Authority: 11 casos pasaron y una conexión se perdió durante la inicialización. Los logs del contenedor no se conservaron; la carrera con el servidor temporal quedó como hipótesis, no causa demostrada.
- PR #392 añadió espera del servidor final, sondas `SELECT 1` y `@@port = 3306`, reintentos acotados solo para errores transitorios y diagnóstico del contenedor. El SHA exacto pasó 20 pruebas (`20 passed in 50.09s`) y todos sus checks; se integró como `8d1d1c49c5795fd6416085084265006fbcb6d69b`. `post-merge-guard` verificó el run `38052737043` SUCCESS y que `master` quedó en ese merge sin carrera.
- La auditoría Tier 3 del SHA #392 fue APROBADO, 0 BLOCK / 0 MAJOR / 0 MINOR. Kimi y GLM/ZCode aprobaron con observaciones bajas no bloqueantes. Las observaciones no cambian la prueba exacta ni el guard.
- El comparador de pisos en `master` se extrae desde el blob de `master` y corre aislado contra el árbol candidato; la historia shallow falla cerrada.
- PR #390 (grant exacto para #387) había quedado ligado a la definición anterior de 12 pruebas. Tras #392, el piso vigente es 20. Se rebasó localmente el grant sobre `master@8d1d1c49`, se actualizaron `floor_definition` y su SHA-256, y se calculó `expected_diff_sha256=aff23332d22338202316a3f4175bb32c29102794c16f837e7bcfc57c37e019fc` con el comparador oficial sobre el candidato simulado de 11 rutas.

## Decisiones

- El grant #390 debe ser un merge dedicado e inmediatamente anterior a #387. Si la base avanza o cambia el árbol del revert, rehacer el grant y el hash; no reutilizar la autorización.
- El owner de #388 conserva su worktree y rama. Codex no los modifica; el owner deberá reapilar tras el revert y revalidar sus pisos.
- El archivo `/etc/jax/authority/trusted-root.json` no está presente en el servidor y no se encontró una restauración autorizada. Nadie debe fabricarlo; producción permanece bloqueada hasta recuperar la fuente autorizada.

## Pendientes

- Confirmar la simulación del merge dedicado #390 con la definición del piso 20 y hash OID exacto.
- Auditoría/CI/preflight/merge/guard de #390 y luego reapilar, auditar y verificar #387.
- Reabrir coordinación con owner de #388 después del revert.
- Recuperar el trusted root de fuente autorizada y completar la verificación operativa de producción.

**Decidió:** Codex, sesión coordinadora con ventana de Fernando abierta, 2026-10-10. Fernando dio GO para resolver el cierre técnico; el GO no sustituye el origen autorizado del trusted root.
