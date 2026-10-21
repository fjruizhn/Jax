# OP343 / OP345 · Office namespace · Codex · 2026-10-20

Fernando autorizó iniciar desarrollos. Rama propia codex/office-namespace-20261020, base98d42f2a; JXP par en codex/virtual-office-20261020. Marca c3d2c999 publicada.

Spec/plan docs/superpowers/{specs,plans}/2026-10-20-office-namespace.md. Sol resolvió usar autoridad global SOLO de Office en JXP, no GLOBAL_FULL_ACCESS para Axioma clásico/Faro. Este repo implementa namespace y creación autorizada atómica, sin habilitar API ni clientes por sí solo.

VERDAD OPERACIONAL: implementación Tier2 verificada por raíz: 50 pruebas enfocadas pasan, incluidas 3 reales MariaDB12.3.3 aislada33991. Namespace CLASSIC en consumidores y creación Office transaccional, idempotencia con actor y reintentos acotados. Harness CI=true solo selecciona DB desechable; sin GITHUB_ACTIONS ni override productivo. Pendiente inmediato: revisión Sol sobre SHA exacto; no consumir API nueva antes del gate. No tocar producción ni otros worktrees. Siguiente lectura: cat docs/superpowers/plans/2026-10-20-office-namespace.md.

Fixround1: raíz58passed67warnings0.43s; LASMANOSCLASSIC, downgradeguard y pruebasreales ampliadas. Evidencia docs/superpowers/reviews/2026-10-20-office-namespace-evidence.md. Dictamen anterior RECHAZADO28c1cb3e; corrección requiere scopedrereview exactSHA.

2026-10-20 · Task2 marker de compatibilidad namespace `office-namespace.1` añadido sin cambiar autoridad. Raíz12pruebasnamespace/inputvalidation0.11s; dictamen independiente pendiente antes de consumir desde JXP. Fuente: prueba y diff local, Codex.

2026-10-20 · Task3 adaptermembership misma transacción diseñado antes de invitacionesJXP, contratooffice-membership.md yplanTask3. Separaprotocolo de creación; grantrolfijoOffice/revokepermite limpieza dormant actualautorizada; marcadorprevistooffice-namespace.2. Todavía sinimplementación. Raízdecidió trasarquitecturaSol office_workflow_contract. Preservarclásico/LASMANOS yrollbackconOfficehistoria.

Task3membership congelada: adaptergrant/revoke caller-ownedtransaction, fixedroles, durableidempotency/replaycurrentauth and truthfulcapabilityprovenance; freshTier2completedrealrace/rollback/replaymatrix. Raíz67passed0.84s en33991. Fuente reviews/2026-10-20-office-membership-evidence.md. No consumir office-namespace.2 hasta dictamenSol exactSHA. Deadlock1213retryable requiere retrytransactioncallerconmismakey, nosilentloop.
