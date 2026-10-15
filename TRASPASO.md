# Traspaso · Rotación token Max 2026-10-15

Objetivo: documentar en Jax la rotación operativa realizada por Fernando y
Codex, sin secretos.

Hecho: credencial nueva validada por Anthropic HTTP 200; servicios reiniciados
y cargaron el mismo valor del archivo; respaldo restaurado a temporal.
Documento de historia añadido en esta rama.

Falta: observar la siguiente corrida del timer, cerrar el pendiente viejo y
crear el recordatorio 2027-09-24 en claude-skills. Auditar e integrar esta
rama; borrar este archivo antes de la auditoría final.

Decisiones: Fernando pidió adelantar la rotación el 15-oct-2026. El procedimiento
con `claude setup-token` y script existente deriva de su decisión del 27-sep.

Siguiente comando: `systemctl list-timers jax-catalogo-modelos.timer --no-pager`.
