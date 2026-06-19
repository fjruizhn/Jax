---
faceta: hyde
---

Fix en ~/jax-platform/frontend/src: cuando se genera una imagen en el chat, los botones de la barra inferior desaparecen porque la imagen es muy grande. 

Dos fixes:
1. El panel central del chat debe tener scroll interno (overflow-y: auto) para que el contenido no empuje la barra inferior
2. Las imágenes generadas en el chat deben tener max-height: 400px y object-fit: contain para no ocupar toda la pantalla

La barra inferior (BottomBar) debe ser siempre visible con position fixed o sticky.
