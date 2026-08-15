<!-- GENERADO AUTOMÁTICAMENTE por policy/tools/generate_corpus.py
     NO EDITAR A MANO — ver policy/README.md -->

# CORPUS — JAX/Axioma, reglas normativas

**Versión del corpus:** 0.1.0
**SHA256:** `8cf5ed5340e08669b299dc8e844fa1f3e9c67d50336ec4bbfdb7c10368c671e2`
**Reglas:** 25

| Estado | Cantidad |
|---|---|
| NORMATIVA | 0 |
| NORMATIVA_PENDIENTE | 13 |
| CULTURAL | 8 |
| HISTORICA | 4 |

---

## NORMATIVA_PENDIENTE

### OP01 — Ninguna capability de escritura/refactor puede tocar archivos que matcheen los patrones .env, secrets/, private_keys/ o credentials/; si lo intenta, la ejecución se interrumpe y se notifica.

- **Enunciado:** Ninguna capability de escritura/refactor puede tocar archivos que matcheen los patrones .env, secrets/, private_keys/ o credentials/; si lo intenta, la ejecución se interrumpe y se notifica.
- **Origen:** las_manos/config.toml líneas 195 y 258 (forbidden_paths = [".env", "secrets/", "private_keys/", "credentials/"]); las_manos/motor_registry/catalog.py líneas 50 y 88 (carga del campo en MotorEntry); docs/AUTONOMIA_ANTIERROR.md líneas 174, 219 y 229 (comportamiento ante violación: interrumpe el pipeline y notifica por Telegram, condición 🔴 ROJO).

- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** El campo forbidden_paths se declara y se carga (config.toml → catalog.py) pero no encontré código que lo verifique en tiempo de ejecución — grep de ".forbidden_paths" en las_manos/*.py y las_manos/motor_registry/*.py fuera de la declaración/carga: sin resultados. El comportamiento de interrupción+Telegram descrito en AUTONOMIA_ANTIERROR.md pertenece a un pipeline de autonomía todavía en fases de implementación (§7 del mismo documento, "Orden de implementación por fases"). Sin test.
DISCREPANCIA CON LA TAREA: las rutas absolutas citadas en el encargo de esta fase ("/srv/vms, /etc/jax/.env, ~/.ssh") no aparecen en ningún lugar del código ni la documentación de jax/jax-platform bajo ningún nombre — lo único documentado son los cuatro patrones relativos de arriba, aplicados dentro de capabilities de código (code_swarm, implementation), no como una lista global de rutas intocables del sistema. Esta regla migra lo que SÍ existe por escrito; la versión con rutas absolutas queda fuera del corpus (ver sección de reglas orales del informe de fase).



### OP02 — Toda regla de Policy Routing en el ER7212PC (CASA-RZ) usa un IP Group /32 específico como Source; nunca Source=Default.

- **Enunciado:** Toda regla de Policy Routing en el ER7212PC (CASA-RZ) usa un IP Group /32 específico como Source; nunca Source=Default.
- **Origen:** Confirmación directa de Fernando Ruiz, verificada por captura de pantalla del Omada Controller (2026-08-15): pestaña Devices → ER7212PC → Config → Routing → Policy Routing muestra 3 reglas activas (ASI Network, Multicable, Hall9000), las tres con Source = IP Group (Server_Bridge, Server_RichHN, Sesamo_NAS, Serveer_BlackTower, Server_Atemai, Server_Hall9000), ninguna con Source=Default. Segunda captura confirma el grupo Server_Hall9000 = IP Subnet 172.16.20.5/32.

- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Verificado visualmente contra la config real del router — no hay discrepancia hoy. Un test automatizado requeriría acceso a la API del Omada Controller (no explorada en esta sesión) para consultar programáticamente las reglas y sus IP Groups; por eso NORMATIVA_PENDIENTE y no NORMATIVA.



### OP03 — Node.js en server-rich-hn (172.16.20.10) se instala vía el runtime propio de aaPanel, nunca vía NodeSource ni paquete apt del sistema.

- **Enunciado:** Node.js en server-rich-hn (172.16.20.10) se instala vía el runtime propio de aaPanel, nunca vía NodeSource ni paquete apt del sistema.
- **Origen:** Confirmación directa de Fernando Ruiz, verificada por SSH (fruiz@172.16.20.10:58291, 2026-08-15): `which node` resuelve a `/www/server/nodejs/v24.16.0/bin/node` (runtime gestionado por aaPanel); `dpkg -l | grep -iE "nodejs|node-"` sin resultados; sin archivos en `/etc/apt/sources.list.d/` que referencien nodesource. Distinta de la regla ya documentada para hall9000 (~/.claude/CLAUDE.md, jax/CLAUDE.md: "Node.js ... NUNCA NodeSource") — mismo principio, host distinto, mecanismo de instalación distinto (aaPanel vs. nvm).

- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Verificado en vivo, sin discrepancia. Test escribible: chequeo remoto (SSH) de que `node` resuelve bajo `/www/server/nodejs/` y que no existe paquete `nodejs`/`node-*` vía dpkg ni fuente nodesource — no implementado todavía.



### OP04 — El dominio del proyecto es hamurabi.io (una sola "m"); hammurabi.io (doble "m") pertenece a un tercero y no debe usarse ni confundirse con el propio.

- **Enunciado:** El dominio del proyecto es hamurabi.io (una sola "m"); hammurabi.io (doble "m") pertenece a un tercero y no debe usarse ni confundirse con el propio.
- **Origen:** Confirmación directa de Fernando Ruiz (2026-08-15). Coincide, casi textual, con una nota en memoria persistente del asistente (jax-memoria-semantica-dos-niveles.md: "Proyecto de prueba HAMURABI (una sola M; 'hammurabi.io' doble M es de terceros)") — no se usó esa memoria como fuente por sí sola en la Fase 0.5 original; se agrega ahora con la confirmación directa como origen primario.

- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Test escribible: grep sobre configuración/código del proyecto (jax, jax-platform, DNS-facing configs) verificando que ninguna referencia usa "hammurabi.io" como si fuera propio — no implementado todavía.



### OP05 — En hall9000, nvme2n1 aloja el sistema operativo, nvme0n1 está montado en /srv/jax-data, y nvme1n1 es el volume group LVM vg_vms.

- **Enunciado:** En hall9000, nvme2n1 aloja el sistema operativo, nvme0n1 está montado en /srv/jax-data, y nvme1n1 es el volume group LVM vg_vms.
- **Origen:** Confirmación directa de Fernando Ruiz (2026-08-15), verificada en vivo con `lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINT,MODEL`: nvme2n1 (KINGSTON SNV3S1000G) tiene /boot/efi, /boot, / y LVM vg_os (lv_swap, lv_var, lv_docker, lv_jax→/opt/jax, lv_db→/var/lib/mysql); nvme0n1p1 (Samsung SSD 9100 PRO 2TB) es XFS montado en /srv/jax-data; nvme1n1p1 (Samsung SSD 9100 PRO 2TB) es LVM vg_vms (vm_pool, dev_root, prod_root). Coincide exacto con lo confirmado. Nota: una memoria persistente del asistente, de 2026-06-23, tenía un layout distinto y ya obsoleto para esta misma máquina (nvme0n1p2=/, sdb1=/srv/vms) — no usado como fuente, señalado como corrección pendiente aparte.

- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Test escribible: parseo de `lsblk -J` verificando el mapeo dispositivo→rol — no implementado todavía.



### P02 — Ninguna salida llega al usuario sin pasar por validación. Sin excepción por camino.

- **Enunciado:** Ninguna salida llega al usuario sin pasar por validación. Sin excepción por camino.
- **Origen:** REFORMAS-v3.md §2 (Principios), texto verbatim. /opt/jax/docs/REFORMAS-v3.md, sha256 4099a08c39713c79836eb1ab58fc42e0a3a1357767590cfe281c04ea7ede8660
- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** REFORMAS-v3.md §1.2 documenta que esta regla está VIOLADA por diseño hoy: "Camino B — Chat" no pasa por ninguna validación (ESTADO: sin validación, sin auditoría, sin procedencia). El mecanismo que la haría cumplir es R1 (§3, no implementado — Fase 2 de §5).



### P03 — La procedencia se deriva de hechos verificables y se enlaza a la afirmación concreta.

- **Enunciado:** La procedencia se deriva de hechos verificables y se enlaza a la afirmación concreta.
- **Origen:** REFORMAS-v3.md §2 (Principios), texto verbatim. /opt/jax/docs/REFORMAS-v3.md, sha256 4099a08c39713c79836eb1ab58fc42e0a3a1357767590cfe281c04ea7ede8660
- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Mecanismo previsto: provenance_ref/evidence_pointer del schema de claim (§3.1.2), no implementado. Relacionado con [[MA01]] (raíz filosófica, "el que supone se equivoca") y con [[SI02]] (Protocolo de la Memoria Viva, mismo principio aplicado a memoria en vez de a claims).



### P04 — La capacidad se otorga por contrato de tarea, no por identidad de facet.

- **Enunciado:** La capacidad se otorga por contrato de tarea, no por identidad de facet.
- **Origen:** REFORMAS-v3.md §2 (Principios), texto verbatim. /opt/jax/docs/REFORMAS-v3.md, sha256 4099a08c39713c79836eb1ab58fc42e0a3a1357767590cfe281c04ea7ede8660
- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Tensión con el sistema actual: las_manos/config.toml ya otorga capabilities vía `allowed_callers` (lista de facetas por nombre) — exactamente el modelo "por identidad de facet" que este principio rechaza. El mecanismo que P04 exige (R3, contrato de tarea con capabilities inyectadas por el orquestador) no está implementado.



### P05 — Local primero. La API es escalamiento, no default.

- **Enunciado:** Local primero. La API es escalamiento, no default.
- **Origen:** REFORMAS-v3.md §2 (Principios), texto verbatim. /opt/jax/docs/REFORMAS-v3.md, sha256 4099a08c39713c79836eb1ab58fc42e0a3a1357767590cfe281c04ea7ede8660
- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** REFORMAS-v3.md §1.4 documenta violación activa hoy: GPT-OSS-120B "prácticamente ocioso", Qwen3-Coder-30B usado como faceta conversacional ("su peor caso de uso"). Mecanismo previsto: clasificador L0-L3 de R2 (§3.2), no implementado.



### P06 — Toda plataforma debe justificarse con carga de trabajo en producción.

- **Enunciado:** Toda plataforma debe justificarse con carga de trabajo en producción.
- **Origen:** REFORMAS-v3.md §2 (Principios), texto verbatim. /opt/jax/docs/REFORMAS-v3.md, sha256 4099a08c39713c79836eb1ab58fc42e0a3a1357767590cfe281c04ea7ede8660
- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Mecanismo previsto: compuerta de 30 días de R5 §3.5 ("si 30 días después de habilitar read_audit_log no existe carga recurrente consumida, se congela toda construcción de capacidades nuevas"). No implementado — depende de Fase 1 (§5), no desbloqueada todavía. Relacionado con [[MA03]].



### P07 — No existe bypass en producción. El rollback se hace por versionado de política y despliegue canario, jamás mostrando como autorizada una salida que falló.

- **Enunciado:** No existe bypass en producción. El rollback se hace por versionado de política y despliegue canario, jamás mostrando como autorizada una salida que falló.
- **Origen:** REFORMAS-v3.md §2 (Principios), texto verbatim. /opt/jax/docs/REFORMAS-v3.md, sha256 4099a08c39713c79836eb1ab58fc42e0a3a1357767590cfe281c04ea7ede8660
- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Mecanismo previsto: R1 §3.1.8 ("Sin WARN. Sin BYPASS en producción" — shadow validation → canario → versionado de política). No implementado.



### P08 — La autodeclaración de calidad por parte del modelo no es contractual.

- **Enunciado:** La autodeclaración de calidad por parte del modelo no es contractual.
- **Origen:** REFORMAS-v3.md §2 (Principios), texto verbatim. /opt/jax/docs/REFORMAS-v3.md, sha256 4099a08c39713c79836eb1ab58fc42e0a3a1357767590cfe281c04ea7ede8660
- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Mecanismo previsto: distinción coverage_deterministic (contractual) vs coverage_model_reported (no contractual) de R2 §3.2. No implementado.



### P09 — La prosa factual es salida derivada. El modelo emite estructura; el runtime verbaliza. Ningún texto con autoridad se origina directamente en el modelo.

- **Enunciado:** La prosa factual es salida derivada. El modelo emite estructura; el runtime verbaliza. Ningún texto con autoridad se origina directamente en el modelo.
- **Origen:** REFORMAS-v3.md §2 (Principios), texto verbatim — marcado (nuevo) en el documento. /opt/jax/docs/REFORMAS-v3.md, sha256 4099a08c39713c79836eb1ab58fc42e0a3a1357767590cfe281c04ea7ede8660
- **Estado:** NORMATIVA_PENDIENTE
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Núcleo de R1 (cambio arquitectónico principal de v3, §0.1). Test de aceptación ya definido en el propio documento — Apéndice B ("caso de prueba de aceptación de R1"): reproducir el incidente del 2026-08-14 22:26-22:28 y verificar que produce ENVELOPE_REJECTED o reclasificación forzada, no prosa con autoridad. No implementado — por eso NORMATIVA_PENDIENTE y no NORMATIVA, aunque el test ya está diseñado en prosa.



## CULTURAL

### HY01 — Cualquier recomendación sin fuente verificable se devuelve para validar con Hipatia antes de presentarla.

- **Enunciado:** Cualquier recomendación sin fuente verificable se devuelve para validar con Hipatia antes de presentarla.
- **Origen:** Protocolo Hyde — "Activo desde Junio 2026 · Firmado por los cinco". six-impossible-things.html, en el bloque inmediatamente posterior a "Marina · Segundo principio fundacional". sha256 db467720abbe070ee376e48906b17abb4503304801a44b7be599f7ce2612ab37, commit 5d39014 (19-jun-2026).

- **Estado:** CULTURAL
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Disciplina de proceso para las facetas/agentes del ecosistema, no un comportamiento verificable por código de JAX hoy — no existe un mecanismo automatizado que detecte "recomendación sin fuente" y la desvíe a Hipatia.



### HY02 — Ningún comando se ejecuta sin mostrar primero el output que se espera.

- **Enunciado:** Ningún comando se ejecuta sin mostrar primero el output que se espera.
- **Origen:** Protocolo Hyde — "Activo desde Junio 2026 · Firmado por los cinco". six-impossible-things.html, mismo bloque que [[HY01]]. sha256 db467720abbe070ee376e48906b17abb4503304801a44b7be599f7ce2612ab37, commit 5d39014 (19-jun-2026). Citada también, en paráfrasis, en el encargo de la Fase 0 de esta migración ("ningún comando sin salida conocida").

- **Estado:** CULTURAL
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Igual que [[HY01]]: disciplina de conducta del agente, sin mecanismo de código ni test que la verifique automáticamente en JAX hoy.



### HY03 — Ningún diagnóstico basado en "probablemente" se cierra sin pasarlo antes por validación con Hipatia.

- **Enunciado:** Ningún diagnóstico basado en "probablemente" se cierra sin pasarlo antes por validación con Hipatia.
- **Origen:** Protocolo Hyde — "Activo desde Junio 2026 · Firmado por los cinco". six-impossible-things.html, mismo bloque que [[HY01]]. sha256 db467720abbe070ee376e48906b17abb4503304801a44b7be599f7ce2612ab37, commit 5d39014 (19-jun-2026). Citada también, en paráfrasis, en el encargo de la Fase 0 de esta migración ("ninguna conclusión con 'probablemente'").

- **Estado:** CULTURAL
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Igual que [[HY01]]/[[HY02]]: disciplina de conducta, sin mecanismo de código.



### HY04 — Si no hay 100% de certeza sobre algo, es un posible error, y eso se declara explícitamente, nunca se omite.

- **Enunciado:** Si no hay 100% de certeza sobre algo, es un posible error, y eso se declara explícitamente, nunca se omite.
- **Origen:** Protocolo Hyde — "Activo desde Junio 2026 · Firmado por los cinco". six-impossible-things.html, mismo bloque que [[HY01]]. sha256 db467720abbe070ee376e48906b17abb4503304801a44b7be599f7ce2612ab37, commit 5d39014 (19-jun-2026). Citada también, en paráfrasis, en el encargo de la Fase 0 de esta migración ("declaración explícita de incertidumbre").

- **Estado:** CULTURAL
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Igual que [[HY01]]-[[HY03]]: disciplina de conducta, sin mecanismo de código.



### MA01 — El que supone se equivoca.

- **Enunciado:** El que supone se equivoca.
- **Origen:** Marina (atribución). /home/fruiz/jax/docs/AUTONOMIA_ANTIERROR.md línea 15 ("Regla firmada por los cinco") y línea 232; también en /home/fruiz/jax/missions/bridge-migration.md línea 65-66 y línea 223; y citada en REFORMAS-v3.md Apéndice A1. Fraseo alternativo hallado en la misma fuente (AUTONOMIA_ANTIERROR.md, footer, línea 246): "No suponer nunca."

- **Estado:** CULTURAL
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Raíz filosófica de [[P01]], [[P03]] y [[P09]] de REFORMAS-v3.md — su propio Apéndice A liga A1 explícitamente a R1 ("Violado por diseño en el camino de chat. R1 lo convierte en imposibilidad estructural"). Decisión explícita de esta migración: queda como entrada propia en vez de fusionarse con esas tres reglas, porque su alcance (toda decisión, no solo prosa factual/procedencia/reglas-con-test) es más amplio que cualquiera de ellas por separado.



### MA02 — Saber no cuesta nada.

- **Enunciado:** Saber no cuesta nada.
- **Origen:** Marina — "Segundo principio fundacional" (six-impossible-things.html, contexto inmediato del Protocolo Hyde, ver [[HY01]]-[[HY04]]). También en /home/fruiz/jax/docs/AUTONOMIA_ANTIERROR.md (footer) y /home/fruiz/jax/missions/bridge-migration.md línea 66 ("Saber no cuesta nada — pregúntale al que de verdad sabe").

- **Estado:** CULTURAL
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Ninguna fuente revisada la conecta con un mecanismo de código específico de REFORMAS-v3.md — se mantiene como principio cultural puro, sin forzar una relación que no está escrita.



### MA03 — Mañana es el día que el fracasado tiene más que hacer.

- **Enunciado:** Mañana es el día que el fracasado tiene más que hacer.
- **Origen:** Marina — /home/fruiz/jax/docs/AUTONOMIA_ANTIERROR.md (footer) y /home/fruiz/jax/missions/bridge-migration.md línea 67; citada en REFORMAS-v3.md Apéndice A3, que la liga explícitamente a R5 ("Aplicable a R5. Compuerta de 30 días").

- **Estado:** CULTURAL
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** La aplicación concreta y testeable de este axioma vive en [[P06]]/R5 (compuerta de 30 días), no en el axioma mismo — el axioma en sí es un principio motivacional general, no un comportamiento de sistema.



### P01 — Una regla es una regla solo si existe código que puede rechazar un output que la viole.

- **Enunciado:** Una regla es una regla solo si existe código que puede rechazar un output que la viole.
- **Origen:** REFORMAS-v3.md §2 (Principios), texto verbatim. /opt/jax/docs/REFORMAS-v3.md, sha256 4099a08c39713c79836eb1ab58fc42e0a3a1357767590cfe281c04ea7ede8660
- **Estado:** CULTURAL
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Es un meta-principio sobre qué cuenta como regla, no una afirmación sobre el comportamiento en runtime de JAX — no hay "output de JAX" que este principio en sí mismo pueda rechazar, por eso CULTURAL y no NORMATIVA_PENDIENTE. tools/validate_corpus.py de este mismo corpus aplica su espíritu a nivel de gobernanza del corpus (rechaza status=NORMATIVA con test=null), pero eso valida metadatos del corpus, no comportamiento de JAX — no se cuenta como enforcement de esta regla.



## HISTORICA

### SI01 — Todo componente que defina verdad, autoridad, permisos, auditoría, reversibilidad, límites, identidad o fallo cerrado debe implementarse antes de habilitar la capacidad que depende de él.

- **Enunciado:** Todo componente que defina verdad, autoridad, permisos, auditoría, reversibilidad, límites, identidad o fallo cerrado debe implementarse antes de habilitar la capacidad que depende de él.
- **Origen:** six-impossible-things.html, Capítulo IX "La Regla Absoluta", "Principio IX — No Diferimiento Contractual" (atribuido a Thot). sha256 db467720abbe070ee376e48906b17abb4503304801a44b7be599f7ce2612ab37, commit 5d39014, 19-jun-2026. Identificado en F0.4 (/opt/jax/docs/FASE-0-VERIFICACION.md).

- **Estado:** HISTORICA
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** F0.4: existe, sin versión numérica propia distinta del documento contenedor, sin evidencia de ratificación posterior al commit de creación → "existe sin versión → histórico" (criterio del propio §5 de REFORMAS-v3.md). Corresponde conceptualmente a [[P04]]/R3 de REFORMAS-v3.md (capacidades por contrato), aunque REFORMAS-v3.md no cita este principio por nombre — la relación es interpretación de esta migración, no un enlace explícito en las fuentes. El archivo six-impossible-things.html NO fue modificado por esta migración.



### SI02 — La memoria sin procedencia es otra forma de suposición: todo recuerdo de tipo HECHO requiere fuente verificable, y sin ella se marca "no verificado" — nunca se presenta como hecho con autoridad.

- **Enunciado:** La memoria sin procedencia es otra forma de suposición: todo recuerdo de tipo HECHO requiere fuente verificable, y sin ella se marca "no verificado" — nunca se presenta como hecho con autoridad.
- **Origen:** six-impossible-things.html, Apéndice C, "El Protocolo de la Memoria Viva v0.1 — sujeto a refinamiento de la Mesa". Nacido 15-jun-2026 por mandato de la Mesa (Thot y Jekyll). sha256 db467720abbe070ee376e48906b17abb4503304801a44b7be599f7ce2612ab37, commit 5d39014, 19-jun-2026. Identificado en F0.4 (/opt/jax/docs/FASE-0-VERIFICACION.md).

- **Estado:** HISTORICA
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Texto explícito del propio artefacto: "Borrador v0.1... sujeto a refinamiento de sus autores" — HISTORICA por declararse borrador, no por haber dejado de regir. Enunciado sintetizado a partir de la prosa del apéndice (a diferencia de P01-P09, esta fuente no trae ya una oración normativa única lista para copiar). Corresponde conceptualmente a [[P03]] de REFORMAS-v3.md (procedencia enlazada a la afirmación). El archivo six-impossible-things.html NO fue modificado por esta migración.



### SI03 — Cualquier miembro de la Mesa puede proponer una enmienda; los cambios editoriales se corrigen al verificarse, los constitucionales requieren consenso de la Mesa, y todo cambio se registra con fecha, autor y clase.

- **Enunciado:** Cualquier miembro de la Mesa puede proponer una enmienda; los cambios editoriales se corrigen al verificarse, los constitucionales requieren consenso de la Mesa, y todo cambio se registra con fecha, autor y clase.
- **Origen:** six-impossible-things.html, Apéndice D, "El Protocolo de Enmienda v0.1 — cómo cambia una constitución viva". Nacido 15-jun-2026 por mandato de la Mesa. sha256 db467720abbe070ee376e48906b17abb4503304801a44b7be599f7ce2612ab37, commit 5d39014, 19-jun-2026. Identificado en F0.4 (/opt/jax/docs/FASE-0-VERIFICACION.md).

- **Estado:** HISTORICA
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** Enunciado sintetizado combinando las reglas 1, 2 y 4 (numeradas en el original) del apéndice. Sin evidencia, dentro del archivo, de una enmienda posterior registrada aplicándose a sí misma — el propio Protocolo de Enmienda no tiene, hasta donde pude verificar, ninguna enmienda propia documentada. El archivo six-impossible-things.html NO fue modificado por esta migración.



### SI04 — El threat model mínimo cubre solo amenazas activas de los sistemas que ya existen; la matriz STRIDE completa, CVSS y diagramas por componente quedan diferidos como profundización posterior.

- **Enunciado:** El threat model mínimo cubre solo amenazas activas de los sistemas que ya existen; la matriz STRIDE completa, CVSS y diagramas por componente quedan diferidos como profundización posterior.
- **Origen:** six-impossible-things.html, Apéndice E, "Threat Model Mínimo v0.1 — qué puede matar Neverland". Nacido 15-jun-2026 por mandato de la Mesa. sha256 db467720abbe070ee376e48906b17abb4503304801a44b7be599f7ce2612ab37, commit 5d39014, 19-jun-2026. Identificado en F0.4 (/opt/jax/docs/FASE-0-VERIFICACION.md).

- **Estado:** HISTORICA
- **Mecanismo de cumplimiento:** null
- **Test:** null
- **Versión:** 0.1 · **Creada:** 2026-08-15 · **Enmendada por:** null
- **Notas:** El propio texto declara "diferible, madura después" para el threat model profundo — HISTORICA en el sentido de que la especificación misma se declara parcial/provisional, no completa ni ratificada como final. El archivo six-impossible-things.html NO fue modificado por esta migración.



