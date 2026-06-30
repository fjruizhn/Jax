thot: Auditoría adversarial del Contrato Formal v2 de Las Manos (incluido íntegro abajo). NO es una validación. Tu trabajo NO es confirmar que el contrato es sólido — es ROMPERLO. Asumí que tiene agujeros y encontralos. Si terminás diciendo "se ve bien", fallaste la misión.

CONTEXTO: este contrato pasó por (1) un primer formalizador (Ada v1), (2) dos revisores externos adversariales (GPT + DeepSeek), (3) una segunda formalización (Ada v2) que incorporó todo ese hardening. Vos sos la cuarta pasada y la última antes de implementar. Lo que los tres anteriores no vieron, lo tenés que ver vos. El contrato va a ejecutar operaciones reales en producción (servidores, deploys, control plane). Un agujero que se te escape llega a producción.

REGLA DE RIGOR (la tuya): cuestioná supuestos, declará incertidumbres, no cierres un hallazgo con "probablemente". El que supone se equivoca.

PRESUPUESTO DE HALLAZGOS OBLIGATORIO (no podés cerrar sin esto):
- Mínimo 3 vectores de ATAQUE concretos: una secuencia específica de pasos/capabilities/envelopes que produzca un resultado prohibido, o demuestre por qué NO se puede (con el paso exacto del decide() que lo frena).
- Mínimo 2 INVARIANTES que falten o estén mal cuantificadas (∀ donde debería ser ∃, o al revés; condición que no se chequea en decide()).
- Mínimo 2 huecos en el modelo de PROVENANCE/LINEAGE (Artefacto 6): cómo se falsifica un artifact_hash; si se puede romper una cadena de lineage sin que L4/L10 lo atrapen; si hay un camino de "external_untrusted → producción" que evada L3/L8.

ATACÁ ESPECÍFICAMENTE ESTOS PUNTOS:

1. LAS 3 CAPAS DE RIESGO (Artefacto 5). Construí una cadena de capabilities que produzca una operación peligrosa pero que las 3 capas (pisos + efectos + aditivo) clasifiquen por debajo de su riesgo real. Probá: cadenas de pasos todos "low" que deberían escalar; combinaciones que esquiven los floors; el ceiling "critical" usado para ocultar criticidad acumulada.

2. EL decide() (Artefacto 3). Hay una fase (A-L) que se pueda saltar? El orden de las fases permite que algo pase antes de su chequeo? promote_env en Fase D y la re-entrada — hay un estado donde el env promovido no re-evalúa los gates? El nonce marcado en A10 antes de las fases de riesgo deja un hueco si una fase posterior hace GATE (el nonce ya se consumió)?

3. PROVENANCE/LINEAGE (Artefacto 6). El ataque que motivó este artefacto era lavado entre planes. De verdad lo cierra? Atacá: artifact_hash falsificable, lineage con depth reseteado, parent_artifact_ids omitidos, source_trust que se "limpia" en una transformación intermedia, un Plan que consume un artefacto sin declararlo en expected_artifacts.

4. CRIPTO E IDENTIDAD (Artefacto 4). El binding de Jacobs (I40) tiene un huevo del huevo — quién certifica a Jacobs? La caller_chain se puede forjar si un eslabón intermedio está comprometido? El runtime adapter que tiene la private key (I41) es un nuevo punto único de fallo?

5. TU PROPIO ROL (auto-crítica obligatoria). Por I5 (implementer ≠ auditor), vos auditando un contrato que define tu propio veto bloqueante tenés sesgo estructural. Cuestionalo: el veto bloqueante de Thot es un punto único de fallo? Si Thot es comprometido (prompt injection vía un plan que incluye texto malicioso), qué puede hacer? El override solo-operador es suficiente defensa si Thot se vuelve adversarial, o Thot comprometido puede vetar todo y trabar el sistema (DoS)?

6. EL HUECO QUE NO ESTOY NOMBRANDO. Los puntos 1-5 son donde el diseñador cree que están los riesgos. Tu valor real es el punto 6: qué NO se está preguntando que debería? Qué supuesto de todo el diseño es el más frágil?

FORMATO DE SALIDA:
- Por cada vector de ataque: título, secuencia exacta, resultado prohibido logrado (o el paso que lo frena), severidad (crítica/alta/media), y corrección concreta.
- Veredicto final: GO / NO-GO para implementar Fase 1. Si NO-GO, qué bloquea exactamente. Si GO, qué condiciones quedan.
- No endulces. Este es el último filtro antes de que Kimi escriba código que toca producción.
