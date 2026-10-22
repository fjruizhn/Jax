# AXIOMA 3.0 — F2-C Web Chat governed boundary

F2-C begins the migration of Web Chat only.  It adds a side-effect-free
`GovernedRenderer` which accepts sealed F2-A `GovernedResponseEnvelope`
objects, revalidates any F2-B current-resolution receipt at render time, and
renders typed content blocks through server-owned templates/notices.

Provider output is buffered as an internal candidate.  It is not a browser
stream.  The server-side Web Chat adapter seals either a non-governed narrative
candidate or a safe degraded/cancelled notice; only renderer-generated chunks
may later be transported.  There is no outbox or durable `OUTPUT_PREPARED`
lifecycle in F2-C.  That dependency remains F2-D work, and this module does
not enable an authoritative production emission path by itself.

The deterministic governed-domain registry is exact-phrase based, not an LLM
classifier.  A registered system proposition appearing in `NARRATIVE_TEXT`
causes server-owned unavailable rendering; it cannot escape as provider prose.
Trusted status/source/authority presentation comes only from typed claims and
server templates.  HTML, Markdown-like payload, terminal controls, Unicode
format controls and nested tool JSON are presented as untrusted literal data.

Existing Web Chat shadow validation may remain as parity telemetry during its
platform migration, but the F2-C renderer/gate is the pre-display enforcement
boundary.  No other model-facing channel is claimed governed by this record.
