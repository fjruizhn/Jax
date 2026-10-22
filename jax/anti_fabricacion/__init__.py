"""Anti-fabricación, capas 2 (detector de estructura determinista) y 4 (consecuencias).

Spec cerrada de escalón 3; pre-registro en
docs/superpowers/specs/2026-10-22-anti-fabricacion-preregistro.md.

Este paquete se importa de dos maneras: `jax.anti_fabricacion` (scripts, REPL) y
`anti_fabricacion` pelado, vía el symlink `las_manos/anti_fabricacion`, que es
como lo ve LAS MANOS (cwd las_manos/, donde `jax.` no es importable). Por eso
dentro del paquete SOLO hay imports relativos.
"""
