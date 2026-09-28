"""La entrega de una misión de código, orquestada (spec 2026-09-28 v1.3 §3.3, DC8).

Corre como `jaxsvc`, FUERA de la jaula, y todo lo que mira lo mira EN EL ESPEJO: nunca corre
git dentro del clon de Qwen ni mide nada en su árbol de trabajo. Puro salvo lo inyectado (los
pasos con git o red llegan como parámetros; por omisión, los de `entrega`).

Orden -- cada paso que falla corta ahí, y nada se empuja antes del paso 9:
1. Traer `axioma/<id>` del clon al espejo (`upload-pack` como la cuenta, fsck).
2. Sin commits nuevos respecto de `origin/<base>` → `sin_cambios`.
3. Sin informe legible de C5 → `sin_informe_c5` (DC8: sin informe no hay PR).
4. Identidad: author Y committer de TODOS los commits = la identidad de Axioma configurada
   (Qwen puede fijar cualquier autor; esto impide que un commit se haga pasar por Fernando).
5. C1 de código sobre el diff neto (`--text`) y barrido de secretos sobre él.
6. Barrido de secretos commit por commit (lo agregado y quitado dentro de la rama también
   viajaría a GitHub en el historial).
7. Tope de tamaño sobre TODOS los blobs que el empuje llevaría (`rev-list --objects` + `cat-file`).
8. Cualquier violación → `rechazada_por_contrato` con (regla, ruta, detalle), sin secretos.
9. Empujar; 10. abrir o actualizar el PR → `abierto`. Un fallo en 1, 9 o 10 (o al leer el
   espejo) → `fallo_entrega` con el motivo saneado.

Lo que sale de aquí va a la bitácora de la plataforma: rutas y detalles pasan por
`tapar_secretos` y el token se reemplaza por `***`."""
from __future__ import annotations

import httpx

from jax.ejecutor.codigo import entrega as E
from jax.ejecutor.codigo.barrido import lineas_con_secretos, tamanos_excedidos, tapar_secretos
from jax.ejecutor.codigo.entrega import EntregaRechazada
from jax.ejecutor.codigo.preparar import Clon, parsear_autor
from jax.ejecutor.codigo.reglas_diff import Violacion, revisar

__all__ = ["entregar", "Clon", "EntregaRechazada"]

TOPE_DETALLE = 300


def pie(mision_id: str, modelo: str) -> str:
    return f"\n\n---\nHecho-por: Axioma (Ejecutor, misión {mision_id}, cerebro {modelo})"


def _limpio(texto: str, token: str) -> str:
    if token:
        texto = texto.replace(token, "***")
    return tapar_secretos(texto)[:TOPE_DETALLE]


def _resultado(estado: str, *, violaciones=(), pr_url: str | None = None, notas=(), token: str = "") -> dict:
    return {"estado_entrega": estado, "pr_url": pr_url,
            "violaciones": [{"regla": v.regla, "ruta": _limpio(v.ruta, token), "detalle": _limpio(v.detalle, token)}
                            for v in violaciones],
            "notas": list(notas)}


def _fallo(motivo: str, token: str) -> dict:
    return _resultado("fallo_entrega", violaciones=(Violacion("entrega", "", motivo),), token=token)


def _identidad(historial, esperada: tuple[str, str]) -> tuple[Violacion, ...]:
    """Una violación por commit; el detalle dice QUÉ papel no coincide, nunca quién aparece."""
    v: list[Violacion] = []
    for c in historial:
        papeles = [p for p, quien in (("author", c.autor), ("committer", c.committer)) if quien != esperada]
        if papeles:
            v.append(Violacion("identidad", "", f"commit {c.sha[:12]}: {' y '.join(papeles)} no es la identidad "
                                                "de Axioma"))
    return tuple(v)


def _sin_repetir(violaciones) -> tuple[Violacion, ...]:
    return tuple(dict.fromkeys(violaciones))


async def entregar(clon: Clon, *, mision_id: str, repo: str, revision_legible: bool, informe: str, token: str,
                   cliente, tope_bytes: int, modelo: str, autor: str, upload_pack: str,
                   traer=E.traer_del_clon, commits=E.commits_de_la_rama, diff=E.diff_en_el_espejo,
                   tamanos=E.tamanos_de_la_rama, empujar=E.empujar, abrir_pr=E.abrir_o_actualizar_pr) -> dict:
    esperada = parsear_autor(autor)
    base = clon.rama_por_omision
    try:
        await traer(clon.espejo, clon.ruta, mision_id=mision_id, upload_pack=upload_pack)
        historial = await commits(clon.espejo, mision_id=mision_id, rama_por_omision=base)
    except EntregaRechazada as exc:  # fail-closed: sin push ni PR; el motivo (saneado) queda en la bitácora
        return _fallo(str(exc), token)
    if not historial:
        return _resultado("sin_cambios")
    if not revision_legible:
        return _resultado("sin_informe_c5")
    try:
        cambios = await diff(clon.espejo, mision_id=mision_id, rama_por_omision=base)
        medidos = await tamanos(clon.espejo, mision_id=mision_id, rama_por_omision=base)
    except EntregaRechazada as exc:  # fail-closed: sin push ni PR
        return _fallo(str(exc), token)
    violaciones = _sin_repetir(
        _identidad(historial, esperada) + revisar(cambios, repo) + lineas_con_secretos(cambios)
        + tuple(v for c in historial for v in lineas_con_secretos(c.cambios))
        + tamanos_excedidos(medidos, tope_bytes=tope_bytes))
    if violaciones:
        return _resultado("rechazada_por_contrato", violaciones=violaciones, token=token)
    try:
        await empujar(clon.espejo, mision_id=mision_id, rama_por_omision=base, token=token)
        pr = await abrir_pr(cliente, repo=repo, rama=clon.rama, base=base, titulo=f"Axioma: misión {mision_id[:8]}",
                            cuerpo=informe + pie(mision_id, modelo))
    except EntregaRechazada as exc:  # fail-closed: el motivo (saneado) queda en la bitácora
        return _fallo(str(exc), token)
    except httpx.HTTPStatusError as exc:  # fail-closed: la rama pudo quedar empujada, el PR no se abrió
        return _fallo(f"github_api: {exc.response.status_code}", token)
    except httpx.HTTPError as exc:  # fail-closed: igual, sin el mensaje (puede traer la URL)
        return _fallo(f"github_api: {type(exc).__name__}", token)
    return _resultado("abierto", pr_url=pr.url, notas=pr.notas)
