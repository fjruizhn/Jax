"""La entrega de una misión de código, orquestada (spec 2026-09-28 v1.3 §3.3, DC8).

Corre como `jaxsvc`, FUERA de la jaula, y todo lo que mira lo mira EN EL ESPEJO: nunca corre
git dentro del clon de Qwen ni mide nada en su árbol de trabajo. Puro salvo lo inyectado (los
pasos con git, red o la pausa llegan como parámetros; por omisión, los de `entrega`).

Orden -- cada paso que falla corta ahí, y nada se empuja antes del paso 10:
1. Traer `axioma/<id>` del clon al espejo (`upload-pack` como la cuenta, fsck).
2. Sin commits nuevos respecto de `origin/<base>` → `sin_cambios` (con la nota
   `pr_previo_sin_cambios_nuevos` si ya hay un PR abierto de la misión).
3. Sin informe legible de C5 → `sin_informe_c5` (DC8: sin informe no hay PR).
4. Identidad: author Y committer de TODOS los commits = la identidad de Axioma configurada.
5. C1 de código sobre el diff neto (`--text`); barrido de secretos sobre lo agregado, las rutas
   (y rutas anteriores), el objeto de cada commit (mensaje y cabeceras) -- neto y commit por
   commit -- y el título y cuerpo del PR (BLOCK-1 de la auditoría de escalón 3).
6. Topes: por archivo (el blob mayor de cada ruta) y total (todo el rango), en el espejo.
7. Cualquier violación → `rechazada_por_contrato` con (regla, ruta, detalle), sin secretos.
8. La pausa se vuelve a leer: puesta → `sin_entregar` (motivo `pausa_puesta`).
9. El SHA de la punta; 10. empujar. Un fallo en 1, 9, 10 (o al leer el espejo) → `fallo_entrega`.
11. DESPUÉS del empuje, todo tiene estado propio (MAJOR-1): la pausa puesta, o cualquier fallo al
    abrir el PR (incluido un cuerpo de respuesta ilegible) → `empujado_sin_pr` con
    `rama_empujada: true` y el `sha`, sin el texto de la excepción. PR abierto → `abierto` (con
    `sin_etiqueta` si falló solo la etiqueta).

El cuerpo del PR tiene un tope (`TOPE_CUERPO`): el informe se recorta con aviso ANTES de empujar.
Lo que sale de aquí va a la bitácora de la plataforma: rutas y detalles pasan por
`tapar_secretos` y el token se reemplaza por `***`."""
from __future__ import annotations

import httpx

from jax.ejecutor.codigo import entrega as E
from jax.ejecutor.codigo.barrido import (hay_secreto, lineas_con_secretos, rutas_con_secretos, tamanos_excedidos,
                                         tapar_secretos, total_excedido)
from jax.ejecutor.codigo.entrega import EntregaRechazada
from jax.ejecutor.codigo.preparar import Clon, parsear_autor
from jax.ejecutor.codigo.reglas_diff import Violacion, revisar

__all__ = ["entregar", "Clon", "EntregaRechazada", "TOPE_CUERPO"]

TOPE_DETALLE = 300
#: El cuerpo de un PR de GitHub admite 65 536 caracteres; se deja margen.
TOPE_CUERPO = 60_000


def pie(mision_id: str, modelo: str) -> str:
    return f"\n\n---\nHecho-por: Axioma (Ejecutor, misión {mision_id}, cerebro {modelo})"


def cuerpo_del_pr(informe: str, mision_id: str, modelo: str) -> str:
    """Informe + pie, con el informe recortado (y el aviso de cuánto) si no entra en `TOPE_CUERPO`."""
    final = pie(mision_id, modelo)
    if len(informe) + len(final) <= TOPE_CUERPO:
        return informe + final
    aviso = f"\n\n[informe recortado: se muestran {{n}} de {len(informe)} caracteres]"
    n = TOPE_CUERPO - len(final) - len(aviso.format(n=len(informe)))
    return informe[:n] + aviso.format(n=n) + final


def _limpio(texto: str, token: str) -> str:
    if token:
        texto = texto.replace(token, "***")
    return tapar_secretos(texto)[:TOPE_DETALLE]


def _resultado(estado: str, *, violaciones=(), pr_url: str | None = None, notas=(), token: str = "",
               sha: str | None = None, rama_empujada: bool | str | None = None, **extra) -> dict:
    """`rama_empujada`: True/False, o "desconocido" (MAJOR-A: ni el empuje ni `ls-remote` respondieron)."""
    return {"estado_entrega": estado, "pr_url": pr_url,
            "violaciones": [{"regla": v.regla, "ruta": _limpio(v.ruta, token), "detalle": _limpio(v.detalle, token)}
                            for v in violaciones],
            "notas": list(notas), "rama_empujada": sha is not None if rama_empujada is None else rama_empujada,
            "sha": sha, **extra}


def _fallo(motivo: str, token: str, rama_empujada: bool | str | None = None, pr_url: str | None = None) -> dict:
    return _resultado("fallo_entrega", violaciones=(Violacion("entrega", "", motivo),), token=token,
                      rama_empujada=rama_empujada, pr_url=pr_url)


def _sin_pr(sha: str, pr_previo: str | None, *, notas=(), motivo: str | None = None) -> dict:
    """MAJOR-B: `empujado_sin_pr` = la rama está en GitHub y el PR de ESTE turno no se confirmó. Si ya
    había un PR abierto de la misión, se da su URL y se avisa que su informe es el de antes."""
    extra = ("pr_con_informe_desactualizado",) if pr_previo else ()
    return _resultado("empujado_sin_pr", sha=sha, pr_url=pr_previo, notas=(*notas, *extra),
                      violaciones=(Violacion("entrega", "", motivo),) if motivo else ())


def _identidad(historial, esperada: tuple[str, str]) -> tuple[Violacion, ...]:
    """Una violación por commit; el detalle dice QUÉ papel no coincide, nunca quién aparece."""
    v: list[Violacion] = []
    for c in historial:
        papeles = [p for p, quien in (("author", c.autor), ("committer", c.committer)) if quien != esperada]
        if papeles:
            v.append(Violacion("identidad", "", f"commit {c.sha[:12]}: {' y '.join(papeles)} no es la identidad "
                                                "de Axioma"))
    return tuple(v)


def _cuerpo_con_secretos(titulo: str, *textos: str, token: str) -> tuple[Violacion, ...]:
    """BLOCK-1: el título y el cuerpo del PR salen a GitHub; el informe lleva texto del modelo
    (dato, línea, comando). Se barren ANTES de empujar: el cuerpo y el informe ENTERO."""
    todos = (titulo, *textos)
    if any(hay_secreto(t) for t in todos) or (token and any(token in t for t in todos)):
        return (Violacion("secretos", "", "patrón de credencial en el título o el cuerpo del PR"),)
    return ()


def _encoding_no_permitido(historial) -> tuple[Violacion, ...]:
    """MINOR-A: el barrido lee cada objeto como UTF-8. Un commit con cabecera `encoding` distinta NO
    se decodifica con ella: se rechaza. Solo cuentan las cabeceras (hasta la primera línea vacía)."""
    v: list[Violacion] = []
    for c in historial:
        for linea in c.crudo.split("\n"):
            if linea == "":
                break
            if linea.startswith("encoding ") and linea[len("encoding "):].strip().lower() not in ("utf-8", "utf8"):
                v.append(Violacion("encoding_no_permitido", "", f"commit {c.sha[:12]}: cabecera encoding no UTF-8"))
                break
    return tuple(v)


def _sin_repetir(violaciones) -> tuple[Violacion, ...]:
    return tuple(dict.fromkeys(violaciones))


async def _pausa(pausa_puesta) -> bool:
    try:
        return bool(await pausa_puesta())
    except Exception:  # fail-closed: si la pausa no se puede leer, está puesta (mismo criterio que pausa.pausa_puesta)
        return True


async def _pr_previo(pr_abierto, cliente, repo: str, rama: str) -> tuple[str | None, tuple[str, ...]]:
    """(URL del PR abierto de la misión o None, notas). Se mira con `_listar` antes de empujar."""
    try:
        url = await pr_abierto(cliente, repo=repo, rama=rama)
        return (url if E.url_de_pr_valida(url) else None), ()
    except Exception:  # fail-soft: no decide ningún empuje; que no se pudo mirar se declara en la nota
        return None, ("pr_previo_no_verificado",)


async def entregar(clon: Clon, *, mision_id: str, repo: str, revision_legible: bool, informe: str, token: str,
                   cliente, tope_bytes: int, tope_total_bytes: int, modelo: str, autor: str, upload_pack: str,
                   pausa_puesta, traer=E.traer_del_clon, commits=E.commits_de_la_rama, diff=E.diff_en_el_espejo,
                   tamanos=E.tamanos_de_la_rama, punta=E.punta_de_la_rama, empujar=E.empujar,
                   abrir_pr=E.abrir_o_actualizar_pr, pr_abierto=E.pr_abierto,
                   consultar_remoto=E.oid_remoto_de_la_rama) -> dict:
    esperada = parsear_autor(autor)
    base = clon.rama_por_omision
    try:
        await traer(clon.espejo, clon.ruta, mision_id=mision_id, upload_pack=upload_pack)
        historial = await commits(clon.espejo, mision_id=mision_id, rama_por_omision=base)
    except EntregaRechazada as exc:  # fail-closed: sin push ni PR; el motivo (saneado) queda en la bitácora
        return _fallo(str(exc), token)
    if not historial:
        previo, notas = await _pr_previo(pr_abierto, cliente, repo, clon.rama)
        return _resultado("sin_cambios", pr_url=previo,  # MAJOR-1: se miraba y no se devolvía
                          notas=(("pr_previo_sin_cambios_nuevos",) if previo else ()) + notas)
    if not revision_legible:
        return _resultado("sin_informe_c5")
    try:
        cambios = await diff(clon.espejo, mision_id=mision_id, rama_por_omision=base)
        medidos = await tamanos(clon.espejo, mision_id=mision_id, rama_por_omision=base)
    except EntregaRechazada as exc:  # fail-closed: sin push ni PR
        return _fallo(str(exc), token)
    titulo = f"Axioma: misión {mision_id[:8]}"
    cuerpo = cuerpo_del_pr(informe, mision_id, modelo)
    violaciones = _sin_repetir(
        _identidad(historial, esperada) + _encoding_no_permitido(historial) + revisar(cambios, repo)
        + lineas_con_secretos(cambios)
        + rutas_con_secretos(cambios)
        + tuple(v for c in historial for v in lineas_con_secretos(c.cambios) + rutas_con_secretos(c.cambios))
        + tuple(Violacion("secretos", "", f"commit {c.sha[:12]}: patrón de credencial en el mensaje o la cabecera")
                for c in historial if hay_secreto(c.crudo))
        + _cuerpo_con_secretos(titulo, cuerpo, informe, token=token)
        + tamanos_excedidos(medidos.por_ruta, tope_bytes=tope_bytes)
        + total_excedido(medidos.total, tope_total_bytes=tope_total_bytes))
    if violaciones:
        return _resultado("rechazada_por_contrato", violaciones=violaciones, token=token)
    # MAJOR-1: desde acá, si ya hay un PR abierto de la misión, cualquier resultado lo declara
    # -- se mira UNA sola vez y se pasa a todo lo que sigue (incluida la relectura de la pausa).
    previo, notas_previas = await _pr_previo(pr_abierto, cliente, repo, clon.rama)
    if await _pausa(pausa_puesta):  # MINOR-1: la pausa pudo ponerse mientras se revisaba
        return _resultado("sin_entregar", motivo="pausa_puesta", pr_url=previo, notas=notas_previas)
    try:
        sha = await punta(clon.espejo, mision_id=mision_id)
    except EntregaRechazada as exc:  # fail-closed: sin push ni PR
        return _fallo(str(exc), token, pr_url=previo)
    try:
        await empujar(clon.espejo, mision_id=mision_id, rama_por_omision=base, token=token)
    except EntregaRechazada as exc:  # fail-closed: MAJOR-A -- antes de decidir, ¿llegó igual?
        try:
            remoto = await consultar_remoto(clon.espejo, mision_id=mision_id, token=token)
        except EntregaRechazada:  # fail-closed: no se sabe si la rama llegó; se declara así
            return _fallo(str(exc), token, rama_empujada="desconocido", pr_url=previo)
        if remoto != sha:
            return _fallo(str(exc), token, rama_empujada=False, pr_url=previo)
        # El remoto tiene exactamente lo que íbamos a empujar: el empuje llegó. Se sigue al PR.
    # --- desde aquí la rama YA está en GitHub: todo resultado lo declara ---
    if await _pausa(pausa_puesta):
        return _sin_pr(sha, previo, notas=("pausa_puesta", *notas_previas))
    try:
        pr = await abrir_pr(cliente, repo=repo, rama=clon.rama, base=base, titulo=titulo, cuerpo=cuerpo)
    except httpx.HTTPStatusError as exc:  # fail-closed: rama empujada, PR no; solo el código HTTP a la bitácora
        return _sin_pr(sha, previo, notas=notas_previas, motivo=f"github_api: {exc.response.status_code}")
    except Exception as exc:  # fail-closed: rama empujada, PR no; solo el tipo (el mensaje puede traer cualquier cosa)
        return _sin_pr(sha, previo, notas=notas_previas, motivo=f"github_api: {type(exc).__name__}")
    if not E.url_de_pr_valida(pr.url):  # MINOR-C: sin URL https válida, el PR no está confirmado
        return _sin_pr(sha, previo, notas=notas_previas, motivo="github_api: pr_url_invalida")
    return _resultado("abierto", pr_url=pr.url, notas=(*pr.notas, *notas_previas), sha=sha)
