"""
Tests de policy/governance/grounding.py — REFORMAS Fase 2 SP3.

Todo acá es PURO: sin I/O, sin DB, sin red. Los ValidationContext se arman a
mano. Numeración de pruebas = spec §9.1 / §9.1b.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOVERNANCE = REPO_ROOT / "policy" / "governance"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GOVERNANCE))

import grounding  # noqa: E402
import validator  # noqa: E402
from las_manos.motor_registry.catalog import MotorCatalog  # noqa: E402


def _ctx(ops: set[str], mutating: set[str] = frozenset({"write_file"})) -> validator.ValidationContext:
    return validator.ValidationContext(
        ops=frozenset(ops),
        mutating_capabilities=frozenset(mutating),
        catalog=MotorCatalog({}),
        config_paths_allowlist=frozenset(),
        repo_root=REPO_ROOT,
    )


# Contexto A del spec: write_file es mutante. Con orden por name,
# 'read_file' < 'write_file', así que write_file queda en /capabilities/1.
CTX_A = _ctx({"write_file", "read_file"})


# --- normalize_args ---------------------------------------------------------

def test_normalize_args_str_and_strip_every_value():
    assert grounding.normalize_args({"name": " write_file ", "mode": 1}) == {"name": "write_file", "mode": "1"}


def test_normalize_args_keeps_keys_untouched():
    # Las claves NO se normalizan: ARGS_MISMATCH (paso 2) es quien juzga claves.
    assert grounding.normalize_args({" Name": "x"}) == {" Name": "x"}


# --- build_snapshot: 7a determinismo -----------------------------------------

def test_7a_same_ctx_same_sha256():
    assert grounding.build_snapshot(CTX_A).sha256 == grounding.build_snapshot(CTX_A).sha256


def test_7a_different_ctx_different_sha256():
    ctx_b = _ctx({"read_file"})
    assert grounding.build_snapshot(CTX_A).sha256 != grounding.build_snapshot(ctx_b).sha256


def test_sha256_is_over_the_canonical_json():
    import hashlib
    snap = grounding.build_snapshot(CTX_A)
    assert snap.sha256 == hashlib.sha256(snap.canonical_json.encode("utf-8")).hexdigest()
    # canónico: sort_keys + separators compactos
    assert snap.canonical_json == json.dumps(
        json.loads(snap.canonical_json), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


# --- 7c: orden por name, no por orden de llegada -----------------------------

def test_7c_entries_and_pointers_ordered_by_name():
    # La garantía real es sorted() en build_snapshot: los punteros y los
    # nombres salen en orden alfabético, sin importar en qué orden vive el
    # set de ops en memoria.
    a = grounding.build_snapshot(_ctx({"write_file", "read_file", "list_dir"}))
    assert [e.pointer for e in a.entries] == ["/capabilities/0", "/capabilities/1", "/capabilities/2"]
    assert [e.args["name"] for e in a.entries] == ["list_dir", "read_file", "write_file"]


def test_snapshot_entry_carries_predicate_and_normalized_args():
    snap = grounding.build_snapshot(CTX_A)
    e = snap.lookup("/capabilities/1")
    assert e is not None
    assert e.predicate == "CAPABILITY_AVAILABLE"
    assert e.args == {"name": "write_file", "mode": "mutating"}
    assert snap.lookup("/capabilities/0").args == {"name": "read_file", "mode": "read_only"}


def test_lookup_returns_none_for_unknown_pointer():
    snap = grounding.build_snapshot(CTX_A)
    assert snap.lookup("/capabilities/99") is None
    assert snap.lookup("/facets/0") is None


# --- 7b: render sin hash ------------------------------------------------------

def test_7b_render_does_not_contain_the_hash_nor_any_prefix_of_it():
    snap = grounding.build_snapshot(CTX_A)
    text = grounding.render(snap)
    assert snap.sha256 not in text
    assert snap.sha256[:12] not in text
    assert "sha256" not in text


def test_render_lists_every_entry_with_its_pointer_and_args():
    snap = grounding.build_snapshot(CTX_A)
    text = grounding.render(snap)
    assert "/capabilities/0: name=read_file, mode=read_only" in text
    assert "/capabilities/1: name=write_file, mode=mutating" in text
    assert "evidence_pointer" in text  # la instrucción de cómo citar


# --- 7d: fallo ruidoso (P10) ---------------------------------------------------

def test_7d_build_snapshot_raises_on_broken_ctx_never_returns_empty():
    class Broken:
        @property
        def ops(self):
            raise OSError("config ilegible")
        mutating_capabilities = frozenset()
    with pytest.raises(grounding.GroundingBuildError):
        grounding.build_snapshot(Broken())


def test_7d_mutating_capabilities_not_a_container_raises_governed_error():
    # Finding Minor #4: el try original solo cubria ctx.ops / ctx.mutating_
    # capabilities como atributos; "name in mutating" corria AFUERA del try,
    # asi que un mutating_capabilities no iterable (p.ej. None) tiraba
    # TypeError pelado en vez de GroundingBuildError (P10).
    class BrokenMutating:
        ops = frozenset({"write_file"})
        mutating_capabilities = None
    with pytest.raises(grounding.GroundingBuildError):
        grounding.build_snapshot(BrokenMutating())


def test_empty_ops_is_a_valid_snapshot_not_an_error():
    # Cero capabilities es una OBSERVACIÓN válida (spec: "predicado con
    # resolver y cero entradas = observación válida, no UNGROUNDED").
    snap = grounding.build_snapshot(_ctx(set()))
    assert snap.entries == ()
    assert len(snap.sha256) == 64


# --- accredit: la citación se verifica, no se cree (spec §2.2, §4.1) --------

SNAP_A = grounding.build_snapshot(CTX_A)   # write_file en /capabilities/1


def _raw(predicate="CAPABILITY_AVAILABLE", args=None, pointer="__absent__"):
    claim = {"predicate": predicate, "args": args or {"name": "write_file", "mode": "mutating"}}
    if pointer != "__absent__":
        claim["evidence_pointer"] = pointer
    return claim


def test_5_valid_pointer_exact_args_is_observado_with_server_written_provenance():
    acc = grounding.accredit(_raw(pointer="/capabilities/1"), SNAP_A)
    assert acc.outcome == "ACCREDITED"
    assert acc.authority == "OBSERVADO"
    assert acc.provenance_ref == f"tool_result:sha256:{SNAP_A.sha256}"
    assert acc.evidence_pointer_raw == "/capabilities/1"


def test_3_no_pointer_is_inferido_no_pointer():
    acc = grounding.accredit(_raw(), SNAP_A)
    assert acc.outcome == "NO_POINTER"
    assert acc.authority == "INFERIDO"
    assert acc.evidence_pointer_raw is None


def test_2_pointer_out_of_range_is_mismatch():
    acc = grounding.accredit(_raw(pointer="/capabilities/99"), SNAP_A)
    assert acc.outcome == "MISMATCH"
    assert acc.authority == "INFERIDO"
    assert "99" in acc.detail


def test_4_valid_pointer_but_args_differ_is_mismatch_the_forged_citation():
    # El test que sostiene el diseño (spec §2.2): puntero real, args falsos.
    acc = grounding.accredit(
        _raw(args={"name": "write_file", "mode": "read_only"}, pointer="/capabilities/1"), SNAP_A
    )
    assert acc.outcome == "MISMATCH"
    assert acc.authority == "INFERIDO"
    assert "read_only" in acc.detail and "mutating" in acc.detail


def test_pointer_to_entry_of_another_predicate_is_mismatch():
    # JOB_STATUS citando una línea de capabilities. En validate() esto nunca
    # llega acá (paso 3 antes del 5) -- pero accredit debe ser correcto solo.
    acc = grounding.accredit(_raw(predicate="JOB_STATUS", args={"job_id": "1", "status": "ok"},
                                  pointer="/capabilities/1"), SNAP_A)
    assert acc.outcome == "MISMATCH"


def test_args_are_compared_with_the_same_normalization_that_built_the_snapshot():
    # Espacios y tipos no producen un PROVENANCE_MISMATCH por formato (spec §5.3).
    acc = grounding.accredit(
        _raw(args={"name": " write_file ", "mode": "mutating"}, pointer="/capabilities/1"), SNAP_A
    )
    assert acc.outcome == "ACCREDITED"


def test_args_not_a_mapping_is_mismatch():
    # Deferred minor de Task 2: la rama "args no es un objeto" de accredit()
    # no tenia test propio.
    acc = grounding.accredit(_raw(args="write_file", pointer="/capabilities/1"), SNAP_A)
    assert acc.outcome == "MISMATCH"
    assert "args" in acc.detail


def test_6_snapshot_error_is_unavailable_even_with_a_pointer():
    acc = grounding.accredit(_raw(pointer="/capabilities/1"), grounding.SnapshotError("config ilegible"))
    assert acc.outcome == "UNAVAILABLE"
    assert acc.authority == "INFERIDO"
    assert "config ilegible" in acc.detail


@pytest.mark.parametrize(
    "bad",
    ["", "capabilities/1", "/capabilities/abc", "/capabilities/-1", "/x" * 150, "/capabilities/1\n"],
)
def test_9_1b_malformed_pointer_is_mismatch_without_exception(bad):
    # "/capabilities/1\n" es el caso del finding Minor #2: con re.match, "$"
    # matchea antes de un "\n" final y el puntero "pasa" el regex; debe
    # tratarse como malformado (fullmatch), no llegar a lookup().
    acc = grounding.accredit(_raw(pointer=bad), SNAP_A)
    assert acc.outcome == "MISMATCH"
    assert acc.authority == "INFERIDO"
    assert acc.evidence_pointer_raw == bad
    assert "malformado" in acc.detail


def test_9_1b_minus_one_never_indexes_from_the_end():
    # Si "-1" se convirtiera a int y se indexara, apuntaría a la ÚLTIMA
    # entrada (write_file) y los args coincidirían -> ACCREDITED. Eso sería
    # fail-open por Python. Debe ser MISMATCH.
    acc = grounding.accredit(_raw(pointer="/capabilities/-1"), SNAP_A)
    assert acc.outcome == "MISMATCH"


def test_non_string_pointer_is_mismatch_not_exception():
    for bad in (7, None, ["/capabilities/1"], {"p": 1}):
        acc = grounding.accredit(_raw(pointer=bad), SNAP_A)
        assert acc.outcome == ("NO_POINTER" if bad is None else "MISMATCH")


# --- validate() con acreditación: el orden 0-6 del spec §4.1 ----------------

import claims  # noqa: E402
import loaders  # noqa: E402

PREDICATES = {
    "CAPABILITY_AVAILABLE": loaders.PredicateSpec("CAPABILITY_AVAILABLE", ("name", "mode"), "Registro de capabilities"),
    "JOB_STATUS": loaders.PredicateSpec("JOB_STATUS", ("job_id", "status"), "Scheduler"),
}


def _validate(raw, grounding_result, ctx=CTX_A):
    acc = grounding.accredit(raw, grounding_result)
    claim = claims.Claim(
        predicate=raw["predicate"],
        args=grounding.normalize_args(raw["args"]),
        authority=acc.authority,
        provenance_ref=acc.provenance_ref,
        evidence_pointer=acc.evidence_pointer_raw if isinstance(acc.evidence_pointer_raw, str) else "",
        scope="test",
    )
    return validator.validate(claim, PREDICATES, ctx, accreditation=acc)


def test_1_job_status_with_invented_pointer_is_resolver_not_implemented_not_mismatch():
    v = _validate(_raw(predicate="JOB_STATUS", args={"job_id": "1", "status": "ok"},
                       pointer="/capabilities/1"), SNAP_A)
    assert v.status == "RESOLVER_NOT_IMPLEMENTED"


def test_2_capability_with_invented_pointer_is_provenance_mismatch_not_resolver():
    v = _validate(_raw(pointer="/capabilities/99"), SNAP_A)
    assert v.status == "PROVENANCE_MISMATCH"


def test_3_capability_without_pointer_is_authority_invalid():
    v = _validate(_raw(), SNAP_A)
    assert v.status == "AUTHORITY_INVALID"


def test_4_forged_citation_is_provenance_mismatch():
    v = _validate(_raw(args={"name": "write_file", "mode": "read_only"}, pointer="/capabilities/1"), SNAP_A)
    assert v.status == "PROVENANCE_MISMATCH"


def test_5_accredited_claim_reaches_the_resolver_and_is_valid():
    v = _validate(_raw(pointer="/capabilities/1"), SNAP_A)
    assert v.status == "VALID"


def test_6_snapshot_error_is_grounding_unavailable_before_anything_else():
    v = _validate(_raw(pointer="/capabilities/1"), grounding.SnapshotError("boom"))
    assert v.status == "GROUNDING_UNAVAILABLE"
    # También para un predicado sin resolver y para uno desconocido: paso 0 va primero.
    v2 = _validate(_raw(predicate="JOB_STATUS", args={"job_id": "1", "status": "ok"}), grounding.SnapshotError("boom"))
    assert v2.status == "GROUNDING_UNAVAILABLE"


def test_8_fact_mismatch_exercised_by_breaking_the_branch_by_hand():
    # Acredita contra SNAP_A (de CTX_A) -> OBSERVADO. Resuelve contra ctx B
    # = A sin write_file -> FACT_MISMATCH. Las dos capas (spec §4.3) con
    # datos distintos a propósito, porque en producción hoy no puede
    # dispararse (spec §9.4).
    ctx_b = _ctx({"read_file"})
    raw = _raw(pointer="/capabilities/1")
    acc = grounding.accredit(raw, SNAP_A)
    assert acc.authority == "OBSERVADO"
    v = _validate(raw, SNAP_A, ctx=ctx_b)
    assert v.status == "FACT_MISMATCH"


def test_validate_without_accreditation_keeps_legacy_behaviour():
    # Los 39 tests existentes llaman validate() con 3 args: INFERIDO corta,
    # OBSERVADO llega al resolver. Nada de eso cambia.
    c = claims.Claim(predicate="CAPABILITY_AVAILABLE", args={"name": "write_file", "mode": "mutating"},
                     authority="INFERIDO", provenance_ref="x", evidence_pointer="x", scope="t")
    assert validator.validate(c, PREDICATES, CTX_A).status == "AUTHORITY_INVALID"
    c2 = c.model_copy(update={"authority": "OBSERVADO"})
    assert validator.validate(c2, PREDICATES, CTX_A).status == "VALID"


def test_validate_without_accreditation_inferido_on_predicate_without_resolver_is_now_resolver_not_implemented():
    # ÚNICO cambio observable del camino legado (finding Important #1 de la
    # revisión final): antes de SP3 esto daba AUTHORITY_INVALID (INFERIDO
    # cortaba en el paso 4 sin mirar si había resolver). Ahora el paso 3
    # (sin resolver) corre ANTES que el paso 4 (sin puntero), así que un
    # predicado sin resolver da RESOLVER_NOT_IMPLEMENTED aunque el llamador
    # ni siquiera sepa que existe `accreditation`.
    c = claims.Claim(
        predicate="JOB_STATUS", args={"job_id": "1", "status": "ok"},
        authority="INFERIDO", provenance_ref="x", evidence_pointer="x", scope="t",
    )
    assert validator.validate(c, PREDICATES, CTX_A).status == "RESOLVER_NOT_IMPLEMENTED"


def test_new_statuses_fit_in_varchar_30():
    for s in ("PROVENANCE_MISMATCH", "GROUNDING_UNAVAILABLE"):
        assert len(s) <= 30
