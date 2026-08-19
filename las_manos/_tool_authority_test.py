#!/usr/bin/env python3
"""
Motor Registry — gate de autoridad de tool_calls (GAP 2, Fase 2).

Corazón de la sesión 2026-08-19 (T4): 22 casos adversariales corridos
primero a mano contra la DB real y el workspace real (ver sesión), todos
con el comportamiento esperado -- ningún caso ejecutó cuando debía
rechazar. Esta suite los deja reproducibles sin tocar la DB/filesystem
reales: WORKSPACE_ROOT se parchea a un tempdir por test, event_append se
mockea (la escritura real a jacobs_events ya se verificó a mano con
evidencia -- acá se confirma que SE LLAMA, no se reprueba MariaDB).

Hallazgo real durante la corrida a mano: jacobs_events.pipeline_id es
VARCHAR(36) (dimensionado para un UUID) -- dos de los job_id sintéticos de
prueba (>36 chars) hicieron que event_append fallara con DataError. El
fail-soft de _reject/_execution_error absorbió el error correctamente (la
decisión de seguridad fue igual de correcta), pero confirma que un job_id
real siempre es exactamente un UUID de 36 chars (job_store.py:
str(uuid.uuid4())) -- no es un bug alcanzable en producción, solo una
fragilidad del harness de prueba. No se "arregla" acá (el esquema es
correcto para su único caller real); se deja testeado tal cual.

Corre desde /home/fruiz/jax/las_manos con:
  PYTHONPATH=/home/fruiz/jax/las_manos \
  /home/fruiz/jax/las_manos/.venv/bin/python \
  /home/fruiz/jax/las_manos/_tool_authority_test.py

En memoria de Jairo Urbina.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from motor_registry import tool_authority
from motor_registry.catalog import MotorCatalog

# Mismos valores que el seed real aprobado (jax-platform/backend/db/
# migrations.py::_seed_file_tools_capabilities) -- si diverge de ahí, este
# test debe actualizarse a propósito, no arrastrar silenciosamente.
_CAP_CFG = {
    "capabilities": {
        "file_read": {
            "allowed_callers": ["jacobs"], "risk_level": "medium",
            "sandbox_only": True, "requires_human_gate": False,
            "max_execution_minutes": 1, "max_recursion_depth": 0,
            "output_schema": "",
            "forbidden_paths": [".env", "secrets/", "private_keys/", "credentials/"],
        },
        "file_write": {
            "allowed_callers": ["jacobs"], "risk_level": "medium",
            "sandbox_only": True, "requires_human_gate": True,
            "max_execution_minutes": 1, "max_recursion_depth": 0,
            "output_schema": "",
            "forbidden_paths": [".env", "secrets/", "private_keys/", "credentials/"],
        },
    },
}


class ToolAuthorityTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.workspace = Path(self._tmpdir.name)
        self.catalog = MotorCatalog(_CAP_CFG)

        (self.workspace / "legit.txt").write_text("contenido legítimo\n")
        (self.workspace / ".env").write_text("FAKE_KEY=no-es-real\n")
        (self.workspace / "secrets").mkdir()
        (self.workspace / "secrets" / "key.txt").write_text("fake-secret\n")
        (self.workspace / "secrets" / "link_to_env").symlink_to(self.workspace / ".env")
        (self.workspace / "unreadable.txt").write_text("sin permisos\n")
        os.chmod(self.workspace / "unreadable.txt", 0o000)
        self.addCleanup(lambda: os.chmod(self.workspace / "unreadable.txt", 0o644))
        (self.workspace / "binary.bin").write_bytes(bytes(range(256)))
        (self.workspace / "large.txt").write_text("x" * (tool_authority.MAX_READ_BYTES + 1))

        outside = Path(self._tmpdir.name).parent / f"_outside_{os.getpid()}.txt"
        outside.write_text("fuera del jail\n")
        self.addCleanup(outside.unlink, missing_ok=True)
        (self.workspace / "escape_symlink.txt").symlink_to(outside)

        self._patchers = [
            patch.object(tool_authority, "WORKSPACE_ROOT", self.workspace.resolve()),
            patch.object(tool_authority, "event_append", AsyncMock()),
        ]
        for p in self._patchers:
            p.start()
        self.addCleanup(self._stop_patchers)

    def _stop_patchers(self):
        for p in self._patchers:
            p.stop()

    async def _call(self, tool_name, args, caller="jacobs", job_id="test-job-id"):
        args_json = args if isinstance(args, str) else json.dumps(args)
        return await tool_authority.authorize_and_execute_tool_call(
            tool_name=tool_name, arguments_json=args_json, caller=caller,
            job_id=job_id, catalog=self.catalog,
        )

    # --- 1. camino feliz ---
    async def test_1_read_file_legitimo_ejecuta(self):
        r = await self._call("read_file", {"path": "legit.txt"})
        assert r["decision"] == "executed", r
        assert r["content"] == "contenido legítimo\n", r
        tool_authority.event_append.assert_not_awaited()  # solo rechazos/errores auditan

    # --- 2. ruta absoluta ---
    async def test_2_ruta_absoluta_rechaza(self):
        r = await self._call("read_file", {"path": "/etc/passwd"})
        assert r["decision"] == "rejected", r
        assert "absoluta" in r["reason"], r
        tool_authority.event_append.assert_awaited_once()

    # --- 3. '..' escapa el workspace ---
    async def test_3_dotdot_escapa_rechaza(self):
        r = await self._call("read_file", {"path": "../fuera.txt"})
        assert r["decision"] == "rejected", r
        assert "escapa" in r["reason"], r

    # --- 4. forbidden_paths: .env ---
    async def test_4_env_prohibido_rechaza(self):
        r = await self._call("read_file", {"path": ".env"})
        assert r["decision"] == "rejected", r
        assert "prohibida" in r["reason"], r

    # --- 5. write_file declarada, no ejecutable -- rechaza limpio, no crashea ---
    async def test_5_write_file_no_ejecutable_rechaza_limpio(self):
        r = await self._call("write_file", {"path": "x.txt", "content": "hola"})
        assert r["decision"] == "rejected", r
        # file_write tiene requires_human_gate=1 en el seed real -- ese
        # check corre ANTES del check de EXECUTABLE_TOOLS (orden a
        # propósito, ver tool_authority.py), así que la razón real es el
        # gate humano, no "no ejecutable". Ambos caminos rechazan limpio;
        # se confirma cuál es el que realmente dispara.
        assert "human_gate" in r["reason"] or "no es ejecutable" in r["reason"], r

    async def test_5b_tool_declarada_no_ejecutable_sin_gate_humano(self):
        """Variante que aísla el check de EXECUTABLE_TOOLS del check de
        requires_human_gate -- una capability sin gate pero no ejecutable
        debe rechazar por la razón correcta, no por la del gate."""
        cfg = {"capabilities": {"file_write": {**_CAP_CFG["capabilities"]["file_write"], "requires_human_gate": False}}}
        catalog = MotorCatalog(cfg)
        r = await tool_authority.authorize_and_execute_tool_call(
            tool_name="write_file", arguments_json=json.dumps({"path": "x.txt", "content": "hola"}),
            caller="jacobs", job_id="test-job-id", catalog=catalog,
        )
        assert r["decision"] == "rejected", r
        assert "no es ejecutable" in r["reason"], r

    # --- 6. tool inventado ---
    async def test_6_tool_inventado_rechaza(self):
        r = await self._call("delete_everything", {"path": "x"})
        assert r["decision"] == "rejected", r
        assert "no mapea a ninguna capability" in r["reason"], r

    # --- 7. caller no autorizado ---
    async def test_7_caller_no_autorizado_rechaza(self):
        r = await self._call("read_file", {"path": "legit.txt"}, caller="rogue_agent")
        assert r["decision"] == "rejected", r
        assert "allowed_callers" in r["reason"], r

    # --- capability inexistente en el catálogo (ambigüedad = rechazo) ---
    async def test_capability_sin_seed_rechaza(self):
        catalog_vacio = MotorCatalog({"capabilities": {}})
        r = await tool_authority.authorize_and_execute_tool_call(
            tool_name="read_file", arguments_json=json.dumps({"path": "legit.txt"}),
            caller="jacobs", job_id="test-job-id", catalog=catalog_vacio,
        )
        assert r["decision"] == "rejected", r
        assert "no encontrada en el catálogo" in r["reason"], r

    # --- vector de Fernando: forbidden_paths sobre forma CANÓNICA, no string cruda ---
    async def test_secrets_sin_barra_final_rechaza(self):
        r = await self._call("read_file", {"path": "secrets"})
        assert r["decision"] == "rejected", r

    async def test_dotslash_secrets_normaliza_y_rechaza(self):
        r = await self._call("read_file", {"path": "./secrets/x"})
        assert r["decision"] == "rejected", r

    async def test_subdotdot_normaliza_a_env_y_rechaza(self):
        r = await self._call("read_file", {"path": "sub/../.env"})
        assert r["decision"] == "rejected", r

    async def test_case_variant_no_bypassa_ni_falsea_bloqueo(self):
        """.ENV/Secrets/ no coinciden con archivos reales en minúscula
        (filesystem case-sensitive) -- el resultado correcto es 'no
        encontrado', NO 'ejecutado' (bypass) NI 'rechazado por
        forbidden_paths' (falso positivo, el string no matchea de verdad)."""
        for bad_path in (".ENV", "Secrets/key.txt"):
            r = await self._call("read_file", {"path": bad_path})
            assert r["decision"] == "execution_error", (bad_path, r)
            assert r["reason"] == "archivo no encontrado", (bad_path, r)

    async def test_symlink_dentro_del_workspace_a_forbidden_rechaza(self):
        r = await self._call("read_file", {"path": "secrets/link_to_env"})
        assert r["decision"] == "rejected", r
        assert "prohibida" in r["reason"], r

    async def test_symlink_escapa_jail_rechaza(self):
        r = await self._call("read_file", {"path": "escape_symlink.txt"})
        assert r["decision"] == "rejected", r
        assert "escapa" in r["reason"], r

    # --- archivos ilegibles / binarios / grandes: NO son rechazos de autoridad ---
    async def test_archivo_no_existe_es_execution_error_no_rejected(self):
        r = await self._call("read_file", {"path": "no_existe.txt"})
        assert r["decision"] == "execution_error", r

    async def test_archivo_sin_permisos_es_execution_error(self):
        r = await self._call("read_file", {"path": "unreadable.txt"})
        assert r["decision"] == "execution_error", r
        assert "permisos" in r["reason"], r

    async def test_archivo_binario_es_execution_error(self):
        r = await self._call("read_file", {"path": "binary.bin"})
        assert r["decision"] == "execution_error", r
        assert "binario" in r["reason"], r

    async def test_archivo_muy_grande_es_execution_error(self):
        r = await self._call("read_file", {"path": "large.txt"})
        assert r["decision"] == "execution_error", r
        assert "excede el límite" in r["reason"], r

    async def test_directorio_en_vez_de_archivo(self):
        (self.workspace / "un_dir").mkdir()
        r = await self._call("read_file", {"path": "un_dir"})
        # "un_dir" no está en forbidden_paths -- pasa el jail, falla en la
        # lectura misma (es directorio), execution_error, no rejected.
        assert r["decision"] == "execution_error", r
        assert "directorio" in r["reason"], r

    # --- arguments malformados ---
    async def test_arguments_no_json_rechaza(self):
        r = await self._call("read_file", "esto no es json")
        assert r["decision"] == "rejected", r
        assert "JSON válido" in r["reason"], r

    async def test_path_ausente_rechaza(self):
        r = await self._call("read_file", {})
        assert r["decision"] == "rejected", r

    # --- nunca ejecuta cuando rechaza (invariante central de T4) ---
    async def test_ningun_rejected_trae_content(self):
        casos = [
            ("read_file", {"path": "/etc/passwd"}),
            ("read_file", {"path": "../x.txt"}),
            ("read_file", {"path": ".env"}),
            ("write_file", {"path": "x.txt", "content": "y"}),
            ("nope", {"path": "x"}),
        ]
        for tool_name, args in casos:
            r = await self._call(tool_name, args)
            assert r["decision"] != "executed", (tool_name, args, r)
            assert r["content"] is None, (tool_name, args, r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
