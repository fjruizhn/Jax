import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "projections" / "lv002_projection.py"
spec = importlib.util.spec_from_file_location("lv002", MODULE)
lv002 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lv002)
FIXTURE = ROOT / "projections" / "fixtures" / "canonical"


class ProjectionContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.canonical = self.base / "agents"
        shutil.copytree(FIXTURE, self.canonical)
        self.output = self.base / "sandbox"

    def tearDown(self):
        self.tmp.cleanup()

    def agents(self):
        return lv002.read_agents(self.canonical)

    def generate(self, target="codex", dry=False):
        args = ["generate", "--canonical", str(self.canonical), "--target", target, "--output", str(self.output)]
        if dry:
            args.append("--dry-run")
        return lv002.main(args)

    def mutate(self, fn):
        path = self.canonical / "fixture-agent.json"
        data = json.loads(path.read_text())
        fn(data)
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_canonical_valid(self):
        self.assertEqual(self.agents()[0]["id"], "fixture-agent")

    def test_canonical_invalid(self):
        self.mutate(lambda x: x.__setitem__("unexpected", True))
        with self.assertRaises(lv002.ContractError):
            self.agents()

    def test_missing_dependency_reference(self):
        self.mutate(lambda x: x.__setitem__("skills", ["absent-skill"]))
        with self.assertRaisesRegex(lv002.ContractError, "missing canonical skill"):
            self.agents()

    def test_generation_is_deterministic(self):
        self.assertEqual(self.generate(), 0)
        first = (self.output / "agents/fixture-agent.toml").read_bytes()
        self.assertEqual(self.generate(), 0)
        self.assertEqual(first, (self.output / "agents/fixture-agent.toml").read_bytes())

    def test_projection_missing(self):
        report = lv002.inspect(self.agents(), "codex", self.output)
        self.assertEqual(report["agents/fixture-agent.toml"], "MISSING")

    def test_projection_exact(self):
        self.generate()
        self.assertEqual(set(lv002.inspect(self.agents(), "codex", self.output).values()), {"IN_SYNC"})

    def test_manual_projection_change_is_drift(self):
        self.generate()
        path = self.output / "agents/fixture-agent.toml"
        path.write_text(path.read_text() + "manual change\n")
        self.assertEqual(lv002.inspect(self.agents(), "codex", self.output)["agents/fixture-agent.toml"], "DRIFT")

    def test_old_projection_is_stale_after_canonical_update(self):
        self.generate()
        self.mutate(lambda x: x.__setitem__("purpose", "Updated canonical purpose."))
        self.assertEqual(lv002.inspect(self.agents(), "codex", self.output)["agents/fixture-agent.toml"], "STALE")

    def test_unsupported_target_fails_closed(self):
        with self.assertRaises(lv002.ContractError):
            lv002.expected_artifacts(self.agents(), "unsupported")

    def test_qwen_is_human_decision_required(self):
        with self.assertRaisesRegex(lv002.ContractError, "HUMAN_DECISION_REQUIRED"):
            lv002.expected_artifacts(self.agents(), "qwen")

    def test_drift_cannot_be_overwritten(self):
        self.generate()
        path = self.output / "agents/fixture-agent.toml"
        path.write_text(path.read_text() + "manual change\n")
        self.assertEqual(self.generate(), 2)

    def test_dry_run_does_not_modify(self):
        self.assertEqual(self.generate(dry=True), 0)
        self.assertFalse(self.output.exists())

    def test_ariadna_remains_not_active(self):
        project_agents = lv002.read_agents(ROOT / "agents")
        self.assertEqual(project_agents[0]["status"], "PROPOSED_NOT_ACTIVE")
        self.assertEqual(project_agents[0]["runtime_targets"], [])
        with self.assertRaises(lv002.ContractError):
            lv002.expected_artifacts([dict(project_agents[0], runtime_targets=["codex"])], "codex")

    def test_inspection_never_reverse_syncs(self):
        original = (self.canonical / "fixture-agent.json").read_bytes()
        lv002.inspect(self.agents(), "codex", self.output)
        self.assertEqual((self.canonical / "fixture-agent.json").read_bytes(), original)

    def test_projected_fixture_contains_no_secret(self):
        self.generate("claude")
        text = (self.output / "agents/fixture-agent.md").read_text().lower()
        self.assertNotIn("password", text)
        self.assertNotIn("api_key", text)

    def test_secret_named_field_is_rejected(self):
        self.mutate(lambda x: x.__setitem__("api_key", "not-allowed"))
        with self.assertRaisesRegex(lv002.ContractError, "secrets"):
            self.agents()

    def test_unmanaged_projection_is_reported(self):
        path = self.output / "agents/fixture-agent.toml"
        path.parent.mkdir(parents=True)
        path.write_text("not generated\n")
        self.assertEqual(lv002.inspect(self.agents(), "codex", self.output)["agents/fixture-agent.toml"], "UNMANAGED")


if __name__ == "__main__":
    unittest.main()
