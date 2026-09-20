from pathlib import Path


def test_block5_has_no_execution_or_las_manos_imports():
    root = Path(__file__).resolve().parents[2] / "policy" / "decision_record"
    text = "\n".join(path.read_text() for path in root.glob("*.py"))
    assert "execution_authorized" not in text
    assert "execution_id" not in text
    assert "las_manos" not in text
