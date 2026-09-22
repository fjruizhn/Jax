# tests/test_ejecutor_contratos_contexto.py
"""`jax/ejecutor/contratos/contexto.py`: el punto único que genera el CLAUDE.md de
axioma (spec §6.1, nunca a mano) y empaqueta las skills declaradas en cerebros.toml.
`claude_md()` depende de la constitución real de esta máquina
(/home/fruiz/claude-skills/common/CLAUDE.md.core); las pruebas que la necesitan se
saltan donde no existe -- mismo criterio que tests/test_ejecutor_generar_claude_md.py."""
import pytest

from jax.ejecutor.contratos import contexto as CX

requiere_constitucion_real = pytest.mark.skipif(
    not CX.constitucion_disponible(), reason="host-bound, Fase 0: no existe en este runner")


def test_skills_declaradas_son_las_tres_del_encargo():
    assert CX.skills_declaradas() == ("migrando-sin-romper", "desde-la-fuente", "endureciendo")


@requiere_constitucion_real
def test_claude_md_no_esta_vacio_y_su_sha256_es_estable():
    doc = CX.claude_md()
    assert doc.startswith(b"# El Ejecutor de Axioma")
    assert CX.sha256_claude_md() == CX.sha256_claude_md()  # determinista


def test_archivos_de_skills_con_una_fuente_de_prueba(tmp_path, monkeypatch):
    fuente = tmp_path / "skills"
    (fuente / "migrando-sin-romper").mkdir(parents=True)
    (fuente / "migrando-sin-romper" / "SKILL.md").write_text("m")
    (fuente / "desde-la-fuente").mkdir(parents=True)
    (fuente / "desde-la-fuente" / "SKILL.md").write_text("d")
    (fuente / "endureciendo" / "references").mkdir(parents=True)
    (fuente / "endureciendo" / "SKILL.md").write_text("e")
    (fuente / "endureciendo" / "references" / "checklist.md").write_text("c")
    monkeypatch.setattr(CX, "skills_fuente", lambda: fuente)

    archivos = CX.archivos_de_skills()
    assert archivos["migrando-sin-romper/SKILL.md"] == b"m"
    assert archivos["desde-la-fuente/SKILL.md"] == b"d"
    assert archivos["endureciendo/SKILL.md"] == b"e"
    assert archivos["endureciendo/references/checklist.md"] == b"c"
    assert len(archivos) == 4


def test_archivos_de_skills_sin_una_carpeta_declarada_es_skillfaltante(tmp_path, monkeypatch):
    fuente = tmp_path / "skills"
    (fuente / "migrando-sin-romper").mkdir(parents=True)
    (fuente / "migrando-sin-romper" / "SKILL.md").write_text("m")
    # "desde-la-fuente" y "endureciendo" NO existen.
    monkeypatch.setattr(CX, "skills_fuente", lambda: fuente)

    with pytest.raises(CX.SkillFaltante) as e:
        CX.archivos_de_skills()
    assert e.value.args[0] == "desde-la-fuente"  # el primero en orden de la lista


def test_las_tres_skills_reales_de_claude_skills_hoy_dan_skillfaltante():
    """Verdad operacional al 2026-09-22: las tres skills NO existen todavía en
    /home/fruiz/claude-skills/common/skills/ (sólo en el worktree sin mergear
    cs-freno-generados, rama fix/freno-generados). Esta prueba documenta el estado
    real y falla sola el día que alguien las mergee -- momento en el que hay que
    borrarla, no arreglarla."""
    fuente = CX.skills_fuente()
    if all((fuente / n).is_dir() for n in CX.skills_declaradas()):
        pytest.skip("las tres skills ya existen en la ruta canónica: mergeadas")
    with pytest.raises(CX.SkillFaltante):
        CX.archivos_de_skills()


def test_renderizar_etapa_escribe_claude_md_sha256_y_skills(tmp_path, monkeypatch):
    fuente = tmp_path / "skills-fuente"
    for nombre in CX.skills_declaradas():
        (fuente / nombre).mkdir(parents=True)
        (fuente / nombre / "SKILL.md").write_text(f"skill {nombre}")
    monkeypatch.setattr(CX, "skills_fuente", lambda: fuente)
    monkeypatch.setattr(CX, "claude_md", lambda: b"# contenido de prueba\n")

    etapa = tmp_path / "etapa"
    CX.renderizar_etapa(etapa)

    assert (etapa / "CLAUDE.md").read_bytes() == b"# contenido de prueba\n"
    import hashlib
    assert (etapa / "CLAUDE.md.sha256").read_text() == hashlib.sha256(b"# contenido de prueba\n").hexdigest()
    for nombre in CX.skills_declaradas():
        assert (etapa / "skills" / nombre / "SKILL.md").read_text() == f"skill {nombre}"


def test_principal_cli_escribe_la_etapa(tmp_path, monkeypatch):
    fuente = tmp_path / "skills-fuente"
    for nombre in CX.skills_declaradas():
        (fuente / nombre).mkdir(parents=True)
        (fuente / nombre / "SKILL.md").write_text("s")
    monkeypatch.setattr(CX, "skills_fuente", lambda: fuente)
    monkeypatch.setattr(CX, "claude_md", lambda: b"# x\n")
    etapa = tmp_path / "etapa"
    assert CX.principal([str(etapa)]) == 0
    assert (etapa / "CLAUDE.md").exists()
