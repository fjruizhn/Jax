"""El lanzador versionado exige el ambiente Faro aprobado antes de tocar Ollama."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops/las-voces/qwen-auto"


def _script_con_env_de_prueba(tmp_path: Path, env_file: Path, *, stat_mode: str = "644") -> Path:
    stat_bin = tmp_path / "stat-root"
    stat_bin.write_text(
        "#!/bin/sh\n"
        "case \"$2\" in\n"
        f"  '%u:%g:%a') printf '0:0:{stat_mode}\\n' ;;\n"
        f"  '%a') printf '{stat_mode}\\n' ;;\n"
        "  *) exit 64 ;;\n"
        "esac\n"
    )
    stat_bin.chmod(0o755)
    target = tmp_path / "qwen-auto"
    target.write_text(SCRIPT.read_text()
                      .replace("/etc/jax/las-voces-faro.env", str(env_file))
                      .replace("/usr/bin/stat", str(stat_bin)))
    target.chmod(0o755)
    return target


def _env_file(path: Path, *, sha: str = "a" * 40) -> None:
    path.write_text(
        "JAX_FARO_REPO=/srv/faro/claude-skills.git\n"
        f"JAX_FARO_SHA={sha}\n"
        "JAX_FARO_ECOSISTEMA_DIR=/srv/faro/ecosistema\n"
        "JAX_FARO_REF_FRESCURA=refs/heads/main\n"
    )
    path.chmod(0o644)


def test_launcher_has_no_runtime_override_or_shell_source_for_faro_env():
    text = SCRIPT.read_text()
    assert 'FARO_ENV_FILE="/etc/jax/las-voces-faro.env"' in text
    assert "source \"$FARO_ENV_FILE\"" not in text
    assert ". \"$FARO_ENV_FILE\"" not in text
    assert "QWEN_AUTO_FARO_ENV_FILE" not in text
    # Contrato previo del lanzador: la capa Faro no elimina configuración ni
    # ergonomía de Qwen/Ollama que ya usaba la sesión normal.
    assert 'url="${QWEN_AUTO_URL:-http://127.0.0.1:11434}"' in text
    assert 'curl -fsS --max-time 3 "$url/api/tags"' in text
    assert "tmux list-sessions -F '#S'" in text
    assert ' -c "$PWD" "$cmd"' in text


def test_launcher_rejects_missing_or_incomplete_faro_env_before_curl(tmp_path):
    env_file = tmp_path / "las-voces-faro.env"
    script = _script_con_env_de_prueba(tmp_path, env_file)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_mark = tmp_path / "curl-called"
    (bin_dir / "curl").write_text(f"#!/bin/sh\ntouch {curl_mark}\n")
    (bin_dir / "curl").chmod(0o755)
    result = subprocess.run([str(script)], text=True, capture_output=True,
                            env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)})
    assert result.returncode != 0
    assert not curl_mark.exists()


def test_launcher_rejects_unsafe_mode_symlink_and_extra_key_before_curl(tmp_path):
    env_file = tmp_path / "las-voces-faro.env"
    _env_file(env_file)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl_mark = tmp_path / "curl-called"
    (bin_dir / "curl").write_text(f"#!/bin/sh\ntouch {curl_mark}\n")
    (bin_dir / "curl").chmod(0o755)
    unsafe_dir = tmp_path / "unsafe"
    unsafe_dir.mkdir()
    # Only the substituted stat path is test-only; production uses /usr/bin/stat.
    unsafe = _script_con_env_de_prueba(unsafe_dir, env_file, stat_mode="664")
    result = subprocess.run([str(unsafe)], text=True, capture_output=True,
                            env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)})
    assert result.returncode != 0 and not curl_mark.exists()
    linked = tmp_path / "linked.env"
    linked.symlink_to(env_file)
    script = _script_con_env_de_prueba(tmp_path, linked)
    result = subprocess.run([str(script)], text=True, capture_output=True,
                            env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)})
    assert result.returncode != 0 and not curl_mark.exists()
    _env_file(env_file)
    env_file.write_text(env_file.read_text() + "EVIL=$(touch /tmp/no)\n")
    script = _script_con_env_de_prueba(tmp_path, env_file)
    result = subprocess.run([str(script)], text=True, capture_output=True,
                            env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)})
    assert result.returncode != 0 and not curl_mark.exists()
    assert not Path("/tmp/no").exists()
    _env_file(env_file)
    env_file.write_text("JAX_FARO_REPO=/srv/faro/claude-skills.git\n")
    result = subprocess.run([str(script)], text=True, capture_output=True,
                            env={"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path)})
    assert result.returncode != 0
    assert not curl_mark.exists()


def test_launcher_passes_only_the_four_readonly_values_to_tmux(tmp_path):
    env_file = tmp_path / "las-voces-faro.env"
    _env_file(env_file, sha="b" * 40)
    script = _script_con_env_de_prueba(tmp_path, env_file)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "tmux.log"
    for command, body in {
        "qwen": "#!/bin/sh\nexit 0\n",
        "curl": "#!/bin/sh\nprintf '{\"name\":\"qwen3.8-mesa-131k\"}\\n'\n",
        "tmux": f"#!/bin/sh\nprintf '%s\\n' \"$*\" > {log}\n",
    }.items():
        p = bin_dir / command
        p.write_text(body)
        p.chmod(0o755)
    result = subprocess.run([str(script), "--help"], text=True, capture_output=True, env={
        "PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(tmp_path), "JAX_FARO_DUENIO_UID": str(os.getuid()),
    })
    assert result.returncode == 0, result.stderr
    command = log.read_text()
    assert "JAX_FARO_SHA=bbbb" in command
    assert "JAX_FARO_REF_FRESCURA=refs/heads/main" in command
    assert "JAX_FARO_DUENIO_UID=" not in command
    assert "-u JAX_FARO_DUENIO_UID" in command
