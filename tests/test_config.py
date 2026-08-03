from pathlib import Path

from ragwatcher import config
from ragwatcher.config import Settings, apply_cli_overrides, load


def test_defaults_load_offline():
    s = Settings()
    assert s.embed.model.startswith("BAAI/")
    assert s.chunk.strategy == "recursive"
    assert s.store.backend == "simple"


def test_per_dir_toml_overrides(tmp_path: Path, monkeypatch):
    per = tmp_path / ".ragwatcher.toml"
    per.write_text(
        """
[embed]
model = "test/model"

[chunk]
size = 256
"""
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    s, origins = load(data_dir=tmp_path)
    assert s.embed.model == "test/model"
    assert s.chunk.size == 256
    assert any("ragwatcher.toml" in v for v in origins.values())


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("RAGWATCHER_LOG__LEVEL", "debug")
    s = Settings()
    assert s.log.level == "debug"


def test_cli_override_beats_env(monkeypatch):
    monkeypatch.setenv("RAGWATCHER_LOG__LEVEL", "debug")
    s = Settings()
    s2 = apply_cli_overrides(s, {"log.level": "warning"})
    assert s2.log.level == "warning"


def test_untrusted_per_dir_skipped(tmp_path: Path, monkeypatch):
    per = tmp_path / ".ragwatcher.toml"
    per.write_text("[embed]\nmodel = 'evil/model'\n")

    # Monkey-patch _trust_ok to return False
    monkeypatch.setattr(config, "_trust_ok", lambda p: False)
    s, _origins = load(data_dir=tmp_path)
    assert s.embed.model != "evil/model"
