"""Smoke-test all subcommands with --help. No side effects."""
from typer.testing import CliRunner

from ragwatcher.cli import app

runner = CliRunner()


def test_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ragwatcher" in result.output.lower()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip()


SUBCOMMANDS = ["serve", "query", "index", "stats", "sources", "purge", "doctor", "config"]


def test_each_subcommand_help():
    for cmd in SUBCOMMANDS:
        r = runner.invoke(app, [cmd, "--help"])
        assert r.exit_code == 0, f"{cmd}: {r.output}"
