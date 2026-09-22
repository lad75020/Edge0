"""CLI coverage for generation-wide context-size handling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import edge0.cli as cli
from edge0.context import MAX_CONTEXT_SIZE


@pytest.mark.parametrize("command", ["demo", "chat", "serve"])
@pytest.mark.parametrize("value", [1, MAX_CONTEXT_SIZE])
def test_generation_commands_accept_context_size_boundaries(
        monkeypatch, command, value):
    seen = {}

    def handler(args):
        seen["context_size"] = args.context_size
        return 0

    monkeypatch.setattr(cli, f"cmd_{command}", handler)

    assert cli.main([command, "--context-size", str(value)]) == 0
    assert seen == {"context_size": value}


@pytest.mark.parametrize("value", ["512k", "512K", "512ki", "512ko", "512KO"])
def test_cli_parses_binary_context_size_suffixes(monkeypatch, value):
    seen = {}

    def handler(args):
        seen["context_size"] = args.context_size
        return 0

    monkeypatch.setattr(cli, "cmd_chat", handler)

    assert cli.main(["chat", "--context-size", value]) == 0
    assert seen == {"context_size": MAX_CONTEXT_SIZE}


def test_cli_accepts_context_size_alias(monkeypatch):
    seen = {}

    def handler(args):
        seen["context_size"] = args.context_size
        return 0

    monkeypatch.setattr(cli, "cmd_serve", handler)

    assert cli.main(["serve", "--ctx-size", "512ko"]) == 0
    assert seen == {"context_size": MAX_CONTEXT_SIZE}


@pytest.mark.parametrize(
    "value", ["0", "-1", str(MAX_CONTEXT_SIZE + 1), "513k"])
def test_cli_rejects_invalid_context_size(capsys, value):
    with pytest.raises(SystemExit) as exc:
        cli.main(["chat", "--context-size", value])

    assert exc.value.code == 2
    assert "context size" in capsys.readouterr().err


def test_cli_rejects_non_integer_context_size(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["serve", "--context-size", "many"])

    assert exc.value.code == 2
    assert "context size must be an integer" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["k", "1.5k", "512kb", "1kiB"])
def test_cli_rejects_malformed_context_size(capsys, value):
    with pytest.raises(SystemExit) as exc:
        cli.main(["demo", "--context-size", value])

    assert exc.value.code == 2
    assert "context size" in capsys.readouterr().err


def test_engine_kwargs_propagates_context_size():
    args = SimpleNamespace(
        no_prerouter=False,
        no_lora=False,
        history_slots=False,
        context_size=131072,
    )

    assert cli._engine_kwargs(args) == {"context_size": 131072}


def test_resolve_model_loads_registered_tiers_lazily():
    model_dir, name = cli._resolve_model(SimpleNamespace(
        model="edge0-35b",
        model_dir=None,
        name=None,
    ))

    assert model_dir is None
    assert name == "edge0-35b"
