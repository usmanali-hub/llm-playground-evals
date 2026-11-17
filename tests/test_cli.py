from pathlib import Path

import pytest

from llm_playground.__main__ import main


def test_fake_chat_prints_reply_and_estimated_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["chat", "What is a token?", "--system", "Answer briefly."]) == 0
    out = capsys.readouterr().out
    assert "You said: What is a token?" in out and "tokens: in=" in out and "estimated" in out


def test_streaming_prints_the_same_reply(capsys: pytest.CaptureFixture[str]) -> None:
    main(["chat", "hello", "--stream"])
    streamed = capsys.readouterr().out
    main(["chat", "hello"])
    plain = capsys.readouterr().out
    assert streamed.splitlines()[0] == plain.splitlines()[0]


def test_price_flags_add_a_cost(capsys: pytest.CaptureFixture[str]) -> None:
    main(["chat", "hello", "--price-in", "3", "--price-out", "15"])
    assert "cost=" in capsys.readouterr().out


def test_real_provider_needs_a_model_and_both_prices_or_neither() -> None:
    with pytest.raises(SystemExit):
        main(["chat", "hi", "--provider", "anthropic"])
    with pytest.raises(SystemExit):
        main(["chat", "hi", "--price-in", "3"])


def test_missing_api_key_is_a_clean_error(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["chat", "hi", "--provider", "anthropic", "--model", "x"]) == 1
    assert "ANTHROPIC_API_KEY is not set" in capsys.readouterr().err


def test_extract_needs_a_schema_and_reports_failure_cleanly(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["extract", "text"])
    schema = tmp_path / "s.json"
    schema.write_text('{"type": "object", "required": ["name"]}')
    # the offline fake echoes text, never JSON, so every attempt fails and the CLI says so
    assert main(["extract", "Aisha is 31", "--schema", str(schema), "--repairs", "1"]) == 1
    assert "no valid output after 2 attempts" in capsys.readouterr().err
    assert main(["extract", "x", "--schema", str(tmp_path / "missing.json")]) == 1
    assert "error:" in capsys.readouterr().err
