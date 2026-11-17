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
