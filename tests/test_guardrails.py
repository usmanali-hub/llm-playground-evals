import pytest

from llm_playground import Chat, FakeProvider
from llm_playground.guardrails import (
    GuardrailError,
    guarded_send,
    max_length,
    min_length,
    must_match,
    no_banned_terms,
    no_pii_patterns,
    run_guardrails,
)


def test_max_length_flags_a_reply_over_the_limit() -> None:
    check = max_length(10)
    assert check("short") == []
    violations = check("this text is definitely over ten characters")
    assert violations and "over the 10 characters limit" in violations[0]


def test_max_length_counts_words_when_asked() -> None:
    check = max_length(3, unit="words")
    assert check("one two three") == []
    assert check("one two three four") != []


def test_min_length_flags_a_suspiciously_short_reply() -> None:
    check = min_length(20)
    assert check("way too short") != []
    assert check("this reply is comfortably over the twenty character minimum") == []


def test_no_banned_terms_is_case_insensitive_substring_match() -> None:
    check = no_banned_terms(["confidential", "internal only"])
    assert check("This document is marked CONFIDENTIAL.") != []
    assert check("A plain public answer.") == []


def test_no_pii_patterns_catches_email_phone_and_card_shaped_numbers() -> None:
    check = no_pii_patterns()
    assert any("email" in v for v in check("contact me at jane@example.com"))
    assert any("phone" in v for v in check("call 555-867-5309 anytime"))
    assert any("digit" in v for v in check("card number 4111 1111 1111 1111"))
    assert check("a perfectly ordinary sentence with no PII in it") == []


def test_must_match_requires_a_pattern_rather_than_forbidding_one() -> None:
    check = must_match(r"^Hello", "must start with Hello")
    assert check("Hello there") == []
    assert check("Hi there") != []


def test_run_guardrails_collects_violations_from_every_guardrail_in_order() -> None:
    violations = run_guardrails(
        "CONFIDENTIAL: email jane@example.com", [no_banned_terms(["confidential"]), no_pii_patterns()]
    )
    assert len(violations) == 2
    assert "confidential" in violations[0]
    assert "email" in violations[1]


def test_guarded_send_passes_through_a_clean_first_reply() -> None:
    chat = Chat(FakeProvider(reply=lambda r: "a perfectly clean reply"), "fake")
    result = guarded_send(chat, "say something", [max_length(100)])
    assert result.attempts == 1
    assert result.text == "a perfectly clean reply"
    assert result.violations_by_attempt == []


def test_guarded_send_retries_after_a_violation_and_succeeds() -> None:
    replies = iter(["CONFIDENTIAL data here", "a clean reply now"])

    def reply(request):  # type: ignore[no-untyped-def]
        return next(replies)

    chat = Chat(FakeProvider(reply=reply), "fake")
    result = guarded_send(chat, "say something", [no_banned_terms(["confidential"])])
    assert result.attempts == 2
    assert result.text == "a clean reply now"
    assert len(result.violations_by_attempt) == 1
    assert "confidential" in result.violations_by_attempt[0][0]


def test_guarded_send_raises_after_exhausting_retries() -> None:
    chat = Chat(FakeProvider(reply=lambda r: "always has jane@example.com in it"), "fake")
    with pytest.raises(GuardrailError) as exc_info:
        guarded_send(chat, "say something", [no_pii_patterns()], max_retries=1)
    err = exc_info.value
    assert err.attempts == 2
    assert err.violations and "email" in err.violations[0]
    assert "jane@example.com" in err.text


def test_guarded_send_repair_prompt_lists_the_violations() -> None:
    seen_prompts = []

    def reply(request):  # type: ignore[no-untyped-def]
        prompt = request.messages[-1].content
        seen_prompts.append(prompt)
        return "short" if len(seen_prompts) == 1 else "a nice long reply that clears the minimum length easily"

    chat = Chat(FakeProvider(reply=reply), "fake")
    guarded_send(chat, "say something", [min_length(20)])
    assert "failed these checks" in seen_prompts[1]
    assert "under the 20 characters minimum" in seen_prompts[1]
