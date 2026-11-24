"""A scripted fake that leaks an email address on its first reply and comes back clean on retry: shows
guarded_send's repair loop catching and correcting a guardrail violation, not any real model's behavior.
"""

from __future__ import annotations

from llm_playground import Chat, FakeProvider
from llm_playground.guardrails import GuardrailError, guarded_send, max_length, no_pii_patterns

replies = iter(
    [
        "Sure, reach out to our support team at support@example.com for help.",
        "Sure, reach out to our support team through the in-app contact form for help.",
    ]
)


def scripted_model(request: object) -> str:  # type: ignore[no-untyped-def]
    return next(replies)


if __name__ == "__main__":
    chat = Chat(FakeProvider(reply=scripted_model), model="fake")
    try:
        result = guarded_send(chat, "How do I contact support?", [no_pii_patterns(), max_length(200)])
    except GuardrailError as err:
        print(f"no reply ever passed: {err}")
    else:
        print(f"accepted after {result.attempts} attempt(s): {result.text}")
        if result.violations_by_attempt:
            print(f"rejected reply's violations: {result.violations_by_attempt[0]}")
