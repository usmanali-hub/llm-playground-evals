"""Shows the repair loop with a scripted offline model that makes three realistic mistakes before getting it right.

The scripted replies are written by hand to look like common failures (a fenced reply with a string where a number
belongs, a missing field, an extra field). This demonstrates how the harness reacts; it says nothing about how often a
real model makes these mistakes.
"""

import json
from pathlib import Path

from llm_playground import Chat, FakeProvider, extract_structured, scripted_reply

SCHEMA = json.loads((Path(__file__).with_name("invoice_schema.json")).read_text())
TEXT = "Invoice from Acme Tools: 3 bolts and 2 nuts, total 12.50 USD."

REPLIES = [
    '```json\n{"vendor": "Acme Tools", "total": "12.50", "currency": "USD", '
    '"items": [{"name": "bolts", "qty": 3}]}\n```',
    '{"vendor": "Acme Tools", "total": 12.5, "currency": "USD"}',
    '{"vendor": "Acme Tools", "total": 12.5, "currency": "USD", "items": [{"name": "bolts", "qty": 3}], "note": "ok"}',
    '{"vendor": "Acme Tools", "total": 12.5, "currency": "USD", '
    '"items": [{"name": "bolts", "qty": 3}, {"name": "nuts", "qty": 2}]}',
]

chat = Chat(FakeProvider(reply=scripted_reply(REPLIES)), "fake")
result = extract_structured(chat, TEXT, SCHEMA, max_repairs=3)
for i, errors in enumerate(result.errors_by_attempt, start=1):
    print(f"attempt {i} rejected:")
    for error in errors:
        print(f"  - {error}")
print(f"attempt {result.attempts} accepted: {json.dumps(result.data)}")
print(f"tokens (estimated by the fake): {result.usage.total}")
