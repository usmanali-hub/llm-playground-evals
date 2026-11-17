"""HTTP transport: the only place that touches the network, kept small so everything else is testable offline."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any, Protocol

from llm_playground.types import ProviderError

TIMEOUT_SECONDS = 60
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
MAX_ERROR_CHARS = 300


class Transport(Protocol):
    def post_json(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]: ...

    def post_stream(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> Iterator[str]: ...


def scrub(text: str, headers: dict[str, str]) -> str:
    """Remove any header value (API keys, bearer tokens) from text before it goes into an error message."""
    for value in headers.values():
        if len(value) >= 8:
            text = text.replace(value, "[redacted]")
    return text[:MAX_ERROR_CHARS]


def parse_retry_after(value: str | None) -> float | None:
    """Seconds from a ``Retry-After`` header (the numeric form only; a date form is ignored)."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if 0 <= seconds < 3600 else None


class UrllibTransport:
    """Standard-library HTTP so the project has no runtime dependencies."""

    def _open(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json", **headers},
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)  # noqa: S310 - https URLs only, set by providers
        except urllib.error.HTTPError as err:
            detail = scrub(err.read().decode("utf-8", "replace"), headers)
            raise ProviderError(
                f"HTTP {err.code} from {url}: {detail}",
                status=err.code,
                retryable=err.code in RETRYABLE_STATUS,
                retry_after=parse_retry_after(err.headers.get("retry-after") if err.headers else None),
            ) from None
        except (urllib.error.URLError, TimeoutError) as err:
            reason = getattr(err, "reason", err)
            raise ProviderError(f"could not reach {url}: {scrub(str(reason), headers)}", retryable=True) from None

    def post_json(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        with self._open(url, headers, body) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ProviderError("provider returned JSON that is not an object")
        return parsed

    def post_stream(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> Iterator[str]:
        with self._open(url, headers, body) as response:
            for raw in response:
                yield raw.decode("utf-8")
