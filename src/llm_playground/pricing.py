"""Cost from token counts. Prices change and differ by model, so none are built in: you pass the ones you pay."""

from __future__ import annotations

from dataclasses import dataclass

from llm_playground.types import Usage


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float | None = None  # None: cache writes cost the plain input price (a conservative default)
    cache_read_per_mtok: float | None = None  # None: cache reads also cost the plain input price

    def __post_init__(self) -> None:
        prices = (self.input_per_mtok, self.output_per_mtok, self.cache_write_per_mtok, self.cache_read_per_mtok)
        if any(p is not None and p < 0 for p in prices):
            raise ValueError("prices cannot be negative")

    def cost(self, usage: Usage) -> float:
        """Cost in the price's currency for these tokens (prices are per million tokens).

        ``usage.cache_write_tokens`` and ``usage.cache_read_tokens`` are already included in ``input_tokens``
        (that is what the provider bills them as if no cache price is given); passing ``cache_write_per_mtok`` or
        ``cache_read_per_mtok`` re-prices just that portion instead, since a cache write usually costs more than a
        plain input token and a cache read much less.
        """
        plain_input = usage.input_tokens - usage.cache_write_tokens - usage.cache_read_tokens
        cache_write_price = self.cache_write_per_mtok if self.cache_write_per_mtok is not None else self.input_per_mtok
        cache_read_price = self.cache_read_per_mtok if self.cache_read_per_mtok is not None else self.input_per_mtok
        total = (
            plain_input * self.input_per_mtok
            + usage.cache_write_tokens * cache_write_price
            + usage.cache_read_tokens * cache_read_price
            + usage.output_tokens * self.output_per_mtok
        )
        return total / 1_000_000
