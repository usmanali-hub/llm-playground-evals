"""Cost from token counts. Prices change and differ by model, so none are built in: you pass the ones you pay."""

from __future__ import annotations

from dataclasses import dataclass

from llm_playground.types import Usage


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float
    output_per_mtok: float

    def __post_init__(self) -> None:
        if self.input_per_mtok < 0 or self.output_per_mtok < 0:
            raise ValueError("prices cannot be negative")

    def cost(self, usage: Usage) -> float:
        """Cost in the price's currency for these tokens (price is per million tokens)."""
        return (usage.input_tokens * self.input_per_mtok + usage.output_tokens * self.output_per_mtok) / 1_000_000
