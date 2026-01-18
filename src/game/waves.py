from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WaveState:
    index: int = 1

    def advance(self) -> None:
        self.index += 1
