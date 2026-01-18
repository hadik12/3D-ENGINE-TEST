from __future__ import annotations

import time


class Time:
    def __init__(self) -> None:
        self.last = time.time()
        self.delta = 0.0
        self.fps = 0.0

    def tick(self) -> None:
        now = time.time()
        self.delta = now - self.last
        self.last = now
        if self.delta > 0:
            self.fps = 1.0 / self.delta
