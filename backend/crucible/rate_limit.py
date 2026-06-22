from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Callable


class RateLimiter:
    """Sliding-window per-key limiter. max<=0 disables it (allow all)."""

    def __init__(self, max_per_window: int, window_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.max = max_per_window
        self.window = window_s
        self.clock = clock
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        if self.max <= 0:
            return True
        now = self.clock()
        dq = self._hits[key]
        while dq and dq[0] <= now - self.window:
            dq.popleft()
        if len(dq) >= self.max:
            return False
        dq.append(now)
        return True
