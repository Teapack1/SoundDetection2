"""Confidence-stream smoothing and decision logic.

Per-class smoothing chain:
    median(window) → EMA(α) → threshold + hysteresis + N-of-N consensus

Why all of this:
  - median kills isolated single-frame spikes
  - EMA flattens normal variance
  - hysteresis keeps the trigger from re-firing during the ring tail
  - N-of-N consensus requires multiple consecutive over-threshold frames so
    transients caused by music/cutlery don't get through

Then a global trigger gate: argmax over POSITIVE classes, must beat the
runner-up by `margin`, and the `_negative` class must be below `negative_max`.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass


@dataclass
class TriggerEvent:
    class_name: str
    confidence: float
    timestamp: float


class ClassSmoother:
    """Per-class smoother. Feed it raw probabilities, it tells you whether the
    smoothed probability is above threshold AND has been so for N frames.

    The smoother itself is *passive* — it doesn't decide what fires. It exposes
    `update()` which advances the smoothing state and reports whether
    fire-conditions are currently met. The arbiter (which knows about
    inter-class margin and the negative gate) decides whether to actually
    fire, then calls `consume_fire()` on the winning smoother — that's what
    arms the hysteresis.

    This keeps an arbiter rejection from burning a class's smoother state
    (which would require waiting for hysteresis even though no trigger
    happened externally).
    """

    def __init__(
        self,
        median_window: int = 3,
        ema_alpha: float = 0.4,
        threshold: float = 0.6,
        hysteresis_low: float = 0.3,
        consecutive_frames: int = 2,
    ):
        self.median_window = max(1, int(median_window))
        self.ema_alpha = float(ema_alpha)
        self.threshold = float(threshold)
        self.hyst_low = float(hysteresis_low)
        self.consec = max(1, int(consecutive_frames))
        self._buf: collections.deque = collections.deque(maxlen=self.median_window)
        self._ema: float = 0.0
        self._above_count: int = 0
        self._armed: bool = True              # ready to fire?

    def update(self, raw: float) -> tuple[bool, float]:
        """Push raw probability; returns (ready_to_fire, smoothed_value).

        `ready_to_fire` is True when the smoothed value has been above
        threshold for `consecutive_frames` AND the smoother is armed
        (hasn't fired since the last hysteresis dip).
        """
        self._buf.append(float(raw))
        sorted_buf = sorted(self._buf)
        med = sorted_buf[len(sorted_buf) // 2]
        self._ema = self.ema_alpha * med + (1 - self.ema_alpha) * self._ema
        s = self._ema

        if s >= self.threshold:
            self._above_count += 1
        else:
            self._above_count = 0

        if s < self.hyst_low:
            self._armed = True

        ready = self._armed and self._above_count >= self.consec
        return ready, s

    def consume_fire(self) -> None:
        """Called by the arbiter ONLY when this class's fire is actually
        emitted as a TriggerEvent. Disarms the smoother so it won't refire
        until hysteresis releases it.
        """
        self._armed = False
        self._above_count = 0


class TriggerArbiter:
    """Decide at each step which (if any) class fires.

    Per the project requirement (minimise FPs even at the cost of recall):
      - Only POSITIVE classes can fire (the `_negative` class is a gatekeeper).
      - winner.score must be > threshold AND > runner_up.score + margin
      - _negative score must be < negative_max
      - Refractory period after any fire on any class.
    """

    def __init__(
        self,
        classes: list[str],
        thresholds: dict[str, float],
        smoothers_cfg: dict,
        negative_index: int,
        negative_max: float,
        margin_to_second_best: float,
        global_refractory_ms: int,
    ):
        self.classes = classes
        self.neg_idx = negative_index
        self.negative_max = negative_max
        self.margin = margin_to_second_best
        self.global_refractory_s = global_refractory_ms / 1000.0
        self._last_fire_ts = 0.0
        self.smoothers: dict[str, ClassSmoother] = {}
        for cls in classes:
            self.smoothers[cls] = ClassSmoother(
                median_window=smoothers_cfg["median_window"],
                ema_alpha=smoothers_cfg["ema_alpha"],
                threshold=thresholds.get(cls, smoothers_cfg["default_threshold"]),
                hysteresis_low=smoothers_cfg["hysteresis_low"],
                consecutive_frames=smoothers_cfg["consecutive_frames"],
            )

    def step(self, probs: dict[str, float], now: float | None = None) -> TriggerEvent | None:
        now = now if now is not None else time.monotonic()
        smoothed: dict[str, float] = {}
        per_class_ready: dict[str, bool] = {}
        for cls, raw in probs.items():
            ready, s = self.smoothers[cls].update(raw)
            smoothed[cls] = s
            per_class_ready[cls] = ready

        # Global refractory across classes
        if now - self._last_fire_ts < self.global_refractory_s:
            return None

        # Negative gate: smoothed _negative score must not dominate
        neg_name = self.classes[self.neg_idx]
        if smoothed.get(neg_name, 0.0) > self.negative_max:
            return None

        positive_smoothed = {
            c: smoothed[c] for c in self.classes if c != neg_name
        }
        if not positive_smoothed:
            return None
        # Pick winner among positives
        winner = max(positive_smoothed, key=positive_smoothed.get)
        winner_score = positive_smoothed[winner]

        # Margin requirement vs runner-up
        runner_up = sorted(positive_smoothed.values(), reverse=True)
        if len(runner_up) > 1 and winner_score - runner_up[1] < self.margin:
            return None

        # Per-class smoother must report ready
        if not per_class_ready.get(winner, False):
            return None

        # All gates passed — fire and consume the winner's readiness
        self.smoothers[winner].consume_fire()
        self._last_fire_ts = now
        return TriggerEvent(class_name=winner, confidence=winner_score, timestamp=now)
