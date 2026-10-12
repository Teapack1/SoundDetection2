"""Confidence-stream smoothing and decision logic.

Per-class smoothing chain:
    median(window) → EMA(α) → threshold + hysteresis + N-of-N consensus

Why all of this:
  - median kills isolated single-frame spikes
  - EMA flattens normal variance
  - hysteresis keeps the trigger from re-firing during the ring tail
  - N-of-N consensus requires multiple consecutive over-threshold frames so
    transients caused by music/cutlery don't get through

Then a global trigger gate: argmax over every class that isn't role="negative"
(see `Config.class_roles` — `_negative` by default, but any class can be given
that role), the winner must beat the runner-up by `margin`, and every
negative/suppressor class in `vetoes` must be below its own ceiling.

The argmax deliberately spans classes that cannot fire (role="suppressor"). A
class trained to absorb false positives (cutlery, metal-on-metal, glass
handling) only reduces them if it is allowed to *win* the window — excluding
it would just hand the decision to the strongest glass class while the model
was saying "cutlery". A non-firing winner therefore abstains. This is also
what makes the live decision identical to the one checkpoint selection and
evaluation measure, which have always compared every non-negative class.
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
    peak_dbfs: float | None = None
    rms_dbfs: float | None = None
    min_dbfs: float | None = None
    best_channel: int | None = None
    mined_path: str | None = None


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
        consecutive_frames: int = 3,
        smoothing_enabled: bool = True,
    ):
        self.smoothing_enabled = bool(smoothing_enabled)
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

        When smoothing is disabled the raw value passes straight through and
        fires the instant it crosses threshold (no median/EMA/hysteresis/
        consecutive state).
        """
        if not self.smoothing_enabled:
            s = float(raw)
            return s >= self.threshold, s

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
      - Every class outside `negative_classes` competes for the window; only a
        *trigger* class can actually fire, so a suppressor winning means
        nothing fires.
      - winner.score must be > threshold AND > runner_up.score + margin
      - Every class in `vetoes` (every negative-role class plus any suppressor
        with its own `veto_max`) must be below its own ceiling.
      - Refractory period after any fire on any class.
    """

    def __init__(
        self,
        classes: list[str],
        thresholds: dict[str, float],
        smoothers_cfg: dict,
        negative_classes: set[str] | None = None,
        margin_to_second_best: float = 0.0,
        global_refractory_ms: int = 0,
        trigger_classes: set[str] | None = None,
        vetoes: dict[str, float] | None = None,
    ):
        self.classes = classes
        # Never compete for the window — every role="negative" class (not just
        # `_negative`), see the module docstring and `Config.class_roles`.
        self.negative_classes = set(negative_classes or ())
        self.margin = margin_to_second_best
        self.global_refractory_s = global_refractory_ms / 1000.0
        self.trigger_classes = trigger_classes
        # Every negative/suppressor ceiling, keyed by class name — negative and
        # suppressor classes are vetoed identically here; what tells them apart
        # is `negative_classes` excluding the former from window competition.
        self.vetoes = dict(vetoes or {})
        self._last_fire_ts = 0.0
        self.last_smoothed: dict[str, float] = {}
        smoothing_enabled = bool(smoothers_cfg.get("smoothing_enabled", True))
        self.smoothers: dict[str, ClassSmoother] = {}
        for cls in classes:
            self.smoothers[cls] = ClassSmoother(
                median_window=smoothers_cfg["median_window"],
                ema_alpha=smoothers_cfg["ema_alpha"],
                threshold=thresholds.get(cls, smoothers_cfg["default_threshold"]),
                hysteresis_low=smoothers_cfg["hysteresis_low"],
                consecutive_frames=smoothers_cfg["consecutive_frames"],
                smoothing_enabled=smoothing_enabled,
            )

    def step(self, probs: dict[str, float], now: float | None = None) -> TriggerEvent | None:
        now = now if now is not None else time.monotonic()
        smoothed: dict[str, float] = {}
        per_class_ready: dict[str, bool] = {}
        for cls, raw in probs.items():
            ready, s = self.smoothers[cls].update(raw)
            smoothed[cls] = s
            per_class_ready[cls] = ready
        self.last_smoothed = dict(smoothed)

        # Global refractory across classes
        if now - self._last_fire_ts < self.global_refractory_s:
            return None

        # Negative/suppressor gate: any class in `vetoes` (every role="negative"
        # class — `_negative` by default, plus any suppressor that set its own
        # `veto_max`) hard-vetoes the window when its smoothed score dominates.
        for cls, ceiling in self.vetoes.items():
            if smoothed.get(cls, 0.0) > ceiling:
                return None

        # Every class competes except role="negative" ones — including
        # suppressors, which cannot fire but can still win and block, see the
        # module docstring. Excluding a suppressor would just hand the window
        # to the strongest firing class while the model was saying "cutlery".
        positive_smoothed = {
            c: smoothed[c] for c in self.classes if c not in self.negative_classes
        }
        if not positive_smoothed:
            return None
        winner = max(positive_smoothed, key=positive_smoothed.get)
        winner_score = positive_smoothed[winner]

        # Margin requirement vs runner-up
        runner_up = sorted(positive_smoothed.values(), reverse=True)
        if len(runner_up) > 1 and winner_score - runner_up[1] < self.margin:
            return None

        # A suppressor won the window: abstain rather than firing the runner-up.
        if self.trigger_classes is not None and winner not in self.trigger_classes:
            return None

        # Per-class smoother must report ready
        if not per_class_ready.get(winner, False):
            return None

        # All gates passed — fire and consume the winner's readiness
        self.smoothers[winner].consume_fire()
        self._last_fire_ts = now
        return TriggerEvent(class_name=winner, confidence=winner_score, timestamp=now)
