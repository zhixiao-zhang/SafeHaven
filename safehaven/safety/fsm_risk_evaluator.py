"""FSM-based risk evaluator — stateful risk assessment using finite state machine.

Implements the ``RiskEvaluator`` protocol from ``interfaces.py``.
Transitions: CALM → CONCERNED → ELEVATED → CRISIS
The FSM is stateful per session and resets on ``clear()``.
Forward-only "ratchet" constraint: risk level never decreases within a session.
"""

from __future__ import annotations

import logging

from safehaven.models import EmotionLabel, RiskLevel, UserState

_log = logging.getLogger(__name__)

# Monotonic rank — used to enforce forward-only transitions
_RANK: dict[str, int] = {"calm": 0, "concerned": 1, "elevated": 2, "crisis": 3}

_STATE_TO_RISK: dict[str, RiskLevel] = {
    "calm": RiskLevel.LOW,
    "concerned": RiskLevel.MEDIUM,
    "elevated": RiskLevel.MEDIUM,
    "crisis": RiskLevel.HIGH,
}

_NEGATIVE_EMOTIONS = {EmotionLabel.SAD, EmotionLabel.ANXIOUS, EmotionLabel.ANGRY, EmotionLabel.FEARFUL}


class FSMRiskEvaluator:
    """Evaluate risk using a finite state machine that tracks escalation.

    States: CALM → CONCERNED → ELEVATED → CRISIS
    Transitions are forward-only (monotonic) within a session.
    Reset on ``clear()`` for a new session.
    """

    def __init__(self) -> None:
        self._state: str = "calm"
        self._consecutive_negative: int = 0

    @property
    def state(self) -> str:
        """Current FSM state label."""
        return self._state

    def _transition_to(self, new_state: str) -> None:
        """Advance FSM state. Enforces monotonic (forward-only) constraint."""
        assert _RANK[new_state] >= _RANK[self._state], (
            f"Illegal FSM downgrade: {self._state!r} → {new_state!r}"
        )
        if new_state != self._state:
            _log.info("[FSM] %s → %s", self._state, new_state)
            self._state = new_state

    def evaluate(self, state: UserState) -> RiskLevel:
        """Determine risk level and advance FSM state if warranted.

        Guards are evaluated highest-severity first (fail-safe principle).
        Skip-states are allowed: CALM can jump directly to CRISIS.

        Guard order:
        1. FEARFUL ≥ 0.9 → CRISIS (immediate, skip-states allowed)
        2. Terminal CRISIS → always return HIGH, no further transitions
        3. 3+ consecutive negative turns → ELEVATED
        4. SAD/ANXIOUS/ANGRY > 0.6 → increment counter; CALM → CONCERNED
        5. Neutral/positive → reset consecutive counter
        """
        emotion = state.current_emotion

        # Guard 1: high-confidence fear → crisis immediately (skip-states allowed)
        if emotion.label == EmotionLabel.FEARFUL and emotion.confidence >= 0.9:
            self._transition_to("crisis")
            return RiskLevel.HIGH

        # Guard 2: terminal CRISIS state — never leave, always HIGH
        if self._state == "crisis":
            return RiskLevel.HIGH

        # Guard 3: update consecutive counter first, then threshold-check
        if emotion.label in _NEGATIVE_EMOTIONS and emotion.confidence > 0.6:
            self._consecutive_negative += 1
        else:
            # Guard 4: neutral/positive → reset consecutive counter
            self._consecutive_negative = 0

        # Guard 5: sustained negative pattern → elevated (checked after incrementing)
        if self._consecutive_negative >= 3 and self._state not in ("elevated", "crisis"):
            self._transition_to("elevated")
        # Guard 6: first negative → concerned
        elif self._consecutive_negative >= 1 and self._state == "calm":
            self._transition_to("concerned")

        return _STATE_TO_RISK[self._state]

    def clear(self) -> None:
        """Reset FSM to initial CALM state (new session)."""
        self._state = "calm"
        self._consecutive_negative = 0
