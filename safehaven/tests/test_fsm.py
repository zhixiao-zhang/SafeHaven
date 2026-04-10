"""Tests for the FSM-based risk evaluator."""

from __future__ import annotations

from safehaven.models import EmotionLabel, EmotionResult, RiskLevel, UserState
from safehaven.safety.fsm_risk_evaluator import FSMRiskEvaluator


def _make_state(
    label: EmotionLabel,
    confidence: float,
    history: list[RiskLevel] | None = None,
) -> UserState:
    return UserState(
        current_emotion=EmotionResult(label, confidence),
        risk_level=RiskLevel.LOW,
        message_count=1,
        escalation_history=history or [],
    )


class TestFSMRiskEvaluator:
    """Test FSM state transitions and risk level mapping."""

    def test_initial_state_is_calm(self) -> None:
        evaluator = FSMRiskEvaluator()
        assert evaluator.state == "calm"

    def test_clear_resets_to_calm(self) -> None:
        evaluator = FSMRiskEvaluator()
        evaluator._state = "elevated"
        evaluator.clear()
        assert evaluator.state == "calm"

    def test_clear_resets_consecutive_counter(self) -> None:
        evaluator = FSMRiskEvaluator()
        evaluator._consecutive_negative = 3
        evaluator.clear()
        assert evaluator._consecutive_negative == 0

    # --- Basic transitions ---

    def test_neutral_stays_calm(self) -> None:
        evaluator = FSMRiskEvaluator()
        result = evaluator.evaluate(_make_state(EmotionLabel.NEUTRAL, 0.9))
        assert evaluator.state == "calm"
        assert result == RiskLevel.LOW

    def test_happy_stays_calm(self) -> None:
        evaluator = FSMRiskEvaluator()
        result = evaluator.evaluate(_make_state(EmotionLabel.HAPPY, 0.9))
        assert evaluator.state == "calm"
        assert result == RiskLevel.LOW

    def test_sad_high_confidence_moves_to_concerned(self) -> None:
        evaluator = FSMRiskEvaluator()
        result = evaluator.evaluate(_make_state(EmotionLabel.SAD, 0.8))
        assert evaluator.state == "concerned"
        assert result == RiskLevel.MEDIUM

    def test_sad_low_confidence_stays_calm(self) -> None:
        evaluator = FSMRiskEvaluator()
        result = evaluator.evaluate(_make_state(EmotionLabel.SAD, 0.5))
        assert evaluator.state == "calm"
        assert result == RiskLevel.LOW

    # --- Sustained escalation to ELEVATED ---

    def test_three_consecutive_negatives_reach_elevated(self) -> None:
        evaluator = FSMRiskEvaluator()
        evaluator.evaluate(_make_state(EmotionLabel.SAD, 0.8))     # → concerned, counter=1
        evaluator.evaluate(_make_state(EmotionLabel.ANXIOUS, 0.8)) # counter=2
        result = evaluator.evaluate(_make_state(EmotionLabel.ANGRY, 0.8))  # counter=3 → elevated
        assert evaluator.state == "elevated"
        assert result == RiskLevel.MEDIUM

    def test_neutral_resets_consecutive_counter(self) -> None:
        evaluator = FSMRiskEvaluator()
        evaluator.evaluate(_make_state(EmotionLabel.SAD, 0.8))    # counter=1
        evaluator.evaluate(_make_state(EmotionLabel.SAD, 0.8))    # counter=2
        evaluator.evaluate(_make_state(EmotionLabel.NEUTRAL, 0.9)) # reset counter=0
        evaluator.evaluate(_make_state(EmotionLabel.SAD, 0.8))    # counter=1 again
        assert evaluator.state == "concerned"  # still concerned, not elevated
        assert evaluator._consecutive_negative == 1

    # --- CRISIS transitions ---

    def test_fearful_high_confidence_jumps_to_crisis(self) -> None:
        evaluator = FSMRiskEvaluator()
        result = evaluator.evaluate(_make_state(EmotionLabel.FEARFUL, 0.95))
        assert evaluator.state == "crisis"
        assert result == RiskLevel.HIGH

    def test_calm_to_crisis_skip_states(self) -> None:
        """FEARFUL ≥ 0.9 can skip CONCERNED and ELEVATED entirely."""
        evaluator = FSMRiskEvaluator()
        evaluator.evaluate(_make_state(EmotionLabel.FEARFUL, 1.0))
        assert evaluator.state == "crisis"

    def test_fearful_low_confidence_does_not_trigger_crisis(self) -> None:
        evaluator = FSMRiskEvaluator()
        result = evaluator.evaluate(_make_state(EmotionLabel.FEARFUL, 0.7))
        assert evaluator.state != "crisis"
        assert result != RiskLevel.HIGH

    def test_fearful_low_confidence_increments_negative_counter(self) -> None:
        """FEARFUL below 0.9 must still count as negative and increment the
        consecutive-negative counter (not reset it).  Regression test for the
        bug where FEARFUL was missing from _NEGATIVE_EMOTIONS."""
        evaluator = FSMRiskEvaluator()
        evaluator.evaluate(_make_state(EmotionLabel.SAD, 0.8))      # counter=1
        evaluator.evaluate(_make_state(EmotionLabel.FEARFUL, 0.7))  # counter=2 (not 0!)
        result = evaluator.evaluate(_make_state(EmotionLabel.ANXIOUS, 0.8))  # counter=3 → elevated
        assert evaluator._consecutive_negative == 3
        assert evaluator.state == "elevated"
        assert result == RiskLevel.MEDIUM

    # --- Terminal CRISIS state ---

    def test_crisis_state_does_not_downgrade_on_happy(self) -> None:
        """Once in CRISIS, the FSM never leaves — even on positive emotion."""
        evaluator = FSMRiskEvaluator()
        evaluator.evaluate(_make_state(EmotionLabel.FEARFUL, 1.0))  # → crisis
        result = evaluator.evaluate(_make_state(EmotionLabel.HAPPY, 0.99))
        assert evaluator.state == "crisis"
        assert result == RiskLevel.HIGH

    def test_crisis_state_persists_across_multiple_turns(self) -> None:
        evaluator = FSMRiskEvaluator()
        evaluator.evaluate(_make_state(EmotionLabel.FEARFUL, 1.0))
        for _ in range(3):
            result = evaluator.evaluate(_make_state(EmotionLabel.NEUTRAL, 0.9))
            assert evaluator.state == "crisis"
            assert result == RiskLevel.HIGH

    # --- Monotonic constraint ---

    def test_forward_only_transition_asserts_on_downgrade(self) -> None:
        import pytest
        evaluator = FSMRiskEvaluator()
        evaluator._state = "elevated"
        with pytest.raises(AssertionError):
            evaluator._transition_to("calm")

    # --- False positives (document known limitations) ---

    def test_false_positive_note_sad_keyword_detection(self) -> None:
        """Keyword-based emotion may still flag indirect language.

        This test documents the known behavior: emotion detection is
        keyword-based and can misclassify. The FSM adds escalation
        tracking on top — a single false positive does not cause crisis.
        """
        evaluator = FSMRiskEvaluator()
        # "dying of laughter" should not be detected as FEARFUL ≥ 0.9
        # (emotion detector handles this; FSM only sees the EmotionResult)
        result = evaluator.evaluate(_make_state(EmotionLabel.HAPPY, 0.8))
        assert result != RiskLevel.HIGH

    def test_single_negative_turn_does_not_reach_crisis(self) -> None:
        """One SAD message should never jump straight to HIGH risk."""
        evaluator = FSMRiskEvaluator()
        result = evaluator.evaluate(_make_state(EmotionLabel.SAD, 0.9))
        assert result != RiskLevel.HIGH
        assert evaluator.state in ("calm", "concerned")
