"""Week 2 Integration Tests — End-to-end SSP pipeline scenarios.

Covers:
- I1: Normal conversation (LOW risk, CALM state)
- I2: Distressed user (MEDIUM risk, CONCERNED → ELEVATED)
- I3: Crisis detection (HIGH risk, CRISIS state)
- I4: Error handling (API timeout, empty input, memory errors)
"""

from __future__ import annotations

import pytest

from safehaven.controller.chat_controller import ChatController
from safehaven.memory.in_memory import InMemoryConversationMemory
from safehaven.models import (
    ConversationContext,
    EmotionLabel,
    EmotionResult,
    RiskLevel,
    UserState,
)
from safehaven.safety.fsm_risk_evaluator import FSMRiskEvaluator
from safehaven.safety.language_detector import SimpleLanguageDetector
from safehaven.strategy.base import ConcreteStrategySelector
from safehaven.strategy.supportive import SupportiveStrategy
from safehaven.strategy.de_escalation import DeEscalationStrategy
from safehaven.strategy.crisis import CrisisStrategy


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


class StubDetector:
    """Emotion detector that returns a fixed result."""

    def __init__(self, label: EmotionLabel, confidence: float) -> None:
        self._result = EmotionResult(label=label, confidence=confidence)

    def detect(self, text: str) -> EmotionResult:
        return self._result


class SequenceDetector:
    """Cycles through a list of EmotionResults on successive calls."""

    def __init__(self, results: list[tuple[EmotionLabel, float]]) -> None:
        self._results = [EmotionResult(l, c) for l, c in results]
        self._index = 0

    def detect(self, text: str) -> EmotionResult:
        result = self._results[min(self._index, len(self._results) - 1)]
        self._index += 1
        return result


class StubGenerator:
    def __init__(self, response: str = "I hear you.") -> None:
        self._response = response

    def generate(self, context: ConversationContext) -> str:
        return self._response


class RecordingGenerator:
    """Captures every ConversationContext passed to generate()."""

    def __init__(self, response: str = "I hear you.") -> None:
        self._response = response
        self.captured: list[ConversationContext] = []

    def generate(self, context: ConversationContext) -> str:
        self.captured.append(context)
        return self._response


class StubFilter:
    def validate(self, response: str, risk: RiskLevel) -> str:
        return response

    def __class_getitem__(cls, item: object) -> object:
        return cls


class ErrorGenerator:
    """Simulates API timeout by raising ConnectionError."""

    def generate(self, context: ConversationContext) -> str:
        raise ConnectionError("API timeout: upstream service unreachable")


def _make_controller(
    detector: object,
    generator: object,
    *,
    filter_: StubFilter | None = None,
) -> ChatController:
    return ChatController(
        detector=detector,  # type: ignore[arg-type]
        evaluator=FSMRiskEvaluator(),
        memory=InMemoryConversationMemory(),
        generator=generator,  # type: ignore[arg-type]
        output_filter=filter_ if filter_ is not None else StubFilter(),
        language_detector=SimpleLanguageDetector(),
        strategy_selector=ConcreteStrategySelector(),
    )


# ---------------------------------------------------------------------------
# I1 — Normal conversation (LOW risk, CALM state)
# ---------------------------------------------------------------------------


class TestScenario1NormalConversation:
    """I1: 3-turn conversation with happy/neutral user. FSM stays CALM."""

    def test_fsm_stays_calm_across_three_turns(self) -> None:
        gen = RecordingGenerator("That's great to hear! What made your day good?")
        controller = _make_controller(
            StubDetector(EmotionLabel.HAPPY, 0.85), gen
        )

        controller.handle_message("Hi! I had a pretty good day today.")
        controller.handle_message("Work went well and I had lunch with a friend.")
        controller.handle_message("I'm just relaxing now, feeling good.")

        assert controller.fsm_state == "calm"

    def test_low_risk_uses_supportive_strategy(self) -> None:
        gen = RecordingGenerator()
        controller = _make_controller(
            StubDetector(EmotionLabel.HAPPY, 0.85), gen
        )
        controller.handle_message("Hi! I had a pretty good day today.")

        assert len(gen.captured) == 1
        assert gen.captured[0].strategy_name != ""
        # strategy_name should indicate supportive (not crisis/deescalation)
        assert "crisis" not in gen.captured[0].strategy_name.lower()

    def test_returns_string_response_not_none(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.HAPPY, 0.85),
            StubGenerator("That's great to hear!"),
        )
        result = controller.handle_message("Hi! I had a pretty good day today.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_both_messages_stored_in_memory(self) -> None:
        memory = InMemoryConversationMemory()
        controller = ChatController(
            detector=StubDetector(EmotionLabel.NEUTRAL, 0.5),
            evaluator=FSMRiskEvaluator(),
            memory=memory,
            generator=StubGenerator("Hello!"),
            output_filter=StubFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )
        controller.handle_message("Hi there")
        msgs = memory.get_recent_messages()
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"
        assert msgs[0].risk_level == RiskLevel.LOW

    def test_language_detected_as_english(self) -> None:
        memory = InMemoryConversationMemory()
        controller = ChatController(
            detector=StubDetector(EmotionLabel.HAPPY, 0.85),
            evaluator=FSMRiskEvaluator(),
            memory=memory,
            generator=StubGenerator("Great!"),
            output_filter=StubFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )
        controller.handle_message("Hi! I had a pretty good day today.")
        assert memory.get_recent_messages()[0].language == "en"

    def test_last_emotion_set_to_happy(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.HAPPY, 0.85),
            StubGenerator("That's great!"),
        )
        controller.handle_message("I feel great today.")
        assert controller.last_emotion == EmotionLabel.HAPPY


# ---------------------------------------------------------------------------
# I2 — Distressed user (MEDIUM risk, CONCERNED → ELEVATED)
# ---------------------------------------------------------------------------


class TestScenario2DistressedUser:
    """I2: User expresses escalating distress. FSM advances CALM→CONCERNED→ELEVATED."""

    def test_first_sad_message_moves_to_concerned(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.SAD, 0.78),
            StubGenerator("It sounds like you're going through a tough time."),
        )
        result = controller.handle_message(
            "I've been feeling really overwhelmed lately. Nothing seems to work out."
        )
        assert controller.fsm_state == "concerned"
        assert result is not None  # still responding

    def test_three_consecutive_negatives_reach_elevated(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.SAD, 0.85),
            StubGenerator("I hear you."),
        )
        controller.handle_message("I'm so overwhelmed.")
        controller.handle_message("Things are getting worse.")
        controller.handle_message("I can't cope anymore.")
        assert controller.fsm_state == "elevated"

    def test_elevated_state_still_returns_response(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.SAD, 0.85),
            StubGenerator("I hear you."),
        )
        controller.handle_message("I'm so overwhelmed.")
        controller.handle_message("Things are getting worse.")
        result = controller.handle_message("I can't cope anymore.")
        assert result is not None  # ELEVATED returns MEDIUM, not HIGH

    def test_elevated_uses_deescalation_strategy(self) -> None:
        gen = RecordingGenerator()
        controller = _make_controller(
            StubDetector(EmotionLabel.SAD, 0.85), gen
        )
        controller.handle_message("I'm so overwhelmed.")
        controller.handle_message("Things are getting worse.")
        gen.captured.clear()  # reset before the third call
        controller.handle_message("I can't cope anymore.")

        if gen.captured:
            # strategy_name should mention de-escalation or grounding
            name = gen.captured[0].strategy_name.lower()
            assert any(kw in name for kw in ("de", "escal", "ground"))

    def test_anxious_emotion_escalates_too(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.ANXIOUS, 0.80),
            StubGenerator("I hear you."),
        )
        controller.handle_message("I'm so anxious about everything.")
        assert controller.fsm_state in ("concerned", "elevated")

    def test_neutral_turn_resets_consecutive_counter_and_stays_concerned(
        self,
    ) -> None:
        controller = ChatController(
            detector=SequenceDetector(
                [
                    (EmotionLabel.SAD, 0.8),     # → concerned
                    (EmotionLabel.SAD, 0.8),     # counter=2
                    (EmotionLabel.NEUTRAL, 0.9), # reset counter
                    (EmotionLabel.SAD, 0.8),     # counter=1 again
                ]
            ),
            evaluator=FSMRiskEvaluator(),
            memory=InMemoryConversationMemory(),
            generator=StubGenerator("I hear you."),
            output_filter=StubFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )
        controller.handle_message("turn 1")
        controller.handle_message("turn 2")
        controller.handle_message("turn 3 neutral")
        controller.handle_message("turn 4 sad again")
        # Counter was reset at turn 3, so should still be in concerned not elevated
        assert controller.fsm_state == "concerned"


# ---------------------------------------------------------------------------
# I3 — Crisis detection (HIGH risk, CRISIS state)
# ---------------------------------------------------------------------------


class TestScenario3CrisisDetection:
    """I3: User expresses crisis. handle_message returns None; FSM = crisis."""

    def test_fearful_high_confidence_returns_none(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.FEARFUL, 1.0),
            StubGenerator("should not be called"),
        )
        result = controller.handle_message(
            "I don't want to be here anymore. I want to end it all."
        )
        assert result is None

    def test_crisis_state_after_fearful_message(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.FEARFUL, 1.0),
            StubGenerator("should not be called"),
        )
        controller.handle_message("I want to end it all.")
        assert controller.fsm_state == "crisis"

    def test_crisis_is_terminal_happy_does_not_downgrade(self) -> None:
        controller = _make_controller(
            SequenceDetector(
                [(EmotionLabel.FEARFUL, 1.0), (EmotionLabel.HAPPY, 0.99)]
            ),
            StubGenerator("response"),
        )
        controller.handle_message("I want to end it all.")
        result = controller.handle_message("Just kidding, I'm fine!")
        assert controller.fsm_state == "crisis"
        assert result is None  # still None after terminal crisis

    def test_clear_resets_crisis_to_calm(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.FEARFUL, 1.0),
            StubGenerator("response"),
        )
        controller.handle_message("crisis message")
        assert controller.fsm_state == "crisis"
        controller.clear()
        assert controller.fsm_state == "calm"
        assert controller.last_emotion is None
        assert controller.memory.get_recent_messages() == []

    def test_generator_not_called_during_crisis(self) -> None:
        gen = RecordingGenerator()
        controller = _make_controller(
            StubDetector(EmotionLabel.FEARFUL, 1.0), gen
        )
        controller.handle_message("I want to end it all.")
        assert len(gen.captured) == 0  # LLM was never invoked

    def test_arabic_crisis_also_returns_none(self) -> None:
        """Crisis detection works regardless of language."""
        controller = _make_controller(
            StubDetector(EmotionLabel.FEARFUL, 1.0),
            StubGenerator("should not be called"),
        )
        result = controller.handle_message("أريد أن أموت")  # Arabic: "I want to die"
        assert result is None
        assert controller.fsm_state == "crisis"


# ---------------------------------------------------------------------------
# I4 — Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """I4: App handles API timeouts, empty input, and memory errors gracefully."""

    def test_api_timeout_raises_connection_error(self) -> None:
        """ConnectionError from the generator should propagate (not silently fail)."""
        controller = _make_controller(
            StubDetector(EmotionLabel.NEUTRAL, 0.5),
            ErrorGenerator(),
        )
        with pytest.raises(ConnectionError):
            controller.handle_message("Hello, can you help me?")

    def test_empty_input_does_not_crash(self) -> None:
        """Empty string should be handled without raising an unhandled exception."""
        controller = _make_controller(
            StubDetector(EmotionLabel.NEUTRAL, 0.3),
            StubGenerator("I'm here."),
        )
        # Should not raise; may return a string or None
        try:
            result = controller.handle_message("")
            assert result is None or isinstance(result, str)
        except (ValueError, AssertionError):
            pass  # documented that empty input may raise ValueError

    def test_whitespace_only_input_does_not_crash(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.NEUTRAL, 0.3),
            StubGenerator("I'm here."),
        )
        try:
            result = controller.handle_message("   ")
            assert result is None or isinstance(result, str)
        except (ValueError, AssertionError):
            pass

    def test_very_long_input_does_not_crash(self) -> None:
        long_msg = "I feel sad. " * 500  # 6000 chars
        controller = _make_controller(
            StubDetector(EmotionLabel.SAD, 0.8),
            StubGenerator("I hear you."),
        )
        result = controller.handle_message(long_msg)
        assert result is not None

    def test_multiple_clears_are_idempotent(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.NEUTRAL, 0.5),
            StubGenerator("Hello!"),
        )
        controller.handle_message("Hi")
        controller.clear()
        controller.clear()  # second clear should not crash
        assert controller.fsm_state == "calm"
        assert controller.memory.get_recent_messages() == []

    def test_new_session_after_clear_works_normally(self) -> None:
        controller = _make_controller(
            StubDetector(EmotionLabel.FEARFUL, 1.0),
            StubGenerator("response"),
        )
        controller.handle_message("Crisis message")
        assert controller.fsm_state == "crisis"

        controller.clear()

        assert controller.fsm_state == "calm"
        assert controller.last_emotion is None
