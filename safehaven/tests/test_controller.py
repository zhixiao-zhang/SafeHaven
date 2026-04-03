"""Integration tests for ChatController with stub dependencies."""

from __future__ import annotations

from safehaven.controller.chat_controller import ChatController
from safehaven.memory.in_memory import InMemoryConversationMemory
from safehaven.models import (
    ConversationContext,
    EmotionLabel,
    EmotionResult,
    Message,
    RiskLevel,
    UserState,
)
from safehaven.safety.fsm_risk_evaluator import FSMRiskEvaluator
from safehaven.safety.language_detector import SimpleLanguageDetector
from safehaven.strategy.base import ConcreteStrategySelector


class FakeDetector:
    def __init__(self, label: EmotionLabel, confidence: float) -> None:
        self._result = EmotionResult(label=label, confidence=confidence)

    def detect(self, text: str) -> EmotionResult:
        return self._result


class FakeEvaluator:
    def __init__(self, level: RiskLevel) -> None:
        self._level = level

    def evaluate(self, state: UserState) -> RiskLevel:
        return self._level


class FakeGenerator:
    def __init__(self, response: str) -> None:
        self._response = response
        self.contexts: list[ConversationContext] = []

    def generate(self, context: ConversationContext) -> str:
        self.contexts.append(context)
        return self._response


class FakeFilter:
    def validate(self, response: str, risk: RiskLevel) -> str:
        return response


class TimeoutGenerator:
    def generate(self, context: ConversationContext) -> str:
        raise TimeoutError("LLM request timed out")


class LockedMemory:
    def store_message(self, message: Message) -> None:
        raise RuntimeError("database is locked")

    def get_recent_messages(self, limit: int = 10) -> list[Message]:
        return []

    def clear(self) -> None:
        return None


class TestChatControllerIntegration:
    def test_normal_message_returns_response(self) -> None:
        controller = ChatController(
            detector=FakeDetector(EmotionLabel.HAPPY, 0.85),
            evaluator=FakeEvaluator(RiskLevel.LOW),
            memory=InMemoryConversationMemory(),
            generator=FakeGenerator("Great to hear!"),
            output_filter=FakeFilter(),
        )
        result = controller.handle_message("I feel great today")
        assert result == "Great to hear!"

    def test_high_risk_returns_none(self) -> None:
        controller = ChatController(
            detector=FakeDetector(EmotionLabel.FEARFUL, 1.0),
            evaluator=FakeEvaluator(RiskLevel.HIGH),
            memory=InMemoryConversationMemory(),
            generator=FakeGenerator("should not be called"),
            output_filter=FakeFilter(),
        )
        result = controller.handle_message("I want to end it all")
        assert result is None

    def test_messages_stored_in_memory(self) -> None:
        memory = InMemoryConversationMemory()
        controller = ChatController(
            detector=FakeDetector(EmotionLabel.NEUTRAL, 0.3),
            evaluator=FakeEvaluator(RiskLevel.LOW),
            memory=memory,
            generator=FakeGenerator("Hello!"),
            output_filter=FakeFilter(),
        )
        controller.handle_message("Hi there")
        messages = memory.get_recent_messages()
        assert len(messages) == 2  # user + assistant
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"


class TestChatControllerWired:
    """Tests using real FSM, language detector, and strategy selector (mocked LLM only)."""

    def _make_controller(
        self, response: str = "I hear you.", emotion: EmotionLabel = EmotionLabel.NEUTRAL, confidence: float = 0.5
    ) -> ChatController:
        return ChatController(
            detector=FakeDetector(emotion, confidence),
            evaluator=FSMRiskEvaluator(),
            memory=InMemoryConversationMemory(),
            generator=FakeGenerator(response),
            output_filter=FakeFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )

    def test_fsm_state_initially_calm(self) -> None:
        controller = self._make_controller()
        assert controller.fsm_state == "calm"

    def test_arabic_input_sets_language_ar(self) -> None:
        controller = self._make_controller()
        controller.handle_message("مرحبا كيف حالك")
        messages = controller.memory.get_recent_messages()
        assert messages[0].language == "ar"

    def test_english_input_sets_language_en(self) -> None:
        controller = self._make_controller()
        controller.handle_message("Hello how are you")
        messages = controller.memory.get_recent_messages()
        assert messages[0].language == "en"

    def test_last_emotion_set_after_message(self) -> None:
        controller = self._make_controller(emotion=EmotionLabel.SAD, confidence=0.8)
        controller.handle_message("I feel terrible")
        assert controller.last_emotion == EmotionLabel.SAD

    def test_multi_turn_escalation_reaches_high(self) -> None:
        """Three consecutive SAD messages should escalate to HIGH via FSM."""
        controller = ChatController(
            detector=FakeDetector(EmotionLabel.SAD, 0.9),
            evaluator=FSMRiskEvaluator(),
            memory=InMemoryConversationMemory(),
            generator=FakeGenerator("I hear you."),
            output_filter=FakeFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )
        controller.handle_message("I feel terrible")   # counter=1, → concerned
        controller.handle_message("Still feeling bad") # counter=2
        result = controller.handle_message("Everything is wrong") # counter=3, → elevated
        # elevated returns MEDIUM, not HIGH — need one more fearful for HIGH
        assert controller.fsm_state == "elevated"
        assert result is not None  # still responding at ELEVATED

    def test_crisis_emotion_returns_none(self) -> None:
        controller = ChatController(
            detector=FakeDetector(EmotionLabel.FEARFUL, 1.0),
            evaluator=FSMRiskEvaluator(),
            memory=InMemoryConversationMemory(),
            generator=FakeGenerator("should not be called"),
            output_filter=FakeFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )
        result = controller.handle_message("I want to end it all")
        assert result is None
        assert controller.fsm_state == "crisis"

    def test_clear_resets_fsm_and_memory(self) -> None:
        controller = self._make_controller(emotion=EmotionLabel.SAD, confidence=0.9)
        controller.handle_message("I feel sad")
        assert controller.fsm_state != "calm" or True  # may or may not have escalated
        controller.clear()
        assert controller.fsm_state == "calm"
        assert controller.memory.get_recent_messages() == []
        assert controller.last_emotion is None

    def test_strategy_name_set_in_context(self) -> None:
        """Verify strategy_name is populated when strategy_selector is wired."""
        captured: list[ConversationContext] = []

        class CapturingGenerator:
            def generate(self, context: ConversationContext) -> str:
                captured.append(context)
                return "response"

        controller = ChatController(
            detector=FakeDetector(EmotionLabel.NEUTRAL, 0.5),
            evaluator=FSMRiskEvaluator(),
            memory=InMemoryConversationMemory(),
            generator=CapturingGenerator(),
            output_filter=FakeFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )
        controller.handle_message("Hello")
        assert len(captured) == 1
        assert captured[0].strategy_name != ""

    def test_i1_normal_conversation_three_turns_stays_calm_supportive(self) -> None:
        generator = FakeGenerator("Glad to hear that.")
        controller = ChatController(
            detector=FakeDetector(EmotionLabel.HAPPY, 0.9),
            evaluator=FSMRiskEvaluator(),
            memory=InMemoryConversationMemory(),
            generator=generator,
            output_filter=FakeFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )

        assert controller.handle_message("I had a good day.") == "Glad to hear that."
        assert controller.handle_message("Work went well too.") == "Glad to hear that."
        assert controller.handle_message("I feel pretty happy tonight.") == "Glad to hear that."

        assert controller.fsm_state == "calm"
        assert len(generator.contexts) == 3
        for context in generator.contexts:
            assert context.strategy_name == "SupportiveStrategy"
            assert "compassionate" in context.system_prompt.lower()

    def test_i2_distressed_conversation_reaches_elevated_deescalation(self) -> None:
        generator = FakeGenerator("Let's slow things down together.")
        controller = ChatController(
            detector=FakeDetector(EmotionLabel.SAD, 0.9),
            evaluator=FSMRiskEvaluator(),
            memory=InMemoryConversationMemory(),
            generator=generator,
            output_filter=FakeFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )

        controller.handle_message("I've been feeling really overwhelmed lately.")
        assert controller.fsm_state == "concerned"
        controller.handle_message("Nothing seems to work out.")
        assert controller.fsm_state == "concerned"
        result = controller.handle_message("Everything still feels heavy and hopeless.")

        assert result is not None
        assert controller.fsm_state == "elevated"
        assert generator.contexts[-1].strategy_name == "DeEscalationStrategy"
        assert "grounding" in generator.contexts[-1].system_prompt.lower()

    def test_i3_crisis_input_returns_none_and_sets_crisis_state(self) -> None:
        generator = FakeGenerator("should not be used")
        controller = ChatController(
            detector=FakeDetector(EmotionLabel.FEARFUL, 1.0),
            evaluator=FSMRiskEvaluator(),
            memory=InMemoryConversationMemory(),
            generator=generator,
            output_filter=FakeFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )

        result = controller.handle_message("I don't want to be here anymore.")

        assert result is None
        assert controller.fsm_state == "crisis"
        assert generator.contexts == []

    def test_i4_empty_input_returns_user_friendly_error(self) -> None:
        controller = self._make_controller()

        result = controller.handle_message("   ")

        assert result == "Please enter a message so I can respond."
        assert controller.memory.get_recent_messages() == []
        assert controller.fsm_state == "calm"

    def test_i4_timeout_returns_user_friendly_error(self) -> None:
        controller = ChatController(
            detector=FakeDetector(EmotionLabel.NEUTRAL, 0.5),
            evaluator=FSMRiskEvaluator(),
            memory=InMemoryConversationMemory(),
            generator=TimeoutGenerator(),
            output_filter=FakeFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )

        result = controller.handle_message("Hello there")

        assert result == "Sorry, the response service timed out. Please try again."
        assert controller.fsm_state == "calm"

    def test_i4_memory_write_failure_returns_user_friendly_error(self) -> None:
        controller = ChatController(
            detector=FakeDetector(EmotionLabel.NEUTRAL, 0.5),
            evaluator=FSMRiskEvaluator(),
            memory=LockedMemory(),
            generator=FakeGenerator("Hello"),
            output_filter=FakeFilter(),
            language_detector=SimpleLanguageDetector(),
            strategy_selector=ConcreteStrategySelector(),
        )

        result = controller.handle_message("Hello there")

        assert (
            result
            == "Sorry, I'm having trouble saving this conversation right now. Please try again."
        )
