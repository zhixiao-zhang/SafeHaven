"""ChatController — orchestrates the full message pipeline."""

from __future__ import annotations

from safehaven.interfaces import (
    ConversationMemory,
    EmotionDetector,
    LanguageDetector,
    OutputFilter,
    ResponseGenerator,
    RiskEvaluator,
    StrategySelector,
)
from safehaven.models import (
    ConversationContext,
    EmotionLabel,
    Message,
    RiskLevel,
    UserState,
)


class ChatController:
    """Orchestrates the full message pipeline.

    Owns no business logic — delegates to injected modules.
    Implements the Stateful Safety Pipeline (SSP) architecture.
    """

    def __init__(
        self,
        detector: EmotionDetector,
        evaluator: RiskEvaluator,
        memory: ConversationMemory,
        generator: ResponseGenerator,
        output_filter: OutputFilter,
        language_detector: LanguageDetector | None = None,
        strategy_selector: StrategySelector | None = None,
    ) -> None:
        self.detector = detector
        self.evaluator = evaluator
        self.memory = memory
        self.generator = generator
        self.output_filter = output_filter
        self.language_detector = language_detector
        self.strategy_selector = strategy_selector
        self._last_emotion: EmotionLabel | None = None

    @property
    def fsm_state(self) -> str:
        """Current FSM state label — used by UI for color indicator."""
        if hasattr(self.evaluator, "state"):
            state = getattr(self.evaluator, "state")
            if isinstance(state, str):
                return state
        return "calm"

    @property
    def last_emotion(self) -> EmotionLabel | None:
        """Emotion detected on the last user message — used by UI for bubble color."""
        return self._last_emotion

    def handle_message(self, user_text: str) -> str | None:
        """Process one user message through the full SSP pipeline.

        Returns the assistant response, or None if crisis path activated.
        """
        if not user_text.strip():
            return "Please enter a message so I can respond."

        # 1. Detect language
        language = "en"
        if self.language_detector is not None:
            try:
                language = self.language_detector.detect_language(user_text)
            except Exception:
                language = "en"

        # 2. Detect emotion
        try:
            emotion = self.detector.detect(user_text)
        except Exception:
            return "Sorry, I couldn't analyze that message. Please try again."
        self._last_emotion = emotion.label

        # 3. Store user message
        user_msg = Message(
            role="user", content=user_text, emotion=emotion.label, language=language
        )
        try:
            self.memory.store_message(user_msg)
        except Exception:
            return "Sorry, I'm having trouble saving this conversation right now. Please try again."

        # 4. Build user state (include escalation history from memory)
        try:
            recent = self.memory.get_recent_messages()
        except Exception:
            return "Sorry, I'm having trouble loading the conversation history right now. Please try again."
        escalation_history = [
            m.risk_level for m in recent if m.role == "user"
        ]
        state = UserState(
            current_emotion=emotion,
            risk_level=RiskLevel.LOW,
            message_count=len(recent),
            escalation_history=escalation_history,
            language=language,
            fsm_state=self.fsm_state,
        )

        # 5. Evaluate risk
        try:
            risk = self.evaluator.evaluate(state)
        except Exception:
            return "Sorry, I couldn't evaluate that message safely. Please try again."

        # 6. Crisis path
        if risk == RiskLevel.HIGH:
            return None  # Signal UI to show crisis screen

        # 7. Select strategy and build system prompt
        system_prompt = ""
        strategy_name = ""
        if self.strategy_selector is not None:
            try:
                strategy = self.strategy_selector.select(risk, self.fsm_state)
                strategy_name = type(strategy).__name__
                context_for_prompt = ConversationContext(
                    recent_messages=recent,
                    user_state=state,
                )
                system_prompt = strategy.build_system_prompt(context_for_prompt)
            except Exception:
                return "Sorry, I couldn't prepare a safe response strategy. Please try again."

        # 8. Generate response
        context = ConversationContext(
            recent_messages=recent,
            user_state=state,
            system_prompt=system_prompt,
            strategy_name=strategy_name,
        )
        try:
            raw_response = self.generator.generate(context)
        except TimeoutError:
            return "Sorry, the response service timed out. Please try again."
        except Exception:
            return "Sorry, I couldn't generate a response right now. Please try again."

        # 9. Filter output
        try:
            safe_response = self.output_filter.validate(raw_response, risk)
        except Exception:
            return "Sorry, I couldn't validate the response safely. Please try again."

        # 10. Post-process via strategy
        if self.strategy_selector is not None:
            try:
                strategy = self.strategy_selector.select(risk, self.fsm_state)
                safe_response = strategy.post_process(safe_response)
            except Exception:
                return "Sorry, I couldn't finalize the response safely. Please try again."

        # 11. Store assistant message
        assistant_msg = Message(
            role="assistant", content=safe_response, risk_level=risk
        )
        try:
            self.memory.store_message(assistant_msg)
        except Exception:
            return "Sorry, I'm having trouble saving this conversation right now. Please try again."

        return safe_response

    def clear(self) -> None:
        """Reset session state — clears memory and FSM."""
        self.memory.clear()
        clear_fn = getattr(self.evaluator, "clear", None)
        if callable(clear_fn):
            clear_fn()
        self._last_emotion = None
