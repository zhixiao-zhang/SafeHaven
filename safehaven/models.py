from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EmotionLabel(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANXIOUS = "anxious"
    ANGRY = "angry"
    FEARFUL = "fearful"


class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


@dataclass
class EmotionResult:
    label: EmotionLabel
    confidence: float  # 0.0 – 1.0


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    emotion: EmotionLabel | None = None
    risk_level: RiskLevel = RiskLevel.LOW


@dataclass
class UserState:
    current_emotion: EmotionResult
    risk_level: RiskLevel
    message_count: int
    escalation_history: list[RiskLevel] = field(default_factory=list)


@dataclass
class ConversationContext:
    recent_messages: list[Message]
    user_state: UserState
    system_prompt: str = ""

    def to_llm_messages(self) -> list[dict[str, str]]:
        """Format as user/assistant message dicts for the Claude API.

        The system prompt is passed separately via the API's ``system``
        parameter; see ``ClaudeResponseGenerator.generate``.
        """
        return [{"role": m.role, "content": m.content} for m in self.recent_messages]
