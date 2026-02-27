# SafeHaven — CLAUDE.md

> Place this file at the root of your repository so every Claude Code session has consistent context.

## Project

SafeHaven is a safety-aware mental health chatbot (Python, Tkinter, Anthropic Claude API). Course project — not for production use.

## Architecture

Layered pipeline: **UI → EmotionDetector → RiskEvaluator → ResponseGenerator → OutputFilter → UI**

Orchestrated by `ChatController.handle_message(user_text) -> str | None` (returns `None` on crisis → UI shows modal).

## Key Conventions

- Python 3.11+
- Type hints everywhere, `mypy --strict` must pass
- Modules depend on **Protocols** from `interfaces.py`, not concrete classes
- All data models in `models.py` — use dataclasses, not dicts
- Tests use `pytest`; mock the LLM in all automated tests
- API keys in `.env` (never committed)
- Crisis keywords in `resources/crisis_keywords.txt` (one per line)

## Folder Structure

```
safehaven/
├── main.py              # Entry point
├── models.py            # Message, EmotionResult, RiskLevel, UserState, ConversationContext
├── interfaces.py        # Protocol classes for all modules
├── ui/                  # Tkinter chat window + crisis modal
├── memory/              # SQLite ConversationMemory implementation
├── safety/              # EmotionDetector, RiskEvaluator, OutputFilter
├── llm/                 # ResponseGenerator (Anthropic Claude API wrapper)
├── controller/          # ChatController (orchestrator)
├── tests/               # pytest tests
└── resources/           # crisis_keywords.txt, crisis_hotlines.json
```

## Data Models (from `models.py`)

```python
class EmotionLabel(Enum):   # NEUTRAL, HAPPY, SAD, ANXIOUS, ANGRY, FEARFUL
class RiskLevel(Enum):      # LOW=1, MEDIUM=2, HIGH=3
class EmotionResult:        # label: EmotionLabel, confidence: float
class Message:              # role, content, timestamp, emotion, risk_level
class UserState:            # current_emotion, risk_level, message_count, escalation_history
class ConversationContext:  # recent_messages, user_state, system_prompt; has to_llm_messages()
```

## Module Interfaces (from `interfaces.py`)

```python
class EmotionDetector(Protocol):
    def detect(self, text: str) -> EmotionResult: ...

class RiskEvaluator(Protocol):
    def evaluate(self, state: UserState) -> RiskLevel: ...

class ConversationMemory(Protocol):
    def store_message(self, message: Message) -> None: ...
    def get_recent_messages(self, limit: int = 10) -> list[Message]: ...
    def clear(self) -> None: ...

class ResponseGenerator(Protocol):
    def generate(self, context: ConversationContext) -> str: ...

class OutputFilter(Protocol):
    def validate(self, response: str, risk: RiskLevel) -> str: ...
```

## Pipeline Flow (in ChatController.handle_message)

1. `EmotionDetector.detect(user_text)` → `EmotionResult`
2. `ConversationMemory.store_message(user_msg)`
3. Build `UserState` from emotion + history
4. `RiskEvaluator.evaluate(state)` → `RiskLevel`
5. If `HIGH` → return `None` (crisis modal)
6. `ResponseGenerator.generate(context)` → raw string
7. `OutputFilter.validate(raw, risk)` → safe string
8. `ConversationMemory.store_message(assistant_msg)`
9. Return safe string to UI

## When Writing Code

- Import models from `safehaven.models`, interfaces from `safehaven.interfaces`
- New modules implement a Protocol — don't subclass, just match the signature
- Controller takes all dependencies via `__init__` (dependency injection)
- Never call the LLM directly from UI code
- All crisis-related logic goes through `RiskEvaluator`, not ad-hoc checks
- SQLite DB file: `safehaven.db` in project root (gitignored)
