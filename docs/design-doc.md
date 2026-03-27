# SafeHaven — Design Document

> **Course Project — Safety-Aware Mental Health Chatbot**
> Version 2.1 · March 2026 — revised for 2-week sprint

---

## 1. Project Overview

**SafeHaven** is a desktop chatbot that provides empathetic conversational support while actively monitoring for signs of emotional distress or crisis. It uses an API-based LLM (Anthropic Claude) for response generation, layered behind a safety pipeline that analyzes emotion, evaluates risk, and filters output.

> **Current vs Target:** The architecture below describes the full target design. The current implementation uses a subset of this pipeline: `UI (Kivy) → EmotionDetector → KeywordRiskEvaluator → ResponseGenerator → OutputFilter → UI`. Components marked *(stub)* below are defined but raise `NotImplementedError`.

### Scope

| In Scope | Out of Scope |
|----------|--------------|
| English + Arabic text chat | Other languages |
| Desktop GUI (Kivy) | Web or mobile deployment |
| API-based LLM (Anthropic Claude) | Fine-tuned models |
| Local LLM option (Ollama) | Cloud-hosted local models |
| FSM-based risk detection | Clinical-grade NLP |
| Strategy-driven response generation | Hardcoded response logic |
| SQLite conversation storage | Cloud database / auth |
| Crisis resource display | Actual crisis intervention |
| Emotional insights dashboard | Real-time analytics |
| 3 demo scenarios | Production deployment |

**Important disclaimer:** SafeHaven is a course project demonstrating safety-aware architecture. It is **not** a clinical tool and must not be used as a substitute for professional mental health support.

---

## 2. Architecture Overview

### Layered Pipeline

```
┌─────────────────────────────────────────────────────┐
│                   PRESENTATION                       │
│                Kivy ScreenManager                    │
│   (welcome, chat, crisis, insights screens)          │
└──────────────────────┬──────────────────────────────┘
                       │ user text
                       ▼
┌─────────────────────────────────────────────────────┐
│                 LANGUAGE DETECTION                    │
│         LanguageDetector.detect_language(text)        │
│              → 'en' | 'ar'                           │
└──────────────────────┬──────────────────────────────┘
                       │ language code
                       ▼
┌─────────────────────────────────────────────────────┐
│                   PROCESSING                         │
│   ┌─────────────────┐   ┌────────────────────┐      │
│   │ EmotionDetector  │   │ ConversationMemory │      │
│   │ detect(text)     │   │ store / retrieve   │      │
│   └────────┬────────┘   └────────┬───────────┘      │
│            │ EmotionResult       │ recent messages   │
│            ▼                     ▼                   │
│         ┌──────────────────────────┐                 │
│         │       UserState          │                 │
│         └────────────┬─────────────┘                 │
└──────────────────────┼──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│                    DECISION                          │
│           FSMRiskEvaluator (stateful)                │
│     evaluate(UserState) → RiskLevel                  │
│     FSM: CALM → CONCERNED → ELEVATED → CRISIS       │
└──────────────────────┬──────────────────────────────┘
                       │ RiskLevel + FSM state
                       ▼
┌─────────────────────────────────────────────────────┐
│               STRATEGY SELECTION                     │
│     StrategySelector.select(risk, fsm_state)         │
│        → ResponseStrategy                            │
│                                                      │
│     CALM/CONCERNED → SupportiveStrategy              │
│     ELEVATED       → DeEscalationStrategy            │
│     CRISIS         → CrisisStrategy                  │
└──────────────────────┬──────────────────────────────┘
                       │ strategy
                       ▼
              ┌────────┴────────┐
              │                 │
         LOW / MEDIUM          HIGH
              │                 │
              ▼                 ▼
┌─────────────────────┐  ┌──────────────┐
│     GENERATION      │  │ CRISIS PATH  │
│  ResponseGenerator  │  │ Lock input,  │
│  generate(context)  │  │ show crisis  │
│  (system prompt     │  │ screen with  │
│   from strategy)    │  │ hotlines     │
└─────────┬───────────┘  └──────────────┘
          │ raw response
          ▼
┌─────────────────────────────────────────────────────┐
│                   VALIDATION                         │
│   OutputFilter.validate(response, risk) → str        │
│   + Strategy.post_process(response) → str            │
│   (strip harmful content, enforce tone)              │
└──────────────────────┬──────────────────────────────┘
                       │ safe response
                       ▼
┌─────────────────────────────────────────────────────┐
│                   PRESENTATION                       │
│   Display response in chat (emotion-colored bubble)  │
└─────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Purpose | Key Rule | Status |
|-------|---------|----------|--------|
| **Presentation** | Renders UI, captures input | Never calls LLM directly | ✅ Implemented (Kivy, 4 screens) |
| **Language Detection** | Identifies input language | Runs before emotion detection | ❌ Stub — `detect_language()` raises `NotImplementedError` |
| **Processing** | Extracts emotion, stores messages | Stateless detection, stateful storage | ✅ Implemented |
| **Decision** | Evaluates risk from UserState | Single source of risk truth | ⚠️ `KeywordRiskEvaluator` active; `FSMRiskEvaluator.evaluate()` stubbed |
| **Strategy Selection** | Picks response strategy by FSM state | Decouples strategy from controller logic | ❌ Stub — `select()` raises `NotImplementedError`; not wired in controller |
| **Generation** | Calls LLM API with strategy-built prompt | Only called if risk ≤ MEDIUM | ✅ Implemented (Claude + Ollama) |
| **Validation** | Filters LLM output + strategy post-processing | Last gate before display | ✅ Implemented |

---

## 3. Data Models

All models live in `safehaven/models.py`.

```python
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class FSMState(Enum):
    CALM = "calm"
    CONCERNED = "concerned"
    ELEVATED = "elevated"
    CRISIS = "crisis"


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
    role: str              # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    emotion: EmotionLabel | None = None
    risk_level: RiskLevel = RiskLevel.LOW
    language: str = "en"


@dataclass
class UserState:
    current_emotion: EmotionResult
    risk_level: RiskLevel
    message_count: int
    escalation_history: list[RiskLevel] = field(default_factory=list)
    language: str = "en"
    fsm_state: str = "calm"


@dataclass
class ConversationContext:
    recent_messages: list[Message]
    user_state: UserState
    system_prompt: str = ""
    strategy_name: str = ""

    def to_llm_messages(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.recent_messages]
```

---

## 4. Module Interfaces

Each module is defined as a Python Protocol so implementations can be swapped (real vs. stub). All interfaces live in `safehaven/interfaces.py`.

```python
from typing import Protocol
from safehaven.models import (
    EmotionResult, RiskLevel, UserState,
    Message, ConversationContext,
)


class EmotionDetector(Protocol):
    def detect(self, text: str) -> EmotionResult:
        """Analyze text and return the dominant emotion with confidence."""
        ...


class RiskEvaluator(Protocol):
    def evaluate(self, state: UserState) -> RiskLevel:
        """Determine risk level from current user state."""
        ...


class ConversationMemory(Protocol):
    def store_message(self, message: Message) -> None:
        """Persist a message to storage."""
        ...

    def get_recent_messages(self, limit: int = 10) -> list[Message]:
        """Retrieve the N most recent messages, oldest first."""
        ...

    def clear(self) -> None:
        """Clear all stored messages (new session)."""
        ...


class ResponseGenerator(Protocol):
    def generate(self, context: ConversationContext) -> str:
        """Call the LLM API and return the raw response text."""
        ...


class OutputFilter(Protocol):
    def validate(self, response: str, risk: RiskLevel) -> str:
        """Sanitize LLM output based on current risk level."""
        ...


class LanguageDetector(Protocol):
    def detect_language(self, text: str) -> str:
        """Return ISO 639-1 language code ('en', 'ar')."""
        ...


class ResponseStrategy(Protocol):
    def build_system_prompt(self, context: ConversationContext) -> str:
        """Return a system prompt tailored to current risk/emotion state."""
        ...

    def post_process(self, response: str) -> str:
        """Optional post-processing of LLM output."""
        ...


class StrategySelector(Protocol):
    def select(self, risk: RiskLevel, fsm_state: str) -> ResponseStrategy:
        """Pick appropriate strategy based on FSM state."""
        ...
```

### ChatController — Orchestrator

> **Note:** The code below reflects the current implementation. The target pipeline will add `LanguageDetector` and `StrategySelector` calls (see architecture diagram above).

```python
class ChatController:
    """Orchestrates the full message pipeline.

    Owns no business logic — delegates to injected modules.
    """

    def __init__(
        self,
        detector: EmotionDetector,
        evaluator: RiskEvaluator,
        memory: ConversationMemory,
        generator: ResponseGenerator,
        output_filter: OutputFilter,
    ) -> None:
        self.detector = detector
        self.evaluator = evaluator
        self.memory = memory
        self.generator = generator
        self.output_filter = output_filter

    def handle_message(self, user_text: str) -> str | None:
        """Process one user message through the full pipeline.

        Returns the assistant response, or None if crisis path activated.
        """
        # 1. Detect emotion
        emotion = self.detector.detect(user_text)

        # 2. Store user message
        user_msg = Message(role="user", content=user_text, emotion=emotion.label)
        self.memory.store_message(user_msg)

        # 3. Build user state
        state = UserState(
            current_emotion=emotion,
            risk_level=RiskLevel.LOW,
            message_count=len(self.memory.get_recent_messages()),
        )

        # 4. Evaluate risk
        risk = self.evaluator.evaluate(state)

        # 5. Crisis path
        if risk == RiskLevel.HIGH:
            return None  # Signal UI to show crisis screen

        # 6. Generate response
        context = ConversationContext(
            recent_messages=self.memory.get_recent_messages(),
            user_state=state,
        )
        raw_response = self.generator.generate(context)

        # 7. Filter output
        safe_response = self.output_filter.validate(raw_response, risk)

        # 8. Store assistant message
        assistant_msg = Message(
            role="assistant", content=safe_response, risk_level=risk
        )
        self.memory.store_message(assistant_msg)

        return safe_response
```

---

## 5. Folder Structure

```
safehaven/
├── main.py                      # Entry point — launches UI
├── models.py                    # Data models (Section 3)
├── interfaces.py                # Protocol definitions (Section 4)
├── logging_config.py            # Structured logging setup
├── ui/
│   ├── __init__.py
│   ├── app.py                   # Kivy App + ScreenManager
│   ├── welcome_screen.py        # Splash/welcome screen
│   ├── chat_screen.py           # Main chat interface
│   ├── crisis_screen.py         # Crisis resource display
│   ├── insights_screen.py       # Emotional trends dashboard (placeholder)
│   └── theme.py                 # Colors, emotion-to-color map
├── memory/
│   ├── __init__.py
│   ├── sqlite_memory.py         # ConversationMemory impl (SQLite)
│   └── schema.sql               # CREATE TABLE statements
├── safety/
│   ├── __init__.py
│   ├── emotion_detector.py      # EmotionDetector impl
│   ├── risk_evaluator.py        # RiskEvaluator impl (keyword-based)
│   ├── fsm_risk_evaluator.py    # FSM RiskEvaluator impl (stateful)
│   ├── language_detector.py     # LanguageDetector impl
│   └── output_filter.py         # OutputFilter impl
├── llm/
│   ├── __init__.py
│   ├── claude_generator.py      # ResponseGenerator impl (Anthropic Claude)
│   └── local_generator.py       # ResponseGenerator impl (Ollama, local)
├── strategy/
│   ├── __init__.py
│   ├── base.py                  # ConcreteStrategySelector
│   ├── supportive.py            # SupportiveStrategy (CALM/CONCERNED)
│   ├── de_escalation.py         # DeEscalationStrategy (ELEVATED)
│   └── crisis.py                # CrisisStrategy (CRISIS)
├── controller/
│   ├── __init__.py
│   └── chat_controller.py       # ChatController (orchestrator)
├── tests/
│   ├── __init__.py
│   ├── test_emotion.py
│   ├── test_risk.py
│   ├── test_filter.py
│   ├── test_memory.py
│   ├── test_controller.py       # Integration test with stubs
│   ├── test_fsm.py              # FSM transition tests
│   ├── test_strategy.py         # Strategy pattern tests
│   ├── test_language.py         # Language detection tests
│   └── test_local_generator.py  # Local model tests
├── resources/
│   ├── crisis_hotlines.json     # Country → hotline mapping
│   ├── crisis_keywords.txt      # English crisis keywords
│   ├── crisis_keywords_ar.txt   # Arabic crisis keywords
│   ├── emotion_keywords_ar.json # Arabic emotion word sets
│   └── safehaven.kv             # Unused Kivy DSL skeleton (UI is built in Python)
CLAUDE.md                            # Shared LLM context for team (repo root)
```

---

## 6. Sprint Plan — 2-Week Task Board

> **Context:** Weeks 1–4 of the original plan are complete. The codebase has working infrastructure (models, interfaces, controller, UI, memory, emotion detection, keyword risk, output filter, Claude/Ollama generators). The remaining work is implementing the three stubbed modules, wiring them into the controller, and preparing the demo and report.

> **Principle:** Each task has a binary "done" check. Aim for ~3 hours per task. The critical path runs through Backend A — unblock it first.

---

### Team Assignments

| Group | People | Owns |
|-------|--------|------|
| **Backend A** | 1 person | `FSMRiskEvaluator`, `LanguageDetector`, wire `ChatController` |
| **Backend B** | 1 person | `SupportiveStrategy`, `DeEscalationStrategy`, `CrisisStrategy`, `ConcreteStrategySelector` |
| **Frontend** | 1 person | `InsightsScreen`, emotion-colored bubbles, Arabic layout hint |
| **Data & Tests** | 2 people | Keyword lists, unit tests for new modules, demo script |
| **Docs** | 1 person | UML diagrams, report assembly, presentation slides |

> Backend A and B can work in parallel once they agree on Day 1 that `fsm_state` is passed as the plain string values defined in `UserState` (`"calm"`, `"concerned"`, `"elevated"`, `"crisis"`). Frontend and Data/Tests are fully independent from Day 1.

---

### Dependency / Critical Path

```
Backend A: FSMRiskEvaluator
                ↓
Backend B: ConcreteStrategySelector  (needs FSM state string contract)
                ↓
Backend A: Wire ChatController        (needs both FSM + strategies done)
                ↓
Data/Tests: Integration tests         (needs wired controller)
```

Frontend → InsightsScreen depends on `ConversationMemory` (already done) — no backend dependency.

---

### Week 1 — Implement Stubs + Wire Controller

#### Backend A

| # | Task | Done When |
|---|------|-----------|
| A1 | Implement `FSMRiskEvaluator.evaluate()` — CALM→CONCERNED→ELEVATED→CRISIS transitions | `test_fsm.py` passes: single negative emotion → CONCERNED; 3 consecutive → ELEVATED; crisis keyword → CRISIS |
| A2 | Implement `SimpleLanguageDetector.detect_language()` — Unicode Arabic script check | `detect("مرحبا")` → `'ar'`, `detect("hello")` → `'en'` |
| A3 | Wire `ChatController`: inject `LanguageDetector` + `FSMRiskEvaluator` + `StrategySelector`; set language on `UserState`; call `StrategySelector` to build system prompt | `test_controller.py` passes with all 3 demo scenarios end-to-end (mocked LLM) |

#### Backend B

| # | Task | Done When |
|---|------|-----------|
| B1 | Implement `SupportiveStrategy.build_system_prompt()` — warm, empathetic prompt for CALM/CONCERNED | Returns a non-empty system prompt string; `test_strategy.py` passes |
| B2 | Implement `DeEscalationStrategy.build_system_prompt()` — grounding, safety-aware prompt for ELEVATED | Returns a non-empty system prompt string; `test_strategy.py` passes |
| B3 | Implement `CrisisStrategy.build_system_prompt()` — minimal LLM framing for CRISIS state | Returns a non-empty system prompt string; `test_strategy.py` passes |
| B4 | Implement `ConcreteStrategySelector.select()` — maps `fsm_state` string to the correct strategy | `select(HIGH, "crisis")` → `CrisisStrategy`; `select(MEDIUM, "elevated")` → `DeEscalationStrategy`; `select(LOW, "calm")` → `SupportiveStrategy` |

#### Frontend

| # | Task | Done When |
|---|------|-----------|
| F1 | Build `InsightsScreen` — emotion timeline from `ConversationMemory` (bar or list of past emotion labels + counts) | Screen shows real data from SQLite on navigation; no crashes on empty history |
| F2 | Emotion-colored message bubbles in `chat_screen.py` — map `EmotionLabel` to bubble background via `theme.py` | SAD messages appear in a distinct color from HAPPY; ANXIOUS/FEARFUL differ visually |
| F3 | Arabic RTL layout hint — if `language == 'ar'`, set `base_direction='rtl'` on the input and bubble labels | Arabic text is right-aligned in chat |

#### Data & Tests

| # | Task | Done When |
|---|------|-----------|
| D1 | Expand `crisis_keywords.txt` to 50+ entries; peer-review for false positives | File has 50+ lines; running `pytest test_risk.py` still passes |
| D2 | Expand `crisis_keywords_ar.txt` — equivalent Arabic phrases | File has 20+ Arabic entries; spot-checked by a native speaker or translation tool |
| D3 | Fill in `test_fsm.py` — edge cases: empty input, neutral emotion (no transition), rapid escalation, `clear()` resets state | All test cases pass |
| D4 | Fill in `test_strategy.py` — each strategy returns a non-empty prompt; selector maps all 4 FSM states correctly | All test cases pass |
| D5 | Fill in `test_language.py` — English, Arabic, mixed, empty string | All test cases pass |

---

### Week 2 — Integration, Testing, Demo, Documentation

#### Backend A + B (together)

| # | Task | Done When |
|---|------|-----------|
| I1 | End-to-end integration test: Scenario 1 (normal conversation, 3 turns, LOW risk) | Passes with mocked LLM; FSM stays CALM; SupportiveStrategy prompt used |
| I2 | End-to-end integration test: Scenario 2 (distressed user, MEDIUM risk, CONCERNED→ELEVATED) | Passes with mocked LLM; FSM advances; DeEscalationStrategy prompt used |
| I3 | End-to-end integration test: Scenario 3 (crisis detection, HIGH risk) | `handle_message()` returns `None`; UI navigates to crisis screen |
| I4 | Error handling: API timeout, empty input, SQLite lock | App shows user-friendly error message; no crash or unhandled exception |

#### Frontend

| # | Task | Done When |
|---|------|-----------|
| F4 | UI polish: timestamps on messages, "thinking…" indicator while LLM call is in-flight | Timestamps visible; spinner or label shown between send and response |
| F5 | Manual smoke test: run all 3 demo scenarios live with real API key | Screenshots captured for `docs/demo/` |

#### Data & Tests

| # | Task | Done When |
|---|------|-----------|
| D6 | Write demo script — exact user inputs and expected outputs for all 3 scenarios | Markdown file committed to `docs/demo-script.md` |
| D7 | Adversarial output filter tests — 5+ LLM responses containing harmful patterns | `test_filter.py` strips all harmful patterns; all pass |
| D8 | `mypy --strict safehaven/` passes with zero errors after all new code | CI check green |

#### Docs

| # | Task | Done When |
|---|------|-----------|
| W1 | Final UML class diagram (PlantUML or draw.io) — reflects actual wired code | PNG/SVG in `docs/`; all 6 design patterns annotated |
| W2 | Sequence diagram for `handle_message()` through the full target pipeline | PNG/SVG in `docs/` |
| W3 | Design patterns writeup — 1 page covering Strategy, FSM, Pipeline, Observer, Repository, DI | Committed to `docs/report/` |
| W4 | Individual report sections — each person writes ~1 page on their contribution | Sections in `docs/report/` |
| W5 | Presentation slides (5–8 slides): architecture, patterns, demo, learnings | Dry-run rehearsal done; everyone knows their part |
| W6 | Final report assembly + proofread | Single PDF ready for submission |

---

## 7. Design Patterns Used

| Pattern | Where | Status | Why |
|---------|-------|--------|-----|
| **Strategy** | `StrategySelector` picks `ResponseStrategy` by FSM state; strategies: `SupportiveStrategy`, `DeEscalationStrategy`, `CrisisStrategy` | ❌ Classes exist, all `build_system_prompt()` / `select()` raise `NotImplementedError`; not wired in controller | Swap response behavior (system prompt + post-processing) without changing controller logic |
| **Finite State Machine** | `FSMRiskEvaluator` manages states: CALM → CONCERNED → ELEVATED → CRISIS | ❌ Class exists, `evaluate()` raises `NotImplementedError`; `KeywordRiskEvaluator` is the active evaluator | Stateful risk tracking across turns; forward-only transitions prevent premature de-escalation |
| **Pipeline** | `ChatController.handle_message()` — EmotionDetector → RiskEvaluator → Generator → Filter | ⚠️ Implemented with partial pipeline (no LanguageDetector, no StrategySelector yet) | Each stage transforms data for the next; easy to insert/reorder |
| **Observer** | UI ← Controller (callback on response) | ✅ Implemented | Decouples UI from business logic |
| **Repository** | `ConversationMemory` backed by SQLite | ✅ Implemented | Abstracts storage (SQLite today, anything tomorrow) |
| **Dependency Injection** | Controller accepts Protocol-typed dependencies | ✅ Implemented | Easy testing with mocks, swappable implementations |

### FSM State Diagram

```
          negative emotion         sustained negative        crisis keyword
  ┌──────┐  (confidence > 0.6)  ┌───────────┐  (3+ turns) ┌──────────┐         ┌────────┐
  │ CALM │ ──────────────────→  │ CONCERNED │ ──────────→  │ ELEVATED │ ──────→ │ CRISIS │
  └──────┘                      └───────────┘              └──────────┘         └────────┘
     │                               │                          │                    │
     └── neutral emotion ←───────────┘ (no de-escalation)       │                    │
         stays in CALM               stays or advances          stays or advances    terminal
```

**Transition rules:**
- CALM → CONCERNED: Negative emotion detected with confidence > 0.6
- CONCERNED → ELEVATED: 3+ consecutive negative emotion turns
- ELEVATED → CRISIS: Crisis keyword detected or rapid escalation
- CRISIS: Terminal state for the session (reset only via `clear()`)
- No backward transitions within a session

---

## 8. Demo Scenarios

### Scenario 1 — Normal Conversation (LOW risk, CALM state)

```
User:  "Hi! I had a pretty good day today."
       → LanguageDetector: 'en'
       → EmotionDetector: HAPPY (0.85)
       → FSMRiskEvaluator: CALM → stays CALM
       → StrategySelector: SupportiveStrategy
       → RiskLevel: LOW
       → LLM generates friendly response (supportive prompt)
       → OutputFilter: passes through unchanged

Bot:   "That's great to hear! What made your day good?"
```

### Scenario 2 — Distressed User (MEDIUM risk, CONCERNED → ELEVATED)

```
User:  "I've been feeling really overwhelmed lately. Nothing seems to work out."
       → LanguageDetector: 'en'
       → EmotionDetector: SAD (0.78)
       → FSMRiskEvaluator: CALM → CONCERNED (negative emotion, high confidence)
       → StrategySelector: SupportiveStrategy
       → RiskLevel: MEDIUM
       → LLM generates response with empathetic system prompt
       → OutputFilter: prepends empathetic framing

Bot:   "It sounds like you're going through a really tough time, and that's
        completely valid. Would you like to talk about what's been weighing
        on you most?"
```

### Scenario 3 — Crisis Detection (HIGH risk, CRISIS state)

```
User:  "I don't want to be here anymore. I want to end it all."
       → LanguageDetector: 'en'
       → EmotionDetector: FEARFUL (0.92)
       → FSMRiskEvaluator: → CRISIS (crisis keyword match)
       → StrategySelector: CrisisStrategy
       → RiskLevel: HIGH
       → ChatController returns None
       → UI shows crisis screen

┌─────────────────────────────────────────┐
│        ⚠ We Care About You              │
│                                         │
│  It sounds like you may be in crisis.   │
│  Please reach out to a professional:    │
│                                         │
│  • 988 Suicide & Crisis Lifeline        │
│    Call or text: 988                     │
│  • Crisis Text Line                     │
│    Text HOME to 741741                  │
│  • International Association for        │
│    Suicide Prevention:                  │
│    https://www.iasp.info/resources/     │
│                                         │
│  [ I understand — continue chatting ]   │
└─────────────────────────────────────────┘
```

---

## 9. Interface Compatibility Checklist

| Producer | Output | Consumer | Input | Match? |
|----------|--------|----------|-------|--------|
| `LanguageDetector.detect_language()` | `str` ('en'/'ar') | `UserState.language` | `str` | ✓ |
| `EmotionDetector.detect()` | `EmotionResult` | `UserState` constructor | `EmotionResult` | ✓ |
| `UserState` | `UserState` | `RiskEvaluator.evaluate()` | `UserState` | ✓ |
| `RiskEvaluator.evaluate()` | `RiskLevel` | `StrategySelector.select()` | `RiskLevel` | ✓ |
| `FSMRiskEvaluator.state` | `str` | `StrategySelector.select()` | `str` (fsm_state) | ✓ |
| `StrategySelector.select()` | `ResponseStrategy` | `ResponseStrategy.build_system_prompt()` | self | ✓ |
| `ConversationMemory.get_recent_messages()` | `list[Message]` | `ConversationContext` | `list[Message]` | ✓ |
| `ConversationContext.to_llm_messages()` | `list[dict]` | Claude API | `list[dict]` | ✓ |
| `ResponseStrategy.build_system_prompt()` | `str` | `ConversationContext.system_prompt` | `str` | ✓ |
| `ResponseGenerator.generate()` | `str` | `OutputFilter.validate()` | `str` | ✓ |
| `ResponseStrategy.post_process()` | `str` | Final response | `str` | ✓ |
| `RiskEvaluator.evaluate()` | `RiskLevel` | `OutputFilter.validate()` | `RiskLevel` | ✓ |
| `ChatController.handle_message()` | `str \| None` | UI callback | `str \| None` | ✓ |

---

*End of design document.*
